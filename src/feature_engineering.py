"""Feature engineering for house price regression: domain features, generic generators, encoders,
imputation options and a temporal-feature design, all driven by ``FeatureEngineeringConfig``.

Pipeline built by :func:`build_fe_preprocessor` (every step is fitted on training data only and
re-fitted inside each CV fold, so nothing leaks from validation data)::

    temporal  -> TemporalFeatureGenerator   (no-op unless temporal.date_columns is set)
    validate  -> DomainValidator            (Stage 2 rules: types, impossible values -> NaN)
    engineer  -> DomainFeatureEngineer      (named domain features, see FEATURE_CATALOGUE)
    columns   -> ColumnTransformer
                   num : impute (median|mean|knn) -> transform (log|yeo-johnson|box-cox|none)
                         -> cap outliers -> GenericFeatureGenerator (interactions|polynomial|bins) -> scale
                   miss: MissingIndicator      (optional; 0/1 flags bypass outlier capping)
                   bin : impute                (0/1 domain flags bypass outlier capping)
                   cat : impute -> encode     (onehot|frequency|target|hash)

Feature selection lives in :mod:`src.feature_selection`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PowerTransformer, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from .fe_config import FeatureEngineeringConfig
from .models import LogTargetRegressor
from .preprocessing import DomainValidator, LogTransformer, OutlierCapper, PreprocessingConfig, _make_scaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Domain features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    description: str
    rationale: str
    kind: str = "numeric"      # "numeric" (imputed, transformed, capped, scaled) or "binary" (imputed only)
    log: bool = False


FEATURE_CATALOGUE: dict[str, FeatureSpec] = {
    "school_premium": FeatureSpec(
        "exp(school_quality_score / school_premium_scale)",
        "Buyers pay disproportionately more for the best school catchments; the school effect on log price "
        "steepens at the top of the scale. An exponential term lets a linear model express that premium."),
    "school_squared": FeatureSpec(
        "school_quality_score ** 2",
        "Simpler curvature alternative to school_premium; lets selection choose between two encodings of the "
        "same idea (they are rank-identical, which tests the redundancy filter)."),
    "neighbourhood_quality": FeatureSpec(
        "log(school_quality_score / crime_rate_in_area)",
        "Good schools and low crime describe one thing to a buyer: a desirable neighbourhood."),
    "size_per_bedroom": FeatureSpec(
        "property_size / number_of_bedrooms",
        "Spaciousness: the same area split into fewer rooms usually means larger, higher-end rooms.", log=True),
    "bath_per_bed": FeatureSpec(
        "number_of_bathrooms / number_of_bedrooms",
        "Bathroom-to-bedroom ratio is a standard luxury indicator (en-suites)."),
    "total_rooms": FeatureSpec(
        "number_of_bedrooms + number_of_bathrooms",
        "Overall size in rooms; a common listing summary figure."),
    "school_x_size": FeatureSpec(
        "school_quality_score * log(property_size)",
        "Interaction: families buying large homes may value school quality more."),
    "top_school": FeatureSpec(
        "1 if school_quality_score >= top_school_threshold",
        "Catchment effect: homes zoned for top-rated schools may carry a step premium.", kind="binary"),
    "new_build": FeatureSpec(
        "1 if age_of_property < new_build_max_age",
        "New-build premium: little maintenance and modern specification.", kind="binary"),
}


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add named domain features to a validated DataFrame. Stateless (formulas only).

    Missing or impossible inputs (already NaN after ``DomainValidator``) give NaN outputs, filled by
    the following imputation step. Division by zero and log of non-positive values give NaN, never inf.
    Output column order is deterministic: input columns, then ``features`` in the order given.
    """

    def __init__(self, features=None, school_premium_scale: float = 3.0, top_school_threshold: float = 8.5,
                 new_build_max_age: float = 5.0):
        self.features = features
        self.school_premium_scale = school_premium_scale
        self.top_school_threshold = top_school_threshold
        self.new_build_max_age = new_build_max_age

    def fit(self, X, y=None):
        unknown = [f for f in (self.features or []) if f not in FEATURE_CATALOGUE]
        if unknown:
            raise ValueError(f"Unknown engineered features: {unknown}; choose from {list(FEATURE_CATALOGUE)}")
        if self.school_premium_scale <= 0:
            raise ValueError("school_premium_scale must be > 0")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        out = a / b.replace(0, np.nan)
        return out.where(np.isfinite(out))

    @staticmethod
    def _safe_log(v: pd.Series) -> pd.Series:
        return np.log(v.where(v > 0))

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = X.copy()
        s = X["school_quality_score"]
        age = X["age_of_property"]
        formulas = {
            "school_premium": lambda: np.exp(s / self.school_premium_scale),
            "school_squared": lambda: s ** 2,
            "neighbourhood_quality": lambda: self._safe_log(self._ratio(s, X["crime_rate_in_area"])),
            "size_per_bedroom": lambda: self._ratio(X["property_size"], X["number_of_bedrooms"]),
            "bath_per_bed": lambda: self._ratio(X["number_of_bathrooms"], X["number_of_bedrooms"]),
            "total_rooms": lambda: X["number_of_bedrooms"] + X["number_of_bathrooms"],
            "school_x_size": lambda: s * self._safe_log(X["property_size"]),
            "top_school": lambda: (s >= self.top_school_threshold).astype(float).where(s.notna()),
            "new_build": lambda: (age < self.new_build_max_age).astype(float).where(age.notna()),
        }
        for f in self.features or []:
            X[f] = formulas[f]().astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + list(self.features or []), dtype=object)


# --------------------------------------------------------------------------- #
# 2. Generic, toggle-driven generators
# --------------------------------------------------------------------------- #
class GenericFeatureGenerator(BaseEstimator, TransformerMixin):
    """Pairwise interactions, polynomial powers and binning for chosen numeric columns.

    Runs on the transformed numeric block (after imputation/log/capping), so ``a__x__b`` is the
    product of transformed values (e.g. log size x school). Names are deterministic:
    ``a__x__b`` (interaction, pairs in input order), ``c__pow2`` (power), ``c__bin1``... (one-hot
    quantile/uniform/kmeans bins, first bin dropped to avoid collinearity). With every option off it
    passes data through unchanged.
    """

    def __init__(self, interactions=None, polynomial=None, degree: int = 2, binning=None, n_bins: int = 5,
                 strategy: str = "quantile"):
        self.interactions = interactions      # None | "all" | list of columns
        self.polynomial = polynomial          # None | list of columns
        self.degree = degree
        self.binning = binning                # None | list of columns
        self.n_bins = n_bins
        self.strategy = strategy

    def _cols(self, spec, available: list[str]) -> list[str]:
        if not spec:
            return []
        cols = available if spec == "all" else list(spec)
        missing = [c for c in cols if c not in available]
        if missing:
            raise ValueError(f"Generator columns not in the numeric block: {missing}")
        return cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        available = list(X.columns)
        self.feature_names_in_ = np.asarray(available, dtype=object)
        self.n_features_in_ = len(available)
        self.interaction_pairs_ = list(combinations(self._cols(self.interactions, available), 2))
        self.poly_cols_ = self._cols(self.polynomial, available)
        self.bin_cols_ = self._cols(self.binning, available)
        self.binners_ = {}
        for c in self.bin_cols_:
            kb = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy,
                                  quantile_method="averaged_inverted_cdf" if self.strategy == "quantile" else "warn")
            kb.fit(X[[c]])
            self.binners_[c] = kb
        return self

    def transform(self, X):
        check_is_fitted(self, "interaction_pairs_")
        X = pd.DataFrame(X).copy()
        new = {}
        for a, b in self.interaction_pairs_:
            new[f"{a}__x__{b}"] = X[a] * X[b]
        for c in self.poly_cols_:
            for k in range(2, self.degree + 1):
                new[f"{c}__pow{k}"] = X[c] ** k
        for c, kb in self.binners_.items():
            codes = kb.transform(X[[c]])[:, 0].astype(int)
            for i in range(1, int(kb.n_bins_[0])):
                new[f"{c}__bin{i}"] = (codes == i).astype(float)
        if new:
            X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "interaction_pairs_")
        names = list(self.feature_names_in_)
        names += [f"{a}__x__{b}" for a, b in self.interaction_pairs_]
        names += [f"{c}__pow{k}" for c in self.poly_cols_ for k in range(2, self.degree + 1)]
        names += [f"{c}__bin{i}" for c, kb in self.binners_.items() for i in range(1, int(kb.n_bins_[0]))]
        return np.asarray(names, dtype=object)


class ColumnPowerTransformer(BaseEstimator, TransformerMixin):
    """Yeo-Johnson or Box-Cox on selected columns only (a learned alternative to a fixed log)."""

    def __init__(self, columns=None, method: str = "yeo-johnson"):
        self.columns = columns
        self.method = method

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.columns_ = [c for c in (self.columns or []) if c in X.columns]
        if self.method == "box-cox" and self.columns_ and (X[self.columns_] <= 0).any().any():
            bad = [c for c in self.columns_ if (X[c] <= 0).any()]
            raise ValueError(f"Box-Cox needs strictly positive values; non-positive values in {bad}. "
                             "Use numeric_transform: yeo-johnson instead.")
        self.pt_ = PowerTransformer(method=self.method, standardize=False).fit(X[self.columns_]) \
            if self.columns_ else None
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        X = pd.DataFrame(X).copy()
        if self.pt_ is not None:
            data = X[self.columns_]
            if self.method == "box-cox":
                data = data.clip(lower=np.finfo(float).tiny)
            X[self.columns_] = self.pt_.transform(data)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def make_numeric_transform(method: str, columns: list[str]):
    if method == "log":
        return LogTransformer(columns=columns)
    if method in ("yeo-johnson", "box-cox"):
        return ColumnPowerTransformer(columns=columns, method=method)
    if method == "none":
        return "passthrough"
    raise ValueError(f"Unknown numeric_transform {method!r}")


# --------------------------------------------------------------------------- #
# 3. Categorical encoders (for when one-hot is not appropriate)
# --------------------------------------------------------------------------- #
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its share of the training rows. Unseen categories -> 0.

    Suits high-cardinality columns where one-hot would explode the width; leaks no target information.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in X.columns}
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = pd.DataFrame(X)
        return pd.DataFrame({f"{c}_freq": X[c].map(self.maps_[c]).astype(float).fillna(0.0) for c in X.columns},
                            index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


class HashingEncoder(BaseEstimator, TransformerMixin):
    """Hash ``column=value`` tokens into a fixed number of columns (constant width, no vocabulary).

    Suits very high-cardinality or open-ended categories (e.g. street names); collisions are the cost.
    """

    def __init__(self, n_features: int = 8):
        self.n_features = n_features

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.hasher_ = FeatureHasher(n_features=self.n_features, input_type="string", alternate_sign=False)
        return self

    def transform(self, X):
        check_is_fitted(self, "hasher_")
        X = pd.DataFrame(X)
        tokens = [[f"{c}={row[c]}" for c in X.columns] for _, row in X.iterrows()]
        mat = self.hasher_.transform(tokens).toarray()
        return pd.DataFrame(mat, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"hash_{i}" for i in range(self.n_features)], dtype=object)


class LogTargetEncoder(TargetEncoder):
    """scikit-learn's cross-fitted ``TargetEncoder`` applied to log(price).

    ``fit_transform`` uses internal K-fold cross-fitting, so a row's own price never contributes to its
    encoding (leakage-safe); ``transform`` uses encodings learned on all training rows. Log space
    matches the target the models are trained on.
    """

    def fit(self, X, y):
        return super().fit(X, np.log(np.asarray(y, float)))

    def fit_transform(self, X, y):
        return super().fit_transform(X, np.log(np.asarray(y, float)))


def make_encoder(fe: FeatureEngineeringConfig, config: PreprocessingConfig, categorical: list[str], seed: int = 42):
    method = fe.encoding.method
    if method == "onehot":
        return OneHotEncoder(categories=[config.categories[c] for c in categorical],
                             drop="first" if config.drop_first_category else None,
                             handle_unknown="ignore", sparse_output=False)
    if method == "frequency":
        return FrequencyEncoder()
    if method == "hash":
        return HashingEncoder(n_features=fe.encoding.hash_n_features)
    if method == "target":
        return LogTargetEncoder(target_type="continuous", smooth=fe.encoding.target_smoothing, cv=5,
                                shuffle=True, random_state=seed)
    raise ValueError(f"Unknown encoding {method!r}")


# --------------------------------------------------------------------------- #
# 4. Temporal features (design; not applicable to this dataset)
# --------------------------------------------------------------------------- #
class TemporalFeatureGenerator(BaseEstimator, TransformerMixin):
    """Turn date columns into model features; a documented no-op when ``date_columns`` is empty.

    This dataset has **no timestamp column**, so the reference run does not use it. It is included so
    the pipeline is ready for data that does (e.g. a sale date), with leakage-safe defaults:

    * ``days_since``: days between the date and a reference date fixed at fit time (the latest training
      date unless given), so new data is measured on the same scale.
    * ``month``, ``quarter``, ``dayofweek``, ``is_weekend``: seasonality.

    Each date column is replaced by ``<col>_<feature>`` columns. Grouped trends that need history
    (rolling averages, time since a previous sale) are in :func:`past_rolling_mean`, which only uses
    *earlier* rows to avoid leaking the future.
    """

    def __init__(self, date_columns=None, reference_date=None, features=("days_since", "month", "quarter",
                                                                           "is_weekend")):
        self.date_columns = date_columns
        self.reference_date = reference_date
        self.features = features

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        cols = list(self.date_columns or [])
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Date columns not found: {missing}")
        if cols:
            latest = max(pd.to_datetime(X[c], errors="coerce").max() for c in cols)
            self.reference_ = pd.Timestamp(self.reference_date) if self.reference_date else latest
        else:
            self.reference_ = None
            logger.debug("No temporal columns configured; TemporalFeatureGenerator is a no-op")
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = pd.DataFrame(X).copy()
        for c in self.date_columns or []:
            d = pd.to_datetime(X.pop(c), errors="coerce")
            calc = {
                "days_since": lambda d=d: (self.reference_ - d).dt.days.astype(float),
                "month": lambda d=d: d.dt.month.astype(float),
                "quarter": lambda d=d: d.dt.quarter.astype(float),
                "dayofweek": lambda d=d: d.dt.dayofweek.astype(float),
                "is_weekend": lambda d=d: (d.dt.dayofweek >= 5).astype(float).where(d.notna()),
            }
            for f in self.features:
                X[f"{c}_{f}"] = calc[f]()
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        dates = set(self.date_columns or [])
        return np.asarray([c for c in self.feature_names_in_ if c not in dates] + self.output_names(), dtype=object)

    def output_names(self) -> list[str]:
        return [f"{c}_{f}" for c in (self.date_columns or []) for f in self.features]


def past_rolling_mean(df: pd.DataFrame, date_col: str, group_col: str, value_col: str, window: str = "365D"
                      ) -> pd.Series:
    """Mean of ``value_col`` over the previous ``window`` within each group, using only strictly earlier
    rows (``closed="left"``), e.g. the average sale price in the same neighbourhood over the past year.
    The current row's own value is never included, so the feature is safe to use as a predictor."""
    work = df[[date_col, group_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["_pos"] = np.arange(len(work))
    work = work.sort_values([group_col, date_col])
    out = (work.set_index(date_col).groupby(group_col)[value_col]
           .rolling(window, closed="left").mean().reset_index(level=0, drop=True))
    result = pd.Series(out.to_numpy(), index=work["_pos"].to_numpy())
    return result.sort_index().set_axis(df.index)


# --------------------------------------------------------------------------- #
# 5. Assembling the pipeline
# --------------------------------------------------------------------------- #
def build_fe_preprocessor(config: PreprocessingConfig | None = None, fe: FeatureEngineeringConfig | None = None,
                          engineered: list[str] | None = None, drop: list[str] | None = None,
                          seed: int = 42) -> Pipeline:
    """Stage 2 preprocessing extended with configurable feature engineering.

    ``engineered`` defaults to ``fe.domain.features``; ``drop`` removes original features. With the
    default ``FeatureEngineeringConfig`` and ``engineered=[]``, numeric and categorical handling match
    Stage 2 exactly.
    """
    config = config or PreprocessingConfig()
    fe = fe or FeatureEngineeringConfig()
    engineered = list(fe.domain.features if engineered is None else engineered)
    drop_set = set(drop or []) | set(config.drop_features)

    temporal = TemporalFeatureGenerator(fe.temporal.date_columns, fe.temporal.reference_date,
                                        tuple(fe.temporal.features))
    temporal_cols = temporal.output_names()
    validator_numeric = config.numeric_features + temporal_cols

    numeric = [c for c in config.numeric_features if c not in drop_set] + temporal_cols
    numeric += [f for f in engineered if FEATURE_CATALOGUE[f].kind == "numeric"]
    categorical = [c for c in config.categorical_features if c not in drop_set]
    binary = [f for f in engineered if FEATURE_CATALOGUE[f].kind == "binary"]
    skewed = [c for c in config.log_features if c in numeric] + [f for f in engineered if FEATURE_CATALOGUE[f].log]

    imp = fe.imputation
    imputer: Any = (KNNImputer(n_neighbors=imp.knn_neighbors) if imp.numeric_strategy == "knn"
                    else SimpleImputer(strategy=imp.numeric_strategy))
    g = fe.generators
    generator = GenericFeatureGenerator(
        interactions=g.interactions.columns if g.interactions.enabled else None,
        polynomial=g.polynomial.columns if g.polynomial.enabled else None, degree=g.polynomial.degree,
        binning=g.binning.columns if g.binning.enabled else None, n_bins=g.binning.n_bins,
        strategy=g.binning.strategy)

    transformers: list[tuple] = [("num", Pipeline([
        ("impute", imputer),
        ("transform", make_numeric_transform(fe.numeric_transform, skewed)),
        ("cap_outliers", OutlierCapper(config.outlier_method, config.iqr_factor, config.quantile_range)),
        ("generate", generator),
        ("scale", _make_scaler(config.scaler)),
    ]), numeric)]
    if imp.add_missing_indicator:
        transformers.append(("miss", MissingIndicator(features="all", error_on_new=False), numeric))
    if binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy=config.categorical_impute_strategy)),
            ("encode", make_encoder(fe, config, categorical, seed)),
        ]), categorical))

    d = fe.domain
    pipe = Pipeline([
        ("temporal", temporal),
        ("validate", DomainValidator(validator_numeric, config.categorical_features, config.valid_ranges,
                                     config.strictly_positive, config.categories)),
        ("engineer", DomainFeatureEngineer(engineered, d.school_premium_scale, d.top_school_threshold,
                                           d.new_build_max_age)),
        ("columns", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)),
    ])
    pipe.set_output(transform="pandas")
    return pipe


def make_fe_model(regressor, config: PreprocessingConfig | None = None, fe: FeatureEngineeringConfig | None = None,
                  engineered: list[str] | None = None, drop: list[str] | None = None, log_target: bool = True,
                  seed: int = 42) -> Pipeline:
    model = LogTargetRegressor(regressor) if log_target else regressor
    return Pipeline([("preprocess", build_fe_preprocessor(config, fe, engineered, drop, seed)), ("model", model)])


def tree_variant(config: PreprocessingConfig) -> PreprocessingConfig:
    """Trees need no scaling or outlier capping."""
    return PreprocessingConfig(**{**config.to_dict(), "scaler": "none", "outlier_method": "none"})


def catalogue_table(fe: FeatureEngineeringConfig | None = None) -> pd.DataFrame:
    fe = fe or FeatureEngineeringConfig()
    return pd.DataFrame([{"feature": n, "enabled": n in fe.domain.features, "formula": f.description,
                          "type": f.kind, "logged": f.log, "domain_rationale": f.rationale}
                         for n, f in FEATURE_CATALOGUE.items()])
