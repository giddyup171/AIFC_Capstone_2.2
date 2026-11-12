"""Reusable preprocessing pipeline for house sale price regression.

Every design choice here comes from the EDA in ``eda.ipynb``:

* The training data has no missing values, but new data may. Invalid values
  (negative sizes, school scores above 10, unknown locations) are converted to
  missing and then imputed, so the pipeline never crashes or silently learns
  from garbage.
* The "outliers" in this data are the long tail of log-normal features, not
  errors. Rows are therefore never dropped; skewed features are log-transformed,
  then extreme values are capped (winsorised) using bounds learned on the
  training set only.
* The target is right-skewed (skew 1.6 -> 0.1 after log), so it is modelled on
  the log scale with an optional smearing correction when converting back.
* ``location_type`` is one-hot encoded with fixed categories, and train/test
  splits are stratified on it because the 500/300/200 split is by design.

All learned parameters (medians, log floors, capping bounds, scaler statistics)
are fitted on training data only, so there is no train/test leakage.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessingConfig:
    """All tunable settings in one place, so experiments change config, not code."""

    target: str = "house_sale_price"
    numeric_features: list[str] = field(default_factory=lambda: [
        "property_size",
        "number_of_bedrooms",
        "number_of_bathrooms",
        "age_of_property",
        "proximity_to_public_transport",
        "crime_rate_in_area",
        "school_quality_score",
    ])
    categorical_features: list[str] = field(default_factory=lambda: ["location_type"])
    categories: dict[str, list[str]] = field(default_factory=lambda: {
        "location_type": ["Rural", "Suburban", "Urban"],
    })

    # EDA: right-skewed and closer to linear with price on the log scale.
    # school_quality_score is left-skewed, so it is deliberately excluded.
    log_features: list[str] = field(default_factory=lambda: [
        "property_size",
        "age_of_property",
        "proximity_to_public_transport",
        "crime_rate_in_area",
    ])
    # Features to drop before modelling (EDA: bathrooms add ~nothing given size).
    # Empty by default so the choice is left to the modelling stage.
    drop_features: list[str] = field(default_factory=list)

    # Domain rules: values outside (lower, upper) are treated as invalid -> NaN.
    # ``None`` means unbounded. Strict lower bound for counts/size (> 0),
    # inclusive lower bound for the rest (>= 0).
    valid_ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=lambda: {
        "property_size": (0, None),
        "number_of_bedrooms": (0, None),
        "number_of_bathrooms": (0, None),
        "age_of_property": (0, None),
        "proximity_to_public_transport": (0, None),
        "crime_rate_in_area": (0, None),
        "school_quality_score": (0, 10),
    })
    strictly_positive: list[str] = field(default_factory=lambda: [
        "property_size", "number_of_bedrooms", "number_of_bathrooms",
    ])

    # Missing values
    numeric_impute_strategy: str = "median"      # median is robust to the skew
    categorical_impute_strategy: str = "most_frequent"

    # Outliers: "iqr" (Tukey fences), "quantile" (percentile caps) or "none"
    outlier_method: str = "iqr"
    iqr_factor: float = 3.0                      # 3.0 = only "far out" points
    quantile_range: tuple[float, float] = (0.005, 0.995)

    # Scaling: "standard", "robust", "minmax" or "none"
    scaler: str = "standard"

    # Encoding: drop first level to avoid the dummy trap in linear models
    drop_first_category: bool = True

    # Target
    log_target: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def model_numeric_features(self) -> list[str]:
        return [c for c in self.numeric_features if c not in self.drop_features]

    @property
    def model_categorical_features(self) -> list[str]:
        return [c for c in self.categorical_features if c not in self.drop_features]


# --------------------------------------------------------------------------- #
# Custom transformers (scikit-learn compatible)
# --------------------------------------------------------------------------- #
class DomainValidator(BaseEstimator, TransformerMixin):
    """Coerce types and replace impossible values with NaN.

    Stateless: the rules come from the config, not the data. It runs inside the
    pipeline, so the same checks apply at training and at prediction time.
    """

    def __init__(self, numeric_features, categorical_features, valid_ranges=None,
                 strictly_positive=None, categories=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.valid_ranges = valid_ranges
        self.strictly_positive = strictly_positive
        self.categories = categories

    def fit(self, X, y=None):
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _check_columns(self, X):
        missing = [c for c in self.numeric_features + self.categorical_features if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._check_columns(X)
        # Select required columns in a fixed order: extra columns (IDs, the
        # target) are dropped and column order in new data does not matter.
        X = X[self.numeric_features + self.categorical_features].copy()
        ranges = self.valid_ranges or {}
        strict = set(self.strictly_positive or [])

        for col in self.numeric_features:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
            X.loc[~np.isfinite(X[col]), col] = np.nan
            lo, hi = ranges.get(col, (None, None))
            if lo is not None:
                bad = X[col] <= lo if col in strict else X[col] < lo
                X.loc[bad, col] = np.nan
            if hi is not None:
                X.loc[X[col] > hi, col] = np.nan

        for col in self.categorical_features:
            s = X[col].astype(object).where(X[col].notna(), None)
            s = s.map(lambda v: v.strip().title() if isinstance(v, str) else v)
            allowed = (self.categories or {}).get(col)
            if allowed is not None:
                s = s.where(s.isin(allowed), np.nan)
            X[col] = s.astype(object)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.numeric_features + self.categorical_features, dtype=object)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Natural log of selected columns; other columns pass through unchanged.

    A per-column floor (half the smallest positive training value) keeps the
    log finite if a future value is zero or below.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        cols = [c for c in (self.columns or []) if c in X.columns]
        self.columns_ = cols
        self.floors_ = {}
        for c in cols:
            positive = X[c][X[c] > 0]
            self.floors_[c] = float(positive.min() / 2) if len(positive) else 1e-6
        return self

    def transform(self, X):
        check_is_fitted(self, "floors_")
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            X[c] = np.log(np.clip(X[c].astype(float), self.floors_[c], None))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Winsorise each column to bounds learned on the training set.

    Capping keeps every row (the EDA found no erroneous records) while limiting
    the leverage of extreme values on linear models.
    """

    def __init__(self, method="iqr", iqr_factor=3.0, quantile_range=(0.005, 0.995)):
        self.method = method
        self.iqr_factor = iqr_factor
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.method not in {"iqr", "quantile", "none"}:
            raise ValueError(f"Unknown outlier method: {self.method!r}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lower_, self.upper_ = {}, {}
        for c in X.columns:
            s = X[c].dropna()
            if self.method == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr
            elif self.method == "quantile":
                lo, hi = s.quantile(list(self.quantile_range))
            else:
                lo, hi = -np.inf, np.inf
            self.lower_[c], self.upper_[c] = float(lo), float(hi)
        self.n_capped_fit_ = self._count_capped(X)
        return self

    def _count_capped(self, X):
        return {c: int(((X[c] < self.lower_[c]) | (X[c] > self.upper_[c])).sum()) for c in X.columns}

    def transform(self, X):
        check_is_fitted(self, "lower_")
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = X[c].clip(self.lower_[c], self.upper_[c])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _make_scaler(name: str):
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "none": "passthrough",
    }
    if name not in scalers:
        raise ValueError(f"Unknown scaler {name!r}; choose from {list(scalers)}")
    return scalers[name]


def build_preprocessor(config: PreprocessingConfig | None = None) -> Pipeline:
    """Return an unfitted scikit-learn Pipeline that maps raw features to model inputs.

    Numeric: impute (median) -> log selected skewed columns -> cap outliers -> scale
    Categorical: impute (mode) -> one-hot encode with fixed categories

    The returned object can be used alone or as the first step of a model
    Pipeline, e.g. ``Pipeline([("prep", build_preprocessor()), ("model", Ridge())])``.
    """
    config = config or PreprocessingConfig()
    num_cols = config.model_numeric_features
    cat_cols = config.model_categorical_features

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=config.numeric_impute_strategy)),
        ("log", LogTransformer(columns=[c for c in config.log_features if c in num_cols])),
        ("cap_outliers", OutlierCapper(config.outlier_method, config.iqr_factor, config.quantile_range)),
        ("scale", _make_scaler(config.scaler)),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=config.categorical_impute_strategy)),
        ("onehot", OneHotEncoder(
            categories=[config.categories[c] for c in cat_cols],
            drop="first" if config.drop_first_category else None,
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])
    columns = ColumnTransformer(
        [("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline([
        ("validate", DomainValidator(
            config.numeric_features, config.categorical_features,
            config.valid_ranges, config.strictly_positive, config.categories,
        )),
        ("columns", columns),
    ])
    pipeline.set_output(transform="pandas")
    return pipeline


# --------------------------------------------------------------------------- #
# Data-level helpers (row operations happen here, not inside the pipeline,
# because a transformer must return the same number of rows it receives)
# --------------------------------------------------------------------------- #
def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path)
    return df


def clean_training_data(df: pd.DataFrame, config: PreprocessingConfig | None = None
                        ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Row-level cleaning for labelled data: drop duplicates and unusable targets.

    Feature-level problems are left for the pipeline, which imputes them.
    """
    config = config or PreprocessingConfig()
    report = {"rows_in": len(df)}

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    y = pd.to_numeric(df[config.target], errors="coerce")
    bad_target = y.isna() | (y <= 0) | ~np.isfinite(y)
    report["invalid_target_removed"] = int(bad_target.sum())
    df = df.loc[~bad_target].copy()
    df[config.target] = y[~bad_target].astype(float)

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def data_quality_report(df: pd.DataFrame, config: PreprocessingConfig | None = None) -> dict[str, Any]:
    """Count missing and rule-breaking values per column on raw data."""
    config = config or PreprocessingConfig()
    validator = DomainValidator(config.numeric_features, config.categorical_features,
                                config.valid_ranges, config.strictly_positive, config.categories).fit(df)
    validated = validator.transform(df)
    cols = config.numeric_features + config.categorical_features
    return {
        "missing_raw": {c: int(df[c].isna().sum()) for c in cols},
        "invalid_converted_to_missing": {
            c: int(validated[c].isna().sum() - df[c].isna().sum()) for c in cols
        },
    }


def split_data(df: pd.DataFrame, config: PreprocessingConfig | None = None,
               test_size: float = 0.2, random_state: int = 42):
    """Train/test split stratified on location so both sets keep the 50/30/20 mix."""
    config = config or PreprocessingConfig()
    X = df.drop(columns=[config.target])
    y = df[config.target]
    strat_col = config.categorical_features[0] if config.categorical_features else None
    stratify = X[strat_col] if strat_col and X[strat_col].notna().all() else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


# --------------------------------------------------------------------------- #
# Target transformation
# --------------------------------------------------------------------------- #
def transform_target(y, config: PreprocessingConfig | None = None) -> pd.Series:
    config = config or PreprocessingConfig()
    y = pd.Series(y, dtype=float)
    return np.log(y) if config.log_target else y


def inverse_transform_target(y_t, config: PreprocessingConfig | None = None,
                             smearing_factor: float = 1.0) -> np.ndarray:
    """Back-transform predictions. ``exp`` gives the median; multiply by Duan's
    smearing factor to estimate the mean."""
    config = config or PreprocessingConfig()
    y_t = np.asarray(y_t, dtype=float)
    return np.exp(y_t) * smearing_factor if config.log_target else y_t


def smearing_factor(log_residuals) -> float:
    """Duan (1983) smearing estimate: mean of exp(residuals) on the log scale."""
    return float(np.mean(np.exp(np.asarray(log_residuals, dtype=float))))


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
class HousePricePreprocessor:
    """Fit once on training data, reuse everywhere (test set, new data, API).

    >>> prep = HousePricePreprocessor()
    >>> X_train_t, y_train_t = prep.fit_transform(X_train, y_train)
    >>> X_test_t = prep.transform(X_test)
    >>> prep.save("artifacts/preprocessor.joblib")
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.pipeline = build_preprocessor(self.config)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "HousePricePreprocessor":
        self.pipeline.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None):
        Xt = self.fit(X).transform(X)
        return (Xt, transform_target(y, self.config)) if y is not None else Xt

    def transform_target(self, y) -> pd.Series:
        return transform_target(y, self.config)

    def inverse_transform_target(self, y_t, smearing: float = 1.0) -> np.ndarray:
        return inverse_transform_target(y_t, self.config, smearing)

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.get_feature_names_out())

    def fitted_summary(self) -> dict[str, Any]:
        """Learned parameters, useful for auditing and the run report."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        num = self.pipeline.named_steps["columns"].named_transformers_["num"]
        imputer, logt, capper = num.named_steps["impute"], num.named_steps["log"], num.named_steps["cap_outliers"]
        return {
            "numeric_impute_values": dict(zip(imputer.feature_names_in_, map(float, imputer.statistics_))),
            "log_floors": logt.floors_,
            "outlier_bounds_transformed_scale": {
                c: [capper.lower_[c], capper.upper_[c]] for c in capper.lower_
            },
            "values_capped_in_training": capper.n_capped_fit_,
            "output_features": self.feature_names,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "HousePricePreprocessor":
        return joblib.load(path)
