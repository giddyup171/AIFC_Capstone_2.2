"""Domain-driven feature engineering and feature selection for house price regression.

This stage sits between preprocessing (Stage 2) and modelling (Stage 3). It answers two questions:

1. **Which new features does domain knowledge suggest, and do they help?**
   ``DomainFeatureEngineer`` creates features from real-estate reasoning and the EDA
   (see ``FEATURE_CATALOGUE`` for the rationale of each one).
2. **Which features should the model actually use?**
   Five selection techniques (filter, redundancy, embedded, wrapper, model-agnostic) score
   every original and engineered feature on the training data; candidate feature sets built
   from them are then compared with paired cross-validation in dollars, and a pre-declared
   rule picks the final set.

Design notes
------------
* Engineered features are computed **inside** the scikit-learn pipeline (after validation,
  before imputation), so they are rebuilt identically at prediction time and are re-fitted
  within every CV fold: no train/test leakage.
* Binary features (0/1 flags) bypass the Stage 2 outlier capper. Its IQR fences collapse to
  [0, 0] for a mostly-zero column and would silently erase the flag; this module routes
  flags through imputation only.
* Selection uses the training split only. The held-out test set is touched once, to confirm
  the chosen set against the original features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .models import LogTargetRegressor
from .preprocessing import DomainValidator, LogTransformer, OutlierCapper, PreprocessingConfig, _make_scaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Feature catalogue: every engineered feature with its domain rationale
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    description: str
    rationale: str
    kind: str = "numeric"      # "numeric" (imputed, optionally logged, capped, scaled) or "binary" (imputed only)
    log: bool = False


FEATURE_CATALOGUE: dict[str, FeatureSpec] = {
    "school_premium": FeatureSpec(
        "exp(school_quality_score / 3)",
        "Buyers pay disproportionately more for the best school catchments; the EDA and Stage 5 splines show "
        "the school effect on log price steepens at the top of the scale. An exponential term lets a linear "
        "model express that accelerating premium."),
    "school_squared": FeatureSpec(
        "school_quality_score ** 2",
        "Simpler curvature alternative to school_premium; included so the selection step can choose between "
        "two encodings of the same idea (they are highly correlated, which tests the redundancy filter)."),
    "neighbourhood_quality": FeatureSpec(
        "log(school_quality_score / crime_rate_in_area)",
        "Good schools and low crime describe the same thing to a buyer: a desirable neighbourhood. One "
        "combined index is less noisy than two separate signals."),
    "size_per_bedroom": FeatureSpec(
        "property_size / number_of_bedrooms",
        "Spaciousness: the same floor area split into fewer rooms usually indicates larger, higher-end rooms.",
        log=True),
    "bath_per_bed": FeatureSpec(
        "number_of_bathrooms / number_of_bedrooms",
        "Bathroom-to-bedroom ratio is a standard luxury indicator (en-suites)."),
    "total_rooms": FeatureSpec(
        "number_of_bedrooms + number_of_bathrooms",
        "Overall house size in rooms; a common listing summary figure."),
    "school_x_size": FeatureSpec(
        "school_quality_score * log(property_size)",
        "Interaction: families buying large homes may value school quality more, so the school premium could "
        "scale with size."),
    "top_school": FeatureSpec(
        "1 if school_quality_score >= 8.5",
        "Catchment effect: homes zoned for top-rated schools often carry a step premium.", kind="binary"),
    "new_build": FeatureSpec(
        "1 if age_of_property < 5",
        "New-build premium: little maintenance and modern specification.", kind="binary"),
}


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered columns to a validated DataFrame. Stateless: formulas only.

    Missing or impossible inputs (already set to NaN by ``DomainValidator``) give NaN outputs,
    which the following imputation step fills; division by zero gives NaN, never inf.
    """

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        unknown = [f for f in (self.features or []) if f not in FEATURE_CATALOGUE]
        if unknown:
            raise ValueError(f"Unknown engineered features: {unknown}; choose from {list(FEATURE_CATALOGUE)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _ratio(a, b):
        out = a / b.replace(0, np.nan)
        return out.where(np.isfinite(out))

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = X.copy()
        s = X["school_quality_score"]
        formulas = {
            "school_premium": lambda: np.exp(s / 3),
            "school_squared": lambda: s ** 2,
            "neighbourhood_quality": lambda: np.log(self._ratio(s, X["crime_rate_in_area"]).where(lambda v: v > 0)),
            "size_per_bedroom": lambda: self._ratio(X["property_size"], X["number_of_bedrooms"]),
            "bath_per_bed": lambda: self._ratio(X["number_of_bathrooms"], X["number_of_bedrooms"]),
            "total_rooms": lambda: X["number_of_bedrooms"] + X["number_of_bathrooms"],
            "school_x_size": lambda: s * np.log(X["property_size"].where(X["property_size"] > 0)),
            "top_school": lambda: (s >= 8.5).astype(float).where(s.notna()),
            "new_build": lambda: (X["age_of_property"] < 5).astype(float).where(X["age_of_property"].notna()),
        }
        for f in self.features or []:
            X[f] = formulas[f]().astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + list(self.features or []), dtype=object)


# --------------------------------------------------------------------------- #
# 2. Pipelines that include engineered features
# --------------------------------------------------------------------------- #
def build_fe_preprocessor(config: PreprocessingConfig | None = None, engineered: list[str] | None = None,
                          drop: list[str] | None = None) -> Pipeline:
    """Stage 2 preprocessing with engineered features added (and optionally some features dropped).

    validate -> engineer -> [numeric: impute, log, cap, scale | binary: impute | categorical: one-hot]
    With ``engineered=[]`` and ``drop=[]`` the numeric/categorical treatment matches Stage 2 exactly.
    """
    config = config or PreprocessingConfig()
    engineered = list(engineered or [])
    drop = set(drop or []) | set(config.drop_features)
    numeric = [c for c in config.numeric_features if c not in drop]
    categorical = [c for c in config.categorical_features if c not in drop]
    numeric += [f for f in engineered if FEATURE_CATALOGUE[f].kind == "numeric"]
    binary = [f for f in engineered if FEATURE_CATALOGUE[f].kind == "binary"]
    logs = [c for c in config.log_features if c in numeric] + \
           [f for f in engineered if FEATURE_CATALOGUE[f].log]

    transformers = [("num", Pipeline([
        ("impute", SimpleImputer(strategy=config.numeric_impute_strategy)),
        ("log", LogTransformer(columns=logs)),
        ("cap_outliers", OutlierCapper(config.outlier_method, config.iqr_factor, config.quantile_range)),
        ("scale", _make_scaler(config.scaler)),
    ]), numeric)]
    if binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy=config.categorical_impute_strategy)),
            ("onehot", OneHotEncoder(categories=[config.categories[c] for c in categorical],
                                     drop="first" if config.drop_first_category else None,
                                     handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    pipe = Pipeline([
        ("validate", DomainValidator(config.numeric_features, config.categorical_features, config.valid_ranges,
                                     config.strictly_positive, config.categories)),
        ("engineer", DomainFeatureEngineer(engineered)),
        ("columns", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)),
    ])
    pipe.set_output(transform="pandas")
    return pipe


def make_fe_model(regressor, config: PreprocessingConfig | None = None, engineered: list[str] | None = None,
                  drop: list[str] | None = None, log_target: bool = True) -> Pipeline:
    model = LogTargetRegressor(regressor) if log_target else regressor
    return Pipeline([("preprocess", build_fe_preprocessor(config, engineered, drop)), ("model", model)])


def tree_variant(config: PreprocessingConfig) -> PreprocessingConfig:
    return PreprocessingConfig(**{**config.to_dict(), "scaler": "none", "outlier_method": "none"})


# --------------------------------------------------------------------------- #
# 3. Feature selection techniques (training data only)
# --------------------------------------------------------------------------- #
def _design(config, X, y):
    """Processed design matrix with every original + engineered feature, and log target."""
    prep = build_fe_preprocessor(config, list(FEATURE_CATALOGUE)).fit(X)
    return prep.transform(X), np.log(np.asarray(y, float))


def _source_feature(column: str, config: PreprocessingConfig) -> str:
    """Map a processed column (e.g. location_type_Urban) back to the feature it came from."""
    for c in config.categorical_features:
        if column.startswith(f"{c}_"):
            return c
    return column


def filter_scores(Z: pd.DataFrame, y_log: np.ndarray, seed: int) -> pd.DataFrame:
    """Filter methods: mutual information and |Spearman| with log price (univariate relevance)."""
    mi = mutual_info_regression(Z, y_log, random_state=seed)
    sp = [abs(stats.spearmanr(Z[c], y_log)[0]) if Z[c].nunique() > 1 else 0.0 for c in Z.columns]
    return pd.DataFrame({"mutual_info": mi, "abs_spearman": sp}, index=Z.columns)


def redundancy_pairs(Z: pd.DataFrame, mi: pd.Series, threshold: float = 0.9) -> pd.DataFrame:
    """Pairs of features correlated above ``threshold``; the one with lower MI is the drop candidate."""
    corr = Z.corr(method="spearman").abs()
    rows = []
    cols = list(Z.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                keep, drop = (a, b) if mi[a] >= mi[b] else (b, a)
                rows.append({"feature_a": a, "feature_b": b, "abs_spearman_corr": corr.loc[a, b],
                             "keep": keep, "drop_candidate": drop})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_spearman_corr", "keep", "drop_candidate"])


def lasso_stability(Z: pd.DataFrame, y_log: np.ndarray, splits: list[tuple]) -> pd.Series:
    """Embedded method: share of CV folds in which LassoCV keeps a non-zero coefficient."""
    keep = pd.Series(0.0, index=Z.columns)
    for tr, _ in splits:
        m = LassoCV(cv=5, random_state=0, max_iter=20000, n_jobs=-1).fit(Z.iloc[tr], y_log[tr])
        keep += (np.abs(m.coef_) > 1e-8).astype(float)
    return keep / len(splits)


def rfecv_ranking(Z: pd.DataFrame, y_log: np.ndarray, seed: int) -> tuple[pd.Series, int]:
    """Wrapper method: recursive feature elimination with CV on a Ridge model (log RMSE)."""
    rfe = RFECV(RidgeCV(alphas=np.logspace(-3, 3, 13)), step=1, min_features_to_select=3,
                cv=KFold(5, shuffle=True, random_state=seed), scoring="neg_root_mean_squared_error", n_jobs=-1)
    rfe.fit(Z, y_log)
    return pd.Series(rfe.ranking_, index=Z.columns), int(rfe.n_features_)


def permutation_scores(config: PreprocessingConfig, X: pd.DataFrame, y, seed: int) -> pd.Series:
    """Model-agnostic method: permutation importance (RMSE increase, $) of a gradient-boosting model
    on a held-out 25% of the training data, per raw/engineered feature."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.75 * len(X))
    tr, va = idx[:cut], idx[cut:]
    model = make_fe_model(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed),
                          tree_variant(config), list(FEATURE_CATALOGUE))
    model.fit(X.iloc[tr], np.asarray(y)[tr])
    # Permute engineered features by permuting the inputs they are built from is not meaningful, so
    # score the processed columns: wrap the fitted estimator on the processed matrix.
    prep, est = model.named_steps["preprocess"], model.named_steps["model"]
    Zva = prep.transform(X.iloc[va])
    r = permutation_importance(est, Zva, np.asarray(y)[va], scoring="neg_root_mean_squared_error",
                               n_repeats=5, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=Zva.columns)


def selection_table(config: PreprocessingConfig, X: pd.DataFrame, y, splits: list[tuple], seed: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Run all techniques; aggregate to one row per feature (categorical dummies merged) with votes."""
    Z, y_log = _design(config, X, y)
    filt = filter_scores(Z, y_log, seed)
    redundant = redundancy_pairs(Z, filt["mutual_info"])
    lasso = lasso_stability(Z, y_log, splits)
    rank, n_rfe = rfecv_ranking(Z, y_log, seed)
    perm = permutation_scores(config, X, y, seed)

    per_col = pd.DataFrame({"mutual_info": filt["mutual_info"], "abs_spearman": filt["abs_spearman"],
                            "lasso_keep_share": lasso, "rfecv_rank": rank, "permutation_rmse_increase": perm})
    per_col["feature"] = [_source_feature(c, config) for c in per_col.index]
    agg = per_col.groupby("feature").agg(mutual_info=("mutual_info", "max"), abs_spearman=("abs_spearman", "max"),
                                         lasso_keep_share=("lasso_keep_share", "max"),
                                         rfecv_rank=("rfecv_rank", "min"),
                                         permutation_rmse_increase=("permutation_rmse_increase", "sum"))
    agg["engineered"] = agg.index.isin(list(FEATURE_CATALOGUE))
    drop_cands = set(redundant["drop_candidate"])
    # Votes: each technique says keep/drop by a stated rule
    agg["vote_filter"] = (agg["mutual_info"] >= 0.01)
    agg["vote_redundancy"] = ~agg.index.isin(drop_cands)
    agg["vote_lasso"] = agg["lasso_keep_share"] >= 0.8
    agg["vote_rfecv"] = agg["rfecv_rank"] == 1
    agg["vote_permutation"] = agg["permutation_rmse_increase"] > 0.002 * float(np.mean(y))
    vote_cols = [c for c in agg.columns if c.startswith("vote_")]
    agg["votes"] = agg[vote_cols].sum(axis=1)
    return agg.sort_values(["votes", "mutual_info"], ascending=False), redundant, n_rfe


# --------------------------------------------------------------------------- #
# 4. Candidate feature sets and their comparison
# --------------------------------------------------------------------------- #
def candidate_sets(table: pd.DataFrame, config: PreprocessingConfig, min_votes: int = 3) -> dict[str, dict]:
    """Feature sets to compare. Each is {'engineered': [...], 'drop': [...]} relative to the original features."""
    original = set(config.numeric_features + config.categorical_features)
    eng_all = list(FEATURE_CATALOGUE)

    def from_mask(mask) -> dict:
        chosen = set(table.index[mask])
        return {"engineered": [f for f in eng_all if f in chosen], "drop": sorted(original - chosen)}

    return {
        "original": {"engineered": [], "drop": []},
        "original_plus_all_engineered": {"engineered": eng_all, "drop": []},
        "lasso_stable": from_mask(table["vote_lasso"]),
        "rfecv": from_mask(table["vote_rfecv"]),
        "redundancy_pruned": from_mask(table["vote_redundancy"]),
        f"consensus_{min_votes}_of_5": from_mask(table["votes"] >= min_votes),
    }


def compare_feature_sets(sets: dict[str, dict], config: PreprocessingConfig, X, y, splits: list[tuple],
                         seed: int, include_trees: bool = True) -> pd.DataFrame:
    """Paired CV (same folds as Stage 3) for the Ridge benchmark, plus gradient boosting as a check that the
    conclusions are not specific to linear models. RMSE in dollars."""
    from .models import cross_validate_models

    models = {}
    for name, s in sets.items():
        models[f"ridge|{name}"] = make_fe_model(RidgeCV(alphas=np.logspace(-3, 3, 25)), config,
                                                s["engineered"], s["drop"])
        if include_trees and name in ("original", "original_plus_all_engineered"):
            models[f"gboost|{name}"] = make_fe_model(
                HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed),
                tree_variant(config), s["engineered"], s["drop"])
    return cross_validate_models(models, X, y, splits)


def choose_feature_set(folds: pd.DataFrame, sets: dict[str, dict], config: PreprocessingConfig | None = None,
                       min_gain: float = 0.005, tie_tolerance: float = 0.005) -> tuple[str, pd.DataFrame, str]:
    """Pre-declared rule.

    1. A candidate must beat the original features on the paired folds (p < 0.05) by at least ``min_gain``
       (0.5%) of RMSE; otherwise the original features are kept (no complexity without evidence).
    2. Among qualifying candidates, all within ``tie_tolerance`` of the best RMSE are tied; the one with the
       fewest features wins (parsimony: fewer things to compute, explain and maintain).
    """
    from .models import compare_models

    ridge = folds[folds.model.str.startswith("ridge|")].copy()
    ridge["model"] = ridge.model.str.replace("ridge|", "", regex=False)
    mean = ridge.groupby("model").rmse.mean()
    base = mean["original"]
    config = config or PreprocessingConfig()
    n_original = len(config.numeric_features) + len(config.categorical_features)
    rows = []
    for name in sets:
        c = compare_models(ridge, "original", name) if name != "original" else None
        rows.append({"feature_set": name, "cv_rmse": mean[name], "gain_vs_original": 1 - mean[name] / base,
                     "folds_better": c["candidate_wins_folds"] if c else "-",
                     "p_value": c["paired_t_p_value"] if c else np.nan,
                     "n_engineered": len(sets[name]["engineered"]), "n_dropped": len(sets[name]["drop"])})
    table = pd.DataFrame(rows).set_index("feature_set")
    table["n_features"] = n_original + table["n_engineered"] - table["n_dropped"]
    ok = table[(table.index != "original") & (table.p_value < 0.05) & (table.gain_vs_original >= min_gain)]
    if ok.empty:
        return "original", table, "No candidate beat the original features significantly; original features kept."
    best = ok.cv_rmse.min()
    tied = ok[ok.cv_rmse <= best * (1 + tie_tolerance)]
    chosen = tied.sort_values(["n_features", "cv_rmse"]).index[0]
    note = (f"{chosen}: {table.loc[chosen, 'gain_vs_original']:.1%} lower CV RMSE than the original features "
            f"(p = {table.loc[chosen, 'p_value']:.2g}), the smallest set within {tie_tolerance:.1%} of the best")
    return chosen, table, note


# --------------------------------------------------------------------------- #
# 5. Plots
# --------------------------------------------------------------------------- #
def plot_selection(table: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    votes = table[[c for c in table.columns if c.startswith("vote_")]].astype(int)
    votes.columns = [c.replace("vote_", "") for c in votes.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(table) + 1.5),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.imshow(votes.values, cmap="Greens", aspect="auto", vmin=0, vmax=1.3)
    ax.set_xticks(range(votes.shape[1]), votes.columns, rotation=30, ha="right")
    labels = [f"{i}  ★" if table.loc[i, "engineered"] else i for i in votes.index]
    ax.set_yticks(range(len(votes)), labels)
    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            ax.text(j, i, "keep" if votes.values[i, j] else "drop", ha="center", va="center", fontsize=8,
                    color="white" if votes.values[i, j] else "#555")
    ax.set_title("Selection votes per technique (★ = engineered)")
    ax = axes[1]
    ax.barh(range(len(table)), table["votes"], color=["#DD8452" if e else "#4C72B0" for e in table["engineered"]])
    ax.set_yticks(range(len(table)), [""] * len(table))
    ax.set(xlabel="Votes to keep (of 5)", xlim=(0, 5), title="Consensus")
    for a in axes:
        a.set_ylim(len(table) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_feature_set_comparison(folds: pd.DataFrame, chosen: str, path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    s = folds.groupby("model").rmse.agg(["mean", "std"]).sort_values("mean", ascending=False)
    colors = ["#DD8452" if m == f"ridge|{chosen}" else ("#8C8C8C" if m.startswith("gboost") else "#4C72B0")
              for m in s.index]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(s) + 1.5))
    ax.barh(range(len(s)), s["mean"], xerr=s["std"], color=colors, capsize=3)
    for i, (m, sd) in enumerate(zip(s["mean"], s["std"])):
        ax.text(m + sd, i, f"  ${m:,.0f}", va="center", fontsize=9)
    ax.set_yticks(range(len(s)), [m.replace("|", ": ") for m in s.index])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set_xlim(0, (s["mean"] + s["std"]).max() * 1.25)
    ax.set(xlabel="CV RMSE (15 paired folds), lower is better",
           title=f"Feature sets compared (orange = selected: {chosen}; grey = gradient-boosting check)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)
