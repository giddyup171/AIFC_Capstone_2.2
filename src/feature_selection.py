"""Feature selection and feature-set comparison (Stage 2b).

Five techniques each cast a keep/drop vote per feature; every threshold comes from
``FeatureEngineeringConfig.selection``. Candidate feature sets (built from the votes, plus generic
alternatives such as interactions, polynomial terms, binning, power transforms, encodings and
imputation variants) are then compared on the same paired CV folds as Stage 3, and a rule from
``FeatureEngineeringConfig.decision`` picks the final set. Training data only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold

from .fe_config import FeatureEngineeringConfig
from .feature_engineering import FEATURE_CATALOGUE, build_fe_preprocessor, make_fe_model, tree_variant
from .preprocessing import PreprocessingConfig
from .profiling import Profiler

logger = logging.getLogger(__name__)

VOTE_COLUMNS = ["vote_filter", "vote_redundancy", "vote_lasso", "vote_rfecv", "vote_permutation"]


# --------------------------------------------------------------------------- #
# Individual techniques
# --------------------------------------------------------------------------- #
def design_matrix(config: PreprocessingConfig, fe: FeatureEngineeringConfig, X: pd.DataFrame, y
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """Processed matrix with every original + enabled domain feature, and the log target."""
    prep = build_fe_preprocessor(config, fe).fit(X, y)
    return prep.transform(X), np.log(np.asarray(y, float))


def source_feature(column: str, config: PreprocessingConfig) -> str:
    """Map a processed column (e.g. location_type_Urban) back to the feature it came from."""
    for c in config.categorical_features:
        if column.startswith(f"{c}_"):
            return c
    return column


def filter_scores(Z: pd.DataFrame, y_log: np.ndarray, seed: int) -> pd.DataFrame:
    """Filter: mutual information and |Spearman| with log price (univariate relevance)."""
    mi = mutual_info_regression(Z, y_log, random_state=seed)
    sp = [abs(stats.spearmanr(Z[c], y_log)[0]) if Z[c].nunique() > 1 else 0.0 for c in Z.columns]
    return pd.DataFrame({"mutual_info": mi, "abs_spearman": sp}, index=Z.columns)


def redundancy_pairs(Z: pd.DataFrame, mi: pd.Series, threshold: float) -> pd.DataFrame:
    """Pairs with |Spearman| >= threshold; the member with lower MI is the drop candidate."""
    corr = Z.corr(method="spearman").abs().to_numpy()
    rows = []
    cols = list(Z.columns)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b, r = cols[j], float(corr[i, j])
            if r >= threshold:
                keep, drop = (a, b) if float(mi[a]) >= float(mi[b]) else (b, a)
                rows.append({"feature_a": a, "feature_b": b, "abs_spearman_corr": r,
                             "keep": keep, "drop_candidate": drop})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_spearman_corr", "keep", "drop_candidate"])


def lasso_stability(Z: pd.DataFrame, y_log: np.ndarray, splits: list[tuple], inner_folds: int) -> pd.Series:
    """Embedded: share of CV folds in which LassoCV keeps a non-zero coefficient."""
    keep = pd.Series(0.0, index=Z.columns)
    for tr, _ in splits:
        m = LassoCV(cv=inner_folds, random_state=0, max_iter=20000, n_jobs=-1).fit(Z.iloc[tr], y_log[tr])
        keep += (np.abs(m.coef_) > 1e-8).astype(float)
    return keep / len(splits)


def rfecv_ranking(Z: pd.DataFrame, y_log: np.ndarray, seed: int, min_features: int, step: int, folds: int
                  ) -> tuple[pd.Series, int]:
    """Wrapper: recursive feature elimination with CV on Ridge (log RMSE)."""
    rfe = RFECV(RidgeCV(alphas=np.logspace(-3, 3, 13)), step=step, min_features_to_select=min_features,
                cv=KFold(folds, shuffle=True, random_state=seed), scoring="neg_root_mean_squared_error", n_jobs=-1)
    rfe.fit(Z, y_log)
    return pd.Series(rfe.ranking_, index=Z.columns), int(rfe.n_features_)


def permutation_scores(config: PreprocessingConfig, fe: FeatureEngineeringConfig, X: pd.DataFrame, y, seed: int
                       ) -> pd.Series:
    """Model-agnostic: RMSE increase ($) when a processed column is shuffled, gradient boosting fitted on
    (1 - holdout) of the training data and scored on the rest."""
    s = fe.selection
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int((1 - s.permutation_holdout) * len(X))
    tr, va = idx[:cut], idx[cut:]
    y = np.asarray(y, float)
    model = make_fe_model(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed),
                          tree_variant(config), fe, seed=seed)
    model.fit(X.iloc[tr], y[tr])
    prep, est = model.named_steps["preprocess"], model.named_steps["model"]
    Zva = prep.transform(X.iloc[va])
    r = permutation_importance(est, Zva, y[va], scoring="neg_root_mean_squared_error",
                               n_repeats=s.permutation_repeats, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=Zva.columns)


def apply_votes(agg: pd.DataFrame, drop_candidates: set[str], mean_price: float,
                fe: FeatureEngineeringConfig) -> pd.DataFrame:
    """Turn technique scores into keep votes using the configured thresholds (pure function, testable)."""
    s = fe.selection
    agg = agg.copy()
    agg["vote_filter"] = agg["mutual_info"] >= s.mutual_info_min
    agg["vote_redundancy"] = ~agg.index.isin(drop_candidates)
    agg["vote_lasso"] = agg["lasso_keep_share"] >= s.lasso_keep_share
    agg["vote_rfecv"] = agg["rfecv_rank"] == 1
    agg["vote_permutation"] = agg["permutation_rmse_increase"] > s.permutation_min_share * mean_price
    agg["votes"] = agg[VOTE_COLUMNS].sum(axis=1)
    return agg.sort_values(["votes", "mutual_info"], ascending=[False, False])


def selection_table(config: PreprocessingConfig, fe: FeatureEngineeringConfig, X: pd.DataFrame, y,
                    splits: list[tuple], seed: int, profiler: Profiler | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Run all five techniques and aggregate to one row per feature (categorical dummies merged)."""
    prof = profiler or Profiler(enabled=False)
    s = fe.selection
    with prof.step("selection: design matrix"):
        Z, y_log = design_matrix(config, fe, X, y)
    with prof.step("selection: filter (mutual information)"):
        filt = filter_scores(Z, y_log, seed)
    with prof.step("selection: redundancy (Spearman)"):
        redundant = redundancy_pairs(Z, filt["mutual_info"], s.redundancy_corr)
    with prof.step("selection: Lasso stability", folds=len(splits)):
        lasso = lasso_stability(Z, y_log, splits, s.lasso_inner_folds)
    with prof.step("selection: RFECV", rfe_step=s.rfecv_step):
        rank, n_rfe = rfecv_ranking(Z, y_log, seed, s.rfecv_min_features, s.rfecv_step, s.rfecv_folds)
    with prof.step("selection: permutation importance"):
        perm = permutation_scores(config, fe, X, y, seed)

    per_col = pd.DataFrame({"mutual_info": filt["mutual_info"], "abs_spearman": filt["abs_spearman"],
                            "lasso_keep_share": lasso, "rfecv_rank": rank, "permutation_rmse_increase": perm})
    per_col["feature"] = [source_feature(c, config) for c in per_col.index]
    agg = per_col.groupby("feature").agg(mutual_info=("mutual_info", "max"), abs_spearman=("abs_spearman", "max"),
                                         lasso_keep_share=("lasso_keep_share", "max"),
                                         rfecv_rank=("rfecv_rank", "min"),
                                         permutation_rmse_increase=("permutation_rmse_increase", "sum"))
    agg["engineered"] = agg.index.isin(list(FEATURE_CATALOGUE))
    table = apply_votes(agg, set(redundant["drop_candidate"]), float(np.mean(np.asarray(y, float))), fe)
    return table, redundant, n_rfe


# --------------------------------------------------------------------------- #
# Candidate feature sets and comparison
# --------------------------------------------------------------------------- #
def candidate_sets(table: pd.DataFrame, config: PreprocessingConfig, fe: FeatureEngineeringConfig
                   ) -> dict[str, dict[str, Any]]:
    """Each candidate: {'engineered': [...], 'drop': [...], 'overrides': {...FE config overrides...}}."""
    original = set(config.numeric_features + config.categorical_features)
    domain = list(fe.domain.features)

    def from_mask(mask) -> dict[str, Any]:
        chosen = set(table.index[mask])
        return {"engineered": [f for f in domain if f in chosen], "drop": sorted(original - chosen), "overrides": {}}

    sets: dict[str, dict[str, Any]] = {
        "original": {"engineered": [], "drop": [], "overrides": {}},
        "original_plus_all_engineered": {"engineered": domain, "drop": [], "overrides": {}},
        "lasso_stable": from_mask(table["vote_lasso"]),
        "rfecv": from_mask(table["vote_rfecv"]),
        "redundancy_pruned": from_mask(table["vote_redundancy"]),
        f"consensus_{fe.selection.consensus_min_votes}_of_5": from_mask(
            table["votes"] >= fe.selection.consensus_min_votes),
    }
    if fe.decision.include_generic_candidates:
        g = fe.generators
        generic = {
            "generic_pairwise_interactions": {"generators": {"interactions": {"enabled": True,
                                                                              "columns": g.interactions.columns}}},
            "generic_polynomial": {"generators": {"polynomial": {"enabled": True, "columns": g.polynomial.columns,
                                                                 "degree": g.polynomial.degree}}},
            "generic_binning": {"generators": {"binning": {"enabled": True, "columns": g.binning.columns,
                                                           "n_bins": g.binning.n_bins,
                                                           "strategy": g.binning.strategy}}},
            "transform_yeo_johnson": {"numeric_transform": "yeo-johnson"},
            "transform_box_cox": {"numeric_transform": "box-cox"},
            "encoding_target": {"encoding": {"method": "target"}},
            "encoding_frequency": {"encoding": {"method": "frequency"}},
            "imputation_knn_with_indicators": {"imputation": {"numeric_strategy": "knn",
                                                              "add_missing_indicator": True}},
        }
        for name, ov in generic.items():
            sets[name] = {"engineered": [], "drop": [], "overrides": ov}
    return sets


def _model_for(regressor, config: PreprocessingConfig, fe: FeatureEngineeringConfig, spec: dict[str, Any],
               seed: int):
    fe_c = fe.with_overrides(spec.get("overrides", {})) if spec.get("overrides") else fe
    return make_fe_model(regressor, config, fe_c, spec["engineered"], spec["drop"], seed=seed)


def _fit_score(name: str, model, X: pd.DataFrame, y: pd.Series, fold: int, tr, va) -> dict[str, Any]:
    from sklearn.base import clone

    from .models import regression_metrics

    m = clone(model).fit(X.iloc[tr], y.iloc[tr])
    return {"model": name, "fold": fold, **regression_metrics(y.iloc[va], m.predict(X.iloc[va]))}


def parallel_cv(models: dict[str, Any], X: pd.DataFrame, y, splits: list[tuple], n_jobs: int = -1) -> pd.DataFrame:
    """Same output as ``models.cross_validate_models`` (one row per model and fold, identical numbers), but
    the independent (model, fold) fits run in parallel. Profiling showed the comparison is ~88% of the
    stage's run time and embarrassingly parallel."""
    from joblib import Parallel, delayed

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y, float))
    rows = Parallel(n_jobs=n_jobs)(delayed(_fit_score)(n, m, X, y, k, tr, va)
                                   for n, m in models.items() for k, (tr, va) in enumerate(splits))
    return pd.DataFrame(rows)


def compare_feature_sets(sets: dict[str, dict[str, Any]], config: PreprocessingConfig, fe: FeatureEngineeringConfig,
                         X, y, splits: list[tuple], seed: int, profiler: Profiler | None = None) -> pd.DataFrame:
    """Paired CV on the same folds as Stage 3 (Ridge benchmark), plus an optional gradient-boosting check."""
    nj = fe.compute.n_jobs
    prof = profiler or Profiler(enabled=False)
    models = {f"ridge|{n}": _model_for(RidgeCV(alphas=np.logspace(-3, 3, 25)), config, fe, s, seed)
              for n, s in sets.items()}
    with prof.step("comparison: Ridge feature sets", n_sets=len(sets), folds=len(splits)):
        folds = parallel_cv(models, X, y, splits, nj)
    if fe.decision.include_gradient_boosting_check:
        gb = {f"gboost|{n}": _model_for(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                      random_state=seed),
                                        tree_variant(config), fe, sets[n], seed)
              for n in ("original", "original_plus_all_engineered")}
        with prof.step("comparison: gradient-boosting check", folds=len(splits)):
            folds = pd.concat([folds, parallel_cv(gb, X, y, splits, nj)], ignore_index=True)
    return folds


def choose_feature_set(folds: pd.DataFrame, sets: dict[str, dict[str, Any]], config: PreprocessingConfig,
                       fe: FeatureEngineeringConfig) -> tuple[str, pd.DataFrame, str]:
    """Decision rule (thresholds from ``fe.decision``):

    1. A candidate must beat the original features on the paired folds (p < alpha) by at least
       ``min_gain`` of RMSE; otherwise the original features are kept.
    2. Among qualifying candidates, those within ``tie_tolerance`` of the best RMSE are tied; the one with
       the fewest model inputs wins (parsimony).
    """
    from .models import compare_models

    d = fe.decision
    ridge = folds[folds.model.str.startswith("ridge|")].copy()
    ridge["model"] = ridge.model.str.replace("ridge|", "", regex=False)
    mean = ridge.groupby("model").rmse.mean()
    base = mean["original"]
    n_original = len(config.numeric_features) + len(config.categorical_features)
    rows = []
    for name, spec in sets.items():
        c = compare_models(ridge, "original", name) if name != "original" else None
        rows.append({"feature_set": name, "kind": "generic" if spec.get("overrides") else "domain/selection",
                     "cv_rmse": mean[name], "gain_vs_original": 1 - mean[name] / base,
                     "folds_better": c["candidate_wins_folds"] if c else "-",
                     "p_value": c["paired_t_p_value"] if c else np.nan,
                     "n_engineered": len(spec["engineered"]), "n_dropped": len(spec["drop"])})
    table = pd.DataFrame(rows).set_index("feature_set")
    table["n_features"] = n_original + table["n_engineered"] - table["n_dropped"]
    ok = table[(table.index != "original") & (table.p_value < d.alpha) & (table.gain_vs_original >= d.min_gain)]
    if ok.empty:
        return "original", table, "No candidate beat the original features significantly; original features kept."
    best = ok.cv_rmse.min()
    tied = ok[ok.cv_rmse <= best * (1 + d.tie_tolerance)]
    chosen = str(tied.sort_values(["n_features", "cv_rmse"]).index[0])
    gain = table["gain_vs_original"].astype(float).to_dict()[chosen]
    p_value = table["p_value"].astype(float).to_dict()[chosen]
    note = (f"{chosen}: {gain:.1%} lower CV RMSE than the original features "
            f"(p = {p_value:.2g}), the smallest set within {d.tie_tolerance:.1%} of the best")
    return chosen, table, note


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_selection(table: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    votes = table[VOTE_COLUMNS].astype(int)
    votes.columns = [c.replace("vote_", "") for c in votes.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(table) + 1.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.imshow(votes.values, cmap="Greens", aspect="auto", vmin=0, vmax=1.3)
    ax.set_xticks(range(votes.shape[1]), votes.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(votes)), [f"{i}  ★" if table.loc[i, "engineered"] else i for i in votes.index])
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

    def color(m: str) -> str:
        if m == f"ridge|{chosen}":
            return "#DD8452"
        if m.startswith("gboost"):
            return "#8C8C8C"
        return "#64B5CD" if m.split("|")[1].startswith(("generic", "transform", "encoding", "imputation")) \
            else "#4C72B0"

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(s) + 1.6))
    ax.barh(range(len(s)), s["mean"], xerr=s["std"], color=[color(m) for m in s.index], capsize=3)
    for i, (m, sd) in enumerate(zip(s["mean"], s["std"])):
        ax.text(m + sd, i, f"  ${m:,.0f}", va="center", fontsize=8)
    ax.set_yticks(range(len(s)), [m.replace("|", ": ") for m in s.index], fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set_xlim(0, (s["mean"] + s["std"]).max() * 1.25)
    ax.set(xlabel="CV RMSE (15 paired folds), lower is better",
           title=f"Feature sets (orange = selected: {chosen}; light blue = generic options; grey = boosting check)")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)
