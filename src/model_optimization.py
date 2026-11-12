"""Model optimisation: cross-validation strategy, hyperparameter tuning and
fixes for over- and under-fitting, applied to every model family.

What Stage 4 left open, and how this module addresses it
--------------------------------------------------------
* Ridge (linear) may be **under-fitting**: it can only draw straight lines in
  log space. ``build_spline_ridge`` adds cubic B-spline expansions of the
  numeric features, so each feature gets a smooth curve while the model stays
  additive and regularised. The search can still pick "no splines".
* Gradient boosting **over-fits** (training RMSE about a third of validation
  RMSE). ``build_regularised_boosting`` searches a space biased towards
  shallow trees, larger leaves, stronger L2, feature subsampling and optional
  early stopping.
* The neural network is **unstable**: its score depends on the random
  starting weights (and on the maths library). ``SeedAveragedRegressor``
  trains several networks from different seeds and averages them, which cuts
  that variance without changing the architecture.

Evaluation
----------
* ``compare_cv_strategies`` measures how much each CV scheme's estimate moves
  when only the random split changes, to justify the scheme used.
* ``nested_evaluate`` re-tunes inside each outer fold (unbiased) and supports
  a ``wrap`` step, so the network can be tuned on one seed and evaluated as a
  seed average exactly as it will be deployed.
* ``validation_curve_table`` and ``diagnose_fit`` show where each model sits
  between under- and over-fitting.
* ``select_final_model`` applies a rule declared before the results were seen.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
    validation_curve,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

from .advanced_models import monotonic_constraints, tree_config
from .models import LogTargetRegressor, make_model, regression_metrics
from .preprocessing import PreprocessingConfig, build_preprocessor

logger = logging.getLogger(__name__)

SCORING = "neg_root_mean_squared_error"
P = "model__regressor__"

OPTIMISED_DESCRIPTIONS = {
    "ridge_splines": "Ridge on log target with cubic spline expansion of numeric features (tuned knots/degree)",
    "gradient_boosting_opt": ("Gradient boosting with a regularisation-focused search "
                              "(shallow trees, big leaves, L2, early stopping)"),
    "neural_network_opt": "MLP tuned on one seed, deployed as an average of 5 networks from different seeds",
}


# --------------------------------------------------------------------------- #
# 1. Cross-validation strategies
# --------------------------------------------------------------------------- #
def cv_strategies(seed: int) -> dict[str, Any]:
    """Candidate validation schemes. Each value yields (train_idx, val_idx) pairs."""
    return {
        "holdout_80_20": "holdout",
        "kfold_5": KFold(5, shuffle=True, random_state=seed),
        "stratified_kfold_5": StratifiedKFold(5, shuffle=True, random_state=seed),
        "repeated_stratified_5x3": RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed),
        "repeated_stratified_10x3": RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed),
    }


def _splits(strategy, X: pd.DataFrame, seed: int, strat: pd.Series) -> list[tuple]:
    idx = np.arange(len(X))
    if strategy == "holdout":
        tr, va = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
        return [(tr, va)]
    if isinstance(strategy, (StratifiedKFold, RepeatedStratifiedKFold)):
        return list(strategy.split(X, strat))
    return list(strategy.split(X))


def compare_cv_strategies(model: Pipeline, X: pd.DataFrame, y, seeds=range(10),
                          stratify_col: str = "location_type") -> pd.DataFrame:
    """How stable is each scheme's RMSE estimate when only the random split changes?

    For each strategy and seed, the mean validation RMSE is recorded. The
    standard deviation across seeds is the estimate's own noise: a scheme
    with a large value can rank models differently just by re-shuffling.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    strat = X[stratify_col]
    rows = []
    for seed in seeds:
        for name, strategy in cv_strategies(seed).items():
            t0 = time.perf_counter()
            scores = []
            for tr, va in _splits(strategy, X, seed, strat):
                m = clone(model).fit(X.iloc[tr], y.iloc[tr])
                scores.append(regression_metrics(y.iloc[va], m.predict(X.iloc[va]))["rmse"])
            rows.append({"strategy": name, "seed": seed, "rmse_estimate": float(np.mean(scores)),
                         "n_fits": len(scores), "seconds": time.perf_counter() - t0})
    raw = pd.DataFrame(rows)
    out = raw.groupby("strategy", sort=False).agg(
        mean_estimate=("rmse_estimate", "mean"),
        sd_across_seeds=("rmse_estimate", "std"),
        min_estimate=("rmse_estimate", "min"),
        max_estimate=("rmse_estimate", "max"),
        fits_per_estimate=("n_fits", "first"),
        seconds_per_estimate=("seconds", "mean"),
    )
    out["range_across_seeds"] = out["max_estimate"] - out["min_estimate"]
    return out


# --------------------------------------------------------------------------- #
# 2. Optimised model families
# --------------------------------------------------------------------------- #
class SeedAveragedRegressor(RegressorMixin, BaseEstimator):
    """Average of ``n_seeds`` copies of ``estimator`` trained from different seeds.

    Reduces variance that comes from random initialisation (and from small
    floating-point differences between machines), at ``n_seeds`` times the cost.
    """

    def __init__(self, estimator=None, n_seeds: int = 5, base_seed: int = 0):
        self.estimator = estimator
        self.n_seeds = n_seeds
        self.base_seed = base_seed

    def fit(self, X, y):
        self.estimators_ = [clone(self.estimator).set_params(random_state=self.base_seed + i).fit(X, y)
                            for i in range(self.n_seeds)]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)


def build_spline_ridge(config: PreprocessingConfig | None = None) -> tuple[Pipeline, list[dict]]:
    """Ridge with optional spline expansion. The search may choose plain Ridge."""
    cfg = config or PreprocessingConfig()
    num = cfg.model_numeric_features
    splines = ColumnTransformer(
        [("splines", SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), num)],
        remainder="passthrough", verbose_feature_names_out=False,
    )
    model = Pipeline([
        ("preprocess", build_preprocessor(cfg)),
        ("expand", splines),
        ("model", LogTargetRegressor(RidgeCV(alphas=np.logspace(-3, 3, 25)))),
    ])
    space = [
        {"expand": ["passthrough"]},  # plain Ridge, so splines must earn their place
        {"expand__splines__n_knots": randint(3, 9),
         "expand__splines__degree": [2, 3],
         "expand__splines__knots": ["uniform", "quantile"]},
    ]
    return model, space


def build_regularised_boosting(X_sample: pd.DataFrame, config: PreprocessingConfig | None = None,
                               random_state: int = 42) -> tuple[Pipeline, dict]:
    cfg = tree_config(config)
    model = make_model(HistGradientBoostingRegressor(random_state=random_state, validation_fraction=0.15,
                                                     n_iter_no_change=30), cfg)
    names = clone(model.named_steps["preprocess"]).fit(X_sample).get_feature_names_out()
    space = {
        P + "learning_rate": loguniform(0.01, 0.3),
        P + "max_iter": randint(50, 1500),
        P + "max_depth": [2, 3, 4, None],
        P + "max_leaf_nodes": randint(3, 16),
        P + "min_samples_leaf": randint(10, 100),
        P + "l2_regularization": loguniform(1e-2, 100),
        P + "max_features": uniform(0.3, 0.7),
        P + "early_stopping": [True, False],
        P + "monotonic_cst": [None, monotonic_constraints(names)],
    }
    return model, space


def build_seed_averaged_network(config: PreprocessingConfig | None = None, random_state: int = 42
                                ) -> tuple[Pipeline, dict, Callable[[Pipeline], Pipeline]]:
    """Single-seed network to tune, plus a ``wrap`` that turns the tuned one into a 5-seed average."""
    cfg = config or PreprocessingConfig()
    model = make_model(MLPRegressor(solver="lbfgs", max_iter=5000, random_state=random_state), cfg)
    space = {
        P + "hidden_layer_sizes": [(4,), (8,), (16,), (32,), (64,), (16, 8)],
        P + "alpha": loguniform(1e-3, 100),
        P + "activation": ["relu", "tanh"],
    }

    def wrap(tuned: Pipeline, n_seeds: int = 5) -> Pipeline:
        net = tuned.named_steps["model"].regressor
        return make_model(SeedAveragedRegressor(net, n_seeds=n_seeds), cfg)

    return model, space, wrap


@dataclass
class Candidate:
    """A model family to optimise: pipeline, search space and optional post-tuning wrap."""
    name: str
    model: Pipeline
    space: Any
    n_iter: int
    wrap: Callable[[Pipeline], Pipeline] | None = None


def get_optimisation_candidates(X_train: pd.DataFrame, config: PreprocessingConfig | None = None,
                                n_iter: int = 20, random_state: int = 42) -> list[Candidate]:
    r_model, r_space = build_spline_ridge(config)
    g_model, g_space = build_regularised_boosting(X_train, config, random_state)
    n_model, n_space, n_wrap = build_seed_averaged_network(config, random_state)
    return [
        Candidate("ridge_splines", r_model, r_space, n_iter),
        Candidate("gradient_boosting_opt", g_model, g_space, n_iter),
        Candidate("neural_network_opt", n_model, n_space, n_iter, n_wrap),
    ]


# --------------------------------------------------------------------------- #
# 3. Tuning and unbiased evaluation
# --------------------------------------------------------------------------- #
def _search(c: Candidate, inner_folds: int, seed: int, n_jobs: int) -> RandomizedSearchCV:
    return RandomizedSearchCV(c.model, c.space, n_iter=c.n_iter, scoring=SCORING, n_jobs=n_jobs,
                              cv=KFold(inner_folds, shuffle=True, random_state=seed + 1),
                              random_state=seed, refit=True)


def _clean(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        k = k.replace(P, "").replace("expand__splines__", "spline_")
        if k == "expand":
            v = "no splines" if v == "passthrough" else "splines"
        elif k == "monotonic_cst":
            v = "EDA constraints" if v is not None else "none"
        elif isinstance(v, (np.floating, float)):
            v = float(f"{v:.4g}")
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


@dataclass
class OptimisedModel:
    name: str
    estimator: Pipeline            # final, refitted on the full training set (wrapped if needed)
    best_params: dict
    inner_cv_rmse: float
    seconds: float
    trials: pd.DataFrame = field(repr=False)


def tune_candidate(c: Candidate, X, y, inner_folds: int = 5, seed: int = 42, n_jobs: int = -1) -> OptimisedModel:
    t0 = time.perf_counter()
    search = _search(c, inner_folds, seed, n_jobs).fit(X, y)
    final = search.best_estimator_
    if c.wrap is not None:
        final = c.wrap(final).fit(X, y)
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    trials = pd.DataFrame({"rank": res["rank_test_score"].to_numpy(),
                           "inner_cv_rmse": -res["mean_test_score"].to_numpy(),
                           "params": [_clean(p) for p in res["params"]]})
    secs = time.perf_counter() - t0
    logger.info("Optimised %-22s in %.0fs: inner CV RMSE=$%s", c.name, secs, f"{-search.best_score_:,.0f}")
    return OptimisedModel(c.name, final, _clean(search.best_params_), float(-search.best_score_), secs, trials)


def nested_evaluate(candidates: list[Candidate], X: pd.DataFrame, y, outer_splits: list[tuple],
                    inner_folds: int = 3, seed: int = 42, n_jobs: int = -1
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested CV: tune (and wrap) inside each outer fold, score on the untouched outer fold."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    rows, chosen = [], []
    for c in candidates:
        for fold, (tr, va) in enumerate(outer_splits):
            search = _search(c, inner_folds, seed + fold, n_jobs).fit(X.iloc[tr], y.iloc[tr])
            final = search.best_estimator_
            if c.wrap is not None:
                final = c.wrap(final).fit(X.iloc[tr], y.iloc[tr])
            pred = final.predict(X.iloc[va])
            train_rmse = regression_metrics(y.iloc[tr], final.predict(X.iloc[tr]))["rmse"]
            rows.append({"model": c.name, "fold": fold, "train_rmse": train_rmse,
                         **regression_metrics(y.iloc[va], pred)})
            chosen.append({"model": c.name, "fold": fold, **_clean(search.best_params_)})
        logger.info("Nested CV done: %-22s RMSE=$%s", c.name,
                    f"{np.mean([r['rmse'] for r in rows if r['model'] == c.name]):,.0f}")
    return pd.DataFrame(rows), pd.DataFrame(chosen)


# --------------------------------------------------------------------------- #
# 4. Over- / under-fitting diagnostics
# --------------------------------------------------------------------------- #
def validation_curve_table(model: Pipeline, param: str, values: list, X, y, label: str,
                           cv_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Training and validation RMSE across one hyperparameter (the bias-variance trade-off)."""
    tr, va = validation_curve(model, X, y, param_name=param, param_range=values, scoring=SCORING,
                              cv=KFold(cv_folds, shuffle=True, random_state=seed), n_jobs=-1)
    return pd.DataFrame({"curve": label, "param": param.replace(P, "").replace("expand__splines__", ""),
                         "value": [str(v) for v in values], "train_rmse": -tr.mean(axis=1),
                         "val_rmse": -va.mean(axis=1), "val_rmse_std": va.std(axis=1)})


def diagnose_fit(train_rmse: float, val_rmse: float, reference_rmse: float,
                 gap_tol: float = 0.15, under_tol: float = 0.05) -> str:
    """Label a model's position on the bias-variance trade-off.

    over-fitting  : validation error exceeds training error by more than ``gap_tol`` (15%)
    under-fitting : small gap, but validation error is more than ``under_tol`` (5%)
                    above the best validation error any model achieved (``reference_rmse``),
                    i.e. the model leaves learnable signal on the table
    good fit      : otherwise
    """
    gap = (val_rmse - train_rmse) / val_rmse
    if gap > gap_tol:
        return "over-fitting"
    if val_rmse > reference_rmse * (1 + under_tol):
        return "under-fitting"
    return "good fit"


def fit_diagnosis_table(models: dict[str, Pipeline], X, y, splits: list[tuple]) -> pd.DataFrame:
    """Mean training and validation RMSE over the given folds, plus a diagnosis label."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    rows = []
    for name, model in models.items():
        tr_s, va_s = [], []
        for tr, va in splits:
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            tr_s.append(regression_metrics(y.iloc[tr], m.predict(X.iloc[tr]))["rmse"])
            va_s.append(regression_metrics(y.iloc[va], m.predict(X.iloc[va]))["rmse"])
        rows.append({"model": name, "train_rmse": np.mean(tr_s), "val_rmse": np.mean(va_s)})
    df = pd.DataFrame(rows).set_index("model")
    df["gap_pct"] = (df["val_rmse"] - df["train_rmse"]) / df["val_rmse"]
    ref = df["val_rmse"].min()
    df["diagnosis"] = [diagnose_fit(t, v, ref) for t, v in zip(df["train_rmse"], df["val_rmse"])]
    return df


def seed_stability(single: Pipeline, averaged_factory: Callable[[int], Pipeline], X, y,
                   splits: list[tuple], n_single: int = 6, n_sets: int = 3) -> pd.DataFrame:
    """How much does CV RMSE move when only the network's random initialisation changes?"""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    def cv_rmse(model):
        return float(np.mean([regression_metrics(y.iloc[va], clone(model).fit(X.iloc[tr], y.iloc[tr])
                                                 .predict(X.iloc[va]))["rmse"] for tr, va in splits]))

    rows = [{"variant": "single network", "init": s,
             "cv_rmse": cv_rmse(clone(single).set_params(**{P + "random_state": s}))} for s in range(n_single)]
    rows += [{"variant": "5-network average", "init": s, "cv_rmse": cv_rmse(averaged_factory(100 * (s + 1)))}
             for s in range(n_sets)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Final selection (rule fixed before results were seen)
# --------------------------------------------------------------------------- #
SELECTION_RULE = (
    "Take the optimised model with the lowest nested-CV RMSE. It becomes the final model only if "
    "(a) it beats the Ridge benchmark significantly on the 15 paired folds (p < 0.05) AND "
    "(b) its nested-CV RMSE is at least 1% lower than Ridge's (practical-significance margin added "
    "after Stage 4 showed that 'lower' alone can be a $556 coin flip). Otherwise Ridge is kept. "
    "Among candidates within 1% of each other on nested RMSE, the simplest (Ridge family) is preferred."
)

SIMPLICITY_ORDER = ["ridge_log_target", "ridge_splines", "neural_network_opt", "gradient_boosting_opt"]


def select_final_model(nested_rmse: pd.Series, paired: dict[str, dict], benchmark: str = "ridge_log_target",
                       margin: float = 0.01) -> tuple[str, str]:
    """Apply ``SELECTION_RULE``. ``paired`` maps candidate -> compare_models() result vs the benchmark."""
    ref = nested_rmse[benchmark]
    eligible = [m for m, c in paired.items()
                if c["is_significant_improvement"] and nested_rmse[m] <= ref * (1 - margin)]
    if not eligible:
        return benchmark, f"No optimised model met both conditions; {benchmark} is kept."
    best = min(eligible, key=lambda m: nested_rmse[m])
    near = [m for m in eligible if nested_rmse[m] <= nested_rmse[best] * (1 + margin)]
    choice = min(near, key=lambda m: SIMPLICITY_ORDER.index(m) if m in SIMPLICITY_ORDER else 99)
    gain = 1 - nested_rmse[choice] / ref
    note = f"{choice} selected: nested-CV RMSE {gain:.1%} below {benchmark} and significant on paired folds"
    if choice != best:
        note += f" (preferred over {best}, which is within {margin:.0%} but more complex)"
    return choice, note


# --------------------------------------------------------------------------- #
# 6. Plots
# --------------------------------------------------------------------------- #
COLORS = {"ridge_log_target": "#4C72B0", "ridge_splines": "#64B5CD", "gradient_boosting": "#55A868",
          "gradient_boosting_opt": "#8BC48A", "neural_network": "#DD8452", "neural_network_opt": "#F0B27A"}


def _money_axis(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def plot_before_after(nested: pd.DataFrame, path) -> None:
    """Nested-CV RMSE before (Stage 3/4) and after optimisation, per family."""
    import matplotlib.pyplot as plt

    pairs = [("Linear", "ridge_log_target", "ridge_splines"),
             ("Gradient boosting", "gradient_boosting", "gradient_boosting_opt"),
             ("Neural network", "neural_network", "neural_network_opt")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(pairs))
    for i, (_label, before, after) in enumerate(pairs):
        for off, m, txt in [(0.2, before, "before"), (-0.2, after, "after")]:
            if m in nested.index:
                v, s = nested.loc[m, "mean"], nested.loc[m, "std"]
                ax.barh(i + off, v, 0.38, xerr=s, capsize=3, color=COLORS.get(m),
                        hatch="" if txt == "after" else "//", edgecolor="white")
                ax.text(v + s, i + off, f"  ${v:,.0f} ({txt}: {m})", va="center", fontsize=8)
    ax.set_yticks(y, [p[0] for p in pairs])
    _money_axis(ax)
    ax.set_xlim(0, (nested["mean"] + nested["std"]).max() * 1.7)
    ax.set_xlabel("Nested-CV RMSE (re-tuned inside each fold), lower is better")
    ax.set_title("Before vs after optimisation (hatched = before)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_validation_curves(vc: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    curves = list(vc["curve"].unique())
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, curves):
        g = vc[vc["curve"] == name]
        x = np.arange(len(g))
        ax.plot(x, g["train_rmse"], "o--", label="training", color="#888")
        ax.plot(x, g["val_rmse"], "o-", label="validation", color="#C44E52")
        ax.fill_between(x, g["val_rmse"] - g["val_rmse_std"], g["val_rmse"] + g["val_rmse_std"],
                        color="#C44E52", alpha=0.12)
        best = int(np.argmin(g["val_rmse"].to_numpy()))
        ax.axvline(best, color="k", lw=0.8, ls=":")
        ax.set_xticks(x, g["value"], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(g["param"].iloc[0])
        ax.set_title(name, fontsize=10)
        _money_axis(ax, "y")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation curves: left of the dotted line = under-fitting side, right = over-fitting side "
                 "(reversed for regularisation strength)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cv_strategies(cvs: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(cvs))
    ax.barh(y, cvs["sd_across_seeds"], color="#4C72B0")
    for i, (sd, n) in enumerate(zip(cvs["sd_across_seeds"], cvs["fits_per_estimate"])):
        ax.text(sd, i, f"  ${sd:,.0f}  ({n} fits)", va="center", fontsize=9)
    ax.set_yticks(y, cvs.index)
    ax.invert_yaxis()
    _money_axis(ax)
    ax.set_xlim(0, cvs["sd_across_seeds"].max() * 1.45)
    ax.set_xlabel("SD of the RMSE estimate across 10 random seeds (lower = more reliable)")
    ax.set_title("How much does each CV scheme's answer depend on the random split?")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
