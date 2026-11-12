"""Advanced models with architectures distinct from the linear baseline.

Two candidates, each chosen to test a specific weakness of the Ridge benchmark:

1. Gradient boosting (``HistGradientBoostingRegressor``): a tree ensemble.
   Learns non-linear effects and interactions automatically, with no
   assumption about functional form. Monotonic constraints encode the EDA's
   directional findings (price rises with size, school quality and bedrooms,
   falls with crime and age); the tuner decides whether to use them.

2. Neural network (``MLPRegressor``): a feed-forward network trained with
   L-BFGS, which suits small datasets better than SGD/Adam. Learns smooth
   non-linear functions of all features at once; strong L2 regularisation
   (``alpha``) is tuned to stop it fitting noise on 800 rows.

Both are trained on log(price) with smearing (``LogTargetRegressor``) and reuse
the Stage 2 preprocessing, so differences come from the model alone.

Evaluation guards against two common traps:

* Tuning bias: scoring a model on the same data used to choose its
  hyperparameters flatters it. ``nested_cross_validate`` re-tunes inside each
  outer fold, giving an unbiased estimate.
* Split luck: all models are scored on identical folds and compared with the
  paired test in ``src.models.compare_models``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, RandomizedSearchCV, learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .models import make_model, regression_metrics
from .preprocessing import PreprocessingConfig

logger = logging.getLogger(__name__)

SCORING = "neg_root_mean_squared_error"
P = "model__regressor__"  # parameter prefix inside Pipeline -> LogTargetRegressor -> estimator

# Direction of each feature's effect on price, from the EDA (1 = up, -1 = down).
EDA_MONOTONIC_DIRECTIONS = {
    "property_size": 1,
    "school_quality_score": 1,
    "number_of_bedrooms": 1,
    "crime_rate_in_area": -1,
    "age_of_property": -1,
}

ADVANCED_DESCRIPTIONS = {
    "gradient_boosting": "HistGradientBoosting tree ensemble, log target, tuned (optional EDA monotonic constraints)",
    "neural_network": "MLP neural network (L-BFGS, L2-regularised), log target, tuned",
}


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #
def tree_config(config: PreprocessingConfig | None = None) -> PreprocessingConfig:
    """Trees are invariant to scaling and robust to outliers, so skip those steps."""
    config = config or PreprocessingConfig()
    return PreprocessingConfig(**{**config.to_dict(), "scaler": "none", "outlier_method": "none"})


def monotonic_constraints(feature_names) -> list[int]:
    return [EDA_MONOTONIC_DIRECTIONS.get(f, 0) for f in feature_names]


def build_gradient_boosting(config: PreprocessingConfig | None = None, random_state: int = 42
                            ) -> tuple[Pipeline, dict]:
    """Return (unfitted pipeline, hyperparameter search space)."""
    cfg = tree_config(config)
    model = make_model(HistGradientBoostingRegressor(random_state=random_state, early_stopping=False), cfg)
    space = {
        P + "learning_rate": loguniform(0.01, 0.2),
        P + "max_iter": randint(100, 800),
        P + "max_leaf_nodes": randint(4, 32),
        P + "min_samples_leaf": randint(5, 60),
        P + "l2_regularization": loguniform(1e-3, 10),
        P + "max_features": uniform(0.5, 0.5),
        # None = unconstrained; the list is filled in by attach_monotonic_option()
        P + "monotonic_cst": [None],
    }
    return model, space


def attach_monotonic_option(model: Pipeline, space: dict, X_sample: pd.DataFrame) -> dict:
    """Add the EDA monotonic constraints as a search option (needs output feature names)."""
    names = clone(model.named_steps["preprocess"]).fit(X_sample).get_feature_names_out()
    space = dict(space)
    space[P + "monotonic_cst"] = [None, monotonic_constraints(names)]
    return space


def build_neural_network(config: PreprocessingConfig | None = None, random_state: int = 42
                         ) -> tuple[Pipeline, dict]:
    """Return (unfitted pipeline, hyperparameter search space).

    Inputs are standardised by the Stage 2 pipeline, which neural networks need.
    """
    cfg = config or PreprocessingConfig()
    model = make_model(MLPRegressor(solver="lbfgs", max_iter=5000, random_state=random_state), cfg)
    space = {
        P + "hidden_layer_sizes": [(8,), (16,), (32,), (64,), (16, 8), (32, 16)],
        P + "alpha": loguniform(1e-4, 30),
        P + "activation": ["relu", "tanh"],
    }
    return model, space


def get_advanced_models(X_train: pd.DataFrame, config: PreprocessingConfig | None = None,
                        random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Both advanced models with their search spaces, ready for tuning."""
    gb, gb_space = build_gradient_boosting(config, random_state)
    gb_space = attach_monotonic_option(gb, gb_space, X_train)
    nn, nn_space = build_neural_network(config, random_state)
    return {"gradient_boosting": (gb, gb_space), "neural_network": (nn, nn_space)}


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
@dataclass
class TuningResult:
    name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_inner_cv_rmse: float
    seconds: float
    top_trials: pd.DataFrame = field(repr=False)


def make_search(model: Pipeline, space: dict, n_iter: int = 20, inner_folds: int = 3,
                random_state: int = 42, n_jobs: int = -1) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        model, space, n_iter=n_iter, scoring=SCORING, n_jobs=n_jobs, random_state=random_state,
        cv=KFold(inner_folds, shuffle=True, random_state=random_state + 1), refit=True,
    )


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        k = k.replace(P, "")
        if k == "monotonic_cst":
            v = "EDA constraints" if v is not None else "none"
        elif isinstance(v, (np.floating, float)):
            v = float(f"{v:.4g}")
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


def tune_model(name: str, model: Pipeline, space: dict, X, y, n_iter: int = 20, inner_folds: int = 5,
               random_state: int = 42, n_jobs: int = -1) -> TuningResult:
    """Randomised search on the full training set. Returns the refitted best pipeline."""
    t0 = time.perf_counter()
    search = make_search(model, space, n_iter, inner_folds, random_state, n_jobs).fit(X, y)
    secs = time.perf_counter() - t0
    trials = pd.DataFrame(search.cv_results_)
    trials = trials.sort_values("rank_test_score").head(10)
    top = pd.DataFrame({
        "rank": trials["rank_test_score"].to_numpy(),
        "inner_cv_rmse": -trials["mean_test_score"].to_numpy(),
        "params": [_clean_params(p) for p in trials["params"]],
    })
    logger.info("Tuned %-18s in %.0fs: inner CV RMSE=$%s", name, secs, f"{-search.best_score_:,.0f}")
    return TuningResult(name, search.best_estimator_, _clean_params(search.best_params_),
                        float(-search.best_score_), secs, top)


# --------------------------------------------------------------------------- #
# Unbiased evaluation
# --------------------------------------------------------------------------- #
def nested_cross_validate(models: dict[str, Pipeline | tuple[Pipeline, dict]], X: pd.DataFrame, y,
                          outer_splits: list[tuple], n_iter: int = 20, inner_folds: int = 3,
                          random_state: int = 42, n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-tune inside every outer fold so the outer score never sees tuning data.

    ``models`` maps name -> pipeline (no tuning) or (pipeline, space) (tuned).
    Returns (fold metrics, hyperparameters chosen in each fold).
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    rows, chosen = [], []
    for name, spec in models.items():
        for fold, (tr, va) in enumerate(outer_splits):
            if isinstance(spec, tuple):
                search = make_search(spec[0], spec[1], n_iter, inner_folds, random_state + fold, n_jobs)
                fitted = search.fit(X.iloc[tr], y.iloc[tr]).best_estimator_
                chosen.append({"model": name, "fold": fold, **_clean_params(search.best_params_)})
            else:
                fitted = clone(spec).fit(X.iloc[tr], y.iloc[tr])
            rows.append({"model": name, "fold": fold,
                         **regression_metrics(y.iloc[va], fitted.predict(X.iloc[va]))})
        logger.info("Nested CV done: %-18s RMSE=$%s", name,
                    f"{np.mean([r['rmse'] for r in rows if r['model'] == name]):,.0f}")
    return pd.DataFrame(rows), pd.DataFrame(chosen)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def timing_table(models: dict[str, Pipeline], X_train, y_train, X_test, repeats: int = 3) -> pd.DataFrame:
    """Fit time on the training set and prediction time per 1,000 rows."""
    rows = []
    for name, model in models.items():
        fit_t, pred_t = [], []
        for _ in range(repeats):
            m = clone(model)
            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            fit_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            m.predict(X_test)
            pred_t.append(time.perf_counter() - t0)
        rows.append({"model": name, "fit_seconds": float(np.median(fit_t)),
                     "predict_ms_per_1000_rows": float(np.median(pred_t) * 1000 * 1000 / len(X_test))})
    return pd.DataFrame(rows).set_index("model")


def permutation_importance_table(fitted: dict[str, Pipeline], X_test, y_test, n_repeats: int = 10,
                                 random_state: int = 42) -> pd.DataFrame:
    """RMSE increase ($) when each raw input column is shuffled, on the test set.

    Model-agnostic, so it compares what the linear, tree and neural models rely on.
    """
    out = {}
    for name, model in fitted.items():
        r = permutation_importance(model, X_test, y_test, scoring=SCORING, n_repeats=n_repeats,
                                   random_state=random_state, n_jobs=-1)
        out[name] = pd.Series(r.importances_mean, index=X_test.columns)
    return pd.DataFrame(out).sort_values(next(iter(out)), ascending=False)


def learning_curve_table(models: dict[str, Pipeline], X, y, train_sizes=(0.2, 0.4, 0.6, 0.8, 1.0),
                         cv_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Validation RMSE as training data grows: would more data help the flexible models?"""
    rows = []
    for name, model in models.items():
        sizes, tr, va = learning_curve(
            model, X, y, train_sizes=list(train_sizes), scoring=SCORING, n_jobs=-1,
            cv=KFold(cv_folds, shuffle=True, random_state=random_state), shuffle=True, random_state=random_state,
        )
        for n, a, b in zip(sizes, tr, va):
            rows.append({"model": name, "train_rows": int(n), "train_rmse": float(-a.mean()),
                         "val_rmse": float(-b.mean()), "val_rmse_std": float(b.std())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
COLORS = {"ridge_log_target": "#4C72B0", "gradient_boosting": "#55A868", "neural_network": "#DD8452"}


def plot_model_comparison(standard: pd.DataFrame, nested: pd.DataFrame | None, path: str | Path) -> Path:
    """CV RMSE per model: tuned-then-CV (15 folds) vs nested CV."""
    import matplotlib.pyplot as plt

    names = list(standard.index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(names))
    h = 0.38 if nested is not None else 0.6
    ax.barh(y + (h / 2 if nested is not None else 0), standard["mean"], h, xerr=standard["std"], capsize=3,
            color=[COLORS.get(n, "#8172B2") for n in names], label="Tuned on full train, 15-fold CV")
    for i, (m, s) in enumerate(zip(standard["mean"], standard["std"])):
        ax.text(m + s, y[i] + (h / 2 if nested is not None else 0), f"  ${m:,.0f}", va="center", fontsize=9)
    if nested is not None:
        nested = nested.reindex(names)
        ax.barh(y - h / 2, nested["mean"], h, xerr=nested["std"], capsize=3, color="#BBBBBB",
                hatch="//", edgecolor="white", label="Nested CV (re-tuned in each fold), 5-fold")
        for i, (m, s) in enumerate(zip(nested["mean"], nested["std"])):
            ax.text(m + s, y[i] - h / 2, f"  ${m:,.0f}", va="center", fontsize=9, color="#555")
    ax.set_yticks(y, names)
    ax.set_xlabel("RMSE ($), lower is better")
    ax.set_title("Advanced models vs Ridge benchmark")
    upper = max((standard["mean"] + standard["std"]).max(),
                (nested["mean"] + nested["std"]).max() if nested is not None else 0)
    ax.set_xlim(0, upper * 1.25)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_segment_bias(band_bias: pd.DataFrame, loc_bias: pd.DataFrame, path: str | Path) -> Path:
    """Mean error (prediction - actual) by price quartile and by location, per model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, df, title in [(axes[0], band_bias, "Bias by predicted-price quartile (test)"),
                          (axes[1], loc_bias, "Bias by location (test)")]:
        x = np.arange(len(df.index))
        w = 0.8 / len(df.columns)
        for i, m in enumerate(df.columns):
            ax.bar(x + (i - (len(df.columns) - 1) / 2) * w, df[m], w, label=m, color=COLORS.get(m))
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x, df.index)
        ax.set_ylabel("Mean error ($): + over, − under")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_learning_curves(lc: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, g in lc.groupby("model", sort=False):
        c = COLORS.get(m)
        ax.plot(g["train_rows"], g["val_rmse"], "o-", color=c, label=f"{m} (validation)")
        ax.plot(g["train_rows"], g["train_rmse"], "--", color=c, alpha=0.6, label=f"{m} (training)")
    ax.set_xlabel("Training rows")
    ax.set_ylabel("RMSE ($)")
    ax.set_title("Learning curves: training vs validation error")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)
