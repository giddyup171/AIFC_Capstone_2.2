"""Baseline regression models and a reusable evaluation framework.

Purpose
-------
Establish the benchmark every later model has to beat, and provide the
tools to compare models fairly:

* A ladder of baselines from naive to informed, so each gain is measured:
  mean / median predictors -> size-only model -> linear models.
* One set of metrics computed in dollars on the original price scale.
* Repeated, location-stratified cross-validation with identical folds for
  every model, so comparisons are paired and not driven by split luck.
* A held-out test evaluation, per-location error breakdown, diagnostics plots,
  and a saved ``benchmark.json`` holding the score to beat.

Every model is a full Pipeline (``build_preprocessor`` -> regressor), so the
preprocessing is re-fitted inside each CV fold and nothing leaks from
validation data into training.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .preprocessing import PreprocessingConfig, build_preprocessor

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "rmse"
# For each metric: True if higher is better.
METRIC_DIRECTION = {
    "r2": True, "rmse": False, "mae": False, "mape": False,
    "median_ae": False, "max_error": False, "rmsle": False, "bias": None,
}


# --------------------------------------------------------------------------- #
# Log-target wrapper
# --------------------------------------------------------------------------- #
class LogTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit any regressor on log(price) and predict back in dollars.

    The EDA showed price is right-skewed and close to log-linear in the
    features. ``exp(prediction)`` estimates the median price, which is
    biased low for the mean, so predictions are multiplied by Duan's smearing
    factor (mean of exp(training residuals)).
    """

    def __init__(self, regressor=None, smearing: bool = True):
        self.regressor = regressor
        self.smearing = smearing

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("LogTargetRegressor requires strictly positive targets.")
        log_y = np.log(y)
        self.regressor_ = clone(self.regressor if self.regressor is not None else LinearRegression())
        self.regressor_.fit(X, log_y)
        residuals = log_y - self.regressor_.predict(X)
        self.smearing_factor_ = float(np.mean(np.exp(residuals))) if self.smearing else 1.0
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        check_is_fitted(self, "regressor_")
        return np.exp(self.regressor_.predict(X)) * self.smearing_factor_


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """All metrics on the original dollar scale.

    rmse      primary metric: penalises large misses on expensive homes
    mae       typical dollar error, robust to a few big misses
    mape      relative error, comparable across cheap and expensive homes
    r2        share of price variance explained
    rmsle     error on the log scale, matches how the models are trained
    median_ae error on a typical house
    max_error worst single miss
    bias      mean (prediction - actual); >0 means systematic over-prediction
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": float(np.sqrt(np.mean((np.log(np.clip(y_pred, 1, None)) - np.log(y_true)) ** 2))),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def make_model(regressor, config: PreprocessingConfig | None = None, log_target: bool = True) -> Pipeline:
    """Wrap any regressor with the preprocessing pipeline (and optional log target)."""
    config = config or PreprocessingConfig()
    model = LogTargetRegressor(regressor) if log_target else regressor
    return Pipeline([("preprocess", build_preprocessor(config)), ("model", model)])


def get_baseline_models(config: PreprocessingConfig | None = None) -> dict[str, Pipeline]:
    """The baseline ladder, from naive to the main benchmark.

    Add new candidates with ``make_model`` and pass them to the same
    evaluation functions to compare like for like.
    """
    config = config or PreprocessingConfig()
    size_only = PreprocessingConfig(**{
        **config.to_dict(),
        "drop_features": [c for c in config.numeric_features + config.categorical_features
                          if c != "property_size"],
    })
    alphas = np.logspace(-3, 3, 25)
    return {
        "mean_predictor": make_model(DummyRegressor(strategy="mean"), config, log_target=False),
        "median_predictor": make_model(DummyRegressor(strategy="median"), config, log_target=False),
        "size_only_loglog": make_model(LinearRegression(), size_only, log_target=True),
        "linear_raw_target": make_model(LinearRegression(), config, log_target=False),
        "linear_log_target": make_model(LinearRegression(), config, log_target=True),
        "ridge_log_target": make_model(RidgeCV(alphas=alphas), config, log_target=True),
    }


BASELINE_DESCRIPTIONS = {
    "mean_predictor": "Predicts the training mean price for every house (floor any model must clear)",
    "median_predictor": "Predicts the training median price for every house",
    "size_only_loglog": "log(price) ~ log(property_size): the single strongest EDA driver",
    "linear_raw_target": "OLS on all processed features, raw dollar target",
    "linear_log_target": "OLS on all processed features, log target with smearing",
    "ridge_log_target": "RidgeCV (alpha tuned by internal CV), log target: the main benchmark",
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def make_cv_splits(X: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3,
                   random_state: int = 42, stratify_col: str = "location_type") -> list[tuple]:
    """Fixed fold indices, reused for every model so comparisons are paired."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(cv.split(X, X[stratify_col].fillna("missing")))


def cross_validate_models(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
                          splits: list[tuple]) -> pd.DataFrame:
    """Return one row per (model, fold) with every metric."""
    rows = []
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    for name, model in models.items():
        for fold, (tr, va) in enumerate(splits):
            fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
            m = regression_metrics(y.iloc[va], fitted.predict(X.iloc[va]))
            rows.append({"model": name, "fold": fold, **m})
        logger.info("CV done: %-20s RMSE=$%s", name,
                    f"{np.mean([r['rmse'] for r in rows if r['model'] == name]):,.0f}")
    return pd.DataFrame(rows)


def summarise_cv(fold_results: pd.DataFrame, reference: str = "mean_predictor") -> pd.DataFrame:
    """Mean and std per model, sorted by the primary metric, with % gain over a reference."""
    metrics = [c for c in fold_results.columns if c not in ("model", "fold")]
    agg = fold_results.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = [f"cv_{m}_{s}" for m, s in agg.columns]
    agg = agg.sort_values(f"cv_{PRIMARY_METRIC}_mean")
    if reference in agg.index:
        ref = agg.loc[reference, f"cv_{PRIMARY_METRIC}_mean"]
        agg[f"rmse_reduction_vs_{reference}"] = 1 - agg[f"cv_{PRIMARY_METRIC}_mean"] / ref
    return agg


def compare_models(fold_results: pd.DataFrame, baseline: str, candidate: str,
                   metric: str = PRIMARY_METRIC) -> dict[str, Any]:
    """Paired comparison on identical folds.

    A candidate counts as an improvement only if it is better on average AND
    the paired t-test across folds is significant (p < 0.05). This stops a
    lucky split from being reported as progress.
    """
    pivot = fold_results.pivot(index="fold", columns="model", values=metric)
    diff = pivot[candidate] - pivot[baseline]
    higher_better = METRIC_DIRECTION.get(metric, False)
    better = diff > 0 if higher_better else diff < 0
    t, p = stats.ttest_rel(pivot[candidate], pivot[baseline])
    mean_diff = float(diff.mean())
    improved = bool((mean_diff > 0 if higher_better else mean_diff < 0) and p < 0.05)
    return {
        "baseline": baseline, "candidate": candidate, "metric": metric,
        "mean_difference": mean_diff, "std_difference": float(diff.std()),
        "candidate_wins_folds": f"{int(better.sum())}/{len(diff)}",
        "paired_t_p_value": float(p), "is_significant_improvement": improved,
    }


def evaluate_on_test(models: dict[str, Pipeline], X_train, y_train, X_test, y_test
                     ) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Fit each model on the full training set and score once on the held-out test set."""
    fitted, preds, rows = {}, {}, []
    for name, model in models.items():
        fitted[name] = clone(model).fit(X_train, y_train)
        preds[name] = fitted[name].predict(X_test)
        rows.append({"model": name, **{f"test_{k}": v for k, v in regression_metrics(y_test, preds[name]).items()}})
    return pd.DataFrame(rows).set_index("model"), fitted, preds


def segment_metrics(y_true, y_pred, segments: pd.Series) -> pd.DataFrame:
    """Error broken down by a segment (e.g. location) to expose uneven performance."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "seg": np.asarray(segments)})
    rows = {seg: {"n": len(g), **regression_metrics(g["y"], g["p"])} for seg, g in df.groupby("seg")}
    return pd.DataFrame(rows).T[["n", "rmse", "mae", "mape", "r2", "bias"]]


def price_band_metrics(y_true, y_pred, n_bands: int = 4, by: str = "predicted") -> pd.DataFrame:
    """Error by price quartile: does the model struggle with cheap or expensive homes?

    Bands are formed on the **predicted** price by default. Grouping on the
    actual price selects on the outcome: the top band collects houses whose
    random noise pushed them up, so any model (even a perfect one plus noise)
    appears to under-predict expensive homes and over-predict cheap ones
    (regression to the mean). Use ``by="actual"`` only to show that effect.

    R² is left out: inside a narrow price band it is not meaningful.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    key = pd.Series(y_pred if by == "predicted" else y_true)
    bands = pd.qcut(key, n_bands, labels=[f"Q{i + 1}" for i in range(n_bands)])
    return segment_metrics(y_true, y_pred, bands).drop(columns="r2")


def linear_coefficients(fitted_pipeline: Pipeline) -> pd.Series | None:
    """Coefficients of a fitted linear model, named by processed feature.

    With a log target and standardised inputs, a coefficient b means a one
    standard deviation increase multiplies price by about exp(b).
    """
    model = fitted_pipeline.named_steps["model"]
    inner = getattr(model, "regressor_", model)
    if not hasattr(inner, "coef_"):
        return None
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(inner.coef_, index=names).sort_values(key=abs, ascending=False)


# --------------------------------------------------------------------------- #
# Plots and persistence
# --------------------------------------------------------------------------- #
def plot_cv_comparison(summary: pd.DataFrame, path: str | Path, highlight: str | None = None) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    path = Path(path)
    s = summary.sort_values("cv_rmse_mean", ascending=False)
    colors = ["#DD8452" if m == highlight else "#4C72B0" for m in s.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(s.index, s["cv_rmse_mean"], xerr=s["cv_rmse_std"], color=colors, capsize=3)
    for i, (v, sd) in enumerate(zip(s["cv_rmse_mean"], s["cv_rmse_std"])):
        ax.text(v + sd, i, f"  ${v:,.0f}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set_xlabel("Cross-validated RMSE, lower is better")
    title = "Baseline models: repeated stratified CV (mean ± std)"
    ax.set_title(title + (f"\nbenchmark highlighted: {highlight}" if highlight else ""))
    ax.set_xlim(0, (s["cv_rmse_mean"] + s["cv_rmse_std"]).max() * 1.22)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_diagnostics(y_true, y_pred, segments, title: str, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_pred - y_true
    pct = 100 * resid / y_true
    seg = pd.Series(np.asarray(segments))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for s in sorted(seg.unique()):
        m = (seg == s).to_numpy()
        ax.scatter(y_true[m], y_pred[m], s=12, alpha=0.6, label=s)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Actual price ($)", ylabel="Predicted price ($)",
           title="Predicted vs actual (log axes)")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(y_pred, pct, s=12, alpha=0.6)
    ax.axhline(0, color="k", lw=1)
    ax.set(xscale="log", xlabel="Predicted price ($)", ylabel="Error (% of actual)",
           title="Percentage error vs prediction")

    ax = axes[1, 0]
    ax.hist(pct, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="Error (% of actual)", ylabel="Count", title="Distribution of percentage errors")

    ax = axes[1, 1]
    groups = sorted(seg.unique())
    ax.boxplot([np.abs(pct[(seg == g).to_numpy()]) for g in groups])
    ax.set_xticks(range(1, len(groups) + 1), groups)
    ax.set(ylabel="Absolute error (% of actual)", title="Absolute % error by location")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_benchmark(path: str | Path, best_name: str, summary: pd.DataFrame, test: pd.DataFrame,
                   comparisons: list[dict], extra: dict | None = None) -> Path:
    """Write the score-to-beat file that later model experiments compare against."""
    path = Path(path)
    row = summary.loc[best_name]
    benchmark = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_model": best_name,
        "description": BASELINE_DESCRIPTIONS.get(best_name, ""),
        "primary_metric": PRIMARY_METRIC,
        "score_to_beat": {
            "cv_rmse_mean": float(row["cv_rmse_mean"]),
            "cv_rmse_std": float(row["cv_rmse_std"]),
            "cv_mae_mean": float(row["cv_mae_mean"]),
            "cv_mape_mean": float(row["cv_mape_mean"]),
            "cv_r2_mean": float(row["cv_r2_mean"]),
            "test_rmse": float(test.loc[best_name, "test_rmse"]),
            "test_r2": float(test.loc[best_name, "test_r2"]),
        },
        "improvement_rule": "A new model must have lower mean CV RMSE on the same folds "
                            "and a paired t-test p < 0.05 (see compare_models).",
        "paired_comparisons": comparisons,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        **(extra or {}),
    }
    path.write_text(json.dumps(benchmark, indent=2, default=float))
    return path


def save_model(model: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
