"""House price regression: preprocessing, feature engineering and modelling, end to end.

Usage examples
--------------
    python main.py                              # run everything
    python main.py --stage preprocess           # preprocessing only
    python main.py --stage features             # preprocessing + feature engineering & selection
    python main.py --stage baseline             # preprocessing + baseline models
    python main.py --stage advanced             # everything incl. advanced models (same as 'all')
    python main.py --n-iter 40                  # wider hyperparameter search
    python main.py --skip-nested                # faster: skip nested cross-validation
    python main.py --stage advanced             # stop after Stage 3 (skip optimisation)
    python main.py --cv-repeats 5               # more CV repeats for tighter estimates
    python main.py --scaler robust --outlier-method quantile

Stage 1: preprocess
    1. Load the raw CSV and report data quality.
    2. Row-level cleaning (duplicates, invalid targets).
    3. Location-stratified train/test split.
    4. Fit the preprocessing pipeline on the training set only; transform both sets.
    5. Save processed data, the fitted preprocessor and a JSON report.
    6. Robustness check: corrupt a copy of the test set and confirm the fitted
       pipeline still returns finite output.

Stage 1b: features
    6a. Create domain features (src/feature_engineering.py) inside the pipeline.
    6b. Score every original and engineered feature with five selection techniques
        (filter, redundancy, Lasso stability, RFECV, permutation importance).
    6c. Compare candidate feature sets on the same 15 paired folds as Stage 2.
    6d. Choose a set with a pre-declared rule; confirm once on the test set.

Stage 2: baseline
    7. Cross-validate the baseline ladder on identical, location-stratified folds
       (training set only).
    8. Paired comparisons between neighbouring baselines.
    9. Fit on the full training set, score once on the held-out test set.
   10. Per-location and per-price-band errors, coefficients, plots.
   11. Save the leaderboard, benchmark.json (the score to beat) and the model.

Stage 3: advanced
   12. Tune gradient boosting and a neural network (randomised search, training set only).
   13. Score the tuned models on the same 15 folds as the baselines; paired comparisons.
   14. Nested CV (re-tune inside each fold) for an unbiased estimate.
   15. Held-out test set, bias by price band and location, permutation importance,
       timing and learning curves.
   16. Decide whether any model replaces the benchmark; save results, plots, models.

Stage 4: optimize
   17. Compare cross-validation strategies (how stable is each estimate?).
   18. Optimise every model family: spline Ridge (under-fitting), regularised
       boosting (over-fitting), seed-averaged neural network (instability).
   19. Paired 15-fold CV and nested CV, before vs after.
   20. Validation curves, train-vs-validation fit diagnosis, seed stability.
   21. Select the final model with a pre-declared rule; save results and model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to files; no display needed

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.advanced_models import (
    ADVANCED_DESCRIPTIONS,
    get_advanced_models,
    learning_curve_table,
    nested_cross_validate,
    permutation_importance_table,
    plot_learning_curves,
    plot_model_comparison,
    plot_segment_bias,
    timing_table,
    tune_model,
)
from src.fe_config import FEConfigError, load_fe_config
from src.feature_engineering import catalogue_table
from src.feature_selection import (
    _model_for,
    candidate_sets,
    choose_feature_set,
    compare_feature_sets,
    plot_feature_set_comparison,
    plot_selection,
    selection_table,
)
from src.model_optimization import (
    SELECTION_RULE,
    SeedAveragedRegressor,
    compare_cv_strategies,
    fit_diagnosis_table,
    get_optimisation_candidates,
    nested_evaluate,
    plot_before_after,
    plot_cv_strategies,
    plot_validation_curves,
    seed_stability,
    select_final_model,
    tune_candidate,
    validation_curve_table,
)
from src.models import (
    BASELINE_DESCRIPTIONS,
    compare_models,
    cross_validate_models,
    evaluate_on_test,
    get_baseline_models,
    linear_coefficients,
    make_cv_splits,
    make_model,
    plot_cv_comparison,
    plot_diagnostics,
    price_band_metrics,
    regression_metrics,
    save_benchmark,
    save_model,
    segment_metrics,
    summarise_cv,
)
from src.preprocessing import (
    HousePricePreprocessor,
    PreprocessingConfig,
    clean_training_data,
    data_quality_report,
    load_data,
    split_data,
)
from src.profiling import Profiler, cprofile_call

logger = logging.getLogger("main")

BENCHMARK_MODEL = "ridge_log_target"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="House price preprocessing + baseline models")
    p.add_argument("--stage", choices=["preprocess", "features", "baseline", "advanced", "optimize", "all"],
                   default="all",
                   help="Run up to and including this stage ('advanced' and 'all' run everything)")
    p.add_argument("--data", default="synthetic_data_208.csv", help="Path to raw CSV")
    p.add_argument("--output-dir", default="artifacts", help="Where to write outputs")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scaler", choices=["standard", "robust", "minmax", "none"], default="standard")
    p.add_argument("--outlier-method", choices=["iqr", "quantile", "none"], default="iqr")
    p.add_argument("--iqr-factor", type=float, default=3.0)
    p.add_argument("--drop-features", nargs="*", default=[], help="Features to exclude")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=3)
    p.add_argument("--fe-config", default="configs/feature_engineering.yaml",
                   help="Feature engineering & selection config (YAML)")
    p.add_argument("--n-iter", type=int, default=20, help="Random-search trials per advanced model")
    p.add_argument("--skip-nested", action="store_true", help="Skip nested CV for advanced models")
    p.add_argument("--no-robustness-check", action="store_true", help="Skip the corrupted-data check")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Stage 1: preprocessing
# --------------------------------------------------------------------------- #
def corrupt(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Inject the problems the pipeline claims to handle."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    n = len(X)
    idx = lambda k: rng.choice(X.index, size=max(1, int(k * n)), replace=False)
    X["location_type"] = X["location_type"].astype(object)
    X.loc[idx(0.10), "property_size"] = np.nan               # missing
    X.loc[idx(0.05), "crime_rate_in_area"] = np.nan          # missing
    X.loc[idx(0.05), "location_type"] = np.nan               # missing category
    X.loc[idx(0.03), "location_type"] = "Coastal"            # unseen category
    X.loc[idx(0.03), "location_type"] = "  urban "           # messy formatting
    X.loc[idx(0.03), "number_of_bedrooms"] = -2              # impossible value
    X.loc[idx(0.03), "school_quality_score"] = 47            # out of range
    X.loc[idx(0.02), "property_size"] = 5_000_000            # extreme outlier
    X.loc[idx(0.02), "age_of_property"] = 0                  # zero before log
    X["number_of_bathrooms"] = X["number_of_bathrooms"].astype(object)
    X.loc[idx(0.02), "number_of_bathrooms"] = "three"        # wrong type
    return X


def run_preprocessing(args, config: PreprocessingConfig, out: Path) -> dict:
    raw = load_data(args.data)
    quality = data_quality_report(raw, config)
    logger.info("Missing values (raw): %s", sum(quality["missing_raw"].values()))
    logger.info("Invalid values converted to missing: %s", sum(quality["invalid_converted_to_missing"].values()))

    df, cleaning = clean_training_data(raw, config)
    logger.info("Cleaning: %s", cleaning)

    X_train, X_test, y_train, y_test = split_data(df, config, args.test_size, args.seed)
    logger.info("Split: train=%d test=%d", len(X_train), len(X_test))

    prep = HousePricePreprocessor(config)
    X_train_t, y_train_t = prep.fit_transform(X_train, y_train)
    X_test_t = prep.transform(X_test)
    y_test_t = prep.transform_target(y_test)
    summary = prep.fitted_summary()
    logger.info("Output features (%d): %s", len(prep.feature_names), prep.feature_names)
    logger.info("Values capped in training: %s", {k: v for k, v in summary["values_capped_in_training"].items() if v})

    X_train_t.to_csv(out / "X_train.csv", index=False)
    X_test_t.to_csv(out / "X_test.csv", index=False)
    pd.DataFrame({"y": y_train_t.values, "price": y_train.values}).to_csv(out / "y_train.csv", index=False)
    pd.DataFrame({"y": y_test_t.values, "price": y_test.values}).to_csv(out / "y_test.csv", index=False)
    prep.save(out / "preprocessor.joblib")

    report = {
        "config": config.to_dict(),
        "data_quality": quality,
        "cleaning": cleaning,
        "split": {"train": len(X_train), "test": len(X_test),
                  "train_location_mix": X_train["location_type"].value_counts(normalize=True).round(3).to_dict(),
                  "test_location_mix": X_test["location_type"].value_counts(normalize=True).round(3).to_dict()},
        "fitted": summary,
        "processed_train_stats": X_train_t.describe().loc[["mean", "std", "min", "max"]].round(3).to_dict(),
    }

    if not args.no_robustness_check:
        bad = corrupt(X_test, args.seed)
        bad_t = prep.transform(bad)
        ok = bool(np.isfinite(bad_t.to_numpy()).all() and bad_t.shape == X_test_t.shape)
        report["robustness_check"] = {"output_shape": list(bad_t.shape), "all_outputs_finite": ok}
        logger.info("Robustness check on corrupted test data: %s", "PASSED" if ok else "FAILED")
        if not ok:
            raise SystemExit("Pipeline produced non-finite output on corrupted data.")

    (out / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("Saved processed data, preprocessor and report to %s/", out)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# --------------------------------------------------------------------------- #
# Stage 1b: feature engineering and selection
# --------------------------------------------------------------------------- #
def run_features(args, config: PreprocessingConfig, data: dict, out: Path) -> dict:
    from sklearn.linear_model import RidgeCV

    try:
        fe = load_fe_config(args.fe_config)
    except FEConfigError as e:
        raise SystemExit(f"Feature-engineering configuration error: {e}") from e
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    fdir = out / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "fe_config_used.json").write_text(json.dumps(fe.to_dict(), indent=2))
    prof = Profiler(enabled=fe.profiling.enabled, track_memory=fe.profiling.track_memory)
    splits = make_cv_splits(X_train, args.cv_folds, args.cv_repeats, args.seed)

    catalogue_table(fe).to_csv(fdir / "engineered_feature_catalogue.csv", index=False)
    logger.info("Domain features enabled: %s", fe.domain.features)
    if not fe.temporal.date_columns:
        logger.info("Temporal features: no date columns configured (this dataset has none); generator is a no-op")

    if fe.profiling.cprofile_top:
        (table, redundant, n_rfe), report = cprofile_call(selection_table, config, fe, X_train, y_train, splits,
                                                          args.seed, prof, top=fe.profiling.cprofile_top)
        (fdir / "cprofile_selection.txt").write_text(report)
    else:
        table, redundant, n_rfe = selection_table(config, fe, X_train, y_train, splits, args.seed, prof)
    table.to_csv(fdir / "feature_selection_scores.csv")
    redundant.to_csv(fdir / "redundant_pairs.csv", index=False)
    plot_selection(table, fdir / "feature_selection_votes.png")

    sets = candidate_sets(table, config, fe)
    logger.info("Comparing %d candidate feature sets on %d paired folds", len(sets), len(splits))
    folds = compare_feature_sets(sets, config, fe, X_train, y_train, splits, args.seed, prof)
    chosen, comparison, note = choose_feature_set(folds, sets, config, fe)
    folds.to_csv(fdir / "feature_set_cv_folds.csv", index=False)
    comparison.to_csv(fdir / "feature_set_comparison.csv")
    plot_feature_set_comparison(folds, chosen, fdir / "feature_set_comparison.png")

    gb = folds[folds.model.str.startswith("gboost|")].groupby("model").rmse.mean()
    gb_gain = 1 - gb["gboost|original_plus_all_engineered"] / gb["gboost|original"] if len(gb) == 2 else None

    with prof.step("test-set confirmation"):
        test_rows = {}
        for name in dict.fromkeys(["original", chosen]):
            m = _model_for(RidgeCV(alphas=np.logspace(-3, 3, 25)), config, fe, sets[name], args.seed)
            test_rows[name] = regression_metrics(y_test, m.fit(X_train, y_train).predict(X_test))
    test = pd.DataFrame(test_rows).T
    test.to_csv(fdir / "feature_set_test.csv")
    profile = prof.table()
    if not profile.empty:
        profile.to_csv(fdir / "profile.csv", index=False)

    result = {"selected_feature_set": chosen, "decision": note, "engineered": sets[chosen]["engineered"],
              "dropped": sets[chosen]["drop"], "overrides": sets[chosen]["overrides"],
              "rfecv_optimal_n_columns": n_rfe, "candidate_sets": sets,
              "gradient_boosting_gain_all_engineered": gb_gain,
              "rule": {"alpha": fe.decision.alpha, "min_gain": fe.decision.min_gain,
                       "tie_tolerance": fe.decision.tie_tolerance},
              "test": test.round(4).to_dict(orient="index"), "config": args.fe_config}
    (fdir / "selected_features.json").write_text(json.dumps(result, indent=2, default=str))

    pd.set_option("display.width", 200)
    print("\nFEATURE SELECTION (votes to keep, of 5)\n" + table[["engineered", "mutual_info", "lasso_keep_share",
          "rfecv_rank", "permutation_rmse_increase", "votes"]].round(4).to_string())
    print("\nFEATURE SETS (Ridge benchmark, 15 paired folds)\n" + comparison.sort_values("cv_rmse").to_string(
        formatters={"cv_rmse": "${:,.0f}".format, "gain_vs_original": "{:.1%}".format,
                    "p_value": lambda v: "-" if pd.isna(v) else f"{v:.2g}"}))
    if gb_gain is not None:
        print(f"\nGradient-boosting check: all engineered features change CV RMSE by {-gb_gain:+.1%}")
    print("\nTEST SET (Ridge benchmark)\n" + test[["rmse", "mae", "mape", "r2"]].to_string(
        formatters={"rmse": "${:,.0f}".format, "mae": "${:,.0f}".format, "mape": "{:.2%}".format,
                    "r2": "{:.3f}".format}))
    if not profile.empty:
        print("\nPROFILE (feature engineering stage)\n" + profile[["step", "wall_seconds", "cpu_seconds",
              "peak_memory_mb", "share_of_wall"]].round(3).to_string(index=False))
    print(f"\nSELECTED FEATURE SET: {note}")
    logger.info("Saved feature engineering results to %s/", fdir)
    return result


# --------------------------------------------------------------------------- #
# Stage 2: baseline models
# --------------------------------------------------------------------------- #
def run_baseline(args, config: PreprocessingConfig, data: dict, out: Path) -> dict:
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)

    models = get_baseline_models(config)

    # 7. Cross-validation on the training set only
    splits = make_cv_splits(X_train, args.cv_folds, args.cv_repeats, args.seed)
    logger.info("Cross-validating %d baselines on %d folds (%dx%d, stratified by location)",
                len(models), len(splits), args.cv_repeats, args.cv_folds)
    folds = cross_validate_models(models, X_train, y_train, splits)
    summary = summarise_cv(folds)

    # 8. Paired comparisons along the ladder
    pairs = [("mean_predictor", "size_only_loglog"),
             ("size_only_loglog", "linear_log_target"),
             ("linear_raw_target", "linear_log_target"),
             ("linear_log_target", "ridge_log_target")]
    comparisons = [compare_models(folds, a, b) for a, b in pairs]

    # 9. Held-out test set
    test, fitted, preds = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    leaderboard = summary.join(test)
    leaderboard.insert(0, "description", [BASELINE_DESCRIPTIONS.get(m, "") for m in leaderboard.index])

    # 10. Diagnostics for the benchmark model
    bench = fitted[BENCHMARK_MODEL]
    by_location = segment_metrics(y_test, preds[BENCHMARK_MODEL], X_test["location_type"])
    by_band = price_band_metrics(y_test, preds[BENCHMARK_MODEL])
    coefs = linear_coefficients(bench)
    alpha = float(bench.named_steps["model"].regressor_.alpha_)
    smear = float(bench.named_steps["model"].smearing_factor_)

    # 11. Save everything
    folds.to_csv(mdir / "cv_fold_results.csv", index=False)
    leaderboard.to_csv(mdir / "baseline_leaderboard.csv")
    by_location.to_csv(mdir / "benchmark_errors_by_location.csv")
    by_band.to_csv(mdir / "benchmark_errors_by_price_band.csv")
    if coefs is not None:
        coefs.rename("coefficient").to_csv(mdir / "benchmark_coefficients.csv")
    plot_cv_comparison(summary, mdir / "cv_rmse_comparison.png", highlight=BENCHMARK_MODEL)
    plot_diagnostics(y_test, preds[BENCHMARK_MODEL], X_test["location_type"],
                     f"Benchmark ({BENCHMARK_MODEL}) on held-out test set", mdir / "benchmark_diagnostics.png")
    save_model(bench, mdir / "baseline_model.joblib")
    save_benchmark(mdir / "benchmark.json", BENCHMARK_MODEL, summary, test, comparisons,
                   extra={"ridge_alpha": alpha, "smearing_factor": smear,
                          "cv": {"folds": args.cv_folds, "repeats": args.cv_repeats, "seed": args.seed,
                                 "stratified_by": "location_type"},
                          "train_rows": len(X_train), "test_rows": len(X_test)})

    # Console summary
    pd.set_option("display.width", 200)
    view = leaderboard[["cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_mape_mean", "cv_r2_mean",
                        "rmse_reduction_vs_mean_predictor", "test_rmse", "test_r2"]].copy()
    fmt = {c: "${:,.0f}".format for c in ["cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "test_rmse"]}
    fmt.update({c: "{:.1%}".format for c in ["cv_mape_mean", "rmse_reduction_vs_mean_predictor"]})
    fmt.update({c: "{:.3f}".format for c in ["cv_r2_mean", "test_r2"]})
    print("\nBASELINE LEADERBOARD (sorted by CV RMSE)\n" + view.to_string(formatters=fmt))
    print("\nPAIRED COMPARISONS (RMSE, same folds)")
    for c in comparisons:
        print(f"  {c['candidate']:<18} vs {c['baseline']:<18} diff=${c['mean_difference']:>+10,.0f}  "
              f"wins {c['candidate_wins_folds']:>5}  p={c['paired_t_p_value']:.2g}  "
              f"significant={c['is_significant_improvement']}")
    print(f"\nBENCHMARK: {BENCHMARK_MODEL} (alpha={alpha:g}, smearing={smear:.4f})")
    print("Errors by location (test):\n" + by_location.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("Errors by predicted-price quartile (test):\n" + by_band.to_string(float_format=lambda v: f"{v:,.3f}"))
    logger.info("Saved leaderboard, benchmark.json, plots and model to %s/", mdir)
    return {"splits": splits, "folds": folds, "summary": summary, "test": test,
            "fitted": fitted, "preds": preds, "models": models}


# --------------------------------------------------------------------------- #
# Stage 3: advanced models
# --------------------------------------------------------------------------- #
def run_advanced(args, config: PreprocessingConfig, data: dict, base: dict, out: Path) -> dict:
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    adir = out / "advanced"
    adir.mkdir(parents=True, exist_ok=True)
    names = [BENCHMARK_MODEL, "gradient_boosting", "neural_network"]

    # 12. Tune on the training set only
    specs = get_advanced_models(X_train, config, args.seed)
    tuned = {n: tune_model(n, m, s, X_train, y_train, n_iter=args.n_iter, inner_folds=5,
                           random_state=args.seed) for n, (m, s) in specs.items()}
    models = {BENCHMARK_MODEL: base["models"][BENCHMARK_MODEL],
              **{n: t.best_estimator for n, t in tuned.items()}}

    # 13. Same 15 folds as the baselines -> paired comparisons
    adv_folds = cross_validate_models({n: tuned[n].best_estimator for n in tuned}, X_train, y_train, base["splits"])
    folds = pd.concat([base["folds"], adv_folds], ignore_index=True)
    std_cv = folds[folds.model.isin(names)].groupby("model")["rmse"].agg(["mean", "std"]).reindex(names)
    paired = [compare_models(folds, BENCHMARK_MODEL, n) for n in tuned]
    paired.append(compare_models(folds, "gradient_boosting", "neural_network"))

    # 14. Nested CV (unbiased): first repeat of the outer folds, re-tuned in each fold
    nested_cv, nested_params, nested_paired, nf = None, pd.DataFrame(), [], None
    if not args.skip_nested:
        outer = base["splits"][: args.cv_folds]
        logger.info("Nested CV: %d outer folds x %d trials x 3 inner folds (this is the slow part)",
                    len(outer), args.n_iter)
        nf, nested_params = nested_cross_validate(
            {BENCHMARK_MODEL: base["models"][BENCHMARK_MODEL], **specs}, X_train, y_train, outer,
            n_iter=args.n_iter, inner_folds=3, random_state=args.seed)
        nested_cv = nf.groupby("model")["rmse"].agg(["mean", "std"]).reindex(names)
        nested_paired = [compare_models(nf, BENCHMARK_MODEL, n) for n in tuned]
        nf.to_csv(adir / "nested_cv_fold_results.csv", index=False)
        nested_params.to_csv(adir / "nested_cv_chosen_hyperparameters.csv", index=False)

    # 15. Test set and analysis
    test, fitted, preds = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    cv_summary = summarise_cv(folds[folds.model.isin(names)], reference=BENCHMARK_MODEL)
    leaderboard = cv_summary.join(test).reindex(names)
    leaderboard.insert(0, "description", [BASELINE_DESCRIPTIONS.get(n) or ADVANCED_DESCRIPTIONS.get(n)
                                          for n in names])
    if nested_cv is not None:
        leaderboard.insert(1, "nested_cv_rmse_mean", nested_cv["mean"])
        leaderboard.insert(2, "nested_cv_rmse_std", nested_cv["std"])
    loc_tbl = {n: segment_metrics(y_test, preds[n], X_test["location_type"]) for n in names}
    band_tbl = {n: price_band_metrics(y_test, preds[n]) for n in names}
    loc_bias = pd.DataFrame({n: t["bias"] for n, t in loc_tbl.items()})
    band_bias = pd.DataFrame({n: t["bias"] for n, t in band_tbl.items()})
    band_rmse = pd.DataFrame({n: t["rmse"] for n, t in band_tbl.items()})
    importance = permutation_importance_table(fitted, X_test, y_test)
    timing = timing_table(models, X_train, y_train, X_test)
    lc = learning_curve_table(models, X_train, y_train)

    # 16. Benchmark decision: a model must win significantly on the 15 paired folds
    #     AND have a lower nested-CV RMSE (if nested CV ran) to replace the benchmark.
    winners = [c["candidate"] for c in paired[:2] if c["is_significant_improvement"]]
    if nested_cv is not None:
        winners = [w for w in winners if nested_cv.loc[w, "mean"] < nested_cv.loc[BENCHMARK_MODEL, "mean"]]
    selected = min(winners, key=lambda w: std_cv.loc[w, "mean"]) if winners else BENCHMARK_MODEL
    decision = (f"{selected} replaces {BENCHMARK_MODEL}" if selected != BENCHMARK_MODEL else
                f"No advanced model beat {BENCHMARK_MODEL} reliably; the benchmark stands")

    # Save
    adv_folds.to_csv(adir / "advanced_cv_fold_results.csv", index=False)
    leaderboard.to_csv(adir / "model_comparison.csv")
    importance.to_csv(adir / "permutation_importance.csv")
    timing.to_csv(adir / "timing.csv")
    lc.to_csv(adir / "learning_curves.csv", index=False)
    band_bias.to_csv(adir / "bias_by_price_band.csv")
    loc_bias.to_csv(adir / "bias_by_location.csv")
    for n, t in tuned.items():
        t.top_trials.to_csv(adir / f"tuning_top10_{n}.csv", index=False)
        save_model(t.best_estimator, adir / f"{n}.joblib")
    save_model(fitted[selected], adir / "selected_model.joblib")
    plot_model_comparison(std_cv, nested_cv, adir / "model_comparison.png")
    plot_segment_bias(band_bias, loc_bias, adir / "segment_bias.png")
    plot_learning_curves(lc, adir / "learning_curves.png")
    for n in tuned:
        plot_diagnostics(y_test, preds[n], X_test["location_type"], f"{n} on held-out test set",
                         adir / f"{n}_diagnostics.png")
    results = {
        "decision": decision, "selected_model": selected,
        "tuning": {n: {"best_params": t.best_params, "inner_cv_rmse": t.best_inner_cv_rmse,
                       "seconds": round(t.seconds, 1)} for n, t in tuned.items()},
        "paired_comparisons_15_folds": paired,
        "nested_cv": None if nested_cv is None else {
            "rmse": nested_cv.round(1).to_dict(orient="index"), "paired_comparisons": nested_paired},
        "test": test.round(4).to_dict(orient="index"),
        "settings": {"n_iter": args.n_iter, "cv_folds": args.cv_folds, "cv_repeats": args.cv_repeats,
                     "seed": args.seed},
    }
    (adir / "advanced_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Console summary
    money = lambda v: f"${v:,.0f}"
    print("\nADVANCED MODELS: TUNED HYPERPARAMETERS")
    for n, t in tuned.items():
        print(f"  {n:<18} inner CV RMSE {money(t.best_inner_cv_rmse)}  {t.best_params}")
    cols = ["cv_rmse_mean", "cv_rmse_std"] + (["nested_cv_rmse_mean"] if nested_cv is not None else []) + \
           ["cv_mae_mean", "cv_mape_mean", "cv_r2_mean", "test_rmse", "test_mape", "test_r2"]
    fmt = {c: money for c in ["cv_rmse_mean", "cv_rmse_std", "nested_cv_rmse_mean", "cv_mae_mean", "test_rmse"]}
    fmt.update({c: "{:.1%}".format for c in ["cv_mape_mean", "test_mape"]})
    fmt.update({c: "{:.3f}".format for c in ["cv_r2_mean", "test_r2"]})
    print("\nMODEL COMPARISON\n" + leaderboard[cols].to_string(formatters=fmt))
    print("\nPAIRED COMPARISONS (RMSE, same 15 folds; negative diff = candidate better)")
    for c in paired + nested_paired:
        tag = " [nested]" if c in nested_paired else ""
        print(f"  {c['candidate']:<18} vs {c['baseline']:<18} diff={c['mean_difference']:>+9,.0f}  "
              f"wins {c['candidate_wins_folds']:>5}  p={c['paired_t_p_value']:.2g}{tag}")
    print("\nRMSE by predicted-price quartile (test)\n" + band_rmse.to_string(float_format=money))
    print("\nBias by predicted-price quartile (test)\n" + band_bias.to_string(float_format=money))
    print("\nPermutation importance (RMSE increase, test)\n" + importance.to_string(float_format=money))
    print("\nTiming\n" + timing.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nDECISION: {decision}")
    logger.info("Saved advanced results, plots and models to %s/", adir)
    return {"folds": folds, "nested_folds": nf, "specs": specs, "tuned": tuned, "fitted": fitted,
            "selected": selected}


# --------------------------------------------------------------------------- #
# Stage 4: optimisation
# --------------------------------------------------------------------------- #
def run_optimize(args, config: PreprocessingConfig, data: dict, base: dict, adv: dict, out: Path) -> None:
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    odir = out / "optimization"
    odir.mkdir(parents=True, exist_ok=True)
    ridge = base["models"][BENCHMARK_MODEL]
    stage4 = {n: t.best_estimator for n, t in adv["tuned"].items()}
    outer = base["splits"][: args.cv_folds]

    # 17. CV strategy comparison (on the cheap benchmark model)
    logger.info("Comparing CV strategies over 10 seeds")
    cvs = compare_cv_strategies(ridge, X_train, y_train, seeds=range(10))

    # 18. Optimise every family on the full training set
    cands = get_optimisation_candidates(X_train, config, n_iter=args.n_iter, random_state=args.seed)
    opt = {c.name: tune_candidate(c, X_train, y_train, inner_folds=5, seed=args.seed) for c in cands}
    optimised = {n: o.estimator for n, o in opt.items()}

    # 19a. Same 15 paired folds as every earlier model
    opt_folds = cross_validate_models(optimised, X_train, y_train, base["splits"])
    folds = pd.concat([adv["folds"], opt_folds], ignore_index=True)
    before_of = {"ridge_splines": BENCHMARK_MODEL, "gradient_boosting_opt": "gradient_boosting",
                 "neural_network_opt": "neural_network"}
    vs_ridge = {n: compare_models(folds, BENCHMARK_MODEL, n) for n in optimised}
    vs_before = {n: compare_models(folds, before_of[n], n) for n in optimised if before_of[n] != BENCHMARK_MODEL}
    all_names = [BENCHMARK_MODEL, "ridge_splines", "gradient_boosting", "gradient_boosting_opt",
                 "neural_network", "neural_network_opt"]
    cv15 = folds[folds.model.isin(all_names)].groupby("model")["rmse"].agg(["mean", "std"]).reindex(all_names)

    # 19b. Nested CV (unbiased); Stage 3 nested results are reused for the "before" models
    nested, nested_params, nested_vs_ridge = None, pd.DataFrame(), {}
    if not args.skip_nested and adv["nested_folds"] is not None:
        logger.info("Nested CV for optimised models (the slow part)")
        nf_opt, nested_params = nested_evaluate(cands, X_train, y_train, outer, inner_folds=3, seed=args.seed)
        nf_all = pd.concat([adv["nested_folds"], nf_opt], ignore_index=True)
        nested = nf_all.groupby("model")["rmse"].agg(["mean", "std"]).reindex(all_names)
        nested_vs_ridge = {n: compare_models(nf_all, BENCHMARK_MODEL, n) for n in optimised}
        nf_all.to_csv(odir / "nested_cv_fold_results.csv", index=False)
        nested_params.to_csv(odir / "nested_cv_chosen_hyperparameters.csv", index=False)

    # 20. Diagnostics
    every = {BENCHMARK_MODEL: ridge, "ridge_splines": optimised["ridge_splines"],
             "gradient_boosting": stage4["gradient_boosting"],
             "gradient_boosting_opt": optimised["gradient_boosting_opt"],
             "neural_network": stage4["neural_network"], "neural_network_opt": optimised["neural_network_opt"]}
    diag = fit_diagnosis_table(every, X_train, y_train, outer)

    spline_base = optimised["ridge_splines"]
    if spline_base.named_steps["expand"] == "passthrough":   # search chose plain Ridge: curve on a spline version
        spline_base = clone(cands[0].model)
    gb_base = optimised["gradient_boosting_opt"]
    nn_single = clone(adv["specs"]["neural_network"][0]).set_params(
        **{f"model__regressor__{k}": v for k, v in
           [("hidden_layer_sizes", opt["neural_network_opt"].best_params["hidden_layer_sizes"]),
            ("activation", opt["neural_network_opt"].best_params["activation"]),
            ("alpha", opt["neural_network_opt"].best_params["alpha"])]})
    logger.info("Computing validation curves")
    vc = pd.concat([
        validation_curve_table(spline_base, "expand__splines__n_knots", [3, 4, 5, 6, 8, 12, 20, 30],
                               X_train, y_train, "Spline Ridge: flexibility (knots)"),
        validation_curve_table(gb_base, "model__regressor__max_leaf_nodes", [2, 3, 4, 6, 8, 16, 32, 64],
                               X_train, y_train, "Gradient boosting: tree size (leaves)"),
        validation_curve_table(nn_single, "model__regressor__alpha", [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100],
                               X_train, y_train, "Neural network: L2 strength (alpha)"),
    ], ignore_index=True)

    logger.info("Measuring neural-network seed stability")
    nn_net = optimised["neural_network_opt"].named_steps["model"].regressor.estimator
    stab = seed_stability(nn_single,
                          lambda b: make_model(SeedAveragedRegressor(nn_net, n_seeds=5, base_seed=b), config),
                          X_train, y_train, outer)
    stab_summary = stab.groupby("variant", sort=False)["cv_rmse"].agg(["mean", "min", "max", "std"])
    stab_summary["range"] = stab_summary["max"] - stab_summary["min"]

    # Test set (touched once, for reporting only)
    test, fitted, preds = evaluate_on_test(every, X_train, y_train, X_test, y_test)

    # 21. Selection
    if nested is not None:
        final, note = select_final_model(nested["mean"], vs_ridge)
    else:
        final, note = select_final_model(cv15["mean"], vs_ridge)
        note += " (nested CV skipped: decided on 15-fold CV, which is optimistic for tuned models)"

    # Save
    cvs.to_csv(odir / "cv_strategy_comparison.csv")
    opt_folds.to_csv(odir / "optimised_cv_fold_results.csv", index=False)
    diag.to_csv(odir / "fit_diagnosis.csv")
    vc.to_csv(odir / "validation_curves.csv", index=False)
    stab.to_csv(odir / "seed_stability.csv", index=False)
    test.to_csv(odir / "test_results.csv")
    for n, o in opt.items():
        o.trials.head(10).to_csv(odir / f"tuning_top10_{n}.csv", index=False)
        save_model(o.estimator, odir / f"{n}.joblib")
    save_model(fitted[final], odir / "final_model.joblib")
    comp = pd.DataFrame({"cv15_rmse_mean": cv15["mean"], "cv15_rmse_std": cv15["std"]})
    if nested is not None:
        comp["nested_rmse_mean"], comp["nested_rmse_std"] = nested["mean"], nested["std"]
    comp = comp.join(diag[["train_rmse", "gap_pct", "diagnosis"]]).join(test[["test_rmse", "test_mape", "test_r2"]])
    comp.to_csv(odir / "optimisation_summary.csv")
    if nested is not None:
        plot_before_after(nested, odir / "before_after.png")
    plot_validation_curves(vc, odir / "validation_curves.png")
    plot_cv_strategies(cvs, odir / "cv_strategies.png")
    results = {
        "final_model": final, "decision": note, "selection_rule": SELECTION_RULE,
        "tuned": {n: {"best_params": o.best_params, "inner_cv_rmse": o.inner_cv_rmse, "seconds": round(o.seconds, 1)}
                  for n, o in opt.items()},
        "paired_vs_ridge_15_folds": vs_ridge, "paired_vs_stage3_15_folds": vs_before,
        "nested_paired_vs_ridge": nested_vs_ridge,
        "seed_stability": stab_summary.round(1).to_dict(orient="index"),
        "settings": {"n_iter": args.n_iter, "seed": args.seed, "outer_folds": len(outer)},
    }
    (odir / "optimization_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Console summary
    money = lambda v: f"${v:,.0f}"
    print("\nCV STRATEGY COMPARISON (benchmark model, 10 seeds)\n"
          + cvs[["mean_estimate", "sd_across_seeds", "range_across_seeds", "fits_per_estimate"]]
          .to_string(formatters={"mean_estimate": money, "sd_across_seeds": money, "range_across_seeds": money}))
    print("\nOPTIMISED HYPERPARAMETERS")
    for n, o in opt.items():
        print(f"  {n:<22} inner CV RMSE {money(o.inner_cv_rmse)}  {o.best_params}")
    fmt = {c: money for c in ["cv15_rmse_mean", "cv15_rmse_std", "nested_rmse_mean", "nested_rmse_std",
                              "train_rmse", "test_rmse"]}
    fmt.update({"gap_pct": "{:.0%}".format, "test_mape": "{:.1%}".format, "test_r2": "{:.3f}".format})
    print("\nOPTIMISATION SUMMARY (before vs after)\n" + comp.to_string(formatters=fmt))
    print("\nPAIRED COMPARISONS (RMSE; negative diff = optimised model better)")
    for label, d in [("vs Ridge, 15 folds", vs_ridge), ("vs Stage 3 version, 15 folds", vs_before),
                     ("vs Ridge, nested", nested_vs_ridge)]:
        for c in d.values():
            print(f"  {c['candidate']:<22} {label:<28} diff={c['mean_difference']:>+9,.0f}  "
                  f"wins {c['candidate_wins_folds']:>5}  p={c['paired_t_p_value']:.2g}")
    print("\nNEURAL NETWORK SEED STABILITY (5-fold CV RMSE)\n" + stab_summary.to_string(float_format=money))
    print(f"\nFINAL MODEL: {final}\n  {note}")
    logger.info("Saved optimisation results, plots and models to %s/", odir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = PreprocessingConfig(
        scaler=args.scaler,
        outlier_method=args.outlier_method,
        iqr_factor=args.iqr_factor,
        drop_features=args.drop_features,
    )
    data = run_preprocessing(args, config, out)
    if args.stage in ("features", "all"):
        run_features(args, config, data, out)
    if args.stage in ("baseline", "advanced", "optimize", "all"):
        base = run_baseline(args, config, data, out)
        if args.stage in ("advanced", "optimize", "all"):
            adv = run_advanced(args, config, data, base, out)
            if args.stage in ("optimize", "all"):
                run_optimize(args, config, data, base, adv, out)


if __name__ == "__main__":
    main()
