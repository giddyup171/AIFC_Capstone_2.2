import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from src.fe_config import FeatureEngineeringConfig, from_dict
from src.feature_engineering import make_fe_model
from src.feature_selection import (
    VOTE_COLUMNS,
    apply_votes,
    candidate_sets,
    choose_feature_set,
    parallel_cv,
    redundancy_pairs,
    selection_table,
)
from src.models import cross_validate_models
from src.profiling import Profiler


@pytest.fixture(scope="module")
def table(data, config, small_splits):
    X_train, _, y_train, _ = data
    return selection_table(config, FeatureEngineeringConfig(), X_train, y_train, small_splits, seed=42)


def test_selection_table_shape_and_votes(table, config):
    tbl, redundant, n_rfe = table
    original = set(config.numeric_features + config.categorical_features)
    expected = original | set(FeatureEngineeringConfig().domain.features)
    assert set(tbl.index) == expected                       # one row per feature, dummies merged
    assert set(VOTE_COLUMNS) <= set(tbl.columns)
    assert tbl["votes"].between(0, 5).all() and n_rfe >= 3
    assert "property_size" in tbl.index[tbl["votes"] == 5]  # the dominant driver is kept by every technique


def test_selection_is_stable_for_a_fixed_seed(data, config, small_splits, table):
    X_train, _, y_train, _ = data
    again, _, _ = selection_table(config, FeatureEngineeringConfig(), X_train, y_train, small_splits, seed=42)
    pd.testing.assert_frame_equal(table[0], again)


def test_votes_follow_configured_thresholds():
    agg = pd.DataFrame({"mutual_info": [0.02, 0.005], "lasso_keep_share": [0.7, 0.9],
                        "rfecv_rank": [1, 2], "permutation_rmse_increase": [300.0, 100.0]}, index=["f1", "f2"])
    strict = apply_votes(agg, {"f2"}, 100_000, FeatureEngineeringConfig())
    assert strict.loc["f1", ["vote_filter", "vote_lasso", "vote_permutation"]].tolist() == [True, False, True]
    loose = apply_votes(agg, set(), 100_000, from_dict({"selection": {"mutual_info_min": 0.001,
                                                                      "lasso_keep_share": 0.6,
                                                                      "permutation_min_share": 0.0005}}))
    assert loose.loc["f2", "votes"] == strict.loc["f2", "votes"] + 3   # filter + redundancy + permutation flip


def test_redundancy_keeps_higher_mi_member(rng):
    a = rng.normal(size=300)
    Z = pd.DataFrame({"a": a, "a_copy": a * 2 + 0.001 * rng.normal(size=300), "b": rng.normal(size=300)})
    pairs = redundancy_pairs(Z, pd.Series({"a": 0.5, "a_copy": 0.4, "b": 0.1}), threshold=0.9)
    assert len(pairs) == 1 and pairs.iloc[0]["keep"] == "a" and pairs.iloc[0]["drop_candidate"] == "a_copy"


def test_candidate_sets_respect_toggles(table, config):
    tbl = table[0]
    with_generic = candidate_sets(tbl, config, FeatureEngineeringConfig())
    without = candidate_sets(tbl, config, from_dict({"decision": {"include_generic_candidates": False}}))
    assert "generic_polynomial" in with_generic and "generic_polynomial" not in without
    assert with_generic["original"] == {"engineered": [], "drop": [], "overrides": {}}
    assert "consensus_3_of_5" in with_generic


def _fold_table(rmse: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame([{"model": f"ridge|{m}", "fold": k, "rmse": v}
                         for m, vals in rmse.items() for k, v in enumerate(vals)])


def _sets(*names):
    return {n: {"engineered": ["x"] * i, "drop": [], "overrides": {}} for i, n in enumerate(names)}


def test_rule_keeps_original_when_nothing_is_significant(config):
    noise = np.random.default_rng(1).normal(0, 50, 15)
    folds = _fold_table({"original": list(1000 + noise), "cand": list(1000 + noise[::-1])})
    chosen, _, note = choose_feature_set(folds, _sets("original", "cand"), config, FeatureEngineeringConfig())
    assert chosen == "original" and "kept" in note


def test_rule_prefers_fewer_features_among_ties(config):
    base = np.linspace(1000, 1100, 15)
    folds = _fold_table({"original": list(base), "small": list(base * 0.90), "big": list(base * 0.899)})
    sets = {"original": {"engineered": [], "drop": [], "overrides": {}},
            "small": {"engineered": ["a"], "drop": [], "overrides": {}},
            "big": {"engineered": ["a", "b", "c"], "drop": [], "overrides": {}}}
    chosen, table, _ = choose_feature_set(folds, sets, config, FeatureEngineeringConfig())
    assert chosen == "small"                             # within tie tolerance of "big", fewer features
    strict = from_dict({"decision": {"tie_tolerance": 0.0}})
    assert choose_feature_set(folds, sets, config, strict)[0] == "big"


def test_parallel_cv_matches_sequential(data, config, small_splits):
    X_train, _, y_train, _ = data
    models = {"m": make_fe_model(RidgeCV(alphas=np.logspace(-3, 3, 7)), config, engineered=["total_rooms"])}
    seq = cross_validate_models(models, X_train, y_train, small_splits)[["model", "fold", "rmse"]]
    par = parallel_cv(models, X_train, y_train, small_splits, n_jobs=2)[["model", "fold", "rmse"]]
    pd.testing.assert_frame_equal(seq.reset_index(drop=True), par.reset_index(drop=True))


def test_profiler_records_steps_and_can_be_disabled():
    p = Profiler(track_memory=True)
    with p.step("work", detail=1):
        sum(range(10_000))
    t = p.table()
    assert t.loc[0, "step"] == "work" and t.loc[0, "detail"] == 1 and t.loc[0, "wall_seconds"] >= 0
    assert t.loc[0, "peak_memory_mb"] >= 0
    off = Profiler(enabled=False)
    with off.step("x"):
        pass
    assert off.table().empty
