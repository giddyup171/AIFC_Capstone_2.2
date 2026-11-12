import numpy as np
import pandas as pd
import pytest

from src.fe_config import FeatureEngineeringConfig
from src.feature_engineering import (
    FEATURE_CATALOGUE,
    ColumnPowerTransformer,
    DomainFeatureEngineer,
    FrequencyEncoder,
    GenericFeatureGenerator,
    HashingEncoder,
    LogTargetEncoder,
    TemporalFeatureGenerator,
    build_fe_preprocessor,
    past_rolling_mean,
)
from src.preprocessing import build_preprocessor


# --------------------------------------------------------------------------- #
# Domain features
# --------------------------------------------------------------------------- #
def engineer(df, features=None, **kw):
    t = DomainFeatureEngineer(list(FEATURE_CATALOGUE) if features is None else features, **kw)
    return t.fit(df).transform(df)


def test_domain_feature_values(make_house):
    out = engineer(make_house())
    r = out.iloc[0]
    assert r.school_premium == pytest.approx(np.exp(3.0))
    assert r.school_squared == 81
    assert r.neighbourhood_quality == pytest.approx(np.log(9 / 4))
    assert r.size_per_bedroom == 500 and r.bath_per_bed == 0.5 and r.total_rooms == 6
    assert r.school_x_size == pytest.approx(9 * np.log(2000))
    assert r.top_school == 1 and r.new_build == 0


def test_shape_and_deterministic_names(make_house):
    df = make_house()
    t = DomainFeatureEngineer(["total_rooms", "school_premium"]).fit(df)
    out = t.transform(df)
    assert out.shape == (1, df.shape[1] + 2)
    assert list(out.columns) == list(df.columns) + ["total_rooms", "school_premium"]   # order as requested
    assert list(t.get_feature_names_out()) == list(out.columns)


@pytest.mark.parametrize("overrides, feature", [
    ({"number_of_bedrooms": 0.0}, "size_per_bedroom"),
    ({"number_of_bedrooms": 0.0}, "bath_per_bed"),
    ({"crime_rate_in_area": 0.0}, "neighbourhood_quality"),
    ({"property_size": 0.0}, "school_x_size"),
    ({"school_quality_score": 0.0}, "neighbourhood_quality"),
])
def test_zero_denominators_and_bad_logs_give_nan_not_inf(make_house, overrides, feature):
    val = engineer(make_house(**overrides))[feature].iloc[0]
    assert np.isnan(val)


def test_missing_inputs_propagate_as_nan(make_house):
    out = engineer(make_house(school_quality_score=np.nan, age_of_property=np.nan))
    for f in ["school_premium", "school_squared", "top_school", "new_build"]:
        assert np.isnan(out[f].iloc[0])


def test_thresholds_come_from_parameters(make_house):
    df = make_house(school_quality_score=8.0, age_of_property=6.0)
    assert engineer(df, ["top_school"]).top_school.iloc[0] == 0
    assert engineer(df, ["top_school"], top_school_threshold=7.5).top_school.iloc[0] == 1
    assert engineer(df, ["new_build"], new_build_max_age=10).new_build.iloc[0] == 1
    assert engineer(df, ["school_premium"], school_premium_scale=4).school_premium.iloc[0] == pytest.approx(np.exp(2))


def test_unknown_feature_and_bad_scale_raise(make_house):
    with pytest.raises(ValueError, match="Unknown engineered"):
        DomainFeatureEngineer(["nope"]).fit(make_house())
    with pytest.raises(ValueError, match="school_premium_scale"):
        DomainFeatureEngineer(["school_premium"], school_premium_scale=0).fit(make_house())


# --------------------------------------------------------------------------- #
# Generic generators and transforms
# --------------------------------------------------------------------------- #
@pytest.fixture
def block(rng):
    return pd.DataFrame({"a": rng.uniform(1, 5, 200), "b": rng.uniform(1, 5, 200), "c": rng.uniform(1, 5, 200)})


def test_generator_off_is_identity(block):
    out = GenericFeatureGenerator().fit(block).transform(block)
    pd.testing.assert_frame_equal(out, block)


def test_interactions_polynomial_binning_names_and_values(block):
    g = GenericFeatureGenerator(interactions="all", polynomial=["a"], degree=3, binning=["b"], n_bins=4).fit(block)
    out = g.transform(block)
    expected_new = ["a__x__b", "a__x__c", "b__x__c", "a__pow2", "a__pow3", "b__bin1", "b__bin2", "b__bin3"]
    assert list(out.columns) == ["a", "b", "c"] + expected_new
    assert list(g.get_feature_names_out()) == list(out.columns)
    assert np.allclose(out["a__x__c"], block.a * block.c)
    assert np.allclose(out["a__pow3"], block.a ** 3)
    bins = out[["b__bin1", "b__bin2", "b__bin3"]]
    assert set(np.unique(bins.values)) <= {0.0, 1.0} and (bins.sum(axis=1) <= 1).all()


def test_generator_rejects_unknown_columns(block):
    with pytest.raises(ValueError, match="not in the numeric block"):
        GenericFeatureGenerator(interactions=["a", "zzz"]).fit(block)


def test_power_transforms(block):
    for method in ["yeo-johnson", "box-cox"]:
        out = ColumnPowerTransformer(["a"], method).fit(block).transform(block)
        assert np.isfinite(out["a"]).all() and np.allclose(out["b"], block.b)   # other columns untouched
    neg = block.assign(a=block.a - 10)
    with pytest.raises(ValueError, match="strictly positive"):
        ColumnPowerTransformer(["a"], "box-cox").fit(neg)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
def test_frequency_encoder_handles_unseen():
    tr = pd.DataFrame({"city": ["A", "A", "A", "B"]})
    enc = FrequencyEncoder().fit(tr)
    out = enc.transform(pd.DataFrame({"city": ["A", "B", "Z"]}))
    assert out["city_freq"].tolist() == [0.75, 0.25, 0.0]
    assert list(enc.get_feature_names_out()) == ["city_freq"]


def test_hashing_encoder_fixed_width_and_deterministic():
    enc = HashingEncoder(n_features=6).fit(pd.DataFrame({"street": ["x"]}))
    many = pd.DataFrame({"street": [f"road_{i}" for i in range(500)]})
    out1, out2 = enc.transform(many), enc.transform(many)
    assert out1.shape == (500, 6)
    pd.testing.assert_frame_equal(out1, out2)


def test_target_encoder_is_cross_fitted(rng):
    """Leakage check: with cross-fitting, a row's own price must not shape its training encoding."""
    n = 300
    X = pd.DataFrame({"area": rng.choice([f"z{i}" for i in range(100)], n)})   # 3 rows per category on average
    y = pd.Series(np.exp(rng.normal(12, 0.5, n)))
    enc = LogTargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=0)
    cross_fitted = np.asarray(enc.fit_transform(X, y)).ravel()
    naive = np.asarray(enc.transform(X)).ravel()
    # Naive (in-sample) encoding correlates with the row's own target; the cross-fitted one should not
    ylog = np.log(y)
    assert np.corrcoef(naive, ylog)[0, 1] > np.corrcoef(cross_fitted, ylog)[0, 1] + 0.1


# --------------------------------------------------------------------------- #
# Temporal design
# --------------------------------------------------------------------------- #
def test_temporal_no_op_without_date_columns(make_house):
    df = make_house()
    t = TemporalFeatureGenerator([]).fit(df)
    pd.testing.assert_frame_equal(t.transform(df), df)
    assert list(t.get_feature_names_out()) == list(df.columns)


def test_temporal_features_with_reference_fixed_at_fit():
    tr = pd.DataFrame({"sale_date": ["2024-01-06", "2024-03-01"], "x": [1, 2]})
    t = TemporalFeatureGenerator(["sale_date"], features=("days_since", "month", "quarter", "is_weekend")).fit(tr)
    new = pd.DataFrame({"sale_date": ["2024-02-25", None], "x": [3, 4]})
    out = t.transform(new)
    assert "sale_date" not in out.columns
    assert out["sale_date_days_since"].iloc[0] == 5          # reference = latest TRAINING date (2024-03-01)
    assert out["sale_date_month"].iloc[0] == 2 and out["sale_date_quarter"].iloc[0] == 1
    assert out["sale_date_is_weekend"].iloc[0] == 1          # 25 Feb 2024 was a Sunday
    assert np.isnan(out["sale_date_days_since"].iloc[1])


def test_past_rolling_mean_excludes_current_and_future_rows():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-10", "2024-01-20", "2024-01-05"],
                       "area": ["A", "A", "A", "B"], "price": [100.0, 200.0, 300.0, 999.0]})
    out = past_rolling_mean(df, "date", "area", "price", window="30D")
    assert np.isnan(out.iloc[0])                    # no earlier sale in area A
    assert out.iloc[1] == 100.0                     # only the earlier sale, not itself
    assert out.iloc[2] == 150.0
    assert np.isnan(out.iloc[3])                    # area B never sees area A's prices


# --------------------------------------------------------------------------- #
# Full preprocessor
# --------------------------------------------------------------------------- #
def test_default_without_engineered_matches_stage2(data, config):
    X_train, X_test, y_train, _ = data
    stage2 = build_preprocessor(config).fit(X_train).transform(X_test)
    prep = build_fe_preprocessor(config, FeatureEngineeringConfig(), engineered=[])
    fe = prep.fit(X_train, y_train).transform(X_test)
    pd.testing.assert_frame_equal(stage2[sorted(stage2.columns)], fe[sorted(fe.columns)])


def test_binary_flags_survive_outlier_capping(data, config):
    """Regression test for the bug found in Stage 2b: IQR capping erased mostly-zero 0/1 features."""
    X_train, _, y_train, _ = data
    out = build_fe_preprocessor(config, engineered=["top_school", "new_build"]).fit(X_train, y_train).transform(X_train)
    assert out["top_school"].nunique() == 2 and out["new_build"].nunique() == 2


@pytest.mark.parametrize("overrides", [
    {},
    {"numeric_transform": "yeo-johnson"},
    {"numeric_transform": "none"},
    {"encoding": {"method": "frequency"}},
    {"encoding": {"method": "hash", "hash_n_features": 4}},
    {"encoding": {"method": "target"}},
    {"imputation": {"numeric_strategy": "knn", "add_missing_indicator": True}},
    {"generators": {"interactions": {"enabled": True}, "polynomial": {"enabled": True},
                    "binning": {"enabled": True}}},
])
def test_every_option_produces_finite_output_on_dirty_data(data, config, overrides):
    X_train, X_test, y_train, _ = data
    fe = FeatureEngineeringConfig().with_overrides(overrides)
    prep = build_fe_preprocessor(config, fe).fit(X_train, y_train)
    dirty = X_test.copy()
    dirty.iloc[0, dirty.columns.get_loc("property_size")] = np.nan
    dirty.iloc[1, dirty.columns.get_loc("number_of_bedrooms")] = 0          # zero denominator
    dirty["location_type"] = dirty["location_type"].astype(object)
    dirty.iloc[2, dirty.columns.get_loc("location_type")] = "Coastal"       # unseen category
    out = prep.transform(dirty)
    assert len(out) == len(dirty) and np.isfinite(out.to_numpy(dtype=float)).all()
    # deterministic names: refitting gives the same columns in the same order
    again = build_fe_preprocessor(config, fe).fit(X_train, y_train).transform(dirty)
    assert list(out.columns) == list(again.columns)


def test_missing_indicator_columns_flag_missing_values(data, config):
    X_train, X_test, y_train, _ = data
    fe = FeatureEngineeringConfig().with_overrides({"imputation": {"add_missing_indicator": True}})
    prep = build_fe_preprocessor(config, fe, engineered=[]).fit(X_train, y_train)
    dirty = X_test.copy()
    dirty.iloc[0, dirty.columns.get_loc("crime_rate_in_area")] = np.nan
    out = prep.transform(dirty)
    flag = [c for c in out.columns if "missingindicator" in c and "crime" in c]
    assert len(flag) == 1 and out[flag[0]].iloc[0] == 1 and out[flag[0]].iloc[1:].sum() == 0
