import pytest
import yaml

from src.fe_config import FeatureEngineeringConfig, FEConfigError, from_dict, load_fe_config

from .conftest import ROOT


def test_yaml_matches_documented_defaults():
    """The shipped YAML must reproduce the dataclass defaults (the reference run)."""
    assert load_fe_config(ROOT / "configs" / "feature_engineering.yaml").to_dict() == \
        FeatureEngineeringConfig().to_dict()


def test_none_path_gives_defaults():
    assert load_fe_config(None) == FeatureEngineeringConfig()


def test_selection_thresholds_are_configurable():
    cfg = from_dict({"selection": {"mutual_info_min": 0.05, "lasso_keep_share": 0.6,
                                   "permutation_min_share": 0.01}})
    assert (cfg.selection.mutual_info_min, cfg.selection.lasso_keep_share,
            cfg.selection.permutation_min_share) == (0.05, 0.6, 0.01)
    assert cfg.selection.redundancy_corr == 0.9          # untouched values keep their defaults


def test_with_overrides_is_validated_and_does_not_mutate():
    base = FeatureEngineeringConfig()
    changed = base.with_overrides({"encoding": {"method": "target"}})
    assert changed.encoding.method == "target" and base.encoding.method == "onehot"
    with pytest.raises(FEConfigError):
        base.with_overrides({"encoding": {"method": "magic"}})


@pytest.mark.parametrize("data, message", [
    ({"selection": {"lasso_keep_share": 1.5}}, "lasso_keep_share"),
    ({"selection": {"redundancy_corr": 1.0}}, "redundancy_corr"),
    ({"selection": {"consensus_min_votes": 6}}, "consensus_min_votes"),
    ({"selection": {"rfecv_step": 0}}, "rfecv_step"),
    ({"decision": {"alpha": 0}}, "alpha"),
    ({"decision": {"min_gain": -0.1}}, "min_gain"),
    ({"numeric_transform": "sqrt"}, "numeric_transform"),
    ({"encoding": {"method": "embedding"}}, "encoding.method"),
    ({"encoding": {"target_smoothing": -1}}, "target_smoothing"),
    ({"imputation": {"numeric_strategy": "mode"}}, "numeric_strategy"),
    ({"generators": {"polynomial": {"degree": 1}}}, "degree"),
    ({"generators": {"polynomial": {"degree": 7}}}, "degree"),
    ({"generators": {"binning": {"strategy": "random"}}}, "strategy"),
    ({"generators": {"interactions": {"columns": ["property_size"]}}}, "interactions.columns"),
    ({"domain": {"features": ["school_premium", "magic_feature"]}}, "unknown feature"),
    ({"domain": {"school_premium_scale": 0}}, "school_premium_scale"),
    ({"temporal": {"features": ["fiscal_week"]}}, "temporal.features"),
    ({"compute": {"n_jobs": 0}}, "n_jobs"),
    ({"selection": {"no_such_threshold": 1}}, "Unknown key"),
    ({"no_such_section": {}}, "Unknown key"),
])
def test_invalid_values_rejected_with_clear_message(data, message):
    with pytest.raises(FEConfigError, match=message):
        from_dict(data)


def test_missing_file_and_non_mapping_yaml(tmp_path):
    with pytest.raises(FEConfigError, match="not found"):
        load_fe_config(tmp_path / "nope.yaml")
    p = tmp_path / "list.yaml"
    p.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(FEConfigError, match="mapping"):
        load_fe_config(p)
