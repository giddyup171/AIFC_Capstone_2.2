"""Typed, validated configuration for feature engineering and selection.

Every threshold that drives a feature-engineering or selection decision is declared here with a
documented default, loaded from ``configs/feature_engineering.yaml`` and validated, so experiments
change configuration, not code. ``FeatureEngineeringConfig()`` with no arguments reproduces the
reference run.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

NUMERIC_TRANSFORMS = ("log", "yeo-johnson", "box-cox", "none")
ENCODINGS = ("onehot", "frequency", "target", "hash")
IMPUTERS = ("median", "mean", "knn")
BIN_STRATEGIES = ("quantile", "uniform", "kmeans")
TEMPORAL_FEATURES = ("days_since", "month", "quarter", "dayofweek", "is_weekend")


class FEConfigError(ValueError):
    """Raised when the feature-engineering configuration is invalid."""


@dataclass
class DomainConfig:
    features: list[str] = field(default_factory=lambda: [
        "school_premium", "school_squared", "neighbourhood_quality", "size_per_bedroom", "bath_per_bed",
        "total_rooms", "school_x_size", "top_school", "new_build"])
    school_premium_scale: float = 3.0
    top_school_threshold: float = 8.5
    new_build_max_age: float = 5.0


@dataclass
class InteractionsConfig:
    enabled: bool = False
    columns: list[str] | str = field(default_factory=lambda: [
        "property_size", "school_quality_score", "crime_rate_in_area", "age_of_property"])


@dataclass
class PolynomialConfig:
    enabled: bool = False
    columns: list[str] = field(default_factory=lambda: ["school_quality_score", "property_size"])
    degree: int = 2


@dataclass
class BinningConfig:
    enabled: bool = False
    columns: list[str] = field(default_factory=lambda: ["school_quality_score"])
    n_bins: int = 5
    strategy: str = "quantile"


@dataclass
class GeneratorsConfig:
    interactions: InteractionsConfig = field(default_factory=InteractionsConfig)
    polynomial: PolynomialConfig = field(default_factory=PolynomialConfig)
    binning: BinningConfig = field(default_factory=BinningConfig)


@dataclass
class EncodingConfig:
    method: str = "onehot"
    hash_n_features: int = 8
    target_smoothing: str | float = "auto"


@dataclass
class ImputationConfig:
    numeric_strategy: str = "median"
    knn_neighbors: int = 5
    add_missing_indicator: bool = False


@dataclass
class TemporalConfig:
    date_columns: list[str] = field(default_factory=list)
    reference_date: str | None = None
    features: list[str] = field(default_factory=lambda: ["days_since", "month", "quarter", "is_weekend"])


@dataclass
class SelectionConfig:
    mutual_info_min: float = 0.01
    redundancy_corr: float = 0.9
    lasso_keep_share: float = 0.8
    lasso_inner_folds: int = 5
    rfecv_min_features: int = 3
    rfecv_step: int = 1
    rfecv_folds: int = 5
    permutation_min_share: float = 0.002
    permutation_repeats: int = 5
    permutation_holdout: float = 0.25
    consensus_min_votes: int = 3


@dataclass
class DecisionConfig:
    alpha: float = 0.05
    min_gain: float = 0.005
    tie_tolerance: float = 0.005
    include_generic_candidates: bool = True
    include_gradient_boosting_check: bool = True


@dataclass
class ProfilingConfig:
    enabled: bool = True
    track_memory: bool = False   # tracemalloc made the stage 3.6x slower in profiling; enable for one-off runs
    cprofile_top: int = 0


@dataclass
class ComputeConfig:
    n_jobs: int = -1             # parallel CV fits in the feature-set comparison (-1 = all cores)


@dataclass
class FeatureEngineeringConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    generators: GeneratorsConfig = field(default_factory=GeneratorsConfig)
    numeric_transform: str = "log"
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any]) -> FeatureEngineeringConfig:
        """Copy with nested values replaced, e.g. {"encoding": {"method": "target"}}. Validated."""
        return from_dict(_deep_merge(self.to_dict(), overrides))


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls, data: dict | None, path: str):
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise FEConfigError(f"Unknown key(s) in {path or 'config'}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if is_dataclass(default):
            kwargs[name] = _build(type(default), data[name], f"{path}.{name}" if path else name)
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def _choice(name: str, value, allowed) -> None:
    if value not in allowed:
        raise FEConfigError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _fraction(name: str, value, low: float = 0.0, high: float = 1.0, inclusive: bool = True) -> None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and (
        low <= value <= high if inclusive else low < value < high)
    if not ok:
        raise FEConfigError(f"{name} must be a number between {low} and {high}, got {value!r}")


def _positive_int(name: str, value, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise FEConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")


def validate(cfg: FeatureEngineeringConfig) -> FeatureEngineeringConfig:
    from .feature_engineering import FEATURE_CATALOGUE  # local import avoids a cycle

    unknown = [f for f in cfg.domain.features if f not in FEATURE_CATALOGUE]
    if unknown:
        raise FEConfigError(f"domain.features has unknown feature(s) {unknown}; choose from {list(FEATURE_CATALOGUE)}")
    if len(set(cfg.domain.features)) != len(cfg.domain.features):
        raise FEConfigError("domain.features contains duplicates")
    if not cfg.domain.school_premium_scale > 0:
        raise FEConfigError("domain.school_premium_scale must be > 0")
    _fraction("domain.top_school_threshold", cfg.domain.top_school_threshold, 0, 10)
    if not cfg.domain.new_build_max_age > 0:
        raise FEConfigError("domain.new_build_max_age must be > 0")

    g = cfg.generators
    if g.interactions.columns != "all" and (not isinstance(g.interactions.columns, list)
                                            or len(g.interactions.columns) < 2):
        raise FEConfigError("generators.interactions.columns must be 'all' or a list of at least 2 columns")
    _positive_int("generators.polynomial.degree", g.polynomial.degree, 2)
    if g.polynomial.degree > 4:
        raise FEConfigError("generators.polynomial.degree above 4 is not supported (numerically unstable)")
    _positive_int("generators.binning.n_bins", g.binning.n_bins, 2)
    _choice("generators.binning.strategy", g.binning.strategy, BIN_STRATEGIES)

    _choice("numeric_transform", cfg.numeric_transform, NUMERIC_TRANSFORMS)
    _choice("encoding.method", cfg.encoding.method, ENCODINGS)
    _positive_int("encoding.hash_n_features", cfg.encoding.hash_n_features, 2)
    ts = cfg.encoding.target_smoothing
    if ts != "auto" and not (isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts >= 0):
        raise FEConfigError("encoding.target_smoothing must be 'auto' or a number >= 0")
    _choice("imputation.numeric_strategy", cfg.imputation.numeric_strategy, IMPUTERS)
    _positive_int("imputation.knn_neighbors", cfg.imputation.knn_neighbors)

    bad = [f for f in cfg.temporal.features if f not in TEMPORAL_FEATURES]
    if bad:
        raise FEConfigError(f"temporal.features has unknown feature(s) {bad}; choose from {list(TEMPORAL_FEATURES)}")

    s = cfg.selection
    _fraction("selection.mutual_info_min", s.mutual_info_min, 0, 10)
    _fraction("selection.redundancy_corr", s.redundancy_corr, 0, 1, inclusive=False)
    _fraction("selection.lasso_keep_share", s.lasso_keep_share)
    _positive_int("selection.lasso_inner_folds", s.lasso_inner_folds, 2)
    _positive_int("selection.rfecv_min_features", s.rfecv_min_features)
    _positive_int("selection.rfecv_step", s.rfecv_step)
    _positive_int("selection.rfecv_folds", s.rfecv_folds, 2)
    _fraction("selection.permutation_min_share", s.permutation_min_share)
    _positive_int("selection.permutation_repeats", s.permutation_repeats)
    _fraction("selection.permutation_holdout", s.permutation_holdout, 0, 1, inclusive=False)
    _positive_int("selection.consensus_min_votes", s.consensus_min_votes)
    if s.consensus_min_votes > 5:
        raise FEConfigError("selection.consensus_min_votes cannot exceed the 5 techniques")

    d = cfg.decision
    _fraction("decision.alpha", d.alpha, 0, 1, inclusive=False)
    _fraction("decision.min_gain", d.min_gain, 0, 1)
    _fraction("decision.tie_tolerance", d.tie_tolerance, 0, 1)
    if not isinstance(cfg.profiling.cprofile_top, int) or cfg.profiling.cprofile_top < 0:
        raise FEConfigError("profiling.cprofile_top must be an integer >= 0")
    nj = cfg.compute.n_jobs
    if not isinstance(nj, int) or isinstance(nj, bool) or nj == 0 or nj < -1:
        raise FEConfigError("compute.n_jobs must be -1 (all cores) or a positive integer")
    return cfg


def from_dict(data: dict[str, Any] | None) -> FeatureEngineeringConfig:
    return validate(_build(FeatureEngineeringConfig, data, ""))


def load_fe_config(path: str | Path | None) -> FeatureEngineeringConfig:
    """Load and validate a YAML file; ``None`` returns the documented defaults."""
    if path is None:
        return validate(FeatureEngineeringConfig())
    path = Path(path)
    if not path.exists():
        raise FEConfigError(f"Feature-engineering config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise FEConfigError(f"{path} must contain a YAML mapping")
    return from_dict(data)
