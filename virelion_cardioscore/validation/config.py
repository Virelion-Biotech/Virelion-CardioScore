"""Central validation for CardioScore pipeline configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping


class ConfigValidationError(ValueError):
    """Raised when a pipeline configuration violates a runtime contract."""


def _require_number(value, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(result):
        raise ConfigValidationError(f"{name} must be finite; got {value!r}.")
    if minimum is not None and result < minimum:
        raise ConfigValidationError(f"{name} must be >= {minimum}; got {result}.")
    if maximum is not None and result > maximum:
        raise ConfigValidationError(f"{name} must be <= {maximum}; got {result}.")
    return result


def validate_pipeline_config(config: Mapping) -> None:
    """Validate settings used by preprocessing, scoring, and inference."""
    if not isinstance(config, Mapping):
        raise ConfigValidationError("Pipeline configuration must be a mapping.")

    qc = config.get("quality_control", {})
    _require_number(qc.get("min_electrodes_per_well", 4), "quality_control.min_electrodes_per_well", minimum=1)
    _require_number(qc.get("max_noise_sd_uv", 25.0), "quality_control.max_noise_sd_uv", minimum=0)
    _require_number(qc.get("min_beat_detection_rate", 0.7), "quality_control.min_beat_detection_rate", minimum=0, maximum=1)
    stv_limit = qc.get("arrhythmia_proxy_max_stv")
    if stv_limit is not None:
        _require_number(stv_limit, "quality_control.arrhythmia_proxy_max_stv", minimum=0)

    preprocessing = config.get("preprocessing", {})
    highpass = _require_number(preprocessing.get("highpass_hz", 0.5), "preprocessing.highpass_hz", minimum=0)
    lowpass = _require_number(preprocessing.get("lowpass_hz", 40.0), "preprocessing.lowpass_hz", minimum=0)
    if lowpass <= highpass:
        raise ConfigValidationError("preprocessing.lowpass_hz must be greater than preprocessing.highpass_hz.")
    notch = preprocessing.get("notch_hz")
    if notch is not None:
        _require_number(notch, "preprocessing.notch_hz", minimum=0)
    _require_number(preprocessing.get("notch_q", 30.0), "preprocessing.notch_q", minimum=0.000001)

    beat = config.get("beat_detection", {})
    _require_number(beat.get("min_prominence_uv", 20.0), "beat_detection.min_prominence_uv", minimum=0)
    _require_number(beat.get("min_distance_ms", 250.0), "beat_detection.min_distance_ms", minimum=0.000001)
    _require_number(beat.get("refractory_ms", 200.0), "beat_detection.refractory_ms", minimum=0.000001)

    units = config.get("experimental_units", {})
    scoring_unit = str(units.get("scoring_unit", "well"))
    if scoring_unit not in {"auto", "well", "biological_replicate", "batch", "plate"}:
        raise ConfigValidationError(f"experimental_units.scoring_unit has unsupported value {scoring_unit!r}.")
    if not isinstance(units.get("fall_back_to_well", False), bool):
        raise ConfigValidationError("experimental_units.fall_back_to_well must be boolean.")

    normalization = config.get("control_normalization", {})
    scope = str(normalization.get("scope", "compound"))
    if scope not in {"auto", "compound", "plate", "batch", "biological_replicate", "global"}:
        raise ConfigValidationError(f"control_normalization.scope has unsupported value {scope!r}.")
    if not isinstance(normalization.get("require_matching_control", True), bool):
        raise ConfigValidationError("control_normalization.require_matching_control must be boolean.")

    concentration = config.get("concentration_response", {})
    if str(concentration.get("replicate_aggregation", "mean")) not in {"mean", "median"}:
        raise ConfigValidationError("concentration_response.replicate_aggregation must be 'mean' or 'median'.")
    if str(concentration.get("concentration_aggregation", "mean_harmful_effect")) not in {"mean_harmful_effect", "max_absolute_effect"}:
        raise ConfigValidationError("concentration_response.concentration_aggregation has an unsupported value.")
    _require_number(concentration.get("effect_threshold_pct", 10.0), "concentration_response.effect_threshold_pct", minimum=0)
    _require_number(concentration.get("min_concentrations", 3), "concentration_response.min_concentrations", minimum=1)
    if not isinstance(concentration.get("require_min_concentrations_for_scoring", False), bool):
        raise ConfigValidationError("concentration_response.require_min_concentrations_for_scoring must be boolean.")
    _require_number(concentration.get("fit_min_concentrations", 4), "concentration_response.fit_min_concentrations", minimum=2)
    _require_number(concentration.get("fit_min_r_squared", 0.8), "concentration_response.fit_min_r_squared", minimum=-math.inf, maximum=1)
    _require_number(concentration.get("fit_min_monotonicity", 0.8), "concentration_response.fit_min_monotonicity", minimum=0, maximum=1)
    _require_number(concentration.get("fit_ec50_boundary_factor", 2.0), "concentration_response.fit_ec50_boundary_factor", minimum=1)
    _require_number(concentration.get("fit_max_ec50_uncertainty_fold", 100.0), "concentration_response.fit_max_ec50_uncertainty_fold", minimum=1)
    _require_number(concentration.get("min_ec50_coverage", 0.10), "concentration_response.min_ec50_coverage", minimum=0, maximum=1)

    inference = config.get("inference", {})
    _require_number(inference.get("n_bootstrap", 2000), "inference.n_bootstrap", minimum=1)
    _require_number(inference.get("confidence", 0.95), "inference.confidence", minimum=0.5, maximum=0.999999)
    cluster_column = inference.get("cluster_column")
    if cluster_column is not None and (not isinstance(cluster_column, str) or not cluster_column.strip()):
        raise ConfigValidationError("inference.cluster_column must be null or a non-blank string.")

    scoring = config.get("scoring", {})
    low = _require_number(scoring.get("low_threshold", 0.30), "scoring.low_threshold", minimum=0, maximum=1)
    moderate = _require_number(scoring.get("moderate_threshold", 0.60), "scoring.moderate_threshold", minimum=0, maximum=1)
    if low >= moderate:
        raise ConfigValidationError("scoring.low_threshold must be less than scoring.moderate_threshold.")
    _require_number(scoring.get("dose_response_weight", 0.0), "scoring.dose_response_weight", minimum=0)
