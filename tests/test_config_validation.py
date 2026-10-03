from __future__ import annotations

import copy

import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.validation.config import ConfigValidationError, validate_pipeline_config


def test_default_pipeline_configuration_is_valid():
    pipeline = CardioScorePipeline.from_defaults()
    assert pipeline.config["scoring"]["low_threshold"] < pipeline.config["scoring"]["moderate_threshold"]


def test_config_rejects_invalid_risk_threshold_order():
    pipeline = CardioScorePipeline.from_defaults()
    config = copy.deepcopy(pipeline.config)
    config["scoring"]["low_threshold"] = 0.8
    config["scoring"]["moderate_threshold"] = 0.6
    with pytest.raises(ConfigValidationError, match="low_threshold must be less than"):
        validate_pipeline_config(config)


def test_config_rejects_invalid_beat_detection_rate_threshold():
    pipeline = CardioScorePipeline.from_defaults()
    config = copy.deepcopy(pipeline.config)
    config["quality_control"]["min_beat_detection_rate"] = 1.2
    with pytest.raises(ConfigValidationError, match="min_beat_detection_rate must be <= 1"):
        validate_pipeline_config(config)


def test_config_rejects_invalid_dose_response_uncertainty_limit():
    pipeline = CardioScorePipeline.from_defaults()
    config = copy.deepcopy(pipeline.config)
    config["concentration_response"]["fit_max_ec50_uncertainty_fold"] = 0.5
    with pytest.raises(ConfigValidationError, match="fit_max_ec50_uncertainty_fold must be >= 1"):
        validate_pipeline_config(config)


def test_config_rejects_fractional_discrete_settings():
    pipeline = CardioScorePipeline.from_defaults()
    for path, value, label in [
        (("quality_control", "min_electrodes_per_well"), 4.5, "min_electrodes_per_well"),
        (("concentration_response", "min_concentrations"), 3.5, "min_concentrations"),
        (("concentration_response", "fit_min_concentrations"), 4.5, "fit_min_concentrations"),
        (("inference", "n_bootstrap"), 2000.5, "n_bootstrap"),
    ]:
        config = copy.deepcopy(pipeline.config)
        config[path[0]][path[1]] = value
        with pytest.raises(ConfigValidationError, match=f"{label} must be an integer"):
            validate_pipeline_config(config)


def test_config_rejects_boolean_discrete_settings():
    pipeline = CardioScorePipeline.from_defaults()
    for path, label in [
        (("quality_control", "min_electrodes_per_well"), "min_electrodes_per_well"),
        (("concentration_response", "min_concentrations"), "min_concentrations"),
        (("inference", "n_bootstrap"), "n_bootstrap"),
    ]:
        config = copy.deepcopy(pipeline.config)
        config[path[0]][path[1]] = True
        with pytest.raises(ConfigValidationError, match=f"{label} must be an integer"):
            validate_pipeline_config(config)


def test_pipeline_constructor_fails_before_runtime_for_invalid_config():
    pipeline = CardioScorePipeline.from_defaults()
    config = copy.deepcopy(pipeline.config)
    config["inference"]["confidence"] = 1.5
    with pytest.raises(ConfigValidationError, match="inference.confidence must be <="):
        CardioScorePipeline(config)
