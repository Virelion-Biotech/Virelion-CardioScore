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


def test_pipeline_constructor_fails_before_runtime_for_invalid_config():
    pipeline = CardioScorePipeline.from_defaults()
    config = copy.deepcopy(pipeline.config)
    config["inference"]["confidence"] = 1.5
    with pytest.raises(ConfigValidationError, match="inference.confidence must be <="):
        CardioScorePipeline(config)
