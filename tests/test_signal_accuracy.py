"""Signal-level validation against exact ground truth, plus documented known limitations.

Passing tests state what the extraction recovers. ``xfail(strict=True)`` tests describe behaviour the
extraction does NOT yet have (found by this benchmark); they turn red the moment a fix lands so the
expectation can be promoted to a normal test and the validation version bumped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.preprocessing.beat_detection import BeatDetectionConfig, detect_beats
from virelion_cardioscore.preprocessing.endpoints import extract_electrode_features
from virelion_cardioscore.preprocessing.filtering import (
    FilterConfig,
    estimate_noise_sd,
    filter_trace,
)
from virelion_cardioscore.validation.signal_accuracy import (
    agreement_metrics,
    canonical_frame_hash,
    event_detection_metrics,
    lins_ccc,
    match_events,
    verify_deterministic_rebuild,
)
from virelion_cardioscore.validation.signal_truth import make_truth_recording

# --- metrics -------------------------------------------------------------------------------------


def test_matching_is_one_to_one_and_counts_errors():
    matched = match_events([1.0, 2.0, 3.0, 4.0], [1.01, 1.02, 2.5, 3.99], tolerance_s=0.05)
    assert (matched["tp"], matched["fp"], matched["fn"]) == (2, 2, 2)  # 1.02 cannot re-match beat 1
    assert sorted(np.round(matched["timing_errors_s"], 3)) == [-0.01, 0.01]


def test_detection_metrics_perfect_and_degenerate_cases():
    perfect = event_detection_metrics([1, 2, 3], [1, 2, 3], 0.01)
    assert perfect["f1"] == 1.0 and perfect["precision"] == 1.0 and perfect["recall"] == 1.0
    missed = event_detection_metrics([1, 2, 3], [], 0.01)
    assert missed["recall"] == 0.0 and missed["f1"] == 0.0 and missed["fn"] == 3
    with pytest.raises(ValueError, match="tolerance"):
        match_events([1], [1], 0)


def test_agreement_reports_bias_not_just_correlation():
    reference = np.linspace(200, 400, 30)
    measured = reference + 25.0  # perfectly correlated but systematically wrong
    metrics = agreement_metrics(reference, measured)
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(25.0) and metrics["mae"] == pytest.approx(25.0)
    assert metrics["lins_ccc"] < 0.95  # CCC penalises the bias that Pearson ignores
    assert agreement_metrics([1, np.nan, 3], [1, 2, 3])["n_missing_pairs"] == 1
    assert lins_ccc(reference, reference) == pytest.approx(1.0)


def test_deterministic_rebuild_check_detects_nondeterminism():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [0.1 + 0.2, 1 / 3]})
    assert verify_deterministic_rebuild(lambda: frame.copy())["verified_rebuild"] is True
    counter = iter(range(100))
    result = verify_deterministic_rebuild(lambda: pd.DataFrame({"a": [float(next(counter))]}))
    assert result["verified_rebuild"] is False
    assert canonical_frame_hash(frame) == canonical_frame_hash(frame.copy())


def test_truth_recording_reports_both_variability_definitions():
    steady = make_truth_recording(1, ibi_jitter_ms=0.0, fpd_jitter_ms=0.0)
    assert (
        steady.ibi_variability() == pytest.approx(0.0, abs=1e-9)
        and steady.fpd_short_term_variability_ms() == 0.0
    )
    jittery = make_truth_recording(1, ibi_jitter_ms=20.0, fpd_jitter_ms=10.0)
    assert jittery.ibi_variability() > 0.01 and jittery.fpd_short_term_variability_ms() > 3.0


# --- what the extraction recovers (clean synthetic signals) -----------------------------------------


@pytest.mark.parametrize("polarity", [1, -1])
def test_beats_are_detected_and_timed_accurately(polarity):
    recording = make_truth_recording(3, polarity=polarity)
    beats = detect_beats(
        filter_trace(recording.trace_uv, recording.fs_hz, FilterConfig()), recording.fs_hz
    )
    metrics = event_detection_metrics(recording.beat_times_s, beats.beat_times_s, 0.020)
    # One false positive at the very end of the record (filter edge transient) is tolerated; no beat is missed.
    assert (
        metrics["recall"] == 1.0
        and metrics["f1"] >= 0.97
        and metrics["timing_median_abs_error_s"] < 0.001
    )


@pytest.mark.parametrize("true_fpd_ms", [200.0, 300.0, 400.0])
def test_fpd_is_recovered_within_two_ms_up_to_400_ms(true_fpd_ms):
    recording = make_truth_recording(4, fpd_ms=true_fpd_ms)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.fpd_ms is not None and abs(features.fpd_ms - true_fpd_ms) < 2.0


def test_beat_rate_and_repolarization_stv_match_truth():
    recording = make_truth_recording(5, ibi_jitter_ms=20.0, fpd_jitter_ms=10.0)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    true_rate = 60.0 / np.mean(recording.ibi_s)
    assert abs(features.beat_rate_bpm - true_rate) < 0.5
    assert features.stv == pytest.approx(recording.fpd_short_term_variability_ms(), abs=2.0)


def test_stv_tracks_repolarization_variability():
    steady = make_truth_recording(6, fpd_jitter_ms=0.0)
    unstable = make_truth_recording(6, fpd_jitter_ms=25.0)
    steady_features = extract_electrode_features(steady.trace_uv, 1000.0)
    unstable_features = extract_electrode_features(unstable.trace_uv, 1000.0)
    assert unstable.fpd_short_term_variability_ms() > 10.0
    assert unstable_features.stv > steady_features.stv + 5.0


# --- regressions fixed before the validation freeze -----------------------------


@pytest.mark.parametrize("true_fpd_ms", [500.0, 600.0, 700.0])
def test_fpd_beyond_previous_450_ms_limit_is_recovered(true_fpd_ms):
    recording = make_truth_recording(7, fpd_ms=true_fpd_ms)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.fpd_ms is not None and abs(features.fpd_ms - true_fpd_ms) < 5.0


def test_noise_estimate_is_within_a_factor_of_three_of_true_noise():
    recording = make_truth_recording(8, noise_sd_uv=20.0)
    estimate = estimate_noise_sd(recording.trace_uv, recording.fs_hz)
    assert 20.0 / 3 <= estimate <= 20.0 * 3
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert 20.0 / 3 <= features.noise_sd_uv <= 20.0 * 3


def test_over_detection_lowers_beat_detection_rate():
    recording = make_truth_recording(9, noise_sd_uv=20.0)
    features = extract_electrode_features(
        recording.trace_uv, recording.fs_hz, beat_config=BeatDetectionConfig()
    )
    assert (
        features.n_beats > 1.5 * recording.beat_times_s.size
    )  # the detector really is over-counting
    assert features.beat_detection_rate < 0.7