"""Signal-level validation against exact ground truth, plus documented known limitations.

Passing tests state what the extraction recovers. These regression tests are part of the pre-freeze
software validation contract; algorithm changes require a new validation version.
"""

from __future__ import annotations

from pathlib import Path

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


def test_beat_rate_and_variability_definitions_match_truth():
    recording = make_truth_recording(5, ibi_jitter_ms=20.0, fpd_jitter_ms=10.0)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    true_rate = 60.0 / np.mean(recording.ibi_s)
    assert abs(features.beat_rate_bpm - true_rate) < 0.5
    assert features.stv == pytest.approx(recording.ibi_variability(), abs=0.01)
    assert features.fpd_stv_ms == pytest.approx(recording.fpd_short_term_variability_ms(), abs=2.0)


def test_repolarization_stv_tracks_fpd_jitter():
    steady = make_truth_recording(6, fpd_jitter_ms=0.0, duration_s=60.0)
    unstable = make_truth_recording(6, fpd_jitter_ms=25.0, duration_s=60.0)
    steady_features = extract_electrode_features(steady.trace_uv, 1000.0)
    unstable_features = extract_electrode_features(unstable.trace_uv, 1000.0)
    assert unstable.fpd_short_term_variability_ms() > 10.0
    assert unstable_features.fpd_stv_ms > steady_features.fpd_stv_ms + 5.0


# --- limits found by the benchmark, now fixed (each was a strict xfail before) ------------------------


@pytest.mark.parametrize("true_fpd_ms", [450.0, 500.0, 600.0, 700.0])
def test_fpd_beyond_the_old_450_ms_ceiling_is_recovered(true_fpd_ms):
    recording = make_truth_recording(7, fpd_ms=true_fpd_ms)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.fpd_ms is not None and abs(features.fpd_ms - true_fpd_ms) < 2.0


def test_fpd_search_is_still_bounded_by_the_next_beat():
    # At 110 bpm (545 ms) a 600 ms "FPD" cannot exist; the search must not run into the next beat.
    recording = make_truth_recording(7, fpd_ms=300.0, bpm=110.0)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.fpd_ms is not None and abs(features.fpd_ms - 300.0) < 2.0


@pytest.mark.parametrize("true_noise_uv", [5.0, 20.0, 40.0])
def test_noise_estimate_tracks_true_noise_on_the_raw_trace(true_noise_uv):
    recording = make_truth_recording(8, noise_sd_uv=true_noise_uv)
    assert estimate_noise_sd(recording.trace_uv, recording.fs_hz) == pytest.approx(
        true_noise_uv, rel=0.15
    )
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.noise_sd_uv == pytest.approx(true_noise_uv, rel=0.15)


def test_noise_no_longer_creates_beats_and_the_detection_rate_is_honest():
    recording = make_truth_recording(9, noise_sd_uv=20.0)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.n_beats <= 1.1 * recording.beat_times_s.size
    assert abs(features.beat_rate_bpm - 60.0 / np.mean(recording.ibi_s)) < 1.0
    assert features.beat_detection_rate >= 0.9


def test_default_qc_rejects_recordings_the_detector_cannot_handle():
    import yaml

    qc = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "virelion_cardioscore/config/default.yaml"
        ).read_text()
    )["quality_control"]
    for noise in (40.0, 60.0):
        features = extract_electrode_features(
            make_truth_recording(9, noise_sd_uv=noise).trace_uv, 1000.0
        )
        assert (
            features.noise_sd_uv > qc["max_noise_sd_uv"]
            or features.beat_detection_rate < qc["min_beat_detection_rate"]
        )


def test_over_detection_is_penalised_by_the_detection_rate():
    from virelion_cardioscore.preprocessing.beat_detection import BeatDetectionResult

    def result(n_beats, expected):
        idx = np.arange(n_beats) * 100
        return BeatDetectionResult(
            idx, idx / 1000.0, np.ones(n_beats), 1000.0, 30.0, expected_beat_count=expected
        )

    assert result(24, 24.0).beat_detection_rate == 1.0
    assert result(72, 24.0).beat_detection_rate == pytest.approx(1 / 3)
    assert result(12, 24.0).beat_detection_rate == pytest.approx(0.5)


def test_adaptive_prominence_scales_with_the_trace_and_can_be_disabled():
    recording = make_truth_recording(10, noise_sd_uv=20.0)
    filtered = filter_trace(recording.trace_uv, recording.fs_hz, FilterConfig())
    adaptive = detect_beats(filtered, recording.fs_hz)
    assert adaptive.effective_prominence_uv > 20.0
    fixed = detect_beats(
        filtered, recording.fs_hz, BeatDetectionConfig(noise_prominence_multiplier=0.0)
    )
    assert fixed.effective_prominence_uv == 20.0 and fixed.n_beats > 1.5 * adaptive.n_beats
    with pytest.raises(ValueError, match="noise_prominence_multiplier"):
        detect_beats(
            filtered, recording.fs_hz, BeatDetectionConfig(noise_prominence_multiplier=-1.0)
        )


@pytest.mark.parametrize("bpm", [20.0, 29.0, 50.0])
def test_slow_and_normal_rhythms_are_scored_as_reliable(bpm):
    recording = make_truth_recording(11, bpm=bpm, duration_s=60.0)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.beat_detection_rate >= 0.9 and abs(features.beat_rate_bpm - bpm) < 0.5


def test_large_slow_events_in_background_clutter_are_not_over_detected():
    recording = make_truth_recording(
        12, bpm=29.0, amplitude_uv=500.0, clutter_sd_uv=12.0, duration_s=60.0
    )
    beats = detect_beats(
        filter_trace(recording.trace_uv, recording.fs_hz, FilterConfig()), recording.fs_hz
    )
    metrics = event_detection_metrics(recording.beat_times_s, beats.beat_times_s, 0.020)
    assert metrics["recall"] == 1.0 and metrics["precision"] >= 0.95
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert abs(features.beat_rate_bpm - 29.0) < 0.5 and features.stv < 0.02


def test_repolarization_stv_is_reported_and_tracks_truth():
    recording = make_truth_recording(13, fpd_jitter_ms=15.0, duration_s=60.0)
    features = extract_electrode_features(recording.trace_uv, recording.fs_hz)
    assert features.fpd_stv_ms == pytest.approx(recording.fpd_short_term_variability_ms(), abs=1.5)
    steady = extract_electrode_features(
        make_truth_recording(13, duration_s=60.0).trace_uv, 1000.0
    )
    assert steady.fpd_stv_ms < 2.0

