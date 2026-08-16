"""
Signal preprocessing for raw MEA field-potential traces.

Exposes the filtering (detrend / highpass / lowpass / notch) and beat
detection functions used to turn raw electrode voltage traces into the
beat-level data that virelion_cardioscore.features consumes.
"""

from virelion_cardioscore.preprocessing.filtering import (
    FilterConfig,
    estimate_noise_sd,
    filter_trace,
)
from virelion_cardioscore.preprocessing.beat_detection import (
    BeatDetectionConfig,
    BeatDetectionResult,
    detect_beats,
)

__all__ = [
    "FilterConfig",
    "filter_trace",
    "estimate_noise_sd",
    "BeatDetectionConfig",
    "BeatDetectionResult",
    "detect_beats",
]
