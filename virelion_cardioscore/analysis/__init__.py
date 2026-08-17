"""Analysis modules for CardioScore."""

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine, ScoreResult
from virelion_cardioscore.analysis.dose_response import (
    DoseResponseFit,
    fit_4pl,
    fit_concentration_series,
    four_parameter_logistic,
)
from virelion_cardioscore.analysis.hierarchy import (
    HierarchySpec,
    count_independent_units,
    detect_hierarchy,
    hierarchy_columns,
    summarize_experimental_units,
)
from virelion_cardioscore.analysis.normalization import (
    CorrectionDiagnostic,
    apply_control_anchor_correction,
)
from virelion_cardioscore.analysis.normalization_validation import (
    NormalizationValidationResult,
    validate_control_anchor_correction,
)
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline, PipelineResult
from virelion_cardioscore.analysis.statistics import (
    BootstrapCI,
    ProfileDifference,
    bootstrap_ci,
    bootstrap_profile_difference,
)
from virelion_cardioscore.analysis.variability import (
    VariabilityDiagnostic,
    control_variability,
    standardized_treatment_separation,
)

__all__ = [
    "CardioScoreEngine",
    "ScoreResult",
    "DoseResponseFit",
    "fit_4pl",
    "fit_concentration_series",
    "four_parameter_logistic",
    "HierarchySpec",
    "detect_hierarchy",
    "hierarchy_columns",
    "summarize_experimental_units",
    "count_independent_units",
    "CorrectionDiagnostic",
    "apply_control_anchor_correction",
    "NormalizationValidationResult",
    "validate_control_anchor_correction",
    "BootstrapCI",
    "ProfileDifference",
    "bootstrap_ci",
    "bootstrap_profile_difference",
    "VariabilityDiagnostic",
    "control_variability",
    "standardized_treatment_separation",
    "CardioScorePipeline",
    "PipelineResult",
]
