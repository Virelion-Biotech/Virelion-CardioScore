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
from virelion_cardioscore.analysis.hierarchical_pipeline import (
    HierarchicalCardioScorePipeline,
    HierarchicalPipelineResult,
)
from virelion_cardioscore.analysis.inference_comparison import (
    compare_effect_estimates,
    summarize_effect_concordance,
)
from virelion_cardioscore.analysis.mixed_effects import MixedEffectsResult, fit_random_intercept
from virelion_cardioscore.analysis.mixed_effects_pipeline import fit_compound_concentration_mixed_effects
from virelion_cardioscore.analysis.normalization import (
    CorrectionDiagnostic,
    apply_control_anchor_correction,
)
from virelion_cardioscore.analysis.normalization_assumptions import (
    NormalizationAssumptionCheck,
    check_additive_correction_assumptions,
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
    "MixedEffectsResult",
    "fit_random_intercept",
    "fit_compound_concentration_mixed_effects",
    "HierarchicalCardioScorePipeline",
    "HierarchicalPipelineResult",
    "compare_effect_estimates",
    "summarize_effect_concordance",
    "CorrectionDiagnostic",
    "apply_control_anchor_correction",
    "NormalizationAssumptionCheck",
    "check_additive_correction_assumptions",
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
