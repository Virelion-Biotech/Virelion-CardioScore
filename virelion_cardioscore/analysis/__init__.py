"""Analysis modules: statistics, dose-response fitting, CardioScore engine, and QC diagnostics."""

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
