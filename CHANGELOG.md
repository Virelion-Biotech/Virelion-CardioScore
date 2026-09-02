# Changelog

## Unreleased

### Added
- Optional exposure-response evidence contribution to CardioScore using only quality-passing 4PL fits.
- Dose-response evidence is calculated from the tested log-concentration range above fitted EC50.
- Default dose-response scoring weight remains `0.0`, preserving the endpoint-only CardioScore unless explicitly enabled.
- Replicate-level bootstrap confidence intervals for concentration-specific endpoint effects.
- Matched concentration-profile bootstrap comparisons for exploratory between-group inference.
- Opt-in pipeline inference output and JSON serialization; inference does not alter CardioScore.
- Hierarchy-aware experimental-unit summaries recognizing optional biological replicate, batch, plate, and experiment metadata.
- Explicit scoring-unit policy API supporting `well`, `biological_replicate`, `batch`, and `plate` with validation of required metadata.
- Explicit vehicle-control normalization scope supporting `compound`, `plate`, `batch`, `biological_replicate`, and `global` controls.
- Optional plate/batch control-stability diagnostics with control CV, between-group SD, and exploratory treatment-to-control separation.
- HTML reporting of control-stability and plate/batch diagnostics when variability analysis is enabled.
- Optional control-anchored batch normalization using vehicle-only group shifts, with before/after control variability reporting and JSON audit metadata.
- Reusable normalization-validation utilities for testing drift reduction and preservation of within-group treatment-control effects on benchmark datasets.
- Fail-closed normalization assumption checks for treatment allocation and additive-vs-scale drift warnings.
- Optional random-intercept mixed-effects inference for endpoint-level treatment effects with plate/batch clustering; `statsmodels` is an optional dependency.
- Concentration-specific mixed-effects pipeline helper that models each compound × concentration × endpoint separately rather than pooling doses.
- Opt-in hierarchical CardioScore pipeline wrapper that augments standard pipeline results with mixed-effects inference and JSON/HTML reporting.
- Conventional-vs-hierarchical effect concordance analysis reporting absolute/relative disagreement and direction agreement.
- Synthetic hierarchical stress-test generator with known treatment effects, additive plate drift, multiplicative scale drift, and treatment-allocation imbalance scenarios.
- Robustness-matrix runner that sweeps plate drift, treatment allocation, replicate count, and noise level and summarizes conventional-estimator bias and recovery rate.
- Runtime-aligned external validation schema requiring QC fields, vehicle metadata, finite endpoint values, and valid concentration semantics.
- Explicit vehicle-structure validation for locked external runs when vehicle normalization is enabled.

### Fixed
- Restored the actual `preprocessing.beat_detection` implementation after a module collision had replaced it with endpoint-extraction code.
- Removed invalid bootstrap p-values that were derived from an observed-effect bootstrap distribution; profile comparisons now report confidence intervals only.
- Made endpoint direction semantics consistent, including correct handling of decrease-only endpoints and feature-table scoring.
- Made configured endpoint files authoritative and validate endpoint weights, thresholds, and directions at load time.
- Made `normalize_by_vehicle`, preprocessing, beat-detection, mixed-effects treatment, correction guardrails, and reporting settings effective rather than decorative.
- Rejected missing or blank experimental-unit identifiers instead of collapsing them into one pseudo-unit.
- Made plate/batch variability diagnostics operate on group-level control means and concentration-specific treatment separation.
- Preserved plate, batch, experiment, and biological-replicate metadata through raw-trace feature extraction.
- Tightened raw-trace validation for finite values, non-negative concentrations, duplicate timestamps, and inconsistent electrode sampling rates.
- Aligned hierarchical mixed-effects inference with the base run's QC/normalization population instead of re-analyzing rejected raw wells.
- Hardened the browser demo against unsafe HTML injection and silent feature fabrication from missing CSV columns.
- Fixed the standardized validation contract so vehicle controls may legitimately use `concentration_uM=0`, while treated wells must have strictly positive concentrations.
- Prevented external validation from proceeding silently with compounds that lack matching vehicle controls under vehicle-normalized scoring.

### Methodology
- The configured experimental unit now controls the analysis frame used for concentration summaries, dose-response fitting, bootstrap inference, and scoring.
- Wells remain the default analysis unit when no higher-level metadata are available.
- When biological-replicate metadata are supplied, technical wells can be summarized within biological units before higher-level inference and scoring.
- Raw well-level effects remain preserved in `PipelineResult.feature_table` for auditability.
- Vehicle normalization is explicitly scoped; missing matching controls are not silently substituted.
- The default vehicle scope remains `compound`, preserving historical behavior unless a study explicitly uses shared plate/batch/global controls.
- Plate/batch variability diagnostics are QC/inference outputs only and do not alter CardioScore.
- The variability layer does not claim a mixed-effects model; it identifies control instability that should be addressed before confirmatory inference.
- Variability diagnostics are exposed through `PipelineResult` and JSON as `variability` and `treatment_separation` tables, and are surfaced in the HTML report when available.
- Control-anchored normalization learns group shifts from vehicle wells only, requires adequate controls in every corrected group by default, and is disabled by default.
- The normalization is additive and should only be enabled for endpoints where additive recentering is scientifically defensible; it is not presented as a validated regulatory batch-correction model.
- Before/after variability outputs are retained so users can determine whether normalization reduced technical drift rather than assuming it did.
- The normalization-validation harness separately measures control-drift reduction and treatment-effect preservation; it does not turn those benchmark results into a regulatory validation claim.
- Normalization now performs assumption checks before applying corrections and fails closed by default when treatment allocation or additive-shift assumptions are not met.
- Symmetric positive/negative additive shifts are treated sign-invariantly in the assumption diagnostic to avoid false alarms from plate offsets in opposite directions.
- The random-intercept mixed-effects model is inference-only: it quantifies treatment effects while accounting for group-level clustering and reports ICC, but it does not automatically alter CardioScore.
- Mixed-effects inference is concentration-specific so distinct exposure levels are not pooled into a single treatment effect.
- The mixed-effects model requires at least two groups and both treatment levels and should use a genuine experimental grouping variable such as plate, batch, or experiment.
- The hierarchical wrapper augments standard CardioScore output without altering the core scoring path or default results.
- Conventional-vs-hierarchical concordance is a diagnostic only. Disagreement does not establish that either estimator is correct; it identifies analyses where clustering may materially affect the estimated treatment effect.
- Synthetic stress tests encode known ground truth so that treatment-effect bias from plate drift or allocation imbalance can be detected independently of the scorer's own assumptions.
- The robustness matrix is an operating-characteristic tool for synthetic data only; recovery thresholds are configurable and are not claims of real-world performance or regulatory validation.
- The hierarchy layer rejects requests for unavailable experimental-unit metadata instead of silently pseudoreplicating wells.
- The hierarchy layer does not claim a mixed-effects model; it is an explicit guard against accidental pseudoreplication.
- The locked external validation path now validates the full runtime feature contract before executing the pipeline and fails closed when normalized validation data lack compound-matched vehicle controls.

## [0.1.1] – 2026-08-16

### Changed
- Biodefense / MCM-oriented framing (vaccines, antitoxins, antivirals, dual-use-aware documentation).
- Added full browser web app under `web/` (presets, weights, CSV upload, QC, export).

## [0.1.0] – 2026-08-16

### Added
- Initial public release of Virelion CardioScore.
- Transparent multi-endpoint CardioScore engine with configurable weights.
- Low / Moderate / High risk classification aligned with CiPA-oriented principles.
- Synthetic MEA dataset generator for end-to-end demonstration.
- Quality-control filters, CLI, HTML report, tests, packaging (GPL-3.0-or-later).
