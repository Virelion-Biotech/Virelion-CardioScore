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

### Methodology
- The configured experimental unit now controls the analysis frame used for concentration summaries, dose-response fitting, bootstrap inference, and scoring.
- Wells remain the default analysis unit when no higher-level metadata are available.
- When biological-replicate metadata are supplied, technical wells can be summarized within biological units before higher-level inference and scoring.
- Raw well-level effects remain preserved in `PipelineResult.feature_table` for auditability.
- Vehicle normalization is now explicitly scoped; missing matching controls are not silently substituted.
- The default vehicle scope remains `compound`, preserving historical behavior unless a study explicitly uses shared plate/batch/global controls.
- Plate/batch variability diagnostics are QC/inference outputs only and do not alter CardioScore.
- The variability layer does not claim a mixed-effects model; it identifies control instability that should be addressed before confirmatory inference.
- Variability diagnostics are exposed through `PipelineResult` and JSON as `variability` and `treatment_separation` tables, and are surfaced in the HTML report when available.
- Control-anchored normalization learns group shifts from vehicle wells only, requires adequate controls in every corrected group by default, and is disabled by default.
- The normalization is additive and should only be enabled for endpoints where additive recentering is scientifically defensible; it is not presented as a validated regulatory batch-correction model.
- Before/after variability outputs are retained so users can determine whether normalization reduced technical drift rather than assuming it did.
- The normalization-validation harness separately measures control-drift reduction and treatment-effect preservation; it does not turn those benchmark results into a regulatory validation claim.
- The hierarchy layer rejects requests for unavailable experimental-unit metadata instead of silently pseudoreplicating wells.
- The hierarchy layer does not claim a mixed-effects model; it is an explicit guard against accidental pseudoreplication.

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
