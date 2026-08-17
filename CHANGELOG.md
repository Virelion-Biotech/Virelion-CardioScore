# Changelog

## [Unreleased]

### Added
- Replicate-aware concentration summaries with mean/median aggregation and SD/SEM reporting.
- Optional four-parameter logistic concentration-response fitting with EC50 and Hill-slope estimates.
- 95% confidence intervals for EC50 and Hill slope, with optional SEM weighting.
- Dose-response quality diagnostics for R-squared, monotonicity, EC50 boundary placement, and EC50 uncertainty.
- Machine-readable dose-response diagnostics in pipeline JSON output and summary tables.
- Regression tests covering numerical fit quality and diagnostic failure modes.

### Changed
- Dose-response fits no longer count as quality-passing solely because numerical optimization converged.
- Poor-quality dose-response fits remain available for inspection and do not replace the heuristic CardioScore aggregation.

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
