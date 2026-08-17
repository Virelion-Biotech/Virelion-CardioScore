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

### Methodology
- Wells remain the default analysis unit when no higher-level metadata are available.
- When biological-replicate metadata are supplied, technical wells are summarized within biological units before higher-level inference.
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
