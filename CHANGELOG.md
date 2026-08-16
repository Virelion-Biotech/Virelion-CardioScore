# Changelog

All notable changes to Virelion CardioScore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] – 2026-08-16

### Added
- Initial public release of Virelion CardioScore.
- Transparent multi-endpoint CardioScore engine with configurable weights.
- Low / Moderate / High risk classification aligned with CiPA-oriented principles.
- Synthetic MEA dataset generator for end-to-end demonstration.
- Quality-control filters (noise, electrode count, beat detection rate).
- Vehicle-normalized effect calculation for FPD, rate, amplitude, STV, triangulation proxy.
- CLI (`cardioscore demo`, `cardioscore run`).
- HTML report generator with per-compound contribution breakdown.
- CSV and JSON export of scores.
- Basic test suite and packaging metadata (GPL-3.0-or-later).

### Notes
- This is a research framework. Scores are engineering constructs and have not been validated as a regulatory assay.
- Future releases will add real MEA format readers and expanded endpoint libraries.
