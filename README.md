# Virelion-CardioScore

CardioScore is a Python framework for analyzing human iPSC-cardiomyocyte microelectrode-array (MEA) field-potential data and combining electrophysiology endpoints into a configurable research risk score.

## What it contains

- Signal and electrode quality checks.
- Field-potential duration, beat rate, amplitude, short-term variability, and triangulation-related features.
- Vehicle normalization and concentration-aware aggregation.
- Technical-well and independent-unit accounting.
- Bootstrap uncertainty and optional mixed-effects analysis.
- Dose-response diagnostics.
- Transparent weighted scoring with configurable thresholds.
- Benchmark manifests and reproducible reports.
- CLI, Python API, and browser demonstration.

## Installation

Python 3.10+ is required.

```bash
pip install -e '.[dev,mixed]'
```

## Usage

Run the demonstration:

```bash
cardioscore demo
```

Run a feature table:

```bash
cardioscore run --config path/to/pipeline.yaml --features path/to/features.csv
```

Run a reference benchmark:

```bash
cardioscore benchmark --manifest benchmarks/reference_manifest.yaml
```

## Inputs and outputs

**Inputs:** MEA field-potential feature tables or supported analysis inputs, well/sample metadata, vehicle and concentration information, analysis configuration, and optional benchmark manifests.

**Outputs:** electrophysiology feature summaries, normalized/aggregated results, uncertainty and statistical analyses, dose-response diagnostics, configurable risk-score classes, benchmark reports, and reproducibility metadata.

The default score classes are implementation parameters, not validated clinical or regulatory cutoffs.

## Validation

Validation includes signal/electrode QC, independent-unit accounting, bootstrap uncertainty, optional mixed-effects analysis, dose-response diagnostics, and benchmark manifests. The repository's software tests should be run with the development test suite.

Validation of the software or score does not establish cardiotoxicity detection, clinical risk, regulatory acceptance, or causal mechanism.

## Limitations

CardioScore is research software, not a regulatory assay. Results depend on recording quality, feature extraction, experimental-unit definitions, normalization, scoring weights, and threshold choices. CiPA-oriented terminology describes analysis concepts and does not make the repository a CiPA-validated implementation or an ICH S7B substitute.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
