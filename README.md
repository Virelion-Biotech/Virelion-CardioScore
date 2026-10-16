# Virelion-CardioScore

CardioScore is a Python framework for analyzing human iPSC-cardiomyocyte microelectrode-array (MEA) field-potential data and combining electrophysiology endpoints into a configurable research risk score.


<!-- VALIDATION_PRODUCT_STATUS:BEGIN -->
## Validation and product completion status

**Reviewed 2026-10-17; implementation base `40b1ef02cf4c`.** **Status: Workbook ingestion/audit evidence; external score qualification incomplete.** This is an evidence-backed scope assessment, not a percentage-complete estimate or a declaration that the product is finished.

| Area | Current evidence / remaining gate |
|---|---|
| Software / operational checks | Colab readiness audit completed: 17 artifact hashes and 161 source fingerprints verified. The official workbook and original derived outputs reproduce locally. Unresolved tokens now retain null typed-event indicators. Local tests: 404 passed, one optional skip; lint, contract and demo pass. |
| Scientific or domain evidence | The 8,859-row, 28-compound audit resolves the quiescence/arrhythmia documentation error. It retains 52 unresolved event-label rows and 560 undocumented ACA rows; corrected resolved A–D count is 1,286. Score cutoffs, predictive accuracy and regulatory endpoints remain unqualified. Aggregate workbook labels are not a substitute for raw MEA waveform truth. |
| Remaining for the declared product scope | Finish provenance/field-semantic reconciliation and complete endpoint mapping; independently qualify raw-waveform extraction; freeze score weights/probability calibration and independent-unit splits; evaluate held-out compounds/sites with uncertainty and clinical-utility boundaries. |

Evidence and limits: [Colab review and correction](docs/COLAB_CIPA_REVIEW_2026-10-17.md), [retained results](validation/colab_cipa_2026_10_17/), and [repository report](docs/cipa_blinova_workbook_audit.md). The [organization status index](https://github.com/Virelion-Biotech/Virelion-HeartTwin/blob/main/docs/STACK_VALIDATION_STATUS.md) separates software, numerical, data-readiness and empirical qualification. Code/schema execution or a completed download is not a substitute for endpoint-specific validation.
<!-- VALIDATION_PRODUCT_STATUS:END -->

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
