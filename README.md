# Virelion-CardioScore

CardioScore is a Python framework for analyzing human iPSC-cardiomyocyte microelectrode-array (MEA) field-potential data and combining electrophysiology endpoints into a configurable research risk score.

## Scope

- signal and electrode quality checks;
- field-potential duration, beat rate, amplitude, short-term variability, and triangulation-related features;
- vehicle normalization and concentration-aware aggregation;
- technical-well and independent-unit accounting;
- bootstrap uncertainty and optional mixed-effects analysis;
- dose-response diagnostics;
- transparent weighted scoring with configurable thresholds;
- benchmark manifests and reproducible reports;
- CLI, Python API, and a browser demonstration.

The framework can be used for general cardiac safety research, including medical-countermeasure programs. It is not a regulatory assay.

## Installation

Python 3.10+ is required.

```bash
pip install -e '.[dev,mixed]'
```

## Quick start

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

## Risk score

The default implementation maps a weighted score to configurable classes. The default thresholds are implementation parameters, not validated clinical or regulatory cutoffs. Reference scores should be independently reviewed and locked before benchmark use.

| Score | Default class |
|---|---|
| `< 0.30` | Low |
| `0.30–<0.60` | Moderate |
| `≥ 0.60` | High |

## Web demonstration

`web/index.html` provides a client-side demonstration using synthetic data and the supported well-level aggregation contract. It does not implement the full Python biological-unit hierarchy or mixed-effects workflow.

## Scientific limitations

CardioScore is research software. A score is a model output and does not establish cardiotoxicity, clinical risk, regulatory acceptance, or causal mechanism. CiPA-oriented terminology describes alignment of analysis concepts; it does not mean that this repository is a CiPA-validated implementation or an ICH S7B substitute.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release, analysis configuration, and source MEA datasets used in a study.
