# Virelion CardioScore

**Automated CiPA-aligned cardiotoxicity risk scoring from human iPSC-derived cardiomyocyte MEA field-potential recordings.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-research%20framework-orange)]()

Virelion CardioScore is an open-source Python framework that extends the spirit and analysis goals of **Cardio PyMEA** (GPL) to deliver a reproducible, transparent, CiPA-oriented cardiac safety scoring pipeline for preclinical screening.

It is designed for early identification of electrophysiological liability (arrhythmia and cardiotoxicity signals) in candidate therapeutics — including vaccines, antitoxins, antiviral agents, and other modalities — using microelectrode array (MEA) field-potential recordings from human iPSC-derived cardiomyocytes (iPSC-CMs).

> **Important**: This is a research and screening framework. It is **not** a validated regulatory assay or a substitute for formal CiPA or ICH S7B evaluations. All risk classifications are interpretable engineering scores intended to support prioritization decisions by pharmacology and safety teams.

---

## Key Features

- **Signal quality control** — automatic well/electrode QC, artifact rejection, and audit logging
- **Field-potential feature extraction** — FPD, amplitude, beat rate, short-term variability, triangulation proxies, and other CiPA-relevant endpoints
- **Concentration–response analysis** — effect detection and simple curve characterization
- **Transparent CardioScore engine** — weighted multi-endpoint score mapped to Low / Moderate / High risk categories
- **Interpretable outputs** — per-compound summary tables, JSON/CSV exports, and HTML reports with plots
- **CLI + Python API** — batch-friendly command line and importable library
- **Synthetic data generator** — run the full pipeline end-to-end without proprietary MEA files
- **Extensible design** — clear extension points for new MEA formats, endpoints, and custom scoring weights

---

## Installation

```bash
git clone https://github.com/Virelion-Biotech/Virelion-CardioScore.git
cd Virelion-CardioScore
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```bash
cardioscore demo
```

This generates synthetic multi-well MEA data, runs QC → feature extraction → CardioScore, and writes `summary.csv`, `scores.json`, and `report.html` under `./outputs/`.

### Python API

```python
from virelion_cardioscore import CardioScorePipeline, load_synthetic_dataset

dataset = load_synthetic_dataset(n_compounds=3, n_concentrations=5)
pipeline = CardioScorePipeline.from_defaults()
results = pipeline.run(dataset)
print(results.summary_table)
results.to_html("report.html")
```

## Default Risk Categories

| CardioScore | Class     | Interpretation |
|-------------|-----------|----------------|
| < 0.30      | Low       | Minimal electrophysiological liability signals |
| 0.30–0.60   | Moderate  | Clear concentration-dependent changes; recommend follow-up |
| > 0.60      | High      | Strong multi-endpoint liability signals; prioritize additional evaluation |

Weights live in `virelion_cardioscore/config/cipa_endpoints.yaml` and are fully overridable.

## Project Layout

```
Virelion-CardioScore/
├── virelion_cardioscore/   # Main package (CLI, scoring engine, synthetic data, reporting)
├── examples/               # Example configs
├── tests/                  # pytest suite
├── docs/, notebooks/, scripts/
└── pyproject.toml, LICENSE (GPL-3.0), CITATION.cff, CONTRIBUTING.md
```

## Relationship to Cardio PyMEA

Designed as a complementary extension focused on standardized multi-endpoint risk aggregation, CiPA-aligned classification, and transparent batch screening. Feature tables from Cardio PyMEA (or any tool) can be fed into the CardioScore engine.

## Citation

```bibtex
@software{virelion_cardioscore_2026,
  title   = {Virelion CardioScore: CiPA-aligned cardiotoxicity scoring for iPSC-CM MEA data},
  author  = {{Virelion Biotech}},
  year    = {2026},
  url     = {https://github.com/Virelion-Biotech/Virelion-CardioScore},
  license = {GPL-3.0-or-later}
}
```

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

## Disclaimer

Research and internal decision-support only. Not a validated regulatory assay. Scores are engineering constructs and must be interpreted by qualified scientists.

---

*Built to make early cardiac safety assessment more reproducible, transparent, and accessible.*
