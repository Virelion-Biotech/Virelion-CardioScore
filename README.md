# Virelion CardioScore

**Biodefense-oriented, CiPA-aligned cardiotoxicity risk scoring for medical countermeasures (MCMs) using human iPSC-CM MEA field-potential data.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-research%20framework-orange)]()

Virelion CardioScore is an open-source framework for **early cardiac safety screening of biodefense and MCM candidates** — vaccines, antitoxins, antiviral agents, monoclonal antibodies, and related modalities — from microelectrode array (MEA) field-potential recordings of human iPSC-derived cardiomyocytes (iPSC-CMs).

It extends the analysis goals of the Cardio PyMEA ecosystem (GPL) with a transparent, multi-endpoint **CardioScore** mapped to Low / Moderate / High risk classes, aligned with CiPA-oriented electrophysiology principles.

> **Important**: Research and prioritization framework only. **Not** a validated regulatory assay (CiPA / ICH S7B substitute). Scores are interpretable engineering constructs for pharmacology, safety, and biodefense program teams.

---

## Biodefense & MCM focus

Cardiac liability can appear late and costly in MCM development. CardioScore is built to:

- Flag **electrophysiological risk signals** (FPD change, rate, amplitude loss, STV, triangulation proxies) early in *in vitro* work
- Support **batch screening** of vaccine, antitoxin, and antiviral concentration series before animal studies
- Keep scoring **transparent and audit-friendly** (weights, thresholds, and per-endpoint contributions are always visible)
- Remain useful for **dual-use awareness** contexts: same pipeline can document cardiac safety rationale for defensive MCM research packages

Default synthetic profiles and example configs emphasize MCM-style series; small-molecule and general screening remain fully supported.

---

## Key features

- Signal quality control (noise, electrodes, beat detection rate) with audit log
- Field-potential endpoints: FPD, beat rate, amplitude, STV, triangulation proxies
- Vehicle-normalized effects and concentration-aware aggregation
- Transparent weighted CardioScore → Low / Moderate / High
- CLI (`cardioscore demo` / `cardioscore run`) and Python API
- Synthetic MEA generator (MCM-oriented profiles)
- **Browser web app** under [`web/`](web/) — full client-side demo: modality presets, weight editing, CSV feature upload, scoring, export

---

## Installation

```bash
git clone https://github.com/Virelion-Biotech/Virelion-CardioScore.git
cd Virelion-CardioScore
pip install -e ".[dev]"
```

Requires Python 3.10+.

---

## Quick start

```bash
cardioscore demo
```

Writes `summary.csv`, `scores.json`, and `report.html` under `./outputs/`.

### Web app (browser)

Open [`web/index.html`](web/index.html) locally, or host the `web/` folder (GitHub Pages / Netlify).
No server required: synthetic generation, weight controls, optional CSV upload, scoring, and JSON/CSV export all run in the browser.

---

## Risk categories (default)

| CardioScore | Class    | Typical use |
|-------------|----------|-------------|
| < 0.30      | Low      | Minimal EP liability signals at tested concentrations |
| 0.30–0.60   | Moderate | Clear concentration-dependent changes; recommend follow-up |
| > 0.60      | High     | Strong multi-endpoint signals; prioritize additional evaluation |

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

## Disclaimer

Research and internal decision-support only. Not a regulatory safety assay.
