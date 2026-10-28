# Run CardioScore as a normal tool

CardioScore is a research analysis tool for human iPSC-cardiomyocyte MEA field-potential data. It is not a clinical diagnostic device or a validated regulatory assay. The risk class is a configurable research score, not a calibrated probability of cardiotoxicity.

## Option A — run locally (recommended for real analyses)

Python 3.10 or newer is required.

```bash
git clone https://github.com/Virelion-Biotech/Virelion-CardioScore.git
cd Virelion-CardioScore
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
cardioscore --help
```

### Generate example readings

```bash
cardioscore demo --n-compounds 4 --n-concentrations 6 --seed 42 --output-dir outputs/demo
```

This produces `summary.csv`, `scores.json`, and `report.html`. These are synthetic demonstration readings, not biological observations.

### Score the included feature-table example

```bash
cardioscore run --config config/default.yaml --features examples/data/example_features.csv --output-dir outputs/example
```

Open `outputs/example/report.html` for the report, `outputs/example/summary.csv` for tabular readings, and `outputs/example/scores.json` for machine-readable results. The included rows are illustrative synthetic fixtures. Replace them with measurements from your own experiment before interpreting results.

### Score raw per-electrode traces

```bash
cardioscore run --config config/default.yaml --raw-traces path/to/raw_traces.csv --output-dir outputs/raw-run
```

Raw trace CSV must satisfy the schema documented in `virelion_cardioscore/io/raw_trace.py`. Check the schema and sampling/voltage units before running; feature extraction does not infer missing experimental metadata or establish that a signal is a valid field potential.

## Option B — browser demonstration

Open `web/index.html` through a local static server (rather than double-clicking the file, because the app fetches `scoring_contract.json`):

```bash
python -m http.server 8000 --directory web
```

Then visit http://localhost:8000. The browser app supports synthetic runs and uploads of pre-extracted feature CSVs. It runs locally in your browser; it does not upload the CSV to a CardioScore server.

Browser mode is intentionally a lightweight scoring preview. It does not replace the Python pipeline's plate/batch normalization, biological-unit inference, bootstrap/mixed-effects analysis, or full validation reporting.

## Reading interpretation

- `CardioScore` is a normalized, configurable composite research score; it is **not** a probability.
- Low / Moderate / High are implementation thresholds. They have not been qualified as clinical or regulatory cutoffs.
- A result depends on endpoint extraction, control matching, QC, experimental-unit definition, scoring weights, and concentrations tested.
- Review the QC log and exclusion/dropout information; signal loss must not be interpreted as no effect.
- Do not use synthetic demo outputs as evidence about a real compound.

For the validation scope and unresolved scientific gates, see [README](../README.md), [the scientific audit](../SCIENTIFIC_AUDIT_2026-10-09.md), and [the external validation runbook](external_validation_runbook.md).
