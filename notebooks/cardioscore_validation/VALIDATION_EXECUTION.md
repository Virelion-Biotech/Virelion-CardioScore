# CardioScore validation execution

Pinned CardioScore revision used by the validation runner: `869150cd5fb5ccf155fb066258404bd4df163ade`

## Recommended mode: one Colab runtime

Use `scripts/validation/run_colab_validation.py`. It replaces notebook-to-notebook handoff with a single stateful run:

```text
Colab cell
  |
  v
GitHub runner
  |
  +--> intake / source hashes
  +--> Blinova/CiPA component audit (when a matching workbook is uploaded)
  +--> raw MEA -> feature extraction (when a supported raw CSV is uploaded)
  +--> locked external validation (only when verified locked inputs exist)
  +--> robustness + QC (only after a complete primary result)
  +--> optional GitHub publication
```

Every stage uses the same runtime workspace:

```text
/content/cardioscore_validation/runs/<RUN_LABEL>/
    input/
    derived/
    work/
    results/
```

The runner writes its handoff artifacts directly into that workspace, so a later stage reads exactly what the earlier stage produced. No notebook variables or manual file copying are required.

### Single Colab cell

Paste this into a fresh Colab cell:

```python
import urllib.request

RUNNER = "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/main/scripts/validation/run_colab_validation.py"
exec(compile(urllib.request.urlopen(RUNNER).read(), "run_colab_validation.py", "exec"))
```

The runner will prompt for a run label and then open one upload dialog. Upload all assets needed for that validation run together.

Optional environment controls can be set before the fetch:

```python
import os
os.environ["CARDIOSCORE_RUN_LABEL"] = "patel_2019_2026-09-19"
os.environ["CARDIOSCORE_PUBLISH"] = "1"  # set to "0" to keep results only in Colab
```

### GitHub publication

When publication is enabled, create a Colab Secret named `GITHUB_TOKEN`. The runner uses it only through Colab Secrets and never prints it.

The publisher:
- clones current `main`;
- creates `validation_results/<RUN_LABEL>/`;
- copies only an allowlisted set of derived JSON/CSV artifacts;
- creates `PUBLISH_MANIFEST.json` containing artifact hashes;
- refuses unexpected Git changes;
- commits and pushes without force;
- verifies the resulting `origin/main` commit SHA.

Raw `.zip`, `.mat`, `.h5/.hdf5`, raw source datasets, and credentials are not published.

## GIGO gates

The runner fails closed on missing schema, unexpected CiPA platform/event labels, non-deterministic raw-to-feature rebuilds, reference/feature leakage, missing verification lineage, and unsupported raw formats.

Technical wells are not treated as independent drugs. Missing endpoints are not fabricated. Q/quiescence is kept separate from A-D arrhythmia-like events.

The released CiPA workbook is a component/semantic validation source, not a five-endpoint CardioScore gold standard. A headline external CardioScore result requires a verified source-to-feature rebuild, explicit vehicle structure, complete scoreable reference coverage, and a separate reference table.

## Legacy modular notebooks

The numbered notebooks remain in the repository as inspectable modular examples. They are no longer required for routine execution; the single runner above is the preferred Colab workflow.
