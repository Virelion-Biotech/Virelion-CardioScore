# CardioScore validation execution

Pinned CardioScore revision used by the validation runner: `d727e0e6f64ec0d3961e7ae50e580a94fd25e7b8`

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
  +--> raw MEA -> feature extraction (canonical long CSV)\n  +--> MC_Data/Cardio PyMEA -> deterministic signal window (component validation only)
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

Paste this into a fresh Colab cell. Fetch the runner by **commit SHA**, not `main`, so the run
manifest identifies exactly which runner code produced the results:

```python
import hashlib
import os
import urllib.request

RUNNER_SHA = "b1b0413e48482492d166a1c86e1543fbfb18390e"
RUNNER_URL = (
    "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/"
    f"{RUNNER_SHA}/scripts/validation/run_colab_validation.py"
)
source = urllib.request.urlopen(RUNNER_URL).read()
os.environ["CARDIOSCORE_RUNNER_SHA"] = RUNNER_SHA
os.environ["CARDIOSCORE_RUNNER_SOURCE_SHA256"] = hashlib.sha256(source).hexdigest()
exec(compile(source, "run_colab_validation.py", "exec"))
```

The runner will prompt for a run label and then open one upload dialog. Upload all assets needed for that validation run together.

`run_manifest.json` records `runner_revision` and `runner_source_sha256`. An unpinned launch remains fail-soft for convenience but records `unpinned-main` and warns that the runner code is moving; use the SHA-pinned launcher for auditable runs.

Optional environment controls can be set before the fetch:

```python
import os
os.environ["CARDIOSCORE_RUN_LABEL"] = "patel_2019_2026-09-19"
os.environ["CARDIOSCORE_PUBLISH"] = "1"  # set to "0" to keep results only in Colab
```


For a real Cardio PyMEA / Multichannel Systems `MC_DataTool` ASCII recording, select the
electrodes and window explicitly before fetching the runner:

```python
import os
os.environ["CARDIOSCORE_MCS_ELECTRODES"] = "El F7,El F8"
os.environ["CARDIOSCORE_MCS_START_S"] = "0"
os.environ["CARDIOSCORE_MCS_DURATION_S"] = "30"
```

If `CARDIOSCORE_MCS_ELECTRODES` is omitted, the runner recognizes the recording but stops the
signal stage and reports the available electrode names. It never chooses an electrode subset on its
own. MC_Data/Cardio PyMEA output is recorded as `signal_component_only`; because the untreated
recording contains no compound/concentration/vehicle metadata, it is never eligible for the locked
drug-risk analysis.

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

The runner fails closed on missing schema, unexpected CiPA platform/event labels, non-deterministic raw-to-feature or MC_Data-window rebuilds, reference/feature leakage, missing verification lineage, a source receipt whose uploaded file/hash does not match, missing authoritative source metadata, ambiguous multiple MC_Data recordings, and unsupported raw formats.

Technical wells are not treated as independent drugs. Missing endpoints are not fabricated. Q/quiescence is kept separate from A-D arrhythmia-like events.

The released CiPA workbook is a component/semantic validation source, not a five-endpoint CardioScore gold standard. A headline external CardioScore result requires a verified source-to-feature rebuild, an uploaded source file whose SHA-256 matches the immutable receipt, explicit authoritative source metadata, explicit vehicle structure, complete scoreable reference coverage, and a separate reference table with documented reference provenance.

## Blinded Cardio PyMEA acceptance gate

`validation/signal_validation_criteria.yaml` is currently **draft**. The blinded
`pymea_beat_validation.py score` command may still report raw metrics and per-threshold checks while
the criteria are draft, but it deliberately returns no confirmatory pass/fail. A confirmatory
`passed` value is produced only after both the criteria document and its `beat_detection` section
are marked `frozen`; the scorer also records the criteria file SHA-256.

Do not freeze these criteria until the detector/configuration is final and the development windows
used for detector tuning are explicitly excluded from the new held-out annotation sheet.

## Legacy modular notebooks

The numbered notebooks remain in the repository as inspectable modular examples. They are no longer required for routine execution; the single runner above is the preferred Colab workflow.


## Blinova stage: hard stops vs. flags

Structural problems stop the stage: missing columns, missing/non-numeric `conc`/`EAD`/`site`,
malformed present-but-non-numeric `ddFPDc`, a canonical panel other than 28 compounds, an
unrecognized risk label, contradictory risk labels within a compound, a compound with no labeled
row, or an A-D/Q event type on a row with `EAD != 1`.

Documented data-quality findings are **flagged, not fatal**, and are never rewritten in the source
rows: blank `risk` cells, platform codes outside `AXN/CLY/ECR/AMD/MCS` (for example `ACA`),
missing `ddFPDc`, and EAD/A-D findings in compounds the paper reports as event-free
(terfenadine, verapamil). Flags are logged, stored in the run manifest, and written to
`derived/blinova_audit_flags.json`. A run with flags is a component/semantic check only; resolve
the flagged provenance questions before quoting any Blinova-derived number.
