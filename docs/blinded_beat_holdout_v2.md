# Blinded beat validation v2 and residual-event investigation

The rhythm QC detector from PR #37 is retained. PR #38 removes nine unused imports; its full CI matrix passed on Python 3.10–3.12, including tests, demo smoke tests and installed-wheel checks.

## Remaining exploratory disagreements

These observations use the existing AI development labels, not new human ground truth.

- **w25:** three unmatched positive events at 2.870, 4.927 and 6.997 s have amplitudes 34–36 uV and widths 145–152 ms. Matched spikes are approximately 265–279 uV and 13–20 ms wide. The unmatched broad waveform components occur roughly 1.36 s after a spike, beyond the existing companion-suppression half-period interval.
- **w38:** four unmatched negative events at 1.004, 3.103, 7.280 and 9.354 s are approximately 25–29 uV, compared with approximately 98 uV for matched events. Width alone does not separate them reliably. Later small events recur about 1.29 s after a matched spike; the first has no preceding in-window matched event.
- **w37:** the extra leading event at 0.764 s is approximately 27 uV with an approximately 1.50-second prominence width; matched events are approximately 212–216 uV. This is consistent with a broad baseline/tail deflection rather than a narrow depolarization. It is admitted by QC, so QC admission cannot substitute for reference agreement.

No detector threshold or companion-suppression interval was tuned to remove these three windows. Independent reviewers must determine which events are beats.

## Validation reporting correction

The previous QC report required both admission and `detection_bad` to populate `admitted_with_poor_detection`, although `detection_bad` itself forces rejection. That condition could never be true.

The corrected report compares admitted windows with the first non-uncertain reference annotation and flags any unmatched event at the chosen tolerance. It includes the per-window matching result and declares that definition. QC uses the same filter configuration as event matching. New tests exercise reference disagreement and sampling with no cross-electrode time overlap.

## Frozen component study

`scripts/validation/blinded_beat_holdout.py` writes a method/configuration freeze before sampling and refuses to overwrite existing studies. It verifies package method files against immutable detector revision `ffef2ae65ab0c5761c69e9b2679bece8a7da7ba6`; the validation tool is separately hashed.

The sample is 30 randomly selected 10-second windows, seed 20261020, from the unchanged Day9 source. All prior 40 review windows and the 20 original development intervals are excluded **by time across every electrode**, with a two-second guard. New windows are time-disjoint across electrodes. Selection does not use detector output, signal quality or annotations. Raw-only plots and arrays are generated; no predictions are produced for the new set.

Acceptance thresholds retain the earlier beat criteria: 50 ms matching tolerance, precision/recall/F1 >=0.95, and median absolute timing error <=10 ms. Annotator A is the prespecified primary reference; B is secondary and supplies inter-reviewer agreement. Neither sheet contains prefilled labels. Each reviewer should work separately and report uncertain traces rather than guessing.

Before scoring, the wrapper verifies method/configuration hashes and fixed study-file hashes, and requires every annotation row to be complete with reviewer identity. The exact annotation bytes are hashed in results. It refuses to score the initial blank sheets.

This is **within-recording temporal-holdout component validation**, not an independently acquired recording or full CardioScore external validation. The original 40 AI-reviewed windows remain development evidence. Human annotation and an independently acquired dataset remain outstanding.

## Reproduction

Prepare (in Colab with the existing campaign cache):
```
python scripts/validation/blinded_beat_holdout.py prepare --repo /content/CardioScore_rhythm_qc_fix --source /content/drive/MyDrive/CardioScore_validation_campaign_v1/raw/cardio_pymea_2022/Day9__Nov1_130pm.txt --development-manifest /content/drive/MyDrive/CardioScore_validation_campaign_v1/results/cardio_pymea_2022/heldout_annotation/sheet_manifest.json --out /content/drive/MyDrive/CardioScore_validation_campaign_v1/results/pymea_blinded_human_temporal_holdout_v2_30_windows_20261020 --detector-commit ffef2ae65ab0c5761c69e9b2679bece8a7da7ba6 --n-windows 30
```

After two independent human reviews, restore the frozen method files and run:
```
python scripts/validation/blinded_beat_holdout.py score --repo /content/CardioScore_rhythm_qc_fix --study /content/drive/MyDrive/CardioScore_validation_campaign_v1/results/pymea_blinded_human_temporal_holdout_v2_20261020
```

Do not run a detector on the held-out windows or alter thresholds before the human labels are locked.

An initial 40-window sampling plan could not fit under the strict global time exclusions and random non-overlap sampling. It failed before saving or viewing any new raw window or generating predictions. The 30-window revision was specified before any new signal inspection; guards, seed, detector settings, and acceptance thresholds were retained.

Verification: 433 tests passed in Colab (83.64 s); repository-wide Ruff passed.
