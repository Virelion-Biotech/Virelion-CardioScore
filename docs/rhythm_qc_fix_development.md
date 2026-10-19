# Rhythm QC correction: development regression evidence

## Defect confirmed

Against baseline `cba05082623ad6e0e96ab3dad4cdec95e65050c8`, QC rejected 16 of 33 windows marked confident in the existing exploratory AI review. Thirteen had exactly the AI reference beat count. In all 13, voltage autocorrelation selected a weak near-half-period peak instead of the stronger approximately 2-second rhythm. This approximately doubled the expected beat count and reduced QC coverage to approximately 0.5.

For example, w00's strongest voltage correlation was at 2.052 s (0.682), but the half-period allowance selected 1.036 s (0.131). There were five detected and five AI-reference beats in the 10-second window.

## Implementation

Estimator version: `slope-autocorrelation-v2`.

- Compute independent rhythm autocorrelation on the first difference of the filtered voltage. Smooth baseline and repolarization structure therefore contributes less than fast depolarization morphology.
- Retain the existing weak half-period allowance for genuine alternating strong/weak beats. Do not define expected count from detector peaks.
- Also allow integer subperiods of three or more when their correlation is at least 50% of the parent peak (and at least 0.05), within the existing 8% lag tolerance. This handles a synthetic jitter case where the strongest lag spans three cycles. Weak incidental short-lag structure cannot use this new allowance.
- Keep the 0.7 QC cutoff, noise limit, detector prominence settings, and reference labels unchanged.

The expected period also informs companion suppression. This is a behavioral development revision; unchanged peak locations on these 40 windows do not guarantee unchanged detections elsewhere.

## Before/after on the same 40 cached PyMEA windows

| Measure | Baseline | Revised |
| --- | ---: | ---: |
| Confident AI windows rejected | 16 / 33 | 2 / 33 |
| Exact-count confident windows rejected | 13 | 0 |
| Windows with changed detector beat indices | — | 0 / 40 |
| Unresolved noise windows w24/w28/w30 rejected | 3 / 3 | 3 / 3 |

w25 (8 detections versus 5 AI beats) and w38 (8 versus 4) remain rejected. w37 is admitted after correction but still has one unmatched detection (5 versus 4 AI beats, potentially a boundary event). QC admission does not establish annotation agreement.

Because all 40 detector index arrays are identical, the earlier exploratory matching results are unchanged: 160 matched beats, 8 unmatched detections, precision 0.95238, recall 1.0, F1 0.97561 among the 33 confident AI windows.

## Verification

Executed in Google Colab through Edge: **431 tests passed**, including 23 new rhythm-QC cases (83.37 seconds, Python 3.13.16). Four dose-response exponential-overflow warnings were emitted by existing tests outside this change. Initial slope-only implementation exposed an existing synthetic jitter regression; the final integer-subperiod check resolved it without changing that test's assertions.

New regression cases cover smooth twice-per-cycle interference, genuine weak alternating beats, sampling rates, detector-threshold independence, flat/short signals, voltage polarity/offset/scale, and jitter across multiple synthetic windows. The existing end-to-end PyMEA QC test assertions remain unchanged.

Reproduce in `notebooks/cardioscore_validation/CardioScore_Rhythm_QC_Fix_v8.ipynb` using the campaign's existing Drive cache. It preserves baseline source from the pinned Git object, verifies raw-window and AI-label hashes, applies the embedded revision, compares all 40 windows, and runs the complete test suite.

## Evidence boundary

This dataset has informed the fix and is now development regression evidence. The 33 confident/7 uncertain classifications are exploratory AI judgments, not human ground truth. Human validation remains incomplete, and an untouched independently annotated set is still required for external performance claims.

No original frozen criteria, preregistration, annotation manifest, raw windows, AI annotation CSV, or human annotation sheet was modified.

Immutable AI annotation SHA-256:
`dc5371641c9cdef3abaf1980c00713584ffc6ba7cf31f623d5122e5e4691dc43`.

Frozen baseline source SHA-256:
`5c93983d76dda12b5e4b6b449b1489968298dbeca92c37c47a1bef2fcbc09faa`.
