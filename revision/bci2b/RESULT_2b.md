# BCI IV 2b pre-specified validation — result

Protocol `bspc-r1-2b-validation-v1`, sha256 `53a30bc8…`, frozen 2026-10-06 23:46:22 (before any 2b E label was read).
One deviation (D-block size for B04/B05/B08), logged in `deviations.md` before evaluation. Evaluation run once, 2026-10-07.

## Primary (only confirmatory) test

AURC(FBCSP-GMM) − AURC(MSP), Concat+LR, one-sided exact Wilcoxon signed-rank, n = 9, α = 0.05

| mean d | 95% bootstrap CI | W+ | one-sided p | rank-biserial r | decision |
|---:|---|---:|---:|---:|---|
| −0.0290 | [−0.0649, −0.0035] | 9 | 0.0645 | −0.60 | **not validated** (p ≥ 0.05) |

FBCSP-GMM had the lower AURC in 6 of 9 subjects (B01, B03, B04, B06, B07, B09) and the higher AURC in 3 (B02, B05, B08).
The direction matches the 2a exploratory result, but the pre-specified criterion was not met.

## Descriptive (no tests)

Means over 9 subjects: Concat+LR acc 0.734, κ 0.469; AURC FBCSP-GMM 0.168, MSP 0.197, ViT-GMM 0.273, oracle 0.055;
correctness AUROC FBCSP-GMM 0.733, MSP 0.682, ViT-GMM 0.483. Per-subject d: B01 −0.019, B02 +0.012, B03 −0.155,
B04 −0.000, B05 +0.001, B06 −0.024, B07 −0.032, B08 +0.006, B09 −0.050 (median −0.019).

- Per-subject values: `out/subject_metrics_2b.csv`; per trial: `out/trials_2b.csv`; summary: `out/primary_2b.json`.
- With 2 classes, temperature scaling does not change the MSP ranking, so AURC(ts_msp) = AURC(msp) in every subject.
- B03 decodes below chance (κ = −0.07), and its difference (−0.155) is the largest; at that accuracy the AURC difference reflects ranking among mostly wrong predictions.
- The ViT-GMM gate is again close to chance (AUROC ≈ 0.38–0.55), matching 2a.
