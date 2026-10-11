# Exploratory analysis X1 — GMM gate on FBCSP features (post hoc)

Written 2026-10-05, AFTER the frozen primary evaluation (protocol sha 49cdf03c…) had been run and read.
This analysis is exploratory. It is reported in the Supplementary Material only, never in the abstract or as a
primary conclusion, and is reported regardless of its result. No further variants will be added.

Motivation (post hoc): the aligned GMM built on ChannelViT embeddings ranked errors of the Concat+LR decoder close to
chance, whereas on EEGNet embeddings it matched the decoder's MSP. Hypothesis: the usefulness of a separate density
gate depends on the quality of the representation it is fitted on.

Specification (identical to the primary protocol except for the GMM input; same tests as the primary analysis):
- Same five members, same frozen representation models; no retraining.
- GMM input: the member's MIBIF-selected FBCSP features (k = 16), i.e. the same FBCSP features the decoder uses.
- GMM configuration: the protocol's conservative head, NOT re-searched (StandardScaler, PCA d = min(8, n_B - 1),
  one diagonal Gaussian per class, reg_covar 1e-3, member seed).
- Fitted on B_j only (never on run C or the evaluation session).
- Ensemble: mean of the five members' GMM posteriors.
- Decoder: Concat+LR ensemble (unchanged); score s = p_GMM-FBCSP(yhat_F | x).
- Temperature: the frozen run-6 value. Threshold for the new score: fitted on run 6 only, q = 0.8.
- Metrics on the clean evaluation session: per-subject AURC, correctness AUROC, accepted accuracy/kappa at
  the run-6 threshold.
- Tests (one Holm family, two-sided exact Wilcoxon, rank-biserial r, subject bootstrap CI, n = 9):
  AURC(FBCSP-GMM) - AURC(MSP); AURC(FBCSP-GMM) - AURC(temperature-scaled MSP).
  Descriptive only: AURC(FBCSP-GMM) - AURC(ViT-GMM).
