# 接管核查报告

仅读取原协议、模型输出与结果，未重训或覆盖原文件。
检查数：22996；数值失败：0。
主差值：0.103028225，95% CI [0.07022504889735348, 0.14294085921043123]，双侧p=0.00390625，r=1.00。

## 方法与表述待核对

- Epoch selection is shared from member 1 (runs 3/4 fit, run 5 validation). B_j for members 2/3/4/5 overlaps these runs. Head trials were not directly used to train that member's representation, but some head labels indirectly influence the shared epoch. Do not claim complete absence from all model selection.
- Whole-session unsupervised EOG OLS is fitted independently on T and E signals. This is offline/transductive preprocessing with E-signal access, not strictly train-only learned preprocessing or prospective online operation. It does not by itself establish E-label leakage.
- Perturbations were applied to cleaned epochs before model-specific transforms, not before all preprocessing. EOG segments and propagation coefficients came from E eye-movement calibration, whereas the plan described T-eye calibration. Document this protocol departure; do not silently relabel it.
- Main max_gmm outputs use the new ensemble. They do not reproduce the old model's tau=0.80 historical result.
- Current random reference is one seeded random permutation. Matched-random gains are comparisons to that realization, not the analytic expectation; the expectation equals full accuracy.
- Previously observed E results cannot become an unseen holdout by switching assistants. New FBCSP-GMM analysis must remain post-hoc/exploratory with all results reported.
- Non-significance on EEGNet or versus random does not demonstrate equality; representation quality as a causal explanation remains a hypothesis, not an established mechanism.

## 核查边界
Numerical and code-path audit of inherited outputs; not an end-to-end retraining or proof of historical freeze timing.