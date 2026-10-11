# FBCSP-GMM 探索性分析

在此前E结果已知后追加；仅限补充材料，不替换主实验或摘要。
固定融合类别不变，复用45个成员的FBCSP/MIBIF，只在原B_j拟合PCA8/单分量diag GMM。
所有新阈值在C确定并在E应用；没有参数搜索或网络重训。

| Score | AURC mean ± sample SD | Correctness AUROC | E coverage at C q=.8 |
| --- | --- | --- | --- |
| fbcsp_gmm | 0.2689 ± 0.1567 | 0.7549 | 0.7832 |
| vit_gmm | 0.4009 ± 0.1282 | 0.5318 | 0.7713 |
| msp | 0.2978 ± 0.1648 | 0.6998 | 0.7524 |
| ts_msp | 0.2978 ± 0.1644 | 0.6992 | 0.7529 |

## Exploratory paired comparisons (Holm over three)

- EXPLORATORY: FBCSP-GMM - ts_msp AURC: Δ=-0.028940, 95% CI=[-0.042268847692884984, -0.015615009107552803], two-sided p=0.003906, Holm p=0.011719, r=-1.0000.
- EXPLORATORY: FBCSP-GMM - msp AURC: Δ=-0.028915, 95% CI=[-0.04184307362624994, -0.016321278914438974], two-sided p=0.003906, Holm p=0.011719, r=-1.0000.
- EXPLORATORY: FBCSP-GMM - vit_gmm AURC: Δ=-0.131944, 95% CI=[-0.17488111924100683, -0.09846720202302178], two-sided p=0.003906, Holm p=0.011719, r=-1.0000.

## Interpretation limits
Changing the representation alone does not isolate a causal mechanism. Prior E exposure and inherited preprocessing/epoch-selection limitations remain. Report favorable and unfavorable comparisons alike; no further variants are selected.
原主实验文件hash核对未变。