"""Additional manuscript figures for the final revision, from saved result files only (no model is re-run).

Fig. 3  representation dependence of the GMM gate: AURC difference to MSP and correctness AUROC for gates built on
        ChannelViT, EEGNet and FBCSP representations, on BCI IV 2a and 2b, labelled by evidence type
Fig. 5  confusion matrices of Concat+LR on all evaluation trials and on trials accepted by temperature-scaled MSP at the
        run-6 threshold (q = 0.8), with per-class rejection rates

Writes PNG/SVG/PDF to the working folder (BCI/BCI返修). Usage: python make_figures_v2.py
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common  # noqa: F401
from common import CLASSES, OUT, REV, SUBJECTS
import stats as ST

DEST = REV.parent / "BCI返修"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(DEST / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ Fig. 3
def fig_representation():
    sm = read_csv(OUT / "subject_metrics_clean.csv")
    st = json.loads((OUT / "stats_clean.json").read_text())
    x1 = json.loads((OUT / "exploratory" / "x1_stats.json").read_text())
    b2 = json.loads((REV / "bci2b" / "out" / "primary_2b.json").read_text())
    s2b = read_csv(REV / "bci2b" / "out" / "subject_metrics_2b.csv")
    x1s = read_csv(OUT / "exploratory" / "x1_subject.csv")

    def col(dec, score, key):
        return np.array([float([r for r in sm if r["subject"] == s and r["decoder"] == dec and r["score"] == score][0][key])
                         for s in SUBJECTS])

    vit2b = np.array([float(r["aurc_vit_gmm"]) - float(r["aurc_msp"]) for r in s2b])
    entries = [
        # label, per-subject diffs, AUROC gate, AUROC MSP, evidence, color
        ("2a · ChannelViT", np.array(st["primary"]["diffs"]), col("concat", "aligned_gmm", "auroc"),
         col("concat", "msp", "auroc"), "pre-specified", "#c0392b"),
        ("2a · EEGNet", np.array(st["family_C"][0]["diffs"]), col("eegnet", "aligned_gmm", "auroc"),
         col("eegnet", "msp", "auroc"), "pre-specified", "#16a085"),
        ("2a · FBCSP", np.array(x1["family_X1"][0]["diffs"]),
         np.array([float(r["auroc_fbcsp_gmm"]) for r in x1s]), col("concat", "msp", "auroc"), "exploratory", "#2980b9"),
        ("2b · ChannelViT", vit2b, np.array([float(r["auroc_vit_gmm"]) for r in s2b]),
         np.array([float(r["auroc_msp"]) for r in s2b]), "descriptive", "#c0392b"),
        ("2b · FBCSP", np.array(b2["primary"]["per_subject_diff"]),
         np.array([float(r["auroc_fbcsp_gmm"]) for r in s2b]), np.array([float(r["auroc_msp"]) for r in s2b]),
         "pre-specified\n(not confirmed)", "#2980b9"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    rng = np.random.default_rng(3)
    for i, (lab, d, _, _, ev, c) in enumerate(entries):
        y = len(entries) - 1 - i
        ax.scatter(d, y + rng.uniform(-0.13, 0.13, len(d)), s=13, color=c, alpha=0.55, zorder=2)
        ci = ST.bootstrap_ci(d)
        ax.errorbar(ci["mean"], y, xerr=[[ci["mean"] - ci["mean_ci"][0]], [ci["mean_ci"][1] - ci["mean"]]],
                    fmt="D", color=c, mec="black", ms=6, capsize=3, zorder=3,
                    mfc=c if ev.startswith("pre") else "white")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([f"{e[0]}\n{e[4].replace(chr(10), ' ')}" for e in entries][::-1], fontsize=7.5)
    ax.set_xlabel("ΔAURC (gate − MSP)\n< 0: gate ranks errors better")
    ax.set_title("(a) Gate versus classifier confidence", fontsize=9, loc="left")
    ax.set_xlim(-0.18, 0.26)
    ax = axes[1]
    for i, (lab, _, ag, am, ev, c) in enumerate(entries):
        y = len(entries) - 1 - i
        ax.plot([am.mean(), ag.mean()], [y, y], color="#bbb", lw=1.2, zorder=1)
        ax.scatter(am.mean(), y, marker="o", color="#2c3e50", s=28, zorder=2)
        ax.scatter(ag.mean(), y, marker="D", color=c, edgecolor="black", s=30, zorder=3)
    ax.axvline(0.5, color="#aaa", ls=":", lw=0.8)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(entries) - 0.4 + 0.9)
    ax.set_xlim(0.4, 0.82)
    ax.set_xlabel("Correctness AUROC (mean)\n0.5 = chance")
    ax.set_title("(b) Error-ranking ability", fontsize=9, loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2c3e50", label="MSP"),
                       Line2D([], [], marker="D", ls="", mfc="#bbbbbb", mec="black", label="GMM gate")],
              frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    axes[0].set_ylim(-0.6, len(entries) - 0.4 + 0.9)
    fig.tight_layout()
    save(fig, "fig3_representation_dependence")


# ------------------------------------------------------------------ Fig. 5
def fig_confusion():
    fp = json.loads((OUT / "fitted_params.json").read_text())
    tr = [t for t in read_csv(OUT / "trials_clean.csv") if t["decoder"] == "concat"]
    C = np.zeros((2, 4, 4), int)
    for t in tr:
        y, yh = int(t["y"]), int(t["yhat"])
        C[0, y, yh] += 1
        if float(t["ts_msp"]) >= fp[t["subject"]]["concat"]["thresholds"]["ts_msp"]["0.8"]:
            C[1, y, yh] += 1
    names = ["Left hand", "Right hand", "Feet", "Tongue"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for k, (ax, title) in enumerate(zip(axes[:2], [f"(a) All evaluation trials (N = {C[0].sum()})",
                                                    f"(b) Accepted at run-6 threshold (N = {C[1].sum()})"])):
        P = C[k] / C[k].sum(1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{C[k, i, j]}\n({P[i, j]:.2f})", ha="center", va="center", fontsize=6.8,
                        color="white" if P[i, j] > 0.55 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels(names if k == 0 else [], fontsize=7.5)
        ax.set_xlabel("Predicted class")
        if k == 0:
            ax.set_ylabel("True class")
        ax.set_title(title, fontsize=8.5, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    ax = axes[2]
    rej = 1 - C[1].sum(1) / C[0].sum(1)
    ax.barh(range(4), rej, color="#7f8c8d")
    for i, v in enumerate(rej):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("Rejection rate")
    ax.set_title("(c) Withheld per true class", fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, "fig5_confusion_rejection")
    return C


if __name__ == "__main__":
    fig_representation()
    C = fig_confusion()
    print("written to", DEST, "accepted", C[1].sum(), "of", C[0].sum())
