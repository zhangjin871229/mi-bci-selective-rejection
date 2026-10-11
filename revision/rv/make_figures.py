"""Manuscript figures generated only from saved result files (no model is re-run).

Fig. 1  pipeline diagram (vector)
Fig. 4  per-participant AURC differences: 2a primary (ViT-GMM - MSP), 2a exploratory (FBCSP-GMM - MSP),
        2b validation (FBCSP-GMM - MSP)
Fig. 5  raw-EEG perturbations: forced accuracy and AURC versus perturbation level

Usage: python make_figures.py
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import common  # noqa: F401
from common import OUT, REV

FIG = OUT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
def box(ax, x, y, w, h, text, fc="#ffffff", ec="#34495e", fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, style="-|>", color="#34495e", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color,
                                 lw=1.0, linestyle=ls))


def fig1():
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fs = 7.5

    # column headers (two short lines each, centred over their column)
    for x, t in [(0.085, "Data\n(per participant)"), (0.335, "Representation\n(fitted on A$_j$)"),
                 (0.61, "Heads\n(fitted on B$_j$)"), (0.88, "Confidence\nscores s(x)")]:
        ax.text(x, 0.955, t, ha="center", va="center", fontsize=8, fontweight="bold")

    # data partitions (left column)
    box(ax, 0.00, 0.66, 0.17, 0.23, "Training runs 1–5\nfive members:\nA$_j$ = 3 runs\nB$_j$ = 2 runs",
        fc="#eaf2f8", fs=fs)
    box(ax, 0.00, 0.40, 0.17, 0.12, "Training run 6\n(held out)", fc="#fef5e7", fs=fs)
    box(ax, 0.00, 0.18, 0.17, 0.16, "Evaluation session\n(used once,\nafter freezing)", fc="#f4ecf7", fs=fs)
    box(ax, 0.00, 0.02, 0.17, 0.12, "Protocol frozen\n(SHA-256) before\nevaluation labels", fc="#f2f3f4", fs=7)

    # representation (fitted on A_j)
    box(ax, 0.22, 0.80, 0.23, 0.09, "FBCSP + MIBIF (k = 16)", fs=fs)
    box(ax, 0.22, 0.69, 0.23, 0.09, "ChannelViT\nembedding (192-d)", fs=fs)
    box(ax, 0.22, 0.58, 0.23, 0.09, "EEGNet\n(second backbone)", fs=fs)

    # heads (fitted on B_j)
    box(ax, 0.50, 0.76, 0.22, 0.13, "Decoder: Concat+LR\n$p_F(c\\mid x)$\n$\\hat y$ = argmax$_c$ $p_F$",
        fc="#eafaf1", fs=fs)
    box(ax, 0.50, 0.56, 0.22, 0.15, "GMM gate:\nPCA 8 + one diagonal\nGaussian per class\n$p_G(c\\mid z)$",
        fc="#fdedec", fs=fs)

    # scores
    box(ax, 0.77, 0.77, 0.22, 0.12, "MSP, entropy\n(± temperature\nscaling)", fc="#eafaf1", fs=fs)
    box(ax, 0.77, 0.58, 0.22, 0.11, "Aligned GMM\n$p_G(\\hat y\\mid z)$", fc="#fdedec", fs=fs)

    # calibration/thresholds, decision, evaluation (centre, wide)
    box(ax, 0.24, 0.38, 0.50, 0.12, "Run 6 only: temperature T (NLL) and per-score threshold $\\tau_q$\n"
        "(score at rank ⌈q·n$_C$⌉; q = 0.8, also 0.7 and 0.9)", fc="#fef5e7", fs=fs)
    box(ax, 0.24, 0.20, 0.50, 0.12, "Accept $\\hat y$ if s(x) ≥ $\\tau_q$, otherwise withhold\n"
        "(all scores on identical trials and identical $\\hat y$)", fc="#f4ecf7", fs=fs)
    box(ax, 0.24, 0.02, 0.50, 0.12, "Evaluation: risk–coverage curves, AURC / E-AURC (oracle, random),\n"
        "matched coverage, accepted accuracy and κ, class-wise rejection", fc="#f2f3f4", fs=7)

    # arrows
    arrow(ax, 0.17, 0.775, 0.22, 0.845)
    arrow(ax, 0.17, 0.775, 0.22, 0.735)
    arrow(ax, 0.17, 0.775, 0.22, 0.625)
    arrow(ax, 0.45, 0.845, 0.50, 0.835)
    arrow(ax, 0.45, 0.735, 0.50, 0.81)
    arrow(ax, 0.45, 0.735, 0.50, 0.65)
    arrow(ax, 0.72, 0.83, 0.77, 0.83)
    arrow(ax, 0.72, 0.635, 0.77, 0.635)
    arrow(ax, 0.61, 0.76, 0.61, 0.71, ls="--")  # yhat passed to the gate
    ax.text(0.618, 0.733, "$\\hat y$", fontsize=8)
    arrow(ax, 0.17, 0.46, 0.24, 0.44)
    arrow(ax, 0.88, 0.58, 0.74, 0.47)
    ax.text(0.85, 0.505, "scores\non run 6", fontsize=6.5, color="#555", ha="left")
    arrow(ax, 0.49, 0.38, 0.49, 0.32)
    arrow(ax, 0.17, 0.26, 0.24, 0.26)
    arrow(ax, 0.49, 0.20, 0.49, 0.14)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- Fig. 4
def paired_panel(ax, diffs, labels, title, p_text, color):
    d = np.asarray(diffs, float)
    order = np.argsort(d)
    y = np.arange(len(d))
    ax.axvline(0, color="#888", lw=0.8)
    ax.hlines(y, 0, d[order], color=color, lw=1.6, alpha=0.6)
    ax.scatter(d[order], y, color=color, s=22, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=7.5)
    m = d.mean()
    ax.axvline(m, color=color, lw=1.0, ls="--")
    # statistics go in the title so that no data point is covered
    ax.set_title(f"{title}\n{p_text}", fontsize=8)
    ax.set_xlabel("ΔAURC (gate − MSP)\n< 0: gate better")


def fig4():
    st = json.loads((OUT / "stats_clean.json").read_text())
    x1 = json.loads((OUT / "exploratory" / "x1_stats.json").read_text())
    b2 = json.loads((REV / "bci2b" / "out" / "primary_2b.json").read_text())
    a_lab = [f"A0{i}" for i in range(1, 10)]
    b_lab = [f"B0{i}" for i in range(1, 10)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), sharey=False)
    p = st["primary"]
    paired_panel(axes[0], p["diffs"], a_lab, "(a) 2a, pre-specified: ViT-GMM gate",
                 f"mean {p['mean']:+.3f}, p = {p['p']:.4f}, 9/9 worse", "#c0392b")
    e = x1["family_X1"][0]
    paired_panel(axes[1], e["diffs"], a_lab, "(b) 2a, exploratory: FBCSP-GMM gate",
                 f"mean {e['mean']:+.3f}, Holm p = {e['p_holm']:.4f}, 9/9 better", "#2980b9")
    v = b2["primary"]
    n_better = int(np.sum(np.asarray(v["per_subject_diff"]) < 0))
    paired_panel(axes[2], v["per_subject_diff"], b_lab, "(c) 2b, pre-specified: FBCSP-GMM gate",
                 f"mean {v['mean']:+.3f}, one-sided p = {v['p_one_sided']:.3f}\n{n_better}/9 better; not confirmed",
                 "#7f8c8d")
    fig.tight_layout()
    save(fig, "fig4_paired_differences")


# ---------------------------------------------------------------- Fig. 5
def fig5():
    rows = list(csv.DictReader(open(OUT / "table_perturbation.csv", encoding="utf-8")))
    clean = {}
    for r in csv.DictReader(open(OUT / "table_scores.csv", encoding="utf-8")):
        clean[(r["decoder"], r["score"])] = float(r["aurc"])
    forced = list(csv.DictReader(open(OUT / "table_forced_choice.csv", encoding="utf-8")))
    mean_row = [r for r in forced if r["subject"] == "mean"][0]
    clean_acc = {"concat": float(mean_row["p_concat_acc"]), "eegnet": float(mean_row["p_eeg_acc"])}
    get = {(r["condition"], r["decoder"]): r for r in rows}
    series = {"Channel dropout": (["drop1", "drop3", "drop5"], [1, 3, 5], "channels removed"),
              "Ocular contamination": (["eog1", "eog2", "eog4"], [1, 2, 4], "EOG gain")}
    colors = {"concat": "#2c3e50", "eegnet": "#16a085"}
    names = {"concat": "Concat+LR", "eegnet": "EEGNet"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for col, (title, (conds, levels, xlab)) in enumerate(series.items()):
        xs = [0] + levels
        ax = axes[0, col]
        for dec in ("concat", "eegnet"):
            ys = [clean_acc[dec]] + [float(get[(c, dec)]["forced_acc"]) for c in conds]
            ax.plot(xs, ys, "o-", color=colors[dec], label=names[dec])
        ax.axhline(0.25, color="#aaa", ls=":", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Forced-choice accuracy")
        ax.set_xticks(xs)
        ax.set_ylim(0.2, 0.62)
        if col == 0:
            ax.legend(frameon=False, fontsize=7.5)
        ax = axes[1, col]
        for dec in ("concat", "eegnet"):
            ts = [clean[(dec, "ts_msp")]] + [float(get[(c, dec)]["ts_msp_aurc"]) for c in conds]
            gm = [clean[(dec, "aligned_gmm")]] + [float(get[(c, dec)]["aligned_gmm_aurc"]) for c in conds]
            ax.plot(xs, ts, "o-", color=colors[dec], label=f"{names[dec]}: TS-MSP")
            ax.plot(xs, gm, "s--", color=colors[dec], mfc="white", label=f"{names[dec]}: aligned GMM")
        ax.set_xlabel(xlab + " (0 = clean)")
        ax.set_ylabel("AURC (lower is better)")
        ax.set_xticks(xs)
        if col == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_perturbations")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1()
    fig4()
    fig5()
    print("figures written to", FIG)
