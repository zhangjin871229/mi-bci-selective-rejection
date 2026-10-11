"""Run the one-shot E evaluation (clean + perturbations), statistics, summary tables and figures.

Usage: python report.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common  # noqa: F401
import stats as ST
from common import CLASSES, OUT, SUBJECTS, load_protocol, protocol_hash
from evaluate import _col, evaluate
from selective import order_desc, tiebreak
from tab import select, write_csv

FIG = OUT / "figures"
SCORE_ORDER = ["aligned_gmm", "max_gmm", "msp", "neg_entropy", "ts_msp", "ts_neg_entropy", "random", "oracle"]
LABELS = {"aligned_gmm": "Aligned GMM", "max_gmm": "Max GMM (original gate)", "msp": "MSP",
          "neg_entropy": "Neg. entropy", "ts_msp": "Temperature-scaled MSP",
          "ts_neg_entropy": "TS neg. entropy", "random": "Random", "oracle": "Oracle"}
COLORS = {"aligned_gmm": "#c0392b", "max_gmm": "#e67e22", "msp": "#2c3e50", "neg_entropy": "#7f8c8d",
          "ts_msp": "#2980b9", "ts_neg_entropy": "#85c1e9", "random": "#bbbbbb", "oracle": "#27ae60"}
DEC_TITLE = {"concat": "FBCSP + ChannelViT (Concat+LR)", "eegnet": "EEGNet"}


def read_trials(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = defaultdict(list)
    for r in rows:
        by[(r["subject"], r["decoder"])].append(r)
    return by


def risk_coverage_figure(by, decoder, path):
    grid = np.linspace(0.05, 1.0, 96)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for sc in ["oracle", "random", "msp", "ts_msp", "neg_entropy", "max_gmm", "aligned_gmm"]:
        curves = []
        for s in SUBJECTS:
            d = by[(s, decoder)]
            c = np.array([r["correct"] == "True" for r in d], float)
            key = tiebreak([r["trial_id"] for r in d])
            if sc == "oracle":
                v = c
            elif sc == "random":
                v = np.random.default_rng(int(s[1:]) + 77).random(len(c))
            else:
                v = np.array([float(r[sc]) for r in d])
            o = order_desc(v, key)
            risk = np.cumsum(1 - c[o]) / np.arange(1, len(c) + 1)
            cov = np.arange(1, len(c) + 1) / len(c)
            curves.append(np.interp(grid, cov, risk))
        ax.plot(grid, np.mean(curves, 0), color=COLORS[sc],
                lw=2.2 if sc in ("aligned_gmm", "msp", "ts_msp") else 1.3,
                ls="--" if sc in ("random", "oracle") else "-", label=LABELS[sc])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (mean of 9 subjects)")
    ax.set_xlim(0.05, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(DEC_TITLE[decoder], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def reliability_figure(calib, decoder, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for ax, kind, title in zip(axes, ["raw", "ts"], ["Raw probabilities", "Temperature-scaled (T fitted on run 6)"]):
        conf, acc, cnt = np.zeros(8), np.zeros(8), np.zeros(8)
        for s in SUBJECTS:
            for i, (lo, hi, n, c, a) in enumerate(calib[f"{s}_{decoder}_{kind}"]["bins"]):
                if n:
                    conf[i] += n * c
                    acc[i] += n * a
                    cnt[i] += n
        m = cnt > 0
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        ax.plot(conf[m] / cnt[m], acc[m] / cnt[m], "o-", color="#2c3e50")
        for x, y_, n in zip(conf[m] / cnt[m], acc[m] / cnt[m], cnt[m]):
            ax.annotate(f"n={int(n)}", (x, y_), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
        ece = np.mean([calib[f"{s}_{decoder}_{kind}"]["ece"] for s in SUBJECTS])
        ax.set_title(f"{title}\nsubject-mean ECE = {ece:.3f}", fontsize=9)
        ax.set_xlabel("Top-label confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(DEC_TITLE[decoder], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def mean_sd(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1))


def score_table(sm, q):
    rows = []
    for dec in ["concat", "eegnet"]:
        for sc in SCORE_ORDER:
            r = {"decoder": dec, "score": sc}
            for col in ["aurc", "e_aurc", "auroc"] + [f"mc{x}_risk" for x in (0.7, 0.8, 0.9)]:
                r[col], r[col + "_sd"] = mean_sd(_col(sm, dec, sc, col))
            if sc not in ("random", "oracle"):
                for col in ("coverage", "acc", "kappa", "risk"):
                    r[f"thr{q}_{col}"], r[f"thr{q}_{col}_sd"] = mean_sd(_col(sm, dec, sc, f"thr{q}_{col}"))
            rows.append(r)
    return rows


def forced_table(sm):
    rows = []
    decs = ["p_concat", "p_fbcsp", "p_vit", "p_eeg", "g_vit", "g_eeg"]
    for s in SUBJECTS:
        r = {"subject": s}
        for d in decs:
            m = select(sm, subject=s, decoder=d, score="forced")[0]
            r[f"{d}_kappa"], r[f"{d}_acc"] = m["kappa"], m["acc"]
        rows.append(r)
    for name, fn in [("mean", np.mean), ("sd_ddof1", lambda v: np.std(v, ddof=1))]:
        rows.append({"subject": name, **{k: float(fn([float(x[k]) for x in rows[:9]])) for k in rows[0] if k != "subject"}})
    return rows


def class_table(q_rows):
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for r in q_rows:
        a = agg[(r["decoder"], r["score"], r["class"])]
        a[0] += r["total"]; a[1] += r["accepted"]; a[2] += r["rejected"]; a[3] += r["errors_accepted"]
    return [{"decoder": d, "score": s, "class": c, "total": t, "accepted": a, "rejected": rj,
             "rejection_rate": rj / t, "accepted_error_rate": (e / a) if a else ""}
            for (d, s, c), (t, a, rj, e) in agg.items()]


def perturbation_conditions(P):
    pp = P["perturbations"]
    return [f"drop{k}" for k in pp["channel_dropout"]["k"]] + [f"eog{g}" for g in pp["eog_mixing"]["gain"]]


def main():
    P = load_protocol()
    q = P["thresholds"]["q_primary"]
    FIG.mkdir(parents=True, exist_ok=True)

    sm, fitted = evaluate("clean")
    by = read_trials(OUT / "trials_clean.csv")
    calib = json.loads((OUT / "calibration_clean.json").read_text())
    for dec in ["concat", "eegnet"]:
        risk_coverage_figure(by, dec, FIG / f"risk_coverage_{dec}.png")
        reliability_figure(calib, dec, FIG / f"reliability_{dec}.png")
    write_csv(score_table(sm, q), OUT / "table_scores.csv")
    write_csv(forced_table(sm), OUT / "table_forced_choice.csv")
    with open(OUT / "class_rejection_clean.csv", encoding="utf-8") as f:
        cr = [{**r, **{k: int(r[k]) for k in ("total", "accepted", "rejected", "errors_accepted")}} for r in csv.DictReader(f)]
    write_csv(class_table(cr), OUT / "table_class_rejection.csv")

    pert, famD = [], []
    for cond in perturbation_conditions(P):
        smp, _ = evaluate(cond, fitted=fitted)
        for dec, fk in [("concat", "p_concat"), ("eegnet", "p_eeg")]:
            r = {"condition": cond, "decoder": dec}
            r["forced_acc"], _ = mean_sd(_col(smp, fk, "forced", "acc"))
            r["forced_kappa"], _ = mean_sd(_col(smp, fk, "forced", "kappa"))
            for sc in ["aligned_gmm", "msp", "ts_msp"]:
                r[f"{sc}_aurc"], _ = mean_sd(_col(smp, dec, sc, "aurc"))
                r[f"{sc}_coverage"], _ = mean_sd(_col(smp, dec, sc, f"thr{q}_coverage"))
                r[f"{sc}_acc"], _ = mean_sd(_col(smp, dec, sc, f"thr{q}_acc"))
            pert.append(r)
        if cond in ("drop5", "eog4"):
            famD.append(ST.paired(_col(smp, "concat", "aligned_gmm", "aurc"), _col(smp, "concat", "ts_msp", "aurc"),
                                  f"{cond} concat: AURC aligned_gmm - ts_msp"))
    write_csv(pert, OUT / "table_perturbation.csv")
    st = json.loads((OUT / "stats_clean.json").read_text())
    st["family_D"] = ST.holm(famD)
    (OUT / "stats_clean.json").write_text(json.dumps(st, indent=1, default=float))
    print("report done; protocol sha", protocol_hash())


if __name__ == "__main__":
    main()
