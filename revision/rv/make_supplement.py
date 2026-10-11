"""Build Supplementary Material S1–S9 from saved result files only (no model is re-run).

Output: revision/supplement/ (S*.csv tables, S6 figure, Supplementary_Material.md)
Usage: python make_supplement.py
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common  # noqa: F401
from common import CACHE, MODELS, OUT, REV, SUBJECTS, CLASSES
from tab import write_csv

SUP = REV / "supplement"
B2 = REV / "bci2b"
SCORES = ["msp", "ts_msp", "neg_entropy", "ts_neg_entropy", "aligned_gmm", "max_gmm"]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "NA"


def md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ S1 data manifest
def s1():
    man = json.loads((CACHE / "cache_manifest.json").read_text())
    rows = []
    for m in man:
        rows.append({"file": m["file"], "trials_kept": m["n_trials"],
                     **{f"run{k}": v for k, v in m["per_run"].items()},
                     **{f"class{k}": v for k, v in m["per_class"].items()},
                     "gdf_md5": m["gdf_md5"]})
    write_csv(rows, SUP / "S1_data_manifest_2a.csv")
    lab = (REV.parent / "true_labels" / "PROVENANCE.md").read_text(encoding="utf-8")
    return rows, lab


# ------------------------------------------------------------------ S2 parameters and superseded values
def s2():
    p = json.loads((REV / "protocol.json").read_text(encoding="utf-8"))
    superseded = [
        {"quantity": "Selective result at tau = 0.80 (accepted accuracy / kappa)", "original_manuscript": "0.676 / 0.559",
         "origin": "Result file of evaluate_reject.py: 0.691 / 0.587, computed on Concat+LR MSP, not GMM", "revision": "Replaced by run-6 thresholds (Table 5)"},
        {"quantity": "Coverage at tau = 0.80", "original_manuscript": "0.660 (1563/2368)",
         "origin": "Concat+LR MSP, pooled, no validation split", "revision": "Replaced (realised coverage reported per score)"},
        {"quantity": "Wilcoxon p, Concat+LR vs ViT / GMM", "original_manuscript": "0.0039 / 0.0020",
         "origin": "One-sided, not stated", "revision": "All tests two-sided (except the single 2b test, pre-specified one-sided)"},
        {"quantity": "Wilcoxon p, Concat+LR vs FBCSP", "original_manuscript": "0.3750",
         "origin": "Not reproducible from Table 4 (two-sided exact 0.496)", "revision": "Recomputed under new protocol"},
        {"quantity": "Friedman chi2", "original_manuscript": "24.36, p = 2.1e-5",
         "origin": "Not reproducible from Table 4 (21.53, p = 8.2e-5)", "revision": "Not used"},
        {"quantity": "GMM kappa", "original_manuscript": "0.1903 / 0.2200 / 0.1910",
         "origin": "Different configurations (no reduction / PCA32 sweep / standalone)", "revision": "Single frozen configuration"},
        {"quantity": "ViT kappa", "original_manuscript": "0.2538 / 0.2592", "origin": "Different runs", "revision": "Single frozen run"},
        {"quantity": "ECE / MCE", "original_manuscript": "0.226 / 0.305",
         "origin": "Concat+LR MSP, 10 bins (text said GMM, 8 bins)", "revision": "8 equal-width bins, raw and temperature-scaled (Section 3.3)"},
        {"quantity": "SDs", "original_manuscript": "ddof = 0", "origin": "Population SD", "revision": "Sample SD (ddof = 1)"},
        {"quantity": "STFT input units", "original_manuscript": "volts", "origin": "log1p ineffective at 1e-7", "revision": "microvolts"},
    ]
    write_csv(superseded, SUP / "S2_superseded_values.csv")
    return p, superseded


# ------------------------------------------------------------------ S3 per-participant selective results
def s3():
    sm = read_csv(OUT / "subject_metrics_clean.csv")
    rows = []
    for dec in ("concat", "eegnet"):
        forced_key = {"concat": "p_concat", "eegnet": "p_eeg"}[dec]
        for s in SUBJECTS:
            fr = [r for r in sm if r["subject"] == s and r["decoder"] == forced_key and r["score"] == "forced"][0]
            for sc in SCORES + ["random", "oracle"]:
                r = [x for x in sm if x["subject"] == s and x["decoder"] == dec and x["score"] == sc][0]
                row = {"decoder": dec, "subject": s, "score": sc, "n_E": fr["n_total"],
                       "full_acc": f(fr["acc"]), "full_kappa": f(fr["kappa"]),
                       "AURC": f(r["aurc"], 4), "E_AURC": f(r["e_aurc"], 4), "AUROC": f(r["auroc"])}
                for q in ("0.7", "0.8", "0.9"):
                    row[f"mc{q}_acc"] = f(r.get(f"mc{q}_acc"))
                    if sc not in ("random", "oracle"):
                        row[f"thr{q}_cov"] = f(r.get(f"thr{q}_coverage"))
                        row[f"thr{q}_n_acc"] = r.get(f"thr{q}_n_accept", "")
                        row[f"thr{q}_acc"] = f(r.get(f"thr{q}_acc"))
                        row[f"thr{q}_kappa"] = f(r.get(f"thr{q}_kappa"))
                rows.append(row)
    write_csv(rows, SUP / "S3_per_participant_selective_2a.csv")
    return rows


# ------------------------------------------------------------------ S4 class-wise rejection
def s4():
    cr = read_csv(OUT / "class_rejection_clean.csv")
    write_csv(cr, SUP / "S4_class_rejection_per_participant_2a.csv")
    pooled = read_csv(OUT / "table_class_rejection.csv")
    write_csv(pooled, SUP / "S4_class_rejection_pooled_2a.csv")
    return pooled


# ------------------------------------------------------------------ S5 calibration bins and thresholds
def s5():
    cal = json.loads((OUT / "calibration_clean.json").read_text())
    rows = []
    for dec in ("concat", "eegnet"):
        for kind in ("raw", "ts"):
            for s in SUBJECTS:
                c = cal[f"{s}_{dec}_{kind}"]
                rows.append({"decoder": dec, "probabilities": kind, "subject": s, "ECE": f(c["ece"]),
                             "MCE": f(c["mce"]), "NLL": f(c["nll"]), "Brier": f(c["brier"]),
                             **{f"bin{i + 1}_n": b[2] for i, b in enumerate(c["bins"])}})
    write_csv(rows, SUP / "S5_calibration_per_participant_2a.csv")
    fp = json.loads((OUT / "fitted_params.json").read_text())
    trows = []
    for s in SUBJECTS:
        for dec in ("concat", "eegnet"):
            r = {"subject": s, "decoder": dec, "T": f(fp[s][dec]["T"]), "T_flag": fp[s][dec]["T_flag"],
                 "n_run6": fp[s][dec]["n_C"]}
            for sc in SCORES:
                r[f"tau0.8_{sc}"] = f(fp[s][dec]["thresholds"][sc]["0.8"], 4)
            trows.append(r)
    write_csv(trows, SUP / "S5_temperatures_thresholds_2a.csv")
    return rows, trows


# ------------------------------------------------------------------ S6 training curves
def s6():
    fig, axes = plt.subplots(3, 6, figsize=(13, 6.5))
    rows = []
    for i, s in enumerate(SUBJECTS):
        ep = json.loads((MODELS / s / "epochs.json").read_text())
        for j, net in enumerate(("vit", "eegnet")):
            h = np.array(ep[f"{net}_history"])
            ax = axes[i // 3, (i % 3) * 2 + j]
            x = np.arange(1, len(h) + 1)
            ax.plot(x, h[:, 0], color="#2c3e50", lw=0.8, label="train loss")
            ax.plot(x, h[:, 1], color="#c0392b", lw=0.8, label="val NLL")
            ax2 = ax.twinx()
            ax2.plot(x, h[:, 2], color="#27ae60", lw=0.6, alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.tick_params(labelsize=6)
            ax.axvline(ep[net], color="#555", ls="--", lw=0.8)
            if ep.get(f"{net}_v1", ep[net]) != ep[net]:
                ax.axvline(ep[f"{net}_v1"], color="#e67e22", ls=":", lw=1.0)
            ax.set_title(f"{s} {'ChannelViT' if net == 'vit' else 'EEGNet'}: {ep[net]} ep", fontsize=7.5)
            ax.tick_params(labelsize=6)
            rows.append({"subject": s, "network": net, "selected_epochs": ep[net],
                         "rule": ep.get(f"{net}_rule", "nll"), "v1_rule_epochs": ep.get(f"{net}_v1", ep[net]),
                         "max_epochs": len(h), "val_nll_at_selected": f(h[ep[net] - 1, 1]),
                         "val_acc_at_selected": f(h[ep[net] - 1, 2])})
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Supplementary Fig. S6. Epoch selection on member 1 (train on two representation runs, validate on the "
                 "third). Dark: training loss; red: validation NLL; green (right axis): validation accuracy; dashed: "
                 "selected epochs; dotted orange: original-rule selection (A02 EEGNet only).", fontsize=8, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(SUP / f"S6_training_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_csv(rows, SUP / "S6_epoch_selection_2a.csv")
    rows2b = []
    for k in range(1, 10):
        p = B2 / "models" / f"B0{k}" / "epochs.json"
        if p.exists():
            e = json.loads(p.read_text())
            rows2b.append({"subject": f"B0{k}", "network": "vit", "selected_epochs": e["vit"], "rule": e.get("vit_rule", "")})
    write_csv(rows2b, SUP / "S6_epoch_selection_2b.csv")
    return rows, rows2b


# ------------------------------------------------------------------ S7 perturbations per participant
def s7():
    rows = []
    for cond in ("drop1", "drop3", "drop5", "eog1", "eog2", "eog4"):
        sm = read_csv(OUT / f"subject_metrics_{cond}.csv")
        for dec, fk in (("concat", "p_concat"), ("eegnet", "p_eeg")):
            for s in SUBJECTS:
                fr = [r for r in sm if r["subject"] == s and r["decoder"] == fk and r["score"] == "forced"][0]
                row = {"condition": cond, "decoder": dec, "subject": s, "forced_acc": f(fr["acc"]),
                       "forced_kappa": f(fr["kappa"])}
                for sc in ("aligned_gmm", "msp", "ts_msp"):
                    r = [x for x in sm if x["subject"] == s and x["decoder"] == dec and x["score"] == sc][0]
                    row[f"{sc}_AURC"] = f(r["aurc"], 4)
                    row[f"{sc}_cov0.8"] = f(r["thr0.8_coverage"])
                    row[f"{sc}_acc0.8"] = f(r["thr0.8_acc"])
                rows.append(row)
    write_csv(rows, SUP / "S7_perturbations_per_participant_2a.csv")
    return rows


# ------------------------------------------------------------------ S8 statistics, amendment, deviations
def s8():
    st = json.loads((OUT / "stats_clean.json").read_text())
    x1 = json.loads((OUT / "exploratory" / "x1_stats.json").read_text())
    b2 = json.loads((B2 / "out" / "primary_2b.json").read_text())
    rows = []

    def add(fam, r, p_key="p", holm=True):
        rows.append({"family": fam, "comparison": r["comparison"], "n": r.get("n", r.get("n_nonzero", "")),
                     "mean_diff": f(r["mean"], 4), "CI95_low": f(r["mean_ci"][0], 4), "CI95_high": f(r["mean_ci"][1], 4),
                     "median_diff": f(r["median"], 4), "W_plus": r["W_plus"], "r_rb": f(r["r_rb"]),
                     "p": f(r[p_key], 4), "p_holm": f(r["p_holm"], 4) if holm and "p_holm" in r else "—"})
    add("Primary (2a, two-sided)", st["primary"], holm=False)
    for fam in ("family_A", "family_B", "family_C", "family_D"):
        for r in st[fam]:
            add(fam.replace("family_", "Family ") + " (2a, two-sided, Holm)", r)
    for r in st["forced_pairs"]:
        add("Forced-choice kappa (2a, two-sided, Holm)", r)
    for r in x1["family_X1"]:
        add("Exploratory X1 (2a, post hoc, two-sided, Holm)", r)
    p2 = dict(b2["primary"])
    p2["comparison"] = "2b: AURC(FBCSP-GMM) - AURC(MSP)"
    add("2b confirmatory (one-sided, single test)", p2, p_key="p_one_sided", holm=False)
    write_csv(rows, SUP / "S8_all_tests.csv")
    proto = json.loads((REV / "protocol.json").read_text(encoding="utf-8"))
    amend = proto.get("amendments", [])
    sha = (REV / "protocol.sha256").read_text(encoding="utf-8")
    dev2b = (B2 / "deviations.md").read_text(encoding="utf-8")
    return rows, amend, sha, dev2b


# ------------------------------------------------------------------ S9 literature (template)
S9_ROWS = [
    {"method": "FBCSP (Ang et al., 2012)", "kappa_2a": "[TO DO: verify]", "protocol": "T→E, subject-specific", "notes": "full training session"},
    {"method": "EEGNet (Lawhern et al., 2018)", "kappa_2a": "[TO DO: verify]", "protocol": "[TO DO]", "notes": ""},
    {"method": "FBCNet (Mane et al.)", "kappa_2a": "[TO DO: verify]", "protocol": "[TO DO]", "notes": ""},
    {"method": "ATCNet / physics-informed ATCNet (Altaheri et al.)", "kappa_2a": "[TO DO: verify]", "protocol": "[TO DO]", "notes": ""},
    {"method": "EEG Conformer (Song et al., 2023)", "kappa_2a": "[TO DO: verify]", "protocol": "[TO DO]", "notes": ""},
    {"method": "This study: FBCSP-only", "kappa_2a": "0.505", "protocol": "T→E, subject-specific", "notes": "each model trained on 3 of 6 training runs"},
    {"method": "This study: Concat+LR", "kappa_2a": "0.422", "protocol": "T→E, subject-specific", "notes": "each model trained on 3 of 6 training runs"},
    {"method": "This study: EEGNet", "kappa_2a": "0.383", "protocol": "T→E, subject-specific", "notes": "each model trained on 3 of 6 training runs"},
]


# ------------------------------------------------------------------ document
def main():
    SUP.mkdir(parents=True, exist_ok=True)
    m1, lab = s1()
    proto, sup = s2()
    r3 = s3()
    r4 = s4()
    r5, t5 = s5()
    r6, r6b = s6()
    r7 = s7()
    r8, amend, sha, dev2b = s8()
    write_csv(S9_ROWS, SUP / "S9_literature_comparison_TEMPLATE.csv")

    L = ["# Supplementary Material", "",
         "All tables were generated by `rv/make_supplement.py` from the frozen result files; machine-readable versions "
         "are provided as CSV files in this folder. Values are per participant unless stated otherwise.", ""]

    L += ["## S1. Data manifest", "",
          "BCI Competition IV 2a after exclusion of artifact-marked trials (event 1023). Trials per run and per class "
          "(1 = left hand, 2 = right hand, 3 = feet, 4 = tongue), and MD5 checksums of the raw GDF files "
          "(`S1_data_manifest_2a.csv`).", "",
          md_table(m1, ["file", "trials_kept", "run1", "run2", "run3", "run4", "run5", "run6", "class1", "class2", "class3", "class4"]),
          "", "**Evaluation labels.** Provenance and verification record:", "", "```", lab.strip(), "```", ""]

    L += ["## S2. Analysis settings and superseded values", "",
          f"Frozen protocol `{proto['protocol_id']}` (file `protocol.json`; the full settings are reproduced in the "
          "released repository). Values reported in the original submission that are superseded by the revision, with "
          "their audited origin (`S2_superseded_values.csv`):", "",
          md_table(sup, ["quantity", "original_manuscript", "origin", "revision"],
                   ["Quantity", "Original manuscript", "Audited origin", "Revision"]), ""]

    keep = [r for r in r3 if r["score"] in ("ts_msp", "aligned_gmm")]
    L += ["## S3. Per-participant selective results (BCI IV 2a)", "",
          "Full table for all scores, q = 0.7/0.8/0.9 and both decoders in `S3_per_participant_selective_2a.csv`. "
          "Excerpt: temperature-scaled MSP and aligned GMM at q = 0.8.", "",
          md_table(keep, ["decoder", "subject", "score", "n_E", "full_acc", "AURC", "AUROC", "thr0.8_cov", "thr0.8_acc", "thr0.8_kappa"],
                   ["Decoder", "Participant", "Score", "N", "Full acc.", "AURC", "AUROC", "Coverage", "Accepted acc.", "Accepted κ"]), ""]

    r4k = [dict(r, rejection_rate=f(r["rejection_rate"]), accepted_error_rate=f(r["accepted_error_rate"])) for r in r4]
    L += ["## S4. Class-wise rejection (BCI IV 2a, q = 0.8)", "",
          "Pooled over participants; per-participant counts in `S4_class_rejection_per_participant_2a.csv`.", "",
          md_table(r4k, ["decoder", "score", "class", "total", "accepted", "rejected", "rejection_rate", "accepted_error_rate"],
                   ["Decoder", "Score", "True class", "Total", "Accepted", "Rejected", "Rejection rate", "Error rate (accepted)"]), ""]

    L += ["## S5. Calibration, temperatures and thresholds (BCI IV 2a)", "",
          "Per-participant ECE, MCE, NLL, Brier and bin counts (8 equal-width bins) in "
          "`S5_calibration_per_participant_2a.csv`. Temperatures and run-6 thresholds:", "",
          md_table([r for r in t5 if r["decoder"] == "concat"],
                   ["subject", "T", "T_flag", "n_run6", "tau0.8_msp", "tau0.8_ts_msp", "tau0.8_aligned_gmm"],
                   ["Participant", "T", "Status", "n (run 6)", "τ0.8 MSP", "τ0.8 TS-MSP", "τ0.8 aligned GMM"]), ""]

    L += ["## S6. Epoch selection and training curves", "",
          "Supplementary Fig. S6 (`S6_training_curves.png`) shows the member-1 training and validation curves. "
          "Selected numbers of epochs (amended rule) and the selection under the original rule:", "",
          md_table(r6, ["subject", "network", "selected_epochs", "rule", "v1_rule_epochs", "val_nll_at_selected", "val_acc_at_selected"],
                   ["Participant", "Network", "Epochs", "Rule", "Original rule", "Val. NLL", "Val. acc."]), "",
          "BCI IV 2b (ChannelViT only):", "",
          md_table(r6b, ["subject", "network", "selected_epochs", "rule"]), ""]

    L += ["## S7. Raw-EEG perturbations per participant", "",
          "Per-participant forced-choice accuracy and κ, AURC and q = 0.8 coverage/accepted accuracy for aligned GMM, "
          "MSP and temperature-scaled MSP, for both decoders and all six perturbation conditions, are given in "
          "`S7_perturbations_per_participant_2a.csv` (108 rows). Subject means are reported in Table 6 and Fig. 6.", ""]

    L += ["## S8. Statistical tests, protocol amendment and deviations", "",
          "All tests performed (`S8_all_tests.csv`): exact Wilcoxon signed-rank tests on participant-level paired "
          "differences, participant-bootstrap 95% CI of the mean (10 000 resamples, seed 2026), matched-pairs "
          "rank-biserial r.", "",
          md_table(r8, ["family", "comparison", "n", "mean_diff", "CI95_low", "CI95_high", "W_plus", "r_rb", "p", "p_holm"],
                   ["Family", "Comparison", "n", "Mean Δ", "CI low", "CI high", "W+", "r", "p", "p (Holm)"]), "",
          "**Protocol checksums (BCI IV 2a).**", "", "```", sha.strip(), "```", "",
          "**Protocol amendment 1 (made before any evaluation label was read).**", ""]
    for a in amend:
        L += [f"- Date: {a['date']}", f"- Reason: {a['reason']}", f"- Change: {a['change']}",
              f"- Consequence: {a['consequence']}", ""]
    L += ["**BCI IV 2b deviation record.**", "", dev2b.strip(), ""]

    L += ["## S9. Published BCI IV 2a results and evaluation protocols", "",
          "**[TO DO — authors: fill in from the original publications and verify each value. Published numbers obtained "
          "under different protocols (random splits, cross-subject training, different windows, data augmentation or "
          "pre-training) are not directly comparable with ours and are listed for context only.]**", "",
          md_table(S9_ROWS, ["method", "kappa_2a", "protocol", "notes"], ["Method", "κ (2a)", "Protocol", "Notes"]), ""]

    (SUP / "Supplementary_Material.md").write_text("\n".join(L), encoding="utf-8")
    print("supplement written:", sorted(p.name for p in SUP.iterdir()))


if __name__ == "__main__":
    main()
