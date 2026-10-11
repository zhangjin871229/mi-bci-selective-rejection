"""One fixed post-hoc FBCSP-GMM gate on the inherited frozen Concat+LR decoder.

No neural retraining, no hyperparameter search, no replacement of primary files.
Phase fit: fit all 45 shallow heads on their original B_j and set C thresholds.
Phase evaluate: apply those saved heads to clean E, once for this exploration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
sys.path.insert(0, str(REV / "rv"))
import common
import selective as S
import stats as ST
from features import fbcsp_window, load_session
from heads import GMMHead
from train_subject import _Unpickler

ROOT = HERE / "fbcsp_gmm_exploratory"
SUBJECTS = common.SUBJECTS
QS = [0.7, 0.8, 0.9]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def mean_decoder(subject, split):
    vals = []
    for member in range(1, 6):
        with np.load(REV / f"out/raw/{subject}_member{member}.npz", allow_pickle=True) as z:
            vals.append(z[f"p_concat_{split}"])
    return np.mean(vals, axis=0)


def frozen_bundle(subject, member):
    # We need only the stored CSP and MIBIF, not the neural weights.
    with (REV / f"models/{subject}/member{member}.pkl").open("rb") as f:
        b = _Unpickler(f).load()
    assert b.vit is None and b.eeg is None
    return b


def fbcsp_features(bundle, X):
    return bundle.mib.transform(bundle.fb.transform(fbcsp_window(X)))


def fit():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "c_predictions").mkdir(exist_ok=True)
    if (ROOT / "FIT_COMPLETE.json").exists():
        raise SystemExit("Fit already completed; no alternative fits are allowed in this analysis.")
    P = common.load_protocol()
    fixed_plan = dict(
        analysis_id="posthoc-fbcsp-gmm-v1", status="EXPLORATORY_AFTER_PRIMARY_E_RESULTS_SEEN",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        user_scope="One FBCSP-GMM gate, all nine subjects; retain inherited main analysis; no variant search.",
        inherited_protocol_sha256=digest(REV / "protocol.json"),
        inherited_primary_stats_sha256=digest(REV / "out/stats_clean.json"),
        code_sha256=digest(Path(__file__)), decoder="Unchanged five-member Concat+LR probability mean",
        representation="Each inherited member's A_j-fitted FBCSP+MIBIF k=16; no refitting CSP/selection",
        head="Member-specific StandardScaler -> PCA up to 8 -> per-class 1 diagonal Gaussian, reg=1e-3; B_j only",
        class_prior="Empirical class counts on B_j, matching inherited GMMHead implementation",
        seed_rule="Inherited member_seed(subject,member,base=2026)",
        gate="Mean five FBCSP-GMM posteriors, indexed by frozen ensemble Concat+LR argmax",
        thresholds="C run6 score quantiles q=.7/.8/.9, accept all ties; do not refit decoder temperature",
        evaluation="Clean E only; no disturbance or representation variants added",
        comparisons=["AURC FBCSP-GMM - TS-MSP", "AURC FBCSP-GMM - raw MSP", "AURC FBCSP-GMM - ViT-GMM"],
        inference="All exploratory; two-sided paired sign-enumeration Wilcoxon, subject bootstrap CI, Holm over 3 comparisons",
        interpretation="Supplement only; no change to primary endpoint, abstract or prior confirmatory claim",
        fit_failures="Missing classes, fewer than 2 class samples, nonfinite features, nonconverged or nonpositive variances: stop and report; no E-driven fallback",
    )
    plan_path = ROOT / "exploratory_plan.json"
    if plan_path.exists():
        raise SystemExit("Plan already exists; review status before resuming rather than replacing it.")
    dump(plan_path, fixed_plan)
    fitted, diagnostics = {}, []
    baseline_manifest = json.loads((HERE / "audit/input_snapshot.json").read_text(encoding="utf-8"))
    dump(ROOT / "inherited_snapshot.json", baseline_manifest)
    for subject in SUBJECTS:
        Td = load_session(subject, "T")
        c_mask = Td["run_id"] == 6
        pC = mean_decoder(subject, "C")
        assert pC.shape == (int(c_mask.sum()), 4)
        yhC = pC.argmax(1)
        cs = []
        for member in range(1, 6):
            t0 = time.time()
            B = np.isin(Td["run_id"], P["split"]["members"][str(member)]["B"])
            b = frozen_bundle(subject, member)
            FB = fbcsp_features(b, Td["X"][B])
            FC = fbcsp_features(b, Td["X"][c_mask])
            yB = Td["y"][B] - 1
            assert np.array_equal(np.unique(yB), np.arange(4))
            assert min(np.bincount(yB, minlength=4)) >= 2
            assert np.isfinite(FB).all() and np.isfinite(FC).all()
            seed = common.member_seed(subject, member, 2026)
            head = GMMHead(d_max=8, n_components=1, reg_covar=1e-3, seed=seed).fit(FB, yB)
            assert all(g.converged_ and np.isfinite(g.covariances_).all() and (g.covariances_ > 0).all() for g in head.gmms)
            p = head.predict_proba(FC)
            assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)
            with (ROOT / f"models/{subject}_member{member}.pkl").open("wb") as f:
                pickle.dump(head, f)
            cs.append(p)
            diagnostics.append(dict(subject=subject, member=member, n_B=int(B.sum()), n_C=int(c_mask.sum()),
                                    n_features=int(FB.shape[1]), pca_dim=head.dim,
                                    class_counts=np.bincount(yB, minlength=4).tolist(),
                                    converged=[bool(g.converged_) for g in head.gmms],
                                    minimum_covariance=float(min(g.covariances_.min() for g in head.gmms)),
                                    inherited_bundle_sha256=digest(REV / f"models/{subject}/member{member}.pkl")))
            print(f"FIT {subject} member {member}/5: {time.time()-t0:.1f}s", flush=True)
        pc = np.mean(cs, axis=0)
        score = pc[np.arange(len(pc)), yhC]
        thresholds = {str(q): S.threshold_from_C(score, q) for q in QS}
        fitted[subject] = dict(thresholds=thresholds, n_C=len(pc),
                               actual_C_coverage={str(q): float((score >= thresholds[str(q)]).mean()) for q in QS})
        np.savez_compressed(ROOT / f"c_predictions/{subject}.npz", p_gmm=pc,
                            p_decoder=pC, trial_id=Td["trial_id"][c_mask], score=score)
    dump(ROOT / "fitted_C_parameters.json", fitted)
    dump(ROOT / "head_diagnostics.json", diagnostics)
    artifacts = [plan_path, ROOT / "fitted_C_parameters.json", ROOT / "head_diagnostics.json",
                 *sorted((ROOT / "models").glob("*.pkl")), *sorted((ROOT / "c_predictions").glob("*.npz"))]
    dump(ROOT / "FIT_COMPLETE.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                          artifacts={str(p.relative_to(ROOT)): digest(p) for p in artifacts},
                                          status="All new fitting and C thresholds fixed before this exploratory E scoring; historical E results already known."))


def evaluate():
    completion = json.loads((ROOT / "FIT_COMPLETE.json").read_text(encoding="utf-8"))
    for path, hash_value in completion["artifacts"].items():
        assert digest(ROOT / path) == hash_value, f"Frozen exploratory artifact changed: {path}"
    if (ROOT / "EVALUATION_COMPLETE.json").exists():
        raise SystemExit("Exploratory E scoring already completed; do not rerun for result selection.")
    fitted = json.loads((ROOT / "fitted_C_parameters.json").read_text())
    old_params = json.loads((REV / "out/fitted_params.json").read_text())
    old_stats = json.loads((REV / "out/stats_clean.json").read_text())
    per_subject, trials, counts, curves = [], [], [], {}
    for subject in SUBJECTS:
        t0 = time.time()
        Ed = load_session(subject, "E")
        pE = mean_decoder(subject, "E")
        y, yh = Ed["y"] - 1, pE.argmax(1)
        correct, key = yh == y, S.tiebreak(Ed["trial_id"])
        gs = []
        for member in range(1, 6):
            b = frozen_bundle(subject, member)
            FE = fbcsp_features(b, Ed["X"])
            with (ROOT / f"models/{subject}_member{member}.pkl").open("rb") as f:
                head = pickle.load(f)
            gs.append(head.predict_proba(FE))
        pg = np.mean(gs, axis=0)
        score = pg[np.arange(len(y)), yh]
        assert np.isfinite(score).all() and np.allclose(pg.sum(1), 1)
        with np.load(REV / f"out/raw/{subject}_member1.npz") as z:
            pass  # Member outputs remain untouched; all reference probabilities follow below.
        gv = []
        for member in range(1, 6):
            with np.load(REV / f"out/raw/{subject}_member{member}.npz") as z:
                gv.append(z["g_vit_E"])
        gv = np.mean(gv, axis=0)
        ref = S.scores(pE, gv, old_params[subject]["concat"]["T"])
        sc = dict(fbcsp_gmm=score, vit_gmm=ref["aligned_gmm"], msp=ref["msp"], ts_msp=ref["ts_msp"])
        curves[subject] = {name: S.risk_curve(correct, values, key).tolist() for name, values in sc.items()}
        curves[subject]["oracle"] = S.risk_curve(correct, correct.astype(float), key).tolist()
        curves[subject]["random_expected"] = [float(1-correct.mean())] * len(y)
        full = S.accepted_metrics(correct, y, yh, np.ones(len(y), bool))
        for name, values in sc.items():
            r = dict(subject=subject, decoder="unchanged_concat", score=name,
                     n_total=len(y), forced_accuracy=full["acc"], forced_kappa=full["kappa"],
                     aurc=S.aurc(correct, values, key), auroc=S.correctness_auroc(correct, values))
            r["e_aurc"] = r["aurc"] - S.aurc_oracle(correct)
            for q in QS:
                threshold = (fitted[subject]["thresholds"][str(q)] if name == "fbcsp_gmm" else
                             old_params[subject]["concat"]["thresholds"]["aligned_gmm" if name == "vit_gmm" else name][str(q)])
                accept = values >= threshold
                m = S.accepted_metrics(correct, y, yh, accept)
                r[f"q{q}_threshold"] = threshold
                for metric, value in m.items(): r[f"q{q}_{metric}"] = value
                mc = S.matched_coverage(correct, y, yh, values, key, q)
                for metric in ["acc", "risk", "kappa"]: r[f"mc{q}_{metric}"] = mc[metric]
                if name == "fbcsp_gmm":
                    for klass in range(4):
                        actual = y == klass
                        na = int((actual & accept).sum()); nt = int(actual.sum())
                        counts.append(dict(subject=subject, q=q, true_class=common.CLASSES[klass],
                                           total=nt, accepted=na, rejected=nt-na,
                                           rejection_rate=(nt-na)/nt,
                                           accepted_accuracy=float(correct[actual & accept].mean()) if na else None))
            per_subject.append(r)
        for i in range(len(y)):
            trials.append(dict(subject=subject, trial_id=str(Ed["trial_id"][i]), y=int(y[i]),
                               yhat=int(yh[i]), correct=bool(correct[i]),
                               **{name: float(v[i]) for name, v in sc.items()},
                               **{f"p_gmm_c{j}": float(pg[i, j]) for j in range(4)}))
        print(f"EVALUATE {subject}: AURC FBCSP-GMM={S.aurc(correct,score,key):.6f}, {time.time()-t0:.1f}s", flush=True)

    def values(name, field):
        return np.array([r[field] for s in SUBJECTS for r in per_subject if r["subject"] == s and r["score"] == name])
    comparisons = ST.holm([ST.paired(values("fbcsp_gmm", "aurc"), values(name, "aurc"),
                                      f"EXPLORATORY: FBCSP-GMM - {name} AURC") for name in ["ts_msp", "msp", "vit_gmm"]])
    summary = []
    for name in sc:
        r = dict(score=name, analysis="post-hoc exploratory only")
        for field in ["aurc", "e_aurc", "auroc", "q0.8_coverage", "q0.8_acc", "q0.8_kappa", "q0.8_risk", "mc0.8_risk"]:
            a = values(name, field); r[field+"_mean"] = float(a.mean()); r[field+"_sd"] = float(a.std(ddof=1))
        summary.append(r)
    write_csv(ROOT / "trials_exploratory.csv", trials)
    write_csv(ROOT / "subject_metrics_exploratory.csv", per_subject)
    write_csv(ROOT / "true_class_rejection_exploratory.csv", counts)
    write_csv(ROOT / "summary_exploratory.csv", summary)
    dump(ROOT / "stats_exploratory.json", dict(label="EXPLORATORY, after prior E results; not a new confirmatory endpoint", comparisons=comparisons))
    dump(ROOT / "risk_curves_exploratory.json", curves)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0.05, 1, 96)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    labels = dict(fbcsp_gmm="FBCSP-GMM (post-hoc)", vit_gmm="ViT-GMM (inherited)",
                  msp="MSP", ts_msp="Temperature-scaled MSP", oracle="Oracle", random_expected="Random (expected)")
    colors = dict(fbcsp_gmm="#8e44ad", vit_gmm="#c0392b", msp="#34495e", ts_msp="#2980b9", oracle="#27ae60", random_expected="#888888")
    for name in labels:
        yy = [np.interp(grid, np.arange(1, len(curves[s][name])+1)/len(curves[s][name]), curves[s][name]) for s in SUBJECTS]
        ax.plot(grid, np.mean(yy, axis=0), label=labels[name], color=colors[name], linestyle="--" if name in ["oracle", "random_expected"] else "-")
    ax.set(xlabel="Coverage", ylabel="Selective risk (subject mean)", title="Exploratory FBCSP-GMM on the unchanged Concat+LR decoder")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(ROOT / "risk_coverage_exploratory.png", dpi=200)
    fig.savefig(ROOT / "risk_coverage_exploratory.svg")
    plt.close(fig)

    snapshot = json.loads((ROOT / "inherited_snapshot.json").read_text())
    changed = [p for p, info in snapshot.items() if digest(REV / p) != info["sha256"]]
    assert not changed, f"Inherited analysis changed: {changed}"
    dump(ROOT / "EVALUATION_COMPLETE.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                                primary_stats_unchanged=digest(REV / "out/stats_clean.json"),
                                                inherited_file_changes=changed, status="Complete; exploratory only"))
    md = ["# FBCSP-GMM 探索性分析", "", "在此前E结果已知后追加；仅限补充材料，不替换主实验或摘要。",
          "固定融合类别不变，复用45个成员的FBCSP/MIBIF，只在原B_j拟合PCA8/单分量diag GMM。",
          "所有新阈值在C确定并在E应用；没有参数搜索或网络重训。", "",
          "| Score | AURC mean ± sample SD | Correctness AUROC | E coverage at C q=.8 |",
          "| --- | --- | --- | --- |"]
    for r in summary:
        md.append(f"| {r['score']} | {r['aurc_mean']:.4f} ± {r['aurc_sd']:.4f} | {r['auroc_mean']:.4f} | {r['q0.8_coverage_mean']:.4f} |")
    md += ["", "## Exploratory paired comparisons (Holm over three)", ""]
    for r in comparisons:
        md.append(f"- {r['comparison']}: Δ={r['mean']:.6f}, 95% CI={r['mean_ci']}, two-sided p={r['p']:.6f}, Holm p={r['p_holm']:.6f}, r={r['r_rb']:.4f}.")
    md += ["", "## Interpretation limits", "Changing the representation alone does not isolate a causal mechanism. Prior E exposure and inherited preprocessing/epoch-selection limitations remain. Report favorable and unfavorable comparisons alike; no further variants are selected.",
           "原主实验文件hash核对未变。"]
    (ROOT / "exploratory_results.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(dict(summary=summary, comparisons=comparisons), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
