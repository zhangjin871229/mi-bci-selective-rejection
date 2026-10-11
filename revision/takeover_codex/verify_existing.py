"""Read-only verification of inherited revision outputs; write only into takeover_codex.

Run using the existing delivery virtual environment. Does not retrain or overwrite
the inherited protocol, models, raw predictions, or report tables.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REV = HERE.parent
SUBJECTS = [f"A0{i}" for i in range(1, 10)]
SCORES = ["aligned_gmm", "max_gmm", "msp", "neg_entropy", "ts_msp", "ts_neg_entropy"]
CONDITIONS = ["clean", "drop1", "drop3", "drop5", "eog1", "eog2", "eog4"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(ids):
    return np.array([int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16) / 2**32 for t in ids])


def curve(c, s, k):
    idx = np.lexsort((k, -s))
    return np.cumsum(1 - c[idx].astype(float)) / np.arange(1, len(c) + 1)


def exact(d):
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    if not len(d):
        return dict(p=1.0, r_rb=0.0, n=0)
    ranks = rankdata(np.abs(d))
    wp, total = ranks[d > 0].sum(), ranks.sum()
    dist = np.array([np.dot(ranks, b) for b in itertools.product([0, 1], repeat=len(d))])
    minimum = min(wp, total - wp)
    p = np.mean((dist <= minimum + 1e-9) | (dist >= total - minimum - 1e-9))
    return dict(p=float(p), r_rb=float((2 * wp - total) / total), n=len(d))


def paired(d):
    d = np.asarray(d, float)
    draws = np.random.default_rng(2026).integers(0, len(d), (10000, len(d)))
    return dict(diffs=d.tolist(), mean=float(d.mean()),
                mean_ci=np.percentile(d[draws].mean(1), [2.5, 97.5]).tolist(), **exact(d))


def main():
    out = HERE / "audit"
    out.mkdir(exist_ok=True)
    protocol = json.loads((REV / "protocol.json").read_text(encoding="utf-8"))
    fitted = json.loads((REV / "out/fitted_params.json").read_text())
    stored_stats = json.loads((REV / "out/stats_clean.json").read_text())
    digest = sha(REV / "protocol.json")
    checks, failures, verified, deltas = 0, [], [], []
    comparison_deltas = {s: [] for s in SCORES}
    forced_kappas = {s: [] for s in ["p_concat", "p_fbcsp", "p_vit", "p_eeg", "g_vit", "g_eeg"]}

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not bool(condition):
            failures.append(label)

    def close(a, b, label):
        check(np.allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True), label)

    check(fitted["protocol_sha"] == digest[:12], "fitted protocol hash")
    check(stored_stats["protocol_sha"] == digest[:12], "stats protocol hash")
    for condition in CONDITIONS:
        rows = csv_rows(REV / f"out/subject_metrics_{condition}.csv")
        metrics = {(r["subject"], r["decoder"], r["score"]): r for r in rows}
        trial_rows = csv_rows(REV / f"out/trials_{condition}.csv")
        trial_lookup = {}
        for row in trial_rows:
            trial_lookup.setdefault((row["subject"], row["decoder"]), []).append(row)
        class_rows = csv_rows(REV / f"out/class_rejection_{condition}.csv")
        classes = {(r["subject"], r["decoder"], r["score"], r["class"]): r for r in class_rows}
        for subject in SUBJECTS:
            with np.load(REV / f"cache/{subject}T.npz", allow_pickle=True) as z:
                run, yt, ids_t = z["run_id"], z["y"] - 1, z["trial_id"]
            with np.load(REV / f"cache/{subject}E.npz", allow_pickle=True) as z:
                y, ids = z["y"] - 1, z["trial_id"]
            nc = int((run == protocol["split"]["C_run"]).sum())
            suffix = "E" if condition == "clean" else f"E_{condition}"
            outputs, c_outputs = {}, {}
            for field in forced_kappas:
                values, cvalues = [], []
                for member in range(1, 6):
                    with np.load(REV / f"out/raw/{subject}_member{member}.npz", allow_pickle=True) as z:
                        values.append(z[f"{field}_{suffix}"])
                        cvalues.append(z[f"{field}_C"])
                        if condition == "clean" and field == "p_concat":
                            check(np.array_equal(z["trial_id_C"], ids_t[run == 6]), f"{subject}/{member} C alignment")
                outputs[field] = np.mean(values, axis=0)
                c_outputs[field] = np.mean(cvalues, axis=0)
                p = outputs[field]
                check(p.shape == (len(y), 4), f"{condition}/{subject}/{field} shape")
                check(np.isfinite(p).all() and (p >= 0).all(), f"{condition}/{subject}/{field} probabilities")
                # Native neural-network probabilities use float32; validate with
                # the precision of the stored values, not a float64-only tolerance.
                prob_tol = max(1e-10, 8 * np.finfo(p.dtype).eps)
                check(np.allclose(p.sum(1), 1, rtol=0, atol=prob_tol),
                      f"{condition}/{subject}/{field} probability sum")
                r = metrics[subject, field, "forced"]
                yh = p.argmax(1)
                close(float(r["acc"]), (yh == y).mean(), f"{condition}/{subject}/{field} accuracy")
                kap = cohen_kappa_score(y, yh, labels=[0, 1, 2, 3])
                close(float(r["kappa"]), kap, f"{condition}/{subject}/{field} kappa")
                if condition == "clean":
                    forced_kappas[field].append(kap)

            for dec, pk, gk in [("concat", "p_concat", "g_vit"), ("eegnet", "p_eeg", "g_eeg")]:
                p, g = outputs[pk], outputs[gk]
                yh = p.argmax(1)
                c = yh == y
                t = fitted[subject][dec]["T"]
                close(fitted[subject][dec]["n_C"], nc, f"{subject}/{dec} C sample count")
                logp = np.log(p + 1e-12) / t
                pt = np.exp(logp - logp.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
                sc = dict(aligned_gmm=g[np.arange(len(y)), yh], max_gmm=g.max(1), msp=p.max(1),
                          neg_entropy=(p * np.log(p + 1e-12)).sum(1), ts_msp=pt.max(1),
                          ts_neg_entropy=(pt * np.log(pt + 1e-12)).sum(1))
                ct = c_outputs[pk]; cg = c_outputs[gk]; cyh = ct.argmax(1)
                cpt = np.exp(np.log(ct + 1e-12) / t - (np.log(ct + 1e-12) / t).max(1, keepdims=True))
                cpt /= cpt.sum(1, keepdims=True)
                csc = dict(aligned_gmm=cg[np.arange(nc), cyh], max_gmm=cg.max(1), msp=ct.max(1),
                           neg_entropy=(ct * np.log(ct + 1e-12)).sum(1), ts_msp=cpt.max(1),
                           ts_neg_entropy=(cpt * np.log(cpt + 1e-12)).sum(1))
                check(np.array_equal(yh, pt.argmax(1)), f"{condition}/{subject}/{dec} TS prediction unchanged")
                check((sc["aligned_gmm"] <= sc["max_gmm"] + 1e-12).all(), f"{condition}/{subject}/{dec} alignment")
                tr = trial_lookup[subject, dec]
                check(len(tr) == len(y), f"{condition}/{subject}/{dec} csv trial count")
                check([r["trial_id"] for r in tr] == ids.tolist(), f"{condition}/{subject}/{dec} csv trial order")
                for name, v in sc.items():
                    close(v, [float(r[name]) for r in tr], f"{condition}/{subject}/{dec}/{name} csv scores")
                    m = metrics[subject, dec, name]
                    a = float(curve(c, v, key(ids)).mean())
                    close(a, float(m["aurc"]), f"{condition}/{subject}/{dec}/{name} AURC")
                    au = roc_auc_score(c, v) if c.any() and not c.all() else np.nan
                    close(au, float(m["auroc"]), f"{condition}/{subject}/{dec}/{name} AUROC")
                    if condition == "clean":
                        verified.append(dict(subject=subject, decoder=dec, score=name, aurc=a, auroc=float(au)))
                        if dec == "concat":
                            comparison_deltas[name].append(a)
                    for q in [0.7, 0.8, 0.9]:
                        threshold = np.sort(csc[name])[::-1][int(np.ceil(q * nc)) - 1]
                        close(threshold, fitted[subject][dec]["thresholds"][name][str(q)], f"{subject}/{dec}/{name}/{q} C threshold")
                        accept = v >= threshold
                        close(accept.mean(), float(m[f"thr{q}_coverage"]), f"{condition}/{subject}/{dec}/{name}/{q} coverage")
                        close(accept.sum(), float(m[f"thr{q}_n_accept"]), f"{condition}/{subject}/{dec}/{name}/{q} accept count")
                        acc = c[accept].mean() if accept.any() else np.nan
                        close(acc, float(m[f"thr{q}_acc"]), f"{condition}/{subject}/{dec}/{name}/{q} accepted accuracy")
                        if q == 0.8:
                            for j, klass in enumerate(["left_hand", "right_hand", "feet", "tongue"]):
                                r = classes[subject, dec, name, klass]
                                close(int(r["total"]), (y == j).sum(), f"{condition}/{subject}/{dec}/{name}/{klass} total")
                                close(int(r["accepted"]), (accept & (y == j)).sum(), f"{condition}/{subject}/{dec}/{name}/{klass} accepted")
                                close(int(r["rejected"]) + int(r["accepted"]), int(r["total"]), f"{condition}/{subject}/{dec}/{name}/{klass} balance")
                if condition == "clean" and dec == "concat":
                    tsrow = metrics[subject, dec, "ts_msp"]
                    deltas.append(float(tsrow["thr0.8_acc"]) - c.mean())
        print(f"Verified {condition}: cumulative checks {checks}, failures {len(failures)}", flush=True)

    result = paired(np.array(comparison_deltas["aligned_gmm"]) - comparison_deltas["msp"])
    for field in ["mean", "mean_ci", "p", "r_rb", "diffs"]:
        close(result[field], stored_stats["primary"][field], f"primary {field}")
    for field, values in forced_kappas.items():
        close(np.mean(values), stored_stats["forced_kappa"][field]["mean"], f"forced summary {field}")
        close(np.std(values, ddof=1), stored_stats["forced_kappa"][field]["sd"], f"forced SD {field}")

    paths = [REV / "protocol.json", REV / "protocol.sha256", *sorted((REV / "rv").glob("*.py")),
             *sorted((REV / "out").glob("*.csv")), *sorted((REV / "out").glob("*.json"))]
    snapshot = {str(p.relative_to(REV)): dict(sha256=sha(p), size=p.stat().st_size) for p in paths}
    (out / "input_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    gaps = [
        "Epoch selection is shared from member 1 (runs 3/4 fit, run 5 validation). B_j for members 2/3/4/5 overlaps these runs. Head trials were not directly used to train that member's representation, but some head labels indirectly influence the shared epoch. Do not claim complete absence from all model selection.",
        "Whole-session unsupervised EOG OLS is fitted independently on T and E signals. This is offline/transductive preprocessing with E-signal access, not strictly train-only learned preprocessing or prospective online operation. It does not by itself establish E-label leakage.",
        "Perturbations were applied to cleaned epochs before model-specific transforms, not before all preprocessing. EOG segments and propagation coefficients came from E eye-movement calibration, whereas the plan described T-eye calibration. Document this protocol departure; do not silently relabel it.",
        "Main max_gmm outputs use the new ensemble. They do not reproduce the old model's tau=0.80 historical result.",
        "Current random reference is one seeded random permutation. Matched-random gains are comparisons to that realization, not the analytic expectation; the expectation equals full accuracy.",
        "Previously observed E results cannot become an unseen holdout by switching assistants. New FBCSP-GMM analysis must remain post-hoc/exploratory with all results reported.",
        "Non-significance on EEGNet or versus random does not demonstrate equality; representation quality as a causal explanation remains a hypothesis, not an established mechanism.",
    ]
    payload = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest,
                   checks=checks, failures=failures, primary_recomputed=result,
                   ts_msp_accepted_accuracy_gain=paired(deltas), findings=gaps,
                   scope="Numerical and code-path audit of inherited outputs; not an end-to-end retraining or proof of historical freeze timing.")
    (out / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "verified_clean_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(verified[0])); w.writeheader(); w.writerows(verified)
    report = ["# 接管核查报告", "", "仅读取原协议、模型输出与结果，未重训或覆盖原文件。",
              f"检查数：{checks}；数值失败：{len(failures)}。",
              f"主差值：{result['mean']:.9f}，95% CI {result['mean_ci']}，双侧p={result['p']:.8f}，r={result['r_rb']:.2f}。",
              "", "## 方法与表述待核对", ""]
    report.extend(f"- {x}" for x in gaps)
    report += ["", "## 核查边界", payload["scope"]]
    (out / "takeover_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(dict(checks=checks, failures=failures, primary=result), indent=2), flush=True)
    if failures:
        raise SystemExit("Verification failed; preserve the inherited files and investigate.")


if __name__ == "__main__":
    main()
