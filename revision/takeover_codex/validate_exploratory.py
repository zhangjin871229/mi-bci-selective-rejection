"""Validate saved exploratory outputs without refitting or generating new E scores."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from verify_existing import paired, curve, key

HERE = Path(__file__).resolve().parent
ROOT = HERE / "fbcsp_gmm_exploratory"
REV = HERE.parent


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    assert (ROOT / "EVALUATION_COMPLETE.json").exists()
    trials = read(ROOT / "trials_exploratory.csv")
    metrics = read(ROOT / "subject_metrics_exploratory.csv")
    stats = json.loads((ROOT / "stats_exploratory.json").read_text())
    thresholds = json.loads((ROOT / "fitted_C_parameters.json").read_text())
    old_trials = read(REV / "out/trials_clean.csv")
    old = {(r["subject"], r["trial_id"]): r for r in old_trials if r["decoder"] == "concat"}
    lookup = {(r["subject"], r["score"]): r for r in metrics}
    differences = {k: [] for k in ["ts_msp", "msp", "vit_gmm"]}
    per_subject = []
    for s in [f"A0{i}" for i in range(1, 10)]:
        rows = [r for r in trials if r["subject"] == s]
        y = np.array([int(r["y"]) for r in rows]); yh = np.array([int(r["yhat"]) for r in rows])
        probs = np.array([[float(r[f"p_gmm_c{c}"]) for c in range(4)] for r in rows])
        score = np.array([float(r["fbcsp_gmm"]) for r in rows])
        assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)
        assert np.allclose(score, probs[np.arange(len(rows)), yh])
        assert len({r["trial_id"] for r in rows}) == len(rows)
        for r in rows:
            baseline = old[s, r["trial_id"]]
            assert int(r["y"]) == int(baseline["y"]) and int(r["yhat"]) == int(baseline["yhat"])
            assert r["correct"] == baseline["correct"]
            for k in ["msp", "ts_msp"]: assert np.isclose(float(r[k]), float(baseline[k]))
            assert np.isclose(float(r["vit_gmm"]), float(baseline["aligned_gmm"]))
        correct = y == yh
        au = curve(correct, score, key([r["trial_id"] for r in rows])).mean()
        assert np.isclose(au, float(lookup[s, "fbcsp_gmm"]["aurc"]))
        for q in [0.7, 0.8, 0.9]:
            acc = score >= thresholds[s]["thresholds"][str(q)]
            assert np.isclose(acc.mean(), float(lookup[s, "fbcsp_gmm"][f"q{q}_coverage"]))
            assert np.isclose(correct[acc].mean(), float(lookup[s, "fbcsp_gmm"][f"q{q}_acc"]))
        for other in differences:
            differences[other].append(au - float(lookup[s, other]["aurc"]))
        row = lookup[s, "fbcsp_gmm"]
        per_subject.append(dict(subject=s, aurc=float(au), q08_coverage=float(row["q0.8_coverage"]),
                                accepted_accuracy_gain=float(row["q0.8_acc"]) - correct.mean()))
    for i, other in enumerate(differences):
        fresh = paired(differences[other])
        stored = stats["comparisons"][i]
        for k in ["diffs", "mean", "mean_ci", "p", "r_rb"]: assert np.allclose(fresh[k], stored[k])
    ps = np.array([r["p"] for r in stats["comparisons"]]); order = np.argsort(ps); cumulative = 0
    for j, i in enumerate(order):
        cumulative = max(cumulative, min(1, (len(ps)-j)*ps[i]))
        assert np.isclose(cumulative, stats["comparisons"][i]["p_holm"])
    completion = json.loads((ROOT / "FIT_COMPLETE.json").read_text())
    for rel, sha in completion["artifacts"].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == sha
    report = dict(status="PASS", trials=len(trials), subjects=9, predictions_unchanged=True,
                  C_thresholds_and_saved_heads_unchanged=True, per_subject=per_subject,
                  note="Independent saved-score verification; no refitting or additional E inference.")
    (ROOT / "verification_exploratory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
