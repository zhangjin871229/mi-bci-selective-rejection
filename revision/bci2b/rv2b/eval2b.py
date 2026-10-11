"""One-shot E evaluation and the single pre-specified test of protocol_2b.json.

The only script that opens cache/<S>E_labels.npz. Refuses to run unless the protocol is frozen and unchanged.
Outputs (out/): trials_2b.csv, subject_metrics_2b.csv, primary_2b.json
Usage: python -I eval2b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2b as C2  # noqa: E402

import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

import selective as S  # noqa: E402  (2a module)
import stats as ST  # noqa: E402  (2a module)
from tab import write_csv  # noqa: E402

Q = 0.8


def wilcoxon_one_sided_lower(d) -> dict:
    """Exact signed-rank test of H1: median(d) < 0. p = P(W+ <= observed W+) under sign symmetry."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    r = ST._ranks(np.abs(d))
    wp = float(r[d > 0].sum())
    dist = np.array([np.dot(r, s) for s in itertools.product([0, 1], repeat=n)])
    return {"n_nonzero": n, "W_plus": wp, "p_one_sided": float(np.mean(dist <= wp + 1e-9)),
            "r_rb": float((wp - r[d < 0].sum()) / r.sum())}


def ensemble(subject):
    ms = [np.load(C2.OUT / "raw" / f"{subject}_member{j}.npz") for j in range(1, 6)]
    get = lambda k: np.mean([m[k] for m in ms], 0)
    for m in ms[1:]:
        assert np.array_equal(m["trial_id_C"], ms[0]["trial_id_C"])
    return {k: get(k) for k in ms[0].files if k.startswith(("p_", "g_"))}, ms[0]["trial_id_C"]


def main():
    P = C2.check_frozen()
    sha = C2.protocol_sha256()
    rows, trials = [], []
    for s in C2.SUBJECTS:
        o, tidC = ensemble(s)
        T = C2.load_npz(C2.CACHE / f"{s}T.npz")
        mC = T["block"] == C2.C_BLOCK
        assert np.array_equal(T["trial_id"][mC], tidC)
        yC = T["y"][mC] - 1
        L = C2.load_npz(C2.CACHE / f"{s}E_labels.npz")
        Ein = np.load(C2.CACHE / f"{s}E_inputs.npz")["trial_id"]
        assert np.array_equal(L["trial_id"], Ein)
        y = L["y"] - 1
        p = o["p_concat_E"]
        yh = p.argmax(1)
        correct = yh == y
        key = S.tiebreak(L["trial_id"])
        Tc, Tflag = S.fit_temperature(o["p_concat_C"], yC)
        idx = np.arange(len(y))
        sc = {
            "fbcsp_gmm": o["g_fbcsp_E"][idx, yh],
            "msp": p.max(1),
            "ts_msp": S.apply_temperature(p, Tc).max(1),
            "vit_gmm": o["g_vit_E"][idx, yh],
        }
        yhC = o["p_concat_C"].argmax(1)
        thr = S.threshold_from_C(o["g_fbcsp_C"][np.arange(len(yhC)), yhC], Q)
        thr_msp = S.threshold_from_C(o["p_concat_C"].max(1), Q)
        row = {"subject": s, "n_E": len(y), "acc": float(correct.mean()),
               "kappa": float(cohen_kappa_score(y, yh, labels=[0, 1])), "T": Tc, "T_flag": Tflag,
               "aurc_oracle": S.aurc_oracle(correct)}
        for k, v in sc.items():
            row[f"aurc_{k}"] = S.aurc(correct, v, key)
            row[f"auroc_{k}"] = S.correctness_auroc(correct, v)
        for k, (v, t) in {"fbcsp_gmm": (sc["fbcsp_gmm"], thr), "msp": (sc["msp"], thr_msp)}.items():
            a = S.accepted_metrics(correct, y, yh, v >= t)
            row[f"thr{Q}_{k}_coverage"], row[f"thr{Q}_{k}_acc"] = a["coverage"], a["acc"]
        rows.append(row)
        for i in range(len(y)):
            trials.append({"subject": s, "trial_id": L["trial_id"][i], "y": int(y[i]), "yhat": int(yh[i]),
                           "correct": bool(correct[i]), **{k: float(v[i]) for k, v in sc.items()}})
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "subject"}, flush=True)

    C2.OUT.mkdir(exist_ok=True)
    write_csv(trials, C2.OUT / "trials_2b.csv")
    write_csv(rows, C2.OUT / "subject_metrics_2b.csv")
    col = lambda k: np.array([r[k] for r in rows])
    d = col("aurc_fbcsp_gmm") - col("aurc_msp")
    test = wilcoxon_one_sided_lower(d)
    test.update(ST.bootstrap_ci(d))
    res = {
        "protocol_id": P["protocol_id"], "protocol_sha256": sha,
        "primary": {"comparison": "AURC(fbcsp_gmm) - AURC(msp), Concat+LR, one-sided exact Wilcoxon (H1: < 0)",
                    "per_subject_diff": d.tolist(), **test, "alpha": 0.05,
                    "validated": bool(test["p_one_sided"] < 0.05)},
        "descriptive_means": {k: float(col(k).mean()) for k in rows[0] if k not in ("subject", "T_flag")},
        "descriptive_sds": {k: float(col(k).std(ddof=1)) for k in rows[0] if k not in ("subject", "T_flag")},
    }
    (C2.OUT / "primary_2b.json").write_text(json.dumps(res, indent=1, default=float))
    pr = res["primary"]
    print(f"\nPRIMARY: mean d = {pr['mean']:+.4f}, 95% CI {[round(x, 4) for x in pr['mean_ci']]}, "
          f"W+ = {pr['W_plus']}, one-sided p = {pr['p_one_sided']:.4f}, r = {pr['r_rb']:+.2f}, "
          f"validated = {pr['validated']}")


if __name__ == "__main__":
    main()
