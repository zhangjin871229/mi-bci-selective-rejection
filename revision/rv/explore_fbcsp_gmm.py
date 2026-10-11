"""Exploratory analysis X1 (see ../exploratory_spec.md): aligned GMM gate fitted on FBCSP features.

Usage: python explore_fbcsp_gmm.py
"""
from __future__ import annotations

import json

import mne
import numpy as np
from sklearn.metrics import cohen_kappa_score

import common  # noqa: F401
import selective as S
import stats as ST
from common import MODELS, OUT, SUBJECTS, load_protocol, member_runs, member_seed
from evaluate import ensemble
from features import fbcsp_window, load_session
from heads import GMMHead
from tab import write_csv
from train_subject import _Unpickler

mne.set_log_level("ERROR")
XDIR = OUT / "exploratory"


def fbcsp_feats(bundle, X):
    return bundle.mib.transform(bundle.fb.transform(fbcsp_window(X)))


def member_gmm(subject, j, T, E, P):
    """Fit the protocol's conservative GMM head on B_j FBCSP features; return C and E posteriors."""
    with open(MODELS / subject / f"member{j}.pkl", "rb") as f:
        b = _Unpickler(f).load()
    r = member_runs(P, j)
    B = np.isin(T["run_id"], r["B"])
    C = T["run_id"] == P["split"]["C_run"]
    g = GMMHead(seed=member_seed(subject, j, P["seeds"]["base"])).fit(fbcsp_feats(b, T["X"][B]), T["y"][B] - 1)
    return g.predict_proba(fbcsp_feats(b, T["X"][C])), g.predict_proba(fbcsp_feats(b, E["X"]))


def main():
    P = load_protocol()
    q = P["thresholds"]["q_primary"]
    fitted = json.loads((OUT / "fitted_params.json").read_text())
    XDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SUBJECTS:
        T, E = load_session(s, "T"), load_session(s, "E")
        gC, gE = zip(*[member_gmm(s, j, T, E, P) for j in range(1, 6)])
        gC, gE = np.mean(gC, 0), np.mean(gE, 0)
        Cens, Eens = ensemble(s, OUT / "raw", "clean")
        Tc = fitted[s]["concat"]["T"]
        y = E["y"] - 1
        p = Eens["p_concat"]
        yh = p.argmax(1)
        correct = yh == y
        key = S.tiebreak(E["trial_id"])
        scE = S.scores(p, gE, Tc)
        scC = S.scores(Cens["p_concat"], gC, Tc)
        thr = S.threshold_from_C(scC["aligned_gmm"], q)
        acc = S.accepted_metrics(correct, y, yh, scE["aligned_gmm"] >= thr)
        vit = S.scores(p, Eens["g_vit"], Tc)["aligned_gmm"]
        rows.append({
            "subject": s,
            "aurc_fbcsp_gmm": S.aurc(correct, scE["aligned_gmm"], key),
            "aurc_vit_gmm": S.aurc(correct, vit, key),
            "aurc_msp": S.aurc(correct, scE["msp"], key),
            "aurc_ts_msp": S.aurc(correct, scE["ts_msp"], key),
            "aurc_oracle": S.aurc_oracle(correct),
            "auroc_fbcsp_gmm": S.correctness_auroc(correct, scE["aligned_gmm"]),
            "auroc_ts_msp": S.correctness_auroc(correct, scE["ts_msp"]),
            "fbcsp_gmm_forced_kappa": float(cohen_kappa_score(y, gE.argmax(1))),
            "thr_coverage": acc["coverage"], "thr_acc": acc["acc"], "thr_kappa": acc["kappa"],
        })
        print(s, {k: round(v, 4) for k, v in rows[-1].items() if k != "subject"}, flush=True)
    write_csv(rows, XDIR / "x1_subject.csv")
    col = lambda k: np.array([r[k] for r in rows])
    res = {
        "spec": "../../exploratory_spec.md",
        "family_X1": ST.holm([
            ST.paired(col("aurc_fbcsp_gmm"), col("aurc_msp"), "AURC fbcsp_gmm - msp"),
            ST.paired(col("aurc_fbcsp_gmm"), col("aurc_ts_msp"), "AURC fbcsp_gmm - ts_msp"),
        ]),
        "descriptive": ST.paired(col("aurc_fbcsp_gmm"), col("aurc_vit_gmm"), "AURC fbcsp_gmm - vit_gmm"),
        "means": {k: float(col(k).mean()) for k in rows[0] if k != "subject"},
        "sds": {k: float(col(k).std(ddof=1)) for k in rows[0] if k != "subject"},
    }
    (XDIR / "x1_stats.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["means"], indent=1))
    for r in res["family_X1"] + [res["descriptive"]]:
        print(r["comparison"], round(r["mean"], 4), [round(x, 4) for x in r["mean_ci"]],
              "p", round(r["p"], 4), "holm", r.get("p_holm"), "r", round(r["r_rb"], 3))


if __name__ == "__main__":
    main()
