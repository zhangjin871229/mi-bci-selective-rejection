"""Ensemble, C-fitted temperature/thresholds, single E evaluation, statistics.

Refuses to run unless protocol.json status starts with "FROZEN".
Outputs (out/): trials_<condition>.csv, subject_metrics_<cond>.csv, class_rejection_<cond>.csv,
calibration_<cond>.json, stats_<cond>.json, fitted_params.json
Usage: python evaluate.py [--condition clean] [--raw-dir out/raw]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import common  # noqa: F401
import selective as S
import stats as ST
from tab import column, select, write_csv
from common import CLASSES, OUT, SUBJECTS, load_protocol, protocol_hash
from features import load_session

DECODERS = {"concat": ("p_concat", "g_vit"), "eegnet": ("p_eeg", "g_eeg")}
FORCED = ["p_concat", "p_fbcsp", "p_vit", "p_eeg", "g_vit", "g_eeg"]
SCORES = ["aligned_gmm", "max_gmm", "msp", "neg_entropy", "ts_msp", "ts_neg_entropy"]


def ensemble(subject, raw_dir, cond):
    ms = [np.load(raw_dir / f"{subject}_member{j}.npz", allow_pickle=True) for j in range(1, 6)]
    suffix = "E" if cond == "clean" else f"E_{cond}"
    C = {k: np.mean([m[f"{k}_C"] for m in ms], 0) for k in FORCED}
    E = {k: np.mean([m[f"{k}_{suffix}"] for m in ms], 0) for k in FORCED}
    return C, E


def fit_params(subject, C, yC, P):
    """Temperatures and thresholds from run C only (never from E)."""
    out = {}
    for dec, (pk, gk) in DECODERS.items():
        T, flag = S.fit_temperature(C[pk], yC)
        sc = S.scores(C[pk], C[gk], T)
        thr = {s: {str(q): S.threshold_from_C(sc[s], q) for q in [P["thresholds"]["q_primary"]] + P["thresholds"]["q_secondary"]}
               for s in SCORES}
        out[dec] = {"T": T, "T_flag": flag, "thresholds": thr, "n_C": int(len(yC))}
    return out


def evaluate(cond="clean", raw_dir=None, fitted=None):
    P = load_protocol()
    if not P["status"].startswith("FROZEN"):
        raise SystemExit("protocol.json is not frozen; refusing to read E labels")
    raw_dir = raw_dir or OUT / "raw"
    qs = [P["thresholds"]["q_primary"]] + P["thresholds"]["q_secondary"]
    trials, subj_rows, cls_rows, calib = [], [], [], {}
    fitted = fitted if fitted is not None else {}
    for s in SUBJECTS:
        Td = load_session(s, "T")
        yC = Td["y"][Td["run_id"] == P["split"]["C_run"]] - 1
        C, E = ensemble(s, raw_dir, cond)
        if s not in fitted:
            fitted[s] = fit_params(s, C, yC, P)
        Ed = load_session(s, "E")
        y = Ed["y"] - 1
        key = S.tiebreak(Ed["trial_id"])
        # forced-choice metrics for all decoders
        for pk in FORCED:
            yh = E[pk].argmax(1)
            subj_rows.append({"subject": s, "decoder": pk, "score": "forced", "q": 1.0,
                              **S.accepted_metrics((yh == y), y, yh, np.ones(len(y), bool))})
        for dec, (pk, gk) in DECODERS.items():
            fp = fitted[s][dec]
            p = E[pk]
            yh = p.argmax(1)
            correct = (yh == y)
            sc = S.scores(p, E[gk], fp["T"])
            rng = np.random.default_rng(int(s[1:]) + 77)
            sc_ref = {"random": rng.random(len(y)), "oracle": correct.astype(float) + 0.0}
            calib[f"{s}_{dec}_raw"] = S.calibration(p, y)
            calib[f"{s}_{dec}_ts"] = S.calibration(S.apply_temperature(p, fp["T"]), y)
            for name, v in {**sc, **sc_ref}.items():
                row = {"subject": s, "decoder": dec, "score": name,
                       "aurc": S.aurc(correct, v, key), "auroc": S.correctness_auroc(correct, v)}
                row["e_aurc"] = row["aurc"] - S.aurc_oracle(correct)
                for q in qs:
                    m = S.matched_coverage(correct, y, yh, v, key, q)
                    row.update({f"mc{q}_{k}": m[k] for k in ("acc", "kappa", "risk")})
                    if name in SCORES:
                        acc = v >= fp["thresholds"][name][str(q)]
                        a = S.accepted_metrics(correct, y, yh, acc)
                        row.update({f"thr{q}_{k}": a[k] for k in ("coverage", "n_accept", "acc", "kappa", "risk")})
                        if q == P["thresholds"]["q_primary"]:
                            for c in range(4):
                                mc = y == c
                                na = int((acc & mc).sum())
                                cls_rows.append({"subject": s, "decoder": dec, "score": name, "class": CLASSES[c],
                                                 "total": int(mc.sum()), "accepted": na, "rejected": int(mc.sum()) - na,
                                                 "errors_accepted": int((acc & mc & ~correct).sum())})
                subj_rows.append(row)
            for i in range(len(y)):
                trials.append({"subject": s, "trial_id": Ed["trial_id"][i], "decoder": dec, "y": int(y[i]),
                               "yhat": int(yh[i]), "correct": bool(correct[i]),
                               **{k: float(v[i]) for k, v in sc.items()}})
    tag = cond
    OUT.mkdir(exist_ok=True)
    write_csv(trials, OUT / f"trials_{tag}.csv")
    sm = subj_rows
    write_csv(sm, OUT / f"subject_metrics_{tag}.csv")
    write_csv(cls_rows, OUT / f"class_rejection_{tag}.csv")
    (OUT / f"calibration_{tag}.json").write_text(json.dumps(calib, indent=0, default=float))
    if cond == "clean":
        (OUT / "fitted_params.json").write_text(json.dumps({"protocol_sha": protocol_hash(), **fitted}, indent=1))
        (OUT / f"stats_{tag}.json").write_text(json.dumps(run_stats(sm, P), indent=1, default=float))
    return sm, fitted


def _col(sm, dec, score, col):
    return column(sm, SUBJECTS, col, decoder=dec, score=score)


def run_stats(sm, P) -> dict:
    out = {"protocol_sha": protocol_hash()}
    out["primary"] = ST.paired(_col(sm, "concat", "aligned_gmm", "aurc"), _col(sm, "concat", "msp", "aurc"),
                               "concat: AURC aligned_gmm - msp")
    out["family_A"] = ST.holm([ST.paired(_col(sm, "concat", "aligned_gmm", "aurc"), _col(sm, "concat", b, "aurc"),
                                         f"concat: AURC aligned_gmm - {b}")
                               for b in ["ts_msp", "neg_entropy", "ts_neg_entropy", "max_gmm"]])
    q = P["thresholds"]["q_primary"]
    full = {k: _col(sm, "p_concat", "forced", k) for k in ("acc", "kappa")}
    rnd = _col(sm, "concat", "random", f"mc{q}_acc")
    famB = []
    for sc in ["aligned_gmm", "ts_msp"]:
        famB.append(ST.paired(_col(sm, "concat", sc, f"thr{q}_acc"), full["acc"], f"{sc}: accepted acc - full acc"))
        famB.append(ST.paired(_col(sm, "concat", sc, f"thr{q}_kappa"), full["kappa"], f"{sc}: accepted kappa - full kappa"))
        famB.append(ST.paired(_col(sm, "concat", sc, f"mc{q}_acc"), rnd, f"{sc}: acc at matched coverage {q} - random rejection"))
    out["family_B"] = ST.holm(famB)
    out["family_C"] = ST.holm([ST.paired(_col(sm, "eegnet", "aligned_gmm", "aurc"), _col(sm, "eegnet", b, "aurc"),
                                         f"eegnet: AURC aligned_gmm - {b}") for b in ["msp", "ts_msp"]])
    forced = {d: _col(sm, d, "forced", "kappa") for d in FORCED}
    out["forced_kappa"] = {d: {"per_subject": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
                           for d, v in forced.items()}
    out["forced_pairs"] = ST.holm([ST.paired(forced["p_concat"], forced[b], f"kappa concat - {b}")
                                   for b in ["p_fbcsp", "p_vit", "p_eeg", "g_vit"]])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="clean")
    a = ap.parse_args()
    evaluate(a.condition)
    print("done", a.condition)
