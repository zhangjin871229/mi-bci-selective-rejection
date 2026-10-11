"""Scores, temperature scaling, thresholds, risk-coverage metrics and calibration (pure functions)."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score

EPS = 1e-12


# ------------------------------------------------------------------ temperature
def fit_temperature(p: np.ndarray, y: np.ndarray, lo=0.25, hi=4.0) -> tuple[float, str]:
    logp = np.log(p + EPS)

    def nll(T):
        z = logp / T
        z = z - z.max(1, keepdims=True)
        lz = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lz[np.arange(len(y)), y].mean()
    try:
        r = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        T = float(r.x)
        flag = "ok" if r.success else "fail"
        if not r.success:
            T = 1.0
        elif min(T - lo, hi - T) < 1e-3:
            flag = "boundary"
        return T, flag
    except Exception:
        return 1.0, "fail"


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(p + EPS) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ scores
def scores(p_dec: np.ndarray, p_gmm: np.ndarray, T: float) -> dict:
    yhat = p_dec.argmax(1)
    pt = apply_temperature(p_dec, T)
    s = {
        "aligned_gmm": p_gmm[np.arange(len(yhat)), yhat],
        "max_gmm": p_gmm.max(1),
        "msp": p_dec.max(1),
        "neg_entropy": (p_dec * np.log(p_dec + EPS)).sum(1),
        "ts_msp": pt.max(1),
        "ts_neg_entropy": (pt * np.log(pt + EPS)).sum(1),
    }
    # identity checks (protocol 7.2)
    same = p_gmm.argmax(1) == yhat
    assert np.allclose(s["aligned_gmm"][same], s["max_gmm"][same])
    assert np.all(s["aligned_gmm"] <= s["max_gmm"] + 1e-12)
    assert np.array_equal(pt.argmax(1), yhat)
    return s


def tiebreak(trial_ids) -> np.ndarray:
    """Fixed, label-free tie-breaking key in [0, 1)."""
    return np.array([int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16) / 2**32 for t in trial_ids])


def order_desc(score: np.ndarray, key: np.ndarray) -> np.ndarray:
    return np.lexsort((key, -score))


# ------------------------------------------------------------------ thresholds
def threshold_from_C(score_C: np.ndarray, q: float) -> float:
    s = np.sort(score_C)[::-1]
    k = int(np.ceil(q * len(s)))
    return float(s[max(k, 1) - 1])


# ------------------------------------------------------------------ risk-coverage
def risk_curve(correct: np.ndarray, score: np.ndarray, key: np.ndarray) -> np.ndarray:
    o = order_desc(score, key)
    err = 1 - correct[o].astype(float)
    return np.cumsum(err) / np.arange(1, len(err) + 1)


def aurc(correct, score, key) -> float:
    return float(risk_curve(correct, score, key).mean())


def aurc_oracle(correct) -> float:
    c = np.sort(correct.astype(float))[::-1]
    err = 1 - c
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


def matched_coverage(correct, y, yhat, score, key, q) -> dict:
    o = order_desc(score, key)
    k = int(np.ceil(q * len(o)))
    acc_idx = o[:k]
    return accepted_metrics(correct, y, yhat, np.isin(np.arange(len(y)), acc_idx))


def accepted_metrics(correct, y, yhat, accept) -> dict:
    n_acc = int(accept.sum())
    out = {"coverage": n_acc / len(y), "n_accept": n_acc, "n_total": len(y)}
    if n_acc == 0:
        out.update(acc=np.nan, risk=np.nan, kappa=np.nan)
        return out
    out["acc"] = float(correct[accept].mean())
    out["risk"] = 1 - out["acc"]
    ya, pa = y[accept], yhat[accept]
    out["kappa"] = float(cohen_kappa_score(ya, pa, labels=[0, 1, 2, 3])) if len(np.unique(ya)) > 1 else np.nan
    return out


def correctness_auroc(correct, score) -> float:
    from sklearn.metrics import roc_auc_score
    if correct.all() or (~correct.astype(bool)).all():
        return np.nan
    return float(roc_auc_score(correct, score))


# ------------------------------------------------------------------ calibration
def calibration(p: np.ndarray, y: np.ndarray, n_bins=8) -> dict:
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows, ece, mce = [], 0.0, 0.0
    for i in range(n_bins):
        m = b == i
        n = int(m.sum())
        if n:
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += n / len(y) * gap
            mce = max(mce, gap)
            rows.append((edges[i], edges[i + 1], n, float(conf[m].mean()), float(correct[m].mean())))
        else:
            rows.append((edges[i], edges[i + 1], 0, np.nan, np.nan))
    onehot = np.eye(p.shape[1])[y]
    return {"ece": float(ece), "mce": float(mce),
            "nll": float(-np.log(p[np.arange(len(y)), y] + EPS).mean()),
            "brier": float(((p - onehot) ** 2).sum(1).mean()), "bins": rows}
