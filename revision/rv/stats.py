"""Paired subject-level statistics: exact Wilcoxon, rank-biserial r, bootstrap CI, Holm."""
from __future__ import annotations

import itertools

import numpy as np


def _ranks(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def wilcoxon_exact(d) -> dict:
    """Two-sided exact signed-rank test by sign enumeration; zero differences dropped; ties get mean ranks."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return {"n": 0, "W_plus": np.nan, "p": 1.0, "r_rb": 0.0}
    r = _ranks(np.abs(d))
    wp = r[d > 0].sum()
    wm = r[d < 0].sum()
    tot = r.sum()
    dist = np.array([np.dot(r, s) for s in itertools.product([0, 1], repeat=n)])
    m = min(wp, tot - wp)
    p = np.mean((dist <= m + 1e-9) | (dist >= tot - m - 1e-9))
    return {"n": n, "W_plus": float(wp), "p": float(min(p, 1.0)), "r_rb": float((wp - wm) / tot)}


def bootstrap_ci(d, n_boot=10000, seed=2026) -> dict:
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means, meds = d[idx].mean(1), np.median(d[idx], 1)
    return {"mean": float(d.mean()), "mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "median": float(np.median(d)), "median_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "sd": float(d.std(ddof=1))}


def paired(a, b, label="") -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    out = {"comparison": label, "diffs": d.tolist()}
    out.update(wilcoxon_exact(d))
    out.update(bootstrap_ci(d))
    return out


def holm(results: list[dict]) -> list[dict]:
    ps = np.array([r["p"] for r in results])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    for r, a in zip(results, adj):
        r["p_holm"] = float(a)
    return results
