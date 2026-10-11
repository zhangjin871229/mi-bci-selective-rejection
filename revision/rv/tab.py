"""Minimal table helpers (no pandas in the frozen environment)."""
from __future__ import annotations

import csv

import numpy as np


def write_csv(rows: list[dict], path) -> None:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def select(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def column(rows, subjects, col, **kw):
    """Values of `col` for each subject in order (exactly one matching row per subject)."""
    out = []
    for s in subjects:
        m = select(rows, subject=s, **kw)
        assert len(m) == 1, (s, kw, len(m))
        out.append(float(m[0][col]))
    return np.array(out)
