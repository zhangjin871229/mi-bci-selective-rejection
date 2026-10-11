"""Mutual Information-based Best Individual Feature (MIBIF)。"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

class MIBIFSelector:
    """按各类别 MI 排序，取前 k 个特征（labels 为 0–3 或 1–4 均可）。"""

    def __init__(self, k: int = 8, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.selected_indices_: np.ndarray | None = None
        self.mi_scores_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MIBIFSelector":
        y = np.asarray(y, dtype=np.int64)
        if y.min() >= 1:
            y = y - 1
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        k = min(self.k, X.shape[1])
        idx = np.argsort(mi)[::-1][:k]
        self.selected_indices_ = np.sort(idx)
        self.mi_scores_ = mi
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

def pick_mibif_k(F_tr: np.ndarray, y_tr: np.ndarray, candidates: list[int], seed: int) -> int:
    """在 T 上 5-fold 选 MIBIF 的 k（FBCSP 特征已固定）。"""
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import StratifiedKFold

    from .nbpw import NBPWClassifier

    y = np.asarray(y_tr, dtype=np.int64)
    y_model = y - 1 if y.min() >= 1 else y
    if len(np.unique(y_model)) < 4 or len(y) < 40:
        return candidates[0]
    best_k, best_score = candidates[0], -1.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k in candidates:
        scores = []
        for tr, va in skf.split(F_tr, y_model):
            sel = MIBIFSelector(k=k, random_state=seed).fit(F_tr[tr], y_model[tr])
            clf = NBPWClassifier().fit(sel.transform(F_tr[tr]), y_model[tr])
            pred = clf.predict(sel.transform(F_tr[va]))
            scores.append(cohen_kappa_score(y_model[va], pred))
        mean_k = float(np.mean(scores))
        if mean_k > best_score:
            best_score, best_k = mean_k, k
    return best_k
