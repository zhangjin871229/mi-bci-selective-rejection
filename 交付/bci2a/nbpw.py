"""Naive Bayes Parzen Window (NBPW) 多类分类器。"""
from __future__ import annotations

import numpy as np

def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)

class NBPWClassifier:
    """
    朴素贝叶斯 + 一维 Parzen 窗估计 p(x_j | w)。
    特征假定条件独立。
    """

    def __init__(self, bandwidth: float | None = None):
        self.bandwidth = bandwidth
        self.classes_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NBPWClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if y.min() >= 1:
            y = y - 1
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.priors_ = counts / counts.sum()
        return self

    def _bw(self, X: np.ndarray) -> float:
        if self.bandwidth is not None:
            return float(self.bandwidth)
        # Silverman-like per feature scale
        std = np.std(X, axis=0) + 1e-8
        return float(np.median(std) * (X.shape[0] ** (-1.0 / (X.shape[1] + 4))))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.X_train_ is not None and self.y_train_ is not None
        assert self.classes_ is not None and self.priors_ is not None
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        n_cls = len(self.classes_)
        h = self._bw(self.X_train_)
        log_p = np.zeros((n, n_cls), dtype=np.float64)
        for ci, c in enumerate(self.classes_):
            Xc = self.X_train_[self.y_train_ == c]
            if len(Xc) == 0:
                log_p[:, ci] = -np.inf
                continue
            log_like = np.zeros(n, dtype=np.float64)
            for j in range(X.shape[1]):
                dif = (X[:, [j]] - Xc[:, j]) / h
                kern = _gaussian_kernel(dif) / h
                p_j = np.mean(kern, axis=1) + 1e-12
                log_like += np.log(p_j)
            log_p[:, ci] = np.log(self.priors_[ci] + 1e-12) + log_like
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True) + 1e-12
        return p.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
