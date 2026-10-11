"""Filter Bank CSP (Ang et al.) — OVR 扩展，m=2 对 → 4 维 log-var 特征/子带/二类。"""
from __future__ import annotations

import numpy as np
from mne.decoding import CSP

# 冠军论文：9 个子带，各 4 Hz，覆盖 4–40 Hz
CHAMPION_FILTER_BANK: list[tuple[float, float]] = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
]

def bandpass_epochs(X: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    nyq = sfreq / 2.0
    lo, hi = max(l_freq / nyq, 1e-4), min(h_freq / nyq, 0.999)
    b, a = butter(4, [lo, hi], btype="band")
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            out[i, c] = filtfilt(b, a, X[i, c])
    return out

def _binary_labels(y: np.ndarray, positive_class: int) -> np.ndarray:
    """官方 1–4 → 0/1，positive_class 为 OVR 正类。"""
    return (y == positive_class).astype(np.int64)

class FBCSPFeatureExtractor:
    """
    OVR-FBCSP：每类 c、每子带 b 训练一个二分类 CSP，提取 4 维 log-var。
    最终特征维 = n_bands * n_classes * 4 = 9*4*4 = 144。
    """

    def __init__(
        self,
        sfreq: float = 250.0,
        filter_bank: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        reg: str | None = "oas",
    ):
        self.sfreq = sfreq
        self.filter_bank = filter_bank or CHAMPION_FILTER_BANK
        self.n_components = n_components
        self.reg = reg
        self.csps: dict[tuple[int, int], CSP] = {}

    @property
    def n_features(self) -> int:
        return len(self.filter_bank) * 4 * self.n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPFeatureExtractor":
        y = np.asarray(y, dtype=np.int64)
        self.csps = {}
        for bi, band in enumerate(self.filter_bank):
            Xb = bandpass_epochs(X, self.sfreq, band[0], band[1])
            for cls in (1, 2, 3, 4):
                y_bin = _binary_labels(y, cls)
                if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
                    continue
                csp = CSP(
                    n_components=self.n_components,
                    reg=self.reg,
                    log=True,
                    norm_trace=False,
                )
                csp.fit(Xb, y_bin)
                self.csps[(bi, cls)] = csp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for bi, band in enumerate(self.filter_bank):
            Xb = bandpass_epochs(X, self.sfreq, band[0], band[1])
            for cls in (1, 2, 3, 4):
                csp = self.csps.get((bi, cls))
                if csp is None:
                    feats.append(np.zeros((X.shape[0], self.n_components), dtype=np.float64))
                else:
                    feats.append(csp.transform(Xb))
        return np.concatenate(feats, axis=1)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
