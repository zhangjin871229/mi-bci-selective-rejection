"""借鉴 FBCSP 思路的滤波器组与幅值变换（仍输出时频图供 ViT 使用）。"""
from __future__ import annotations

import numpy as np

# 常用 μ/β 子带（Hz），与 FBCSP 文献接近
DEFAULT_FILTER_BANK: list[tuple[float, float]] = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 24.0),
    (24.0, 30.0),
]

def bandpass_1d(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=np.float64)
    nyq = sfreq / 2.0
    lo, hi = l_freq / nyq, h_freq / nyq
    if lo <= 0:
        lo = 1e-4
    if hi >= 1:
        hi = 0.999
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x).astype(np.float32)

def post_stft_magnitude(mag: np.ndarray, *, log1p: bool) -> np.ndarray:
    m = mag.astype(np.float32)
    if log1p:
        m = np.log1p(np.maximum(m, 0.0))
    return m
