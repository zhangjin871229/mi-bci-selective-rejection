"""时频图 (C, F, T) 训练增强；仅用于 train loader。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class TFAugmentConfig:
    noise_std: float = 0.12
    time_mask_max: int = 10
    freq_mask_max: int = 6
    n_time_masks: int = 1
    n_freq_masks: int = 1
    channel_drop_p: float = 0.12
    time_shift_max: int = 6
    scale_std: float = 0.08

def _mask_band(x: np.ndarray, axis: int, width: int, rng: np.random.Generator) -> None:
    """将连续频段/时段置零（Z-score 后 0 ≈ 均值）。"""
    n = x.shape[axis]
    if width <= 0 or n <= 1:
        return
    width = min(width, n)
    start = int(rng.integers(0, n - width + 1))
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(start, start + width)
    x[tuple(sl)] = 0.0

def apply_tf_augment(x: np.ndarray, cfg: TFAugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """
    对单 trial 时频图增强。x: (C, F, T) float32，原地修改副本。
    """
    x = np.array(x, dtype=np.float32, copy=True)

    if cfg.time_shift_max > 0:
        shift = int(rng.integers(-cfg.time_shift_max, cfg.time_shift_max + 1))
        if shift:
            x = np.roll(x, shift, axis=2)

    for _ in range(cfg.n_freq_masks):
        w = int(rng.integers(1, cfg.freq_mask_max + 1))
        _mask_band(x, axis=1, width=w, rng=rng)

    for _ in range(cfg.n_time_masks):
        w = int(rng.integers(1, cfg.time_mask_max + 1))
        _mask_band(x, axis=2, width=w, rng=rng)

    if cfg.channel_drop_p > 0:
        n_ch = x.shape[0]
        drop = rng.random(n_ch) < cfg.channel_drop_p
        if drop.any() and not drop.all():
            x[drop] = 0.0

    if cfg.noise_std > 0:
        x += rng.normal(0.0, cfg.noise_std, size=x.shape).astype(np.float32)

    if cfg.scale_std > 0:
        scale = float(rng.normal(1.0, cfg.scale_std))
        x *= max(scale, 0.5)

    return x
