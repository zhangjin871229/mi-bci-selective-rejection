"""ChannelViT (original architecture) and EEGNet-8,2 with simple CPU training loops."""
from __future__ import annotations

import common  # noqa: F401  (adds legacy package to sys.path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bci2a.augment import TFAugmentConfig, apply_tf_augment
from bci2a.config import TrainConfig
from bci2a.models import ChannelViT

torch.set_num_threads(8)


# ---------------------------------------------------------------- ChannelViT
def make_vit() -> ChannelViT:
    cfg = TrainConfig()
    cfg.freq_bins, cfg.time_frames = 33, 64
    return ChannelViT(cfg, n_channels=22)


class TFNorm:
    def __init__(self, tf_train: np.ndarray):
        self.mu = tf_train.mean(0).astype(np.float32)
        self.sd = (tf_train.std(0) + 1e-6).astype(np.float32)

    def __call__(self, tf: np.ndarray) -> np.ndarray:
        return ((tf - self.mu) / self.sd).astype(np.float32)


_AUG = TFAugmentConfig()


def _vit_batch(xn: np.ndarray, rng, augment: bool) -> torch.Tensor:
    if augment:
        xn = np.stack([apply_tf_augment(x, _AUG, rng) for x in xn])
    return torch.from_numpy(np.ascontiguousarray(xn))


# ---------------------------------------------------------------- EEGNet
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018)."""

    def __init__(self, n_ch=22, n_t=876, n_cls=4, F1=8, D=2, F2=16, k1=64, drop=0.5):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_feat = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_t))).numel()
        self.classifier = nn.Linear(n_feat, n_cls)

    def encode(self, x):
        return self.b2(self.b1(x.unsqueeze(1))).flatten(1)

    def forward(self, x):
        return self.classifier(self.encode(x))


class ChanNorm:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(axis=(0, 2), keepdims=True)[0]
        self.sd = X.std(axis=(0, 2), keepdims=True)[0] + 1e-6

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


# ---------------------------------------------------------------- generic loop
def train(kind: str, Xtr: np.ndarray, ytr: np.ndarray, epochs: int, seed: int,
          Xva: np.ndarray | None = None, yva: np.ndarray | None = None, log=None):
    """Train ViT ('vit', X = normalized TF maps) or EEGNet ('eegnet', X = normalized EEG).

    Labels are 0-3. Returns (model, history) where history[e] = (train_loss, val_nll, val_acc).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if kind == "vit":
        model = make_vit()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    else:
        model = EEGNet(n_t=Xtr.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = None
    yt = torch.from_numpy(ytr.astype(np.int64))
    hist = []
    n = len(ytr)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, 32):
            idx = perm[s:s + 32]
            if len(idx) < 2:
                continue
            xb = _vit_batch(Xtr[idx], rng, True) if kind == "vit" else torch.from_numpy(Xtr[idx])
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if sched is not None:
            sched.step()
        rec = [tot / n, np.nan, np.nan]
        if Xva is not None:
            p = predict_proba(model, Xva)
            rec[1] = float(-np.mean(np.log(p[np.arange(len(yva)), yva] + 1e-12)))
            rec[2] = float((p.argmax(1) == yva).mean())
        hist.append(rec)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            log(f"  {kind} ep {ep + 1}/{epochs} train {rec[0]:.4f} val_nll {rec[1]:.4f} val_acc {rec[2]:.3f}")
    return model, np.array(hist)


@torch.no_grad()
def _batched(fn, X, bs=64):
    return np.concatenate([fn(torch.from_numpy(np.ascontiguousarray(X[i:i + bs]))).numpy()
                           for i in range(0, len(X), bs)])


def predict_logits(model, X):
    model.eval()
    return _batched(model, X)


def predict_proba(model, X):
    z = predict_logits(model, X)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def embed(model, X):
    model.eval()
    return _batched(model.encode, X)


def _moving_average(v: np.ndarray, window: int = 5) -> np.ndarray:
    k = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, v[0]), v, np.full(window // 2, v[-1])])
    return np.convolve(padded, k, mode="valid")[: len(v)]


def select_epoch(val_nll: np.ndarray, window: int = 5) -> int:
    """Original rule (protocol v1): epoch minimising the moving-average validation NLL (earliest on ties)."""
    return int(np.argmin(_moving_average(val_nll, window))) + 1


def select_epoch_amended(history: np.ndarray, window: int = 5, min_epoch: int = 10) -> tuple[int, str]:
    """Amendment 1: minimum moving-average validation NLL over epochs >= min_epoch; if no such epoch
    improves on epoch 1, use the epoch maximising moving-average validation accuracy (earliest on ties)."""
    nll = _moving_average(history[:, 1], window)
    acc = _moving_average(history[:, 2], window)
    i = min_epoch - 1 + int(np.argmin(nll[min_epoch - 1:]))
    if nll[i] < nll[0]:
        return i + 1, "nll"
    return int(np.argmax(acc)) + 1, "acc_fallback"
