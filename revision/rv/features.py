"""Model-specific transforms of cleaned epochs (microvolts, 0-4 s at 250 Hz)."""
from __future__ import annotations

import common  # noqa: F401  (adds legacy package to sys.path)

import numpy as np
from scipy import signal

from common import CACHE, SFREQ
from preprocess_bci2a import wavelet_denoise_1d


def load_session(subject: str, session: str) -> dict:
    d = np.load(CACHE / f"{subject}{session}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


_BP_8_30 = signal.butter(4, [8 / (SFREQ / 2), 30 / (SFREQ / 2)], btype="band")
_BP_4_38 = signal.butter(4, [4 / (SFREQ / 2), 38 / (SFREQ / 2)], btype="band")


def stft_maps(X: np.ndarray) -> np.ndarray:
    """(n, 22, 1001) uV -> (n, 22, 33, 64) log1p STFT magnitude in uV."""
    n, c, _ = X.shape
    out = np.empty((n, c, 33, 64), np.float32)
    b, a = _BP_8_30
    for i in range(n):
        for ch in range(c):
            x = wavelet_denoise_1d(X[i, ch])
            x = signal.filtfilt(b, a, x)
            _, _, Z = signal.stft(x, fs=SFREQ, window="hann", nperseg=64, noverlap=48, nfft=64,
                                  boundary="zeros", padded=True)
            out[i, ch] = np.log1p(np.abs(Z))
    return out


def eegnet_input(X: np.ndarray) -> np.ndarray:
    """0.5-4.0 s, 4-38 Hz band-pass -> (n, 22, 876) float32."""
    b, a = _BP_4_38
    Xf = signal.filtfilt(b, a, X, axis=-1)
    return Xf[:, :, 125:].astype(np.float32)


def fbcsp_window(X: np.ndarray) -> np.ndarray:
    """0.5-2.5 s window used by FBCSP -> (n, 22, 501) float64."""
    return X[:, :, 125:626].astype(np.float64)


def tf_cache(subject: str, session: str) -> np.ndarray:
    """Clean STFT maps, cached on disk."""
    p = CACHE / f"{subject}{session}_tf.npy"
    if p.exists():
        return np.load(p)
    tf = stft_maps(load_session(subject, session)["X"])
    np.save(p, tf)
    return tf
