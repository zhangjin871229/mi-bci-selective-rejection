"""Build 2b epoch caches from the BNCI 004-2014 MAT files (protocol_2b.json, data + split).

Outputs per subject (cache/):
  B0xT.npz         X (n, 3, 1001) uV, y 1-2, block 1-6, session, trial_id   (training data, labels allowed)
  B0xE_inputs.npz  X (n, 3, 1001) uV, trial_id                              (no labels)
  B0xE_labels.npz  y, trial_id                                               (read only by eval2b.py)
Cleaning per session: whole-session EOG OLS regression (intercept + 3 EOG), 50 Hz notch. No CAR.
Usage: python -I prep2b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2b as C2  # noqa: E402  (sets sys.path)

import json

import mne
import numpy as np
import scipy.io as sio

from common import file_md5

mne.set_log_level("ERROR")
CUE = 750          # cue = trial marker + 3 s
N_T = 1001         # 0-4 s after cue


def clean_session(r) -> np.ndarray:
    """(n_samples, 6) -> cleaned EEG (3, n_samples) in uV."""
    raw = np.asarray(r.X, np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("non-finite samples in session")
    eeg, eog = raw[:, :3], raw[:, 3:]
    D = np.concatenate([np.ones((len(eog), 1)), eog], 1)
    beta = np.linalg.lstsq(D, eeg, rcond=None)[0]
    clean = (eeg - D @ beta).T
    return mne.filter.notch_filter(clean, C2.SFREQ, 50.0)


def epochs(r, clean):
    t = np.asarray(r.trial, np.int64) - 1          # MATLAB 1-based sample index
    X = np.stack([clean[:, s + CUE:s + CUE + N_T] for s in t]).astype(np.float32)
    art = np.atleast_1d(np.asarray(r.artifacts)).astype(bool)
    y = np.atleast_1d(np.asarray(r.y)).astype(np.int64)
    assert len(art) == len(t) == len(y) and X.shape[2] == N_T
    return X, y, art


def build(subject: str) -> dict:
    info = {"subject": subject}
    # ---------------- training sessions 01T, 02T, 03T
    fT = C2.DATA / f"{subject}T.mat"
    sess = sio.loadmat(fT, squeeze_me=True, struct_as_record=False)["data"]
    assert len(sess) == 3, len(sess)
    Xs, ys, arts, ses_id, idx = [], [], [], [], []
    for k, r in enumerate(sess, 1):
        X, y, art = epochs(r, clean_session(r))
        Xs.append(X), ys.append(y), arts.append(art)
        ses_id += [k] * len(y)
        idx += list(range(len(y)))
    X, y, art = np.concatenate(Xs), np.concatenate(ys), np.concatenate(arts)
    ses_id, idx = np.array(ses_id), np.array(idx)
    n = [len(v) for v in ys]
    assert n[2] == 160, n
    # deviation 1 (deviations.md): D = all cue trials before the last 40 of 03T, cut into 5 equal contiguous blocks
    n_D = sum(n) - 40
    block = np.concatenate([np.repeat([1, 2, 3, 4, 5], [len(b) for b in np.array_split(np.arange(n_D), 5)]),
                            np.full(40, C2.C_BLOCK)])
    info["T_sessions"] = n
    keep = ~art
    tid = np.array([f"{subject}-0{s}T-t{i:03d}" for s, i in zip(ses_id, idx)])
    np.savez_compressed(C2.CACHE / f"{subject}T.npz", X=X[keep], y=y[keep], block=block[keep],
                        session=ses_id[keep], trial_id=tid[keep])
    info["T_md5"] = file_md5(fT)
    info["T_per_block"] = {int(b): int((block[keep] == b).sum()) for b in range(1, 7)}
    info["T_per_class"] = {int(c): int((y[keep] == c).sum()) for c in (1, 2)}
    info["T_artifacts"] = int(art.sum())
    info["eeg_std_uv"] = float(X[keep].std())

    # ---------------- evaluation sessions 04E, 05E (labels written to a separate file)
    fE = C2.DATA / f"{subject}E.mat"
    sess = sio.loadmat(fE, squeeze_me=True, struct_as_record=False)["data"]
    assert len(sess) == 2, len(sess)
    Xs, ys, tids = [], [], []
    n_art = 0
    for k, r in zip((4, 5), sess):
        Xe, ye, art = epochs(r, clean_session(r))
        Xs.append(Xe[~art]), ys.append(ye[~art])
        tids += [f"{subject}-0{k}E-t{i:03d}" for i in np.flatnonzero(~art)]
        n_art += int(art.sum())
    tids = np.array(tids)
    np.savez_compressed(C2.CACHE / f"{subject}E_inputs.npz", X=np.concatenate(Xs), trial_id=tids)
    np.savez_compressed(C2.CACHE / f"{subject}E_labels.npz", y=np.concatenate(ys), trial_id=tids)
    info["E_md5"] = file_md5(fE)
    info["E_n"] = int(len(tids))
    info["E_artifacts"] = n_art
    return info


if __name__ == "__main__":
    C2.check_frozen()
    C2.CACHE.mkdir(parents=True, exist_ok=True)
    subjects = sys.argv[1:] or C2.SUBJECTS
    manifest = []
    for s in subjects:
        info = build(s)
        print(info, flush=True)
        manifest.append(info)
    if subjects == C2.SUBJECTS:
        (C2.CACHE / "cache_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
