"""Raw-EEG perturbations of the E session, passed through frozen members (no refitting).

Writes the perturbed member outputs into out/raw/<S>_member<j>.npz as extra keys
<output>_E_<condition>, e.g. p_concat_E_drop3, g_vit_E_eog2.
Usage: python perturb.py A01
"""
from __future__ import annotations

import sys
import time

import numpy as np

import common  # noqa: F401
from common import OUT, load_protocol
from features import load_session, stft_maps
from train_subject import load_bundle


def channel_dropout(X, k, perm):
    Xp = X.copy()
    bad, good = perm[:k], perm[k:]
    Xp[:, bad, :] = X[:, good, :].mean(1, keepdims=True)
    return Xp


def eog_mixing(X, calib_eog, beta, gain, rng):
    """Add EOG from the eye-movement block, propagated with beta, re-referenced to common average."""
    n, _, n_t = X.shape
    m = calib_eog.shape[1]
    Xp = X.copy()
    for i in range(n):
        s = int(rng.integers(0, m - n_t))
        seg = calib_eog[:, s:s + n_t]
        seg = seg - seg.mean(1, keepdims=True)
        contam = beta[:, 1:] @ seg                      # (22, n_t)
        contam = contam - contam.mean(0, keepdims=True)  # common average reference
        Xp[i] += gain * contam
    return Xp


def conditions(subject, Ed, P):
    pp = P["perturbations"]
    perm = np.random.default_rng(7 + int(subject[1:])).permutation(22)
    for k in pp["channel_dropout"]["k"]:
        yield f"drop{k}", channel_dropout(Ed["X"], k, perm)
    for g in pp["eog_mixing"]["gain"]:
        rng = np.random.default_rng(11 + int(subject[1:]))
        yield f"eog{g}", eog_mixing(Ed["X"], Ed["calib_eog"], Ed["beta_eog"], g, rng)


def main(subject):
    P = load_protocol()
    Ed = load_session(subject, "E")
    bundles = [load_bundle(subject, j) for j in range(1, 6)]
    paths = [OUT / "raw" / f"{subject}_member{j}.npz" for j in range(1, 6)]
    store = [dict(np.load(p, allow_pickle=True)) for p in paths]
    for name, Xp in conditions(subject, Ed, P):
        t0 = time.time()
        tf = stft_maps(Xp)
        for b, st in zip(bundles, store):
            for k, v in b.outputs(Xp, tf).items():
                st[f"{k}_E_{name}"] = v
        print(f"{subject} {name} done ({time.time() - t0:.0f}s)", flush=True)
    for p, st in zip(paths, store):
        np.savez_compressed(p, **st)


if __name__ == "__main__":
    main(sys.argv[1])
