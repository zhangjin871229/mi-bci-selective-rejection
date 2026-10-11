"""Train the 5-member ensemble for one 2b subject and store C-block and E outputs (protocol_2b.json).

Never reads E labels: only cache/<S>E_inputs.npz is opened.
Outputs: models/<S>/epochs.json, models/<S>/member<j>_vit.pt, out/raw/<S>_member<j>.npz
Usage: python -I train2b.py B01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2b as C2  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import mne  # noqa: E402
import torch  # noqa: E402

mne.set_log_level("ERROR")

import nets  # noqa: E402  (2a module)
from bci2a.config import TrainConfig  # noqa: E402
from bci2a.fbcsp import FBCSPFeatureExtractor, bandpass_epochs  # noqa: E402
from bci2a.mibif import MIBIFSelector  # noqa: E402
from bci2a.models import ChannelViT  # noqa: E402
from features import fbcsp_window, stft_maps  # noqa: E402
from heads import GMMHead, lr_head  # noqa: E402
from mne.decoding import CSP  # noqa: E402


def make_vit_2b() -> ChannelViT:
    cfg = TrainConfig()
    cfg.freq_bins, cfg.time_frames = 33, 64
    return ChannelViT(cfg, n_classes=C2.N_CLS, n_channels=C2.N_CH)


nets.make_vit = make_vit_2b          # nets.train builds its ViT through this name


class FBCSP2b(FBCSPFeatureExtractor):
    """2a FBCSP with CSP components = min(4, n_channels) = 3 and OVR over the 2 classes present."""

    CLASSES = (1, 2)

    def __init__(self):
        super().__init__(sfreq=C2.SFREQ, n_components=min(4, C2.N_CH))

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.int64)
        self.csps = {}
        for bi, band in enumerate(self.filter_bank):
            Xb = bandpass_epochs(X, self.sfreq, band[0], band[1])
            for cls in self.CLASSES:
                csp = CSP(n_components=self.n_components, reg=self.reg, log=True, norm_trace=False)
                self.csps[(bi, cls)] = csp.fit(Xb, (y == cls).astype(np.int64))
        return self

    def transform(self, X):
        feats = []
        for bi, band in enumerate(self.filter_bank):
            Xb = bandpass_epochs(X, self.sfreq, band[0], band[1])
            feats += [self.csps[(bi, cls)].transform(Xb) for cls in self.CLASSES]
        return np.concatenate(feats, axis=1)


def logger(subject):
    C2.LOGS.mkdir(parents=True, exist_ok=True)
    f = open(C2.LOGS / f"train_{subject}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def select_epochs(subject, T, tfT, log) -> dict:
    path = C2.MODELS / subject / "epochs.json"
    if path.exists():
        return json.loads(path.read_text())
    r = C2.MEMBERS[1]
    fit, val = np.isin(T["block"], r["A_fit"]), np.isin(T["block"], r["A_val"])
    y = T["y"] - 1
    t0 = time.time()
    norm = nets.TFNorm(tfT[fit])
    _, h = nets.train("vit", norm(tfT[fit]), y[fit], 200, C2.member_seed(subject, 1),
                      norm(tfT[val]), y[val], log)
    ep, rule = nets.select_epoch_amended(h)
    out = {"vit": ep, "vit_rule": rule, "vit_history": h.tolist()}
    log(f"ViT epoch selection: {ep} ({rule}) ({time.time() - t0:.0f}s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def run_member(subject, j, T, tfT, XE, tfE, ep, log):
    raw_path = C2.OUT / "raw" / f"{subject}_member{j}.npz"
    if raw_path.exists():
        log(f"member {j}: exists, skipped")
        return
    r = C2.MEMBERS[j]
    A, B = np.isin(T["block"], r["A"]), np.isin(T["block"], r["B"])
    C = T["block"] == C2.C_BLOCK
    y = T["y"] - 1
    seed = C2.member_seed(subject, j)
    t0 = time.time()

    # representation layer on A_j
    vnorm = nets.TFNorm(tfT[A])
    vit, _ = nets.train("vit", vnorm(tfT[A]), y[A], ep["vit"], seed)
    Xf = fbcsp_window(T["X"])
    fb = FBCSP2b().fit(Xf[A], T["y"][A])
    mib = MIBIFSelector(k=16, random_state=seed).fit(fb.transform(Xf[A]), y[A])

    def feats(X, tf):
        return mib.transform(fb.transform(fbcsp_window(X))), nets.embed(vit, vnorm(tf))

    # heads on B_j
    fB, zB = feats(T["X"][B], tfT[B])
    concat = lr_head(np.hstack([fB, zB]), y[B])
    g_fb = GMMHead(seed=seed).fit(fB, y[B])
    g_vit = GMMHead(seed=seed).fit(zB, y[B])

    out = {}
    for tag, (X, tf) in {"C": (T["X"][C], tfT[C]), "E": (XE, tfE)}.items():
        f, z = feats(X, tf)
        out[f"p_concat_{tag}"] = concat.predict_proba(np.hstack([f, z]))
        out[f"g_fbcsp_{tag}"] = g_fb.predict_proba(f)
        out[f"g_vit_{tag}"] = g_vit.predict_proba(z)
    out["trial_id_C"] = T["trial_id"][C]
    out["gmm_dim_fbcsp"] = g_fb.dim
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **out)
    torch.save(vit.state_dict(), C2.MODELS / subject / f"member{j}_vit.pt")
    log(f"member {j}: done ({time.time() - t0:.0f}s), n_A={A.sum()} n_B={B.sum()} n_C={C.sum()}, "
        f"fbcsp-gmm d={g_fb.dim}, mibif {mib.selected_indices_.tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    a = ap.parse_args()
    C2.check_frozen()
    log = logger(a.subject)
    log(f"=== {a.subject} protocol_2b sha {C2.protocol_sha256()[:12]}")
    T = C2.load_npz(C2.CACHE / f"{a.subject}T.npz")
    tfT = stft_maps(T["X"])
    ep = select_epochs(a.subject, T, tfT, log)
    E = C2.load_npz(C2.CACHE / f"{a.subject}E_inputs.npz")       # inputs only, no labels
    tfE = stft_maps(E["X"])
    for j in range(1, 6):
        run_member(a.subject, j, T, tfT, E["X"], tfE, ep, log)
    log(f"=== {a.subject} finished")


if __name__ == "__main__":
    main()
