"""Train the 5-member representation/head-separated ensemble for one subject.

Never reads E-session labels: E inputs are only passed through frozen models.
Outputs: models/<S>/epochs.json, models/<S>/member<j>.pkl (+ .pt weights), out/raw/<S>_member<j>.npz
Usage: python train_subject.py A01 [--pilot]   (--pilot: epoch selection only, then stop)
"""
from __future__ import annotations

import common  # noqa: F401  (adds legacy package to sys.path)

import argparse
import json
import pickle
import time

import numpy as np
import torch

import mne
import nets

mne.set_log_level("ERROR")
from common import MODELS, OUT, LOGS, load_protocol, member_runs, member_seed, protocol_hash
from features import eegnet_input, fbcsp_window, load_session, tf_cache
from heads import GMMHead, lr_head
from bci2a.fbcsp import FBCSPFeatureExtractor
from bci2a.mibif import MIBIFSelector


def logger(subject):
    LOGS.mkdir(exist_ok=True)
    f = open(LOGS / f"train_{subject}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def apply_amendment(ep: dict) -> dict:
    """Protocol amendment 1: recompute both epoch counts from the stored validation histories."""
    for k in ("vit", "eegnet"):
        ep.setdefault(f"{k}_v1", ep[k])
        ep[k], ep[f"{k}_rule"] = nets.select_epoch_amended(np.array(ep[f"{k}_history"]))
    return ep


def select_epochs(subject, T, tfT, P, log) -> dict:
    path = MODELS / subject / "epochs.json"
    if path.exists():
        ep = apply_amendment(json.loads(path.read_text()))
        path.write_text(json.dumps(ep))
        log(f"epochs (amendment 1): vit {ep['vit']} ({ep['vit_rule']}, v1 {ep['vit_v1']}), "
            f"eegnet {ep['eegnet']} ({ep['eegnet_rule']}, v1 {ep['eegnet_v1']})")
        return ep
    r = member_runs(P, 1)
    fit = np.isin(T["run_id"], r["A_fit"])
    val = np.isin(T["run_id"], r["A_val"])
    y = T["y"] - 1
    seed = member_seed(subject, 1, P["seeds"]["base"])
    out = {}
    # ViT
    t0 = time.time()
    norm = nets.TFNorm(tfT[fit])
    _, h = nets.train("vit", norm(tfT[fit]), y[fit], P["vit"]["max_epochs"], seed,
                      norm(tfT[val]), y[val], log)
    out["vit"] = nets.select_epoch(h[:, 1])
    out["vit_history"] = h.tolist()
    log(f"ViT epoch selection: {out['vit']} ({time.time() - t0:.0f}s)")
    # EEGNet
    t0 = time.time()
    XeT = eegnet_input(T["X"])
    cn = nets.ChanNorm(XeT[fit])
    _, h = nets.train("eegnet", cn(XeT[fit]), y[fit], P["eegnet"]["max_epochs"], seed,
                      cn(XeT[val]), y[val], log)
    out["eegnet"] = nets.select_epoch(h[:, 1])
    out["eegnet_history"] = h.tolist()
    log(f"EEGNet epoch selection: {out['eegnet']} ({time.time() - t0:.0f}s)")
    out = apply_amendment(out)
    log(f"epochs (amendment 1): vit {out['vit']} ({out['vit_rule']}, v1 {out['vit_v1']}), "
        f"eegnet {out['eegnet']} ({out['eegnet_rule']}, v1 {out['eegnet_v1']})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def run_member(subject, j, T, tfT, E_inputs, ep, P, log):
    raw_path = OUT / "raw" / f"{subject}_member{j}.npz"
    if raw_path.exists():
        log(f"member {j}: exists, skipped")
        return
    r = member_runs(P, j)
    A = np.isin(T["run_id"], r["A"])
    B = np.isin(T["run_id"], r["B"])
    C = T["run_id"] == P["split"]["C_run"]
    y = T["y"] - 1
    seed = member_seed(subject, j, P["seeds"]["base"])
    t0 = time.time()

    # --- representation layer on A_j
    vnorm = nets.TFNorm(tfT[A])
    vit, _ = nets.train("vit", vnorm(tfT[A]), y[A], ep["vit"], seed)
    XeT = eegnet_input(T["X"])
    cn = nets.ChanNorm(XeT[A])
    eeg, _ = nets.train("eegnet", cn(XeT[A]), y[A], ep["eegnet"], seed)
    Xf = fbcsp_window(T["X"])
    fb = FBCSPFeatureExtractor(sfreq=250.0).fit(Xf[A], T["y"][A])
    mib = MIBIFSelector(k=P_K(P), random_state=seed).fit(fb.transform(Xf[A]), y[A])
    log(f"member {j}: representation trained ({time.time() - t0:.0f}s)")

    bundle = Bundle(vit, vnorm, eeg, cn, fb, mib)
    featB = bundle.features(T["X"][B], tfT[B])
    heads = {
        "concat": lr_head(np.hstack([featB["fbcsp"], featB["z_vit"]]), y[B]),
        "fbcsp": lr_head(featB["fbcsp"], y[B]),
        "gmm_vit": GMMHead(seed=seed).fit(featB["z_vit"], y[B]),
        "gmm_eeg": GMMHead(seed=seed).fit(featB["z_eeg"], y[B]),
    }
    bundle.heads = heads

    outs = {"C": bundle.outputs(T["X"][C], tfT[C]), "E": bundle.outputs(*E_inputs)}
    flat = {f"{k}_{s}": v for s, d in outs.items() for k, v in d.items()}
    flat["trial_id_C"] = T["trial_id"][C]
    flat["gmm_dim"] = heads["gmm_vit"].dim
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **flat)
    mdir = MODELS / subject
    torch.save(vit.state_dict(), mdir / f"member{j}_vit.pt")
    torch.save(eeg.state_dict(), mdir / f"member{j}_eegnet.pt")
    bundle.vit = bundle.eeg = None
    with open(mdir / f"member{j}.pkl", "wb") as f:
        pickle.dump(bundle, f)
    log(f"member {j}: done ({time.time() - t0:.0f}s), n_A={A.sum()} n_B={B.sum()} n_C={C.sum()}")


def P_K(P):
    return 16


class Bundle:
    """Frozen member: representation models, normalizers and B-fitted heads."""

    def __init__(self, vit, vnorm, eeg, cn, fb, mib):
        self.vit, self.vnorm, self.eeg, self.cn, self.fb, self.mib = vit, vnorm, eeg, cn, fb, mib
        self.heads = None

    def features(self, X, tf):
        return {
            "fbcsp": self.mib.transform(self.fb.transform(fbcsp_window(X))),
            "z_vit": nets.embed(self.vit, self.vnorm(tf)),
            "z_eeg": nets.embed(self.eeg, self.cn(eegnet_input(X))),
        }

    def outputs(self, X, tf):
        f = self.features(X, tf)
        h = self.heads
        return {
            "p_concat": h["concat"].predict_proba(np.hstack([f["fbcsp"], f["z_vit"]])),
            "p_fbcsp": h["fbcsp"].predict_proba(f["fbcsp"]),
            "p_vit": nets.predict_proba(self.vit, self.vnorm(tf)),
            "p_eeg": nets.predict_proba(self.eeg, self.cn(eegnet_input(X))),
            "g_vit": h["gmm_vit"].predict_proba(f["z_vit"]),
            "g_eeg": h["gmm_eeg"].predict_proba(f["z_eeg"]),
        }


class _Unpickler(pickle.Unpickler):
    """Bundles were pickled while this module ran as __main__; resolve the class here instead."""

    def find_class(self, module, name):
        if module == "__main__" and name == "Bundle":
            return Bundle
        return super().find_class(module, name)


def load_bundle(subject, j):
    mdir = MODELS / subject
    with open(mdir / f"member{j}.pkl", "rb") as f:
        b = _Unpickler(f).load()
    b.vit = nets.make_vit()
    b.vit.load_state_dict(torch.load(mdir / f"member{j}_vit.pt"))
    eeg = nets.EEGNet()
    eeg.load_state_dict(torch.load(mdir / f"member{j}_eegnet.pt"))
    b.eeg = eeg
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    P = load_protocol()
    log = logger(a.subject)
    log(f"=== {a.subject} protocol {P['protocol_id']} sha {protocol_hash()} pilot={a.pilot}")
    T = load_session(a.subject, "T")
    tfT = tf_cache(a.subject, "T")
    ep = select_epochs(a.subject, T, tfT, P, log)
    if a.pilot:
        return
    Ed = load_session(a.subject, "E")       # inputs only; labels are not read here
    E_inputs = (Ed["X"], tf_cache(a.subject, "E"))
    for j in range(1, 6):
        run_member(a.subject, j, T, tfT, E_inputs, ep, P, log)
    log(f"=== {a.subject} finished")


if __name__ == "__main__":
    main()
