"""Build per-subject epoch caches directly from the raw GDF files.

Output per subject/session (cache/A0xT.npz, cache/A0xE.npz):
  X      (n, 22, 1001) float32  cleaned EEG in microvolts, 0-4 s after cue
  eog    (n, 3, 1001)  float32  raw EOG in microvolts, same window
  y      (n,) official labels 1-4
  run_id (n,) MI run 1-6
  trial_id (n,) "A01T-r1-t003" style identifiers (index among all 288 cues)
  calib_eog   (3, m) raw EOG during the eye-movement block (event 1072), microvolts
  beta_eog    (22, 4) OLS propagation [intercept, 3 EOG] from raw EEG on EOG in the 1072 block
Preprocessing follows the original pipeline: EOG regression (whole session, unsupervised),
50 Hz notch, common average reference. Artifact trials (event 1023) are excluded.
"""
from __future__ import annotations

import common  # noqa: F401  (adds legacy package to sys.path)

import json

import mne
import numpy as np

from common import CACHE, GDF_DIR, LABEL_DIR, SUBJECTS, file_md5
from bci2a.gdf_meta import load_eval_labels_from_mat, trial_table_eval_gdf, trial_table_training_gdf
from preprocess_bci2a import pick_eeg_indices, pick_eog_indices, regress_eog_from_eeg

mne.set_log_level("ERROR")
TMIN, TMAX = 0.0, 4.0
UV = 1e6


def _eye_block(raw) -> tuple[float, float]:
    ann = raw.annotations
    for d, o, du in zip(ann.description, ann.onset, ann.duration):
        if d == "1072":
            return float(o), float(o + du)
    raise RuntimeError("no 1072 eye-movement block")


def build(subject: str, session: str) -> dict:
    stem = f"{subject}{session}"
    gdf = GDF_DIR / f"{stem}.gdf"
    raw0 = mne.io.read_raw_gdf(gdf, preload=True)
    sf = float(raw0.info["sfreq"])
    eeg_i, eog_i = pick_eeg_indices(raw0.ch_names), pick_eog_indices(raw0.ch_names)
    data0 = raw0.get_data()

    # eye-movement calibration block (raw, before any cleaning)
    t0, t1 = _eye_block(raw0)
    s0, s1 = int(t0 * sf), int(t1 * sf)
    E = data0[eog_i, s0:s1].T * UV
    Ed = np.concatenate([np.ones((E.shape[0], 1)), E], axis=1)
    beta = np.stack([np.linalg.lstsq(Ed, data0[c, s0:s1] * UV, rcond=None)[0] for c in eeg_i])

    # cleaned EEG, identical steps to the original pipeline
    raw = regress_eog_from_eeg(raw0.copy())
    raw.pick(pick_eeg_indices(raw.ch_names))
    raw.notch_filter(50.0, picks="all")
    raw.set_eeg_reference("average", projection=False)

    if session == "T":
        rows = trial_table_training_gdf(gdf)
    else:
        rows = trial_table_eval_gdf(gdf, load_eval_labels_from_mat(LABEL_DIR / f"{stem}.mat"))
        # trial_table_eval_gdf already drops artifact trials; recover original cue index for ids
    all_idx = list(range(len(rows)))
    if session == "T":
        keep = [i for i in all_idx if not rows[i]["is_artifact"]]
        cue_index = keep
    else:
        keep = all_idx
        cue_index = _eval_cue_indices(gdf)
    rows = [rows[i] for i in keep]

    n0 = int(round(TMIN * sf))
    n_t = int(round((TMAX - TMIN) * sf)) + 1
    X = np.empty((len(rows), len(eeg_i), n_t), np.float32)
    O = np.empty((len(rows), len(eog_i), n_t), np.float32)
    d_clean = raw.get_data()
    for k, r in enumerate(rows):
        s = int(round(r["cue_onset_sec"] * sf)) + n0
        X[k] = d_clean[:, s:s + n_t] * UV
        O[k] = data0[eog_i, s:s + n_t] * UV
    y = np.array([r["label"] for r in rows], np.int64)
    run_id = np.array([r["run_id"] for r in rows], np.int64)
    trial_id = np.array([f"{stem}-r{r['run_id']}-t{ci:03d}" for r, ci in zip(rows, cue_index)])
    np.savez_compressed(
        CACHE / f"{stem}.npz", X=X, eog=O, y=y, run_id=run_id, trial_id=trial_id,
        calib_eog=(data0[eog_i, s0:s1] * UV).astype(np.float32), beta_eog=beta.astype(np.float32),
        ch_names=np.array([raw.ch_names[i] for i in range(len(eeg_i))]), sfreq=sf, tmin=TMIN, tmax=TMAX,
    )
    return {"file": stem, "gdf_md5": file_md5(gdf), "n_trials": len(rows),
            "per_run": {int(r): int((run_id == r).sum()) for r in np.unique(run_id)},
            "per_class": {int(c): int((y == c).sum()) for c in (1, 2, 3, 4)},
            "calib_seconds": round(t1 - t0, 1), "eeg_std_uv": float(X.std())}


def _eval_cue_indices(gdf) -> list[int]:
    """Indices (0-287) of non-artifact trials in an E file, in cue order."""
    raw = mne.io.read_raw_gdf(gdf, preload=False)
    ann = raw.annotations
    order = np.argsort(ann.onset)
    desc = [ann.description[i] for i in order]
    on = [ann.onset[i] for i in order]
    starts = [i for i, d in enumerate(desc) if d == "768"]
    keep = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(desc)
        if "1023" not in desc[i:j]:
            keep.append(k)
    return keep


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in SUBJECTS:
        for ses in "TE":
            info = build(s, ses)
            print(info, flush=True)
            manifest.append(info)
    (CACHE / "cache_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
