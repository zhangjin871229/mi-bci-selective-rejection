"""从 GDF 加载 MI epoch（与 preprocess_bci2a 一致的前处理 + 0.5–2.5s 窗）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .gdf_meta import (
    load_eval_labels_from_mat,
    trial_table_eval_gdf,
    trial_table_training_gdf,
)

def default_gdf_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "BCICIV_2a_gdf"

def default_mat_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "true_labels"

def _preprocess_raw(raw, *, eog_regress: bool, notch_50: bool):
    from preprocess_bci2a import pick_eeg_indices, regress_eog_from_eeg

    if eog_regress:
        raw = regress_eog_from_eeg(raw)
    raw.pick(pick_eeg_indices(raw.ch_names))
    if notch_50:
        raw.notch_filter(50.0, picks="all", verbose="WARNING")
    raw.set_eeg_reference("average", projection=False, verbose="WARNING")
    return raw

def _epochs_from_rows(raw, rows: list[dict], tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mne

    sfreq = float(raw.info["sfreq"])
    events = []
    kept = []
    for row in rows:
        if row.get("is_artifact"):
            continue
        sample = int(round(row["cue_onset_sec"] * sfreq)) + raw.first_samp
        events.append([sample, 0, int(row["label"])])
        kept.append(row)
    if not events:
        raise ValueError("无可用 trial")
    events = np.asarray(events, dtype=np.int64)
    event_id = {f"MI_{i}": i for i in (1, 2, 3, 4)}
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="WARNING",
    )
    y = np.array([int(r["label"]) for r in kept], dtype=np.int64)
    run_id = np.array([int(r["run_id"]) for r in kept], dtype=np.int64)
    return epochs.get_data().astype(np.float64), y, run_id

def load_subject_epochs(
    subject: str,
    session: str,
    *,
    gdf_dir: Path | None = None,
    mat_dir: Path | None = None,
    tmin: float = 0.5,
    tmax: float = 2.5,
    eog_regress: bool = True,
    notch_50: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 X (n, n_ch, n_times), y 官方 1–4, run_id。
    session: 'T' 或 'E'
    """
    import mne

    gdf_dir = (gdf_dir or default_gdf_dir()).expanduser().resolve()
    mat_dir = (mat_dir or default_mat_dir()).expanduser().resolve()
    sub = subject[:3].upper()
    stem = f"{sub}{session.upper()}"
    gdf_path = gdf_dir / f"{stem}.gdf"
    if not gdf_path.is_file():
        raise FileNotFoundError(gdf_path)

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="WARNING")
    raw = _preprocess_raw(raw, eog_regress=eog_regress, notch_50=notch_50)

    if session.upper() == "T":
        rows = trial_table_training_gdf(gdf_path)
    else:
        mat_path = mat_dir / f"{stem}.mat"
        labels = load_eval_labels_from_mat(mat_path)
        rows = trial_table_eval_gdf(gdf_path, labels)

    return _epochs_from_rows(raw, rows, tmin, tmax)
