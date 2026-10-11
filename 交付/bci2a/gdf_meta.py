"""从 BCICIV 2a GDF 解析 MI run、trial 与官方 1–4 类标签。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CUE_CODES_T = ("769", "770", "771", "772")
# 官方四类：769→1 左手, 770→2 右手, 771→3 脚, 772→4 舌
EVENT_TO_LABEL = {769: 1, 770: 2, 771: 3, 772: 4}

def mi_run_boundaries_sec(raw) -> list[float]:
    """6 个 MI run 的起始时间（秒），由 32766 分段且该段内含 768 或 769–772。"""
    ann = raw.annotations
    order = sorted(range(len(ann)), key=lambda i: ann.onset[i])
    runs = [float(ann.onset[i]) for i in order if ann.description[i] == "32766"]
    bounds: list[float] = []
    for ri, t0 in enumerate(runs):
        t1 = runs[ri + 1] if ri + 1 < len(runs) else float(raw.times[-1]) + 1.0
        in_seg = [i for i in order if t0 <= ann.onset[i] < t1]
        has_mi = any(
            ann.description[i] in CUE_CODES_T or ann.description[i] == "768"
            for i in in_seg
        )
        if has_mi:
            bounds.append(t0)
    if len(bounds) != 6:
        raise RuntimeError(f"预期 6 个 MI run，解析到 {len(bounds)} 个")
    return bounds

def run_id_for_onset(onset_sec: float, run_bounds: list[float]) -> int:
    """MI run 编号 1–6（与 PDF 一致）。"""
    for i, t0 in enumerate(run_bounds):
        t1 = run_bounds[i + 1] if i + 1 < len(run_bounds) else 1e9
        if t0 <= onset_sec < t1:
            return i + 1
    return len(run_bounds)

def bad_cue_onsets_sec(raw) -> set[float]:
    """含 1023 的 trial 内所有 769–772 提示 onset（秒）。"""
    ann = raw.annotations
    if len(ann) == 0:
        return set()

    order = np.argsort(ann.onset)
    onsets = ann.onset
    desc = ann.description
    idx768 = [i for i in order if desc[i] == "768"]
    bad_cue: set[float] = set()
    for j, i768 in enumerate(idx768):
        t0 = float(onsets[i768])
        t1 = float(onsets[idx768[j + 1]]) if j + 1 < len(idx768) else float(raw.times[-1]) + 1.0
        in_seg = [k for k in order if t0 <= onsets[k] < t1]
        if not any(desc[k] == "1023" for k in in_seg):
            continue
        for k in in_seg:
            if desc[k] in CUE_CODES_T:
                bad_cue.add(float(onsets[k]))
    return bad_cue

def trial_table_training_gdf(gdf_path: Path) -> list[dict]:
    """
    训练 session (T)：每个 trial 一条记录。
    label: 1–4；cue_onset_sec；run_id: 1–6；is_artifact。
    """
    import mne

    gdf_path = gdf_path.expanduser().resolve()
    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="WARNING")
    run_bounds = mi_run_boundaries_sec(raw)
    bad = bad_cue_onsets_sec(raw)

    ann = raw.annotations
    order = sorted(range(len(ann)), key=lambda i: ann.onset[i])
    rows: list[dict] = []
    for i in order:
        code = ann.description[i]
        if code not in CUE_CODES_T:
            continue
        t = float(ann.onset[i])
        evt = int(code)
        rows.append(
            {
                "label": EVENT_TO_LABEL[evt],
                "cue_onset_sec": t,
                "run_id": run_id_for_onset(t, run_bounds),
                "is_artifact": t in bad or any(abs(t - b) < 1e-4 for b in bad),
                "event_code": evt,
            }
        )
    return rows

def trial_table_eval_gdf(
    gdf_path: Path,
    labels_1_4: np.ndarray,
) -> list[dict]:
    """
    评估 session (E)：按 768 顺序与官方 mat 标签对齐；以 783 为 epoch 零点。
    含 1023 的 trial 丢弃，并同步丢弃对应标签（与 T 一致）。
    """
    import mne

    gdf_path = gdf_path.expanduser().resolve()
    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="WARNING")
    run_bounds = mi_run_boundaries_sec(raw)

    ann = raw.annotations
    order = sorted(range(len(ann)), key=lambda i: ann.onset[i])
    idx768 = [i for i in order if ann.description[i] == "768"]
    labels_1_4 = np.asarray(labels_1_4, dtype=np.int64).ravel()
    if len(labels_1_4) != len(idx768):
        raise ValueError(
            f"{gdf_path.name}: mat 标签数 {len(labels_1_4)} != GDF trial 数 {len(idx768)}"
        )

    rows: list[dict] = []
    for k, i768 in enumerate(idx768):
        t768 = float(ann.onset[i768])
        t1 = (
            float(ann.onset[idx768[k + 1]])
            if k + 1 < len(idx768)
            else float(raw.times[-1]) + 1.0
        )
        in_seg = [j for j in order if t768 <= ann.onset[j] < t1]
        if any(ann.description[j] == "1023" for j in in_seg):
            continue
        t783 = None
        for j in in_seg:
            if ann.description[j] == "783":
                t783 = float(ann.onset[j])
                break
        if t783 is None:
            raise RuntimeError(f"{gdf_path.name}: trial {k} 无 783 提示")
        lab = int(labels_1_4[k])
        if lab not in (1, 2, 3, 4):
            raise ValueError(f"标签须为 1–4，得到 {lab}")
        rows.append(
            {
                "label": lab,
                "cue_onset_sec": t783,
                "run_id": run_id_for_onset(t783, run_bounds),
                "is_artifact": False,
                "event_code": 783,
            }
        )
    return rows

def load_eval_labels_from_mat(mat_path: Path) -> np.ndarray:
    """读取 AxxE.mat 中官方 classlabel（1–4）。"""
    from scipy.io import loadmat

    mat_path = mat_path.expanduser().resolve()
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    for key in ("classlabel", "labels", "y"):
        if key in d:
            y = np.asarray(d[key], dtype=np.int64).ravel()
            return y
    # BNCI / 部分发行版嵌在 data 结构体
    if "data" in d:
        data = d["data"]
        if hasattr(data, "y"):
            return np.asarray(data.y, dtype=np.int64).ravel()
        if hasattr(data, "classlabel"):
            return np.asarray(data.classlabel, dtype=np.int64).ravel()
    raise KeyError(f"{mat_path} 中未找到 classlabel/labels")

def subject_id_from_stem(stem: str) -> str:
    """A01T -> A01, A01E -> A01"""
    return stem[:3].upper()
