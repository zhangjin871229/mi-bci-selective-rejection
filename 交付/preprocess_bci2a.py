"""
BCICIV 2a GDF 预处理（顺序与 desc_2a.pdf 及常见 MI 流程一致）：

1) 连续数据阶段（PDF：去眼电影响、带通/陷波；EOG 不参与分类）
   - 可选：对 22 路 EEG 做三路 EOG 的线性回归去伪迹（文档建议的回归类方法之一）
   - 可选：50 Hz 陷波（采集端已开陷波，可再压残余工频）
   - 可选：带通（采集为 0.5–100 Hz；分类常用再加一段如 8–30 Hz 突出 μ/β）
   - 平均参考（CAR）
   - 可选：剔除专家标为伪迹的 trial（事件 1023，按相邻「768」划 trial）

2) 分段：以提示 769–772 为 0 s，取 [tmin, tmax]（默认 0–4 s）

3) 你补充的三步（放在分段之后、进模型之前）
   - 小波软阈值去噪（每通道时间序列）
   - STFT 幅值谱
   - 对每张时频图 Z-score

命令行：单个文件用 --file；全部被试训练集用 --all（A01T…A09T）。

依赖：mne、numpy<2、scipy、PyWavelets（见 requirements.txt）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_DL_ROOT = Path(__file__).resolve().parent
if str(_DL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DL_ROOT))

# 四类运动想象：提示事件（与 desc_2a.pdf 一致，官方标签 1–4）
CUE_EVENT_ID = {"769": 1, "770": 2, "771": 3, "772": 4}


def _check_numpy_for_gdf() -> None:
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print(
            "警告：NumPy >= 2 时 MNE 读 GDF 可能失败，请先: pip install \"numpy<2\"\n",
            file=sys.stderr,
        )


def default_gdf_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "BCICIV_2a_gdf"


def default_out_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def default_mat_dir() -> Path:
    """官方真值标签 AxxT.mat / AxxE.mat（classlabel 1–4）。"""
    return Path(__file__).resolve().parent.parent / "true_labels"


def list_training_gdfs(gdf_dir: Path) -> list[Path]:
    """BCICIV 2a 训练 session：A01T…A09T。"""
    gdf_dir = gdf_dir.expanduser().resolve()
    pat = re.compile(r"^A\d{2}T$", re.IGNORECASE)
    return sorted(p for p in gdf_dir.glob("A*.gdf") if pat.match(p.stem))


def list_eval_gdfs(gdf_dir: Path) -> list[Path]:
    """评估 session：A01E…A09E（标签需配套 AxxE.mat）。"""
    gdf_dir = gdf_dir.expanduser().resolve()
    pat = re.compile(r"^A\d{2}E$", re.IGNORECASE)
    return sorted(p for p in gdf_dir.glob("A*.gdf") if pat.match(p.stem))


def _epochs_from_cues(
    raw: "mne.io.BaseRaw",
    trials: list[dict],
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 cue_onset_sec 切段，返回 data, y(1–4), run_id(1–6)。"""
    import mne

    sfreq = float(raw.info["sfreq"])
    events = []
    for row in trials:
        sample = int(round(row["cue_onset_sec"] * sfreq)) + raw.first_samp
        events.append([sample, 0, int(row["label"])])
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
    y = np.array([int(row["label"]) for row in trials], dtype=np.int64)
    run_id = np.array([int(row["run_id"]) for row in trials], dtype=np.int64)
    return epochs.get_data(), y, run_id


def pick_eeg_indices(ch_names: list[str]) -> list[int]:
    """保留非 EOG 通道（本数据集为前 22 路 EEG + 3 路 EOG）。"""
    idx = [i for i, n in enumerate(ch_names) if "EOG" not in n]
    if len(idx) != 22:
        # 仍按「不含 EOG」名称筛选；若数量异常则给出提示但不强行改
        print(f"提示：当前选中 {len(idx)} 路非 EOG 通道（预期 22）。")
    return idx


def pick_eog_indices(ch_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(ch_names) if "EOG" in n]


def regress_eog_from_eeg(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
    """
    用三路 EOG 对每路 EEG 做 OLS 回归，从 EEG 中减去 EOG 可解释部分（离线简化版）。
    需在仍含 EOG 通道的 Raw 上调用；返回仍含全部通道的 Raw（仅 EEG 行被改）。
    """
    import mne

    ch_names = raw.ch_names
    picks_eeg = pick_eeg_indices(ch_names)
    picks_eog = pick_eog_indices(ch_names)
    if not picks_eog or not picks_eeg:
        return raw

    ann = raw.annotations.copy()

    X = raw.get_data().astype(np.float64, copy=True)
    E = X[picks_eog].T
    E = np.concatenate([np.ones((E.shape[0], 1), dtype=np.float64), E], axis=1)
    for ci in picks_eeg:
        y = X[ci]
        beta, _, _, _ = np.linalg.lstsq(E, y, rcond=None)
        X[ci] = y - E @ beta

    raw_new = mne.io.RawArray(X.astype(np.float32), raw.info, verbose="WARNING")
    raw_new.set_annotations(ann)
    return raw_new


def bad_cue_onsets_sec(raw: "mne.io.BaseRaw") -> set[float]:
    """
    按相邻「768」trial 起划分段；若该段内出现「1023」则该 trial 内所有提示视为伪迹 trial。
    onset 单位为秒，与 MNE annotations 一致。
    """
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
        # 含 trial 起点（768 可与 1023 同一时刻）
        in_seg = [k for k in order if t0 <= onsets[k] < t1]
        if not any(desc[k] == "1023" for k in in_seg):
            continue
        for k in in_seg:
            if desc[k] in ("769", "770", "771", "772"):
                bad_cue.add(float(onsets[k]))
    return bad_cue


def filter_events_drop_artifact_cues(
    raw: "mne.io.BaseRaw",
    events: np.ndarray,
) -> tuple[np.ndarray, int]:
    """剔除落在伪迹 trial 内的提示事件（与 768 分段 + 1023 一致）。"""
    bad_onsets = bad_cue_onsets_sec(raw)
    if not bad_onsets:
        return events, 0

    sfreq = float(raw.info["sfreq"])
    tol = 1.0 / sfreq
    idx = (events[:, 0] - raw.first_samp).astype(np.int64)
    idx = np.clip(idx, 0, len(raw.times) - 1)
    evt_times = raw.times[idx]
    keep = np.array(
        [not any(abs(float(t) - b) <= tol for b in bad_onsets) for t in evt_times],
        dtype=bool,
    )
    n_drop = int(np.sum(~keep))
    return events[keep], n_drop


def wavelet_denoise_1d(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "soft",
) -> np.ndarray:
    """对单通道时间序列做小波分解软阈值去噪。"""
    import pywt

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    detail_coeffs = coeffs[1:]
    if not detail_coeffs:
        return x.astype(np.float32)
    # 用最高频细节估计噪声标准差（通用启发式）
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = float(np.std(detail_coeffs[-1]) + 1e-12)
    uth = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
    new_coeffs = [coeffs[0]]
    for c in detail_coeffs:
        new_coeffs.append(pywt.threshold(c, value=uth, mode=mode))
    recon = pywt.waverec(new_coeffs, wavelet, mode="symmetric")
    recon = recon[:n]
    return recon.astype(np.float32)


def stft_magnitude(
    x: np.ndarray,
    sfreq: float,
    nperseg: int,
    noverlap: int,
    nfft: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单通道 STFT 幅值谱。"""
    from scipy import signal

    nfft = nfft or nperseg
    f, t, Zxx = signal.stft(
        x,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(Zxx).astype(np.float32)
    return f.astype(np.float32), t.astype(np.float32), mag


def zscore_tf(mag: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对单张时频幅值矩阵做整体 Z-score。"""
    m = np.mean(mag)
    s = np.std(mag)
    if s < eps:
        s = eps
    return ((mag - m) / s).astype(np.float32)


def run(
    gdf_path: Path,
    out_dir: Path,
    *,
    tmin: float,
    tmax: float,
    use_wavelet: bool,
    wavelet: str,
    nperseg: int,
    noverlap: int,
    nfft: int | None,
    eog_regress: bool,
    notch_50: bool,
    bandpass: tuple[float, float] | None,
    drop_artifact_trials: bool,
    mat_dir: Path | None = None,
    zscore_per_trial: bool = True,
    filter_bands: list[tuple[float, float]] | None = None,
    log_magnitude: bool = False,
) -> Path:
    import mne

    gdf_path = gdf_path.expanduser().resolve()
    if not gdf_path.is_file():
        raise FileNotFoundError(gdf_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="WARNING")
    sfreq = float(raw.info["sfreq"])

    # EOG 回归
    if eog_regress:
        raw = regress_eog_from_eeg(raw)

    # 仅保留 EEG 用于后续分类特征（PDF：EOG 不得用于分类）
    eeg_idx = pick_eeg_indices(raw.ch_names)
    raw.pick(eeg_idx)

    # 陷波 / 带通
    if notch_50:
        raw.notch_filter(50.0, picks="all", verbose="WARNING")
    if bandpass is not None:
        lf, hf = bandpass
        raw.filter(l_freq=lf, h_freq=hf, picks="all", verbose="WARNING")

    raw.set_eeg_reference("average", projection=False, verbose="WARNING")

    from bci2a.gdf_meta import (
        load_eval_labels_from_mat,
        trial_table_eval_gdf,
        trial_table_training_gdf,
    )

    stem = gdf_path.stem
    session = "E" if stem.upper().endswith("E") else "T"
    if session == "T":
        all_rows = trial_table_training_gdf(gdf_path)
    else:
        if mat_dir is None:
            raise ValueError(
                f"{stem}: 评估 session 需提供 --mat-dir（含官方 AxxE.mat，classlabel 1–4）"
            )
        mat_path = mat_dir / f"{stem}.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        labels = load_eval_labels_from_mat(mat_path)
        all_rows = trial_table_eval_gdf(gdf_path, labels)

    n_drop = sum(1 for r in all_rows if r["is_artifact"])
    if drop_artifact_trials:
        trials = [r for r in all_rows if not r["is_artifact"]]
    else:
        trials = all_rows
    if not trials:
        raise RuntimeError("剔除伪迹后无剩余 trial。")

    data, y, run_id = _epochs_from_cues(raw, trials, tmin, tmax)

    from bci2a.feature_utils import DEFAULT_FILTER_BANK, bandpass_1d, post_stft_magnitude

    bands = filter_bands if filter_bands else [None]
    n_ep, n_ch, _ = data.shape
    f_ref: np.ndarray | None = None
    t_ref: np.ndarray | None = None
    ref_tf_shape: tuple[int, int] | None = None
    tf_list: list[np.ndarray] = []

    for ei in range(n_ep):
        ch_mags: list[np.ndarray] = []
        for band in bands:
            lf, hf = band if band is not None else (None, None)
            for ci in range(n_ch):
                x = data[ei, ci].astype(np.float32, copy=False)
                if use_wavelet:
                    x = wavelet_denoise_1d(x, wavelet=wavelet)
                if band is not None:
                    x = bandpass_1d(x, sfreq, lf, hf)
                f, t_stft, mag = stft_magnitude(
                    x, sfreq, nperseg=nperseg, noverlap=noverlap, nfft=nfft
                )
                mag = post_stft_magnitude(mag, log1p=log_magnitude)
                mag_out = zscore_tf(mag) if zscore_per_trial else mag.astype(np.float32)
                if ref_tf_shape is None:
                    f_ref, t_ref = f, t_stft
                    ref_tf_shape = mag_out.shape
                elif mag_out.shape != ref_tf_shape:
                    raise RuntimeError(
                        f"STFT 形状不一致: 期望 {ref_tf_shape}, 得到 {mag_out.shape} "
                        f"(trial={ei}, ch={ci})"
                    )
                ch_mags.append(mag_out)
        tf_list.append(np.stack(ch_mags, axis=0))

    X = np.stack(tf_list, axis=0).astype(np.float32)
    assert f_ref is not None and t_ref is not None

    out_path = out_dir / f"{stem}_tfwave.npz"
    ch_names = np.array(raw.ch_names, dtype=object)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        run_id=run_id,
        session=session,
        label_offset=np.int32(1),
        stft_freqs=f_ref,
        stft_times=t_ref,
        sfreq=sfreq,
        ch_names=ch_names,
        source_file=str(gdf_path),
        tmin=tmin,
        tmax=tmax,
        use_wavelet=use_wavelet,
        wavelet=wavelet,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=int(nfft or nperseg),
        eog_regress=eog_regress,
        notch_50=notch_50,
        bandpass_l=float(bandpass[0]) if bandpass else np.nan,
        bandpass_h=float(bandpass[1]) if bandpass else np.nan,
        bandpass_off=bandpass is None,
        drop_artifact_trials=drop_artifact_trials,
        n_artifact_trials_dropped=n_drop,
        zscore_per_trial=zscore_per_trial,
        log_magnitude=log_magnitude,
        n_eeg_channels=np.int32(n_ch),
        n_filter_bands=np.int32(len(bands) if filter_bands else 1),
        filter_banks=np.array(filter_bands if filter_bands else [], dtype=np.float32),
    )
    return out_path


def main() -> None:
    _check_numpy_for_gdf()

    p = argparse.ArgumentParser(
        description="BCICIV 2a：PDF 常规预处理 + 小波 + STFT + Z-score，保存 npz"
    )
    p.add_argument(
        "--gdf-dir",
        type=Path,
        default=default_gdf_dir(),
        help="GDF 目录",
    )
    p.add_argument(
        "--file",
        type=str,
        default="A01T.gdf",
        help="单个 GDF 文件名（与 --all 二选一）",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="处理全部被试训练 session A01T…A09T",
    )
    p.add_argument(
        "--all-eval",
        action="store_true",
        help="处理评估 session A01E…A09E（须 --mat-dir）",
    )
    p.add_argument(
        "--mat-dir",
        type=Path,
        default=default_mat_dir(),
        help="真值标签目录（默认 ../true_labels，含 classlabel 1–4）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir(),
        help="输出目录（默认 DL/data）",
    )
    p.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="相对提示 onset 的 epoch 起点（秒），默认 0",
    )
    p.add_argument(
        "--tmax",
        type=float,
        default=4.0,
        help="相对提示 onset 的 epoch 终点（秒），默认 4（约 4 s MI）",
    )
    p.add_argument(
        "--no-eog-regress",
        action="store_true",
        help="关闭 EOG 线性回归（默认开启，符合文档去眼电思路）",
    )
    p.add_argument(
        "--no-notch",
        action="store_true",
        help="关闭 50 Hz 陷波（默认开启）",
    )
    p.add_argument(
        "--no-bandpass",
        action="store_true",
        help="关闭额外带通（默认 8–30 Hz，突出运动想象频段）",
    )
    p.add_argument("--l-freq", type=float, default=8.0, help="带通下限 Hz")
    p.add_argument("--h-freq", type=float, default=30.0, help="带通上限 Hz")
    p.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留含 1023 的伪迹 trial（默认剔除）",
    )
    p.add_argument(
        "--no-zscore",
        action="store_true",
        help="不做 per-trial Z-score，保留 STFT 幅值（配合训练集归一化）",
    )
    p.add_argument(
        "--no-wavelet",
        action="store_true",
        help="跳过小波去噪，仅 STFT + Z-score",
    )
    p.add_argument("--wavelet", type=str, default="db4", help="PyWavelets 小波名")
    p.add_argument("--nperseg", type=int, default=64, help="STFT 窗长（样本点）")
    p.add_argument("--noverlap", type=int, default=48, help="STFT 重叠（样本点）")
    p.add_argument("--nfft", type=int, default=None, help="FFT 长度，默认等于 nperseg")
    p.add_argument(
        "--filter-bank",
        action="store_true",
        help="FBCSP 式滤波器组：每子带×22 通道分别 STFT，通道维堆叠",
    )
    p.add_argument(
        "--log-mag",
        action="store_true",
        help="STFT 幅值 log1p（FBCSP/CSP 常见）",
    )
    p.add_argument(
        "--preset-fbcsp",
        action="store_true",
        help="冠军启发预设：filter-bank + tmin/tmax=0.5/2.5 + log-mag + no-zscore，输出 data_fb/",
    )
    args = p.parse_args()

    if args.preset_fbcsp:
        args.filter_bank = True
        args.log_mag = True
        args.no_zscore = True
        args.tmin = 0.5
        args.tmax = 2.5
        args.no_bandpass = True
        if args.out_dir == default_out_dir():
            args.out_dir = Path(__file__).resolve().parent / "data_fb"

    gdf_dir = args.gdf_dir.expanduser().resolve()
    mat_dir = args.mat_dir.expanduser().resolve()
    if args.all and args.all_eval:
        raise SystemExit("请只指定 --all 或 --all-eval 之一")
    if args.all:
        paths = list_training_gdfs(gdf_dir)
        if not paths:
            raise FileNotFoundError(f"未在 {gdf_dir} 找到 AxxT.gdf")
    elif args.all_eval:
        paths = list_eval_gdfs(gdf_dir)
        if not paths:
            raise FileNotFoundError(f"未在 {gdf_dir} 找到 AxxE.gdf")
        if mat_dir is None:
            raise SystemExit("--all-eval 需要 --mat-dir")
    else:
        path = Path(args.file)
        if not path.is_absolute():
            path = gdf_dir / path
        paths = [path]
        if path.stem.upper().endswith("E") and mat_dir is None:
            raise SystemExit(f"处理 {path.name} 需要 --mat-dir")

    bandpass = None if args.no_bandpass else (args.l_freq, args.h_freq)
    if args.filter_bank and not args.no_bandpass:
        print("提示：--filter-bank 已启用，建议 --no-bandpass，子带内再滤波。")
    from bci2a.feature_utils import DEFAULT_FILTER_BANK

    fb = DEFAULT_FILTER_BANK if args.filter_bank else None
    out_dir = args.out_dir.expanduser().resolve()

    failures: list[tuple[str, str]] = []
    for idx, path in enumerate(paths, start=1):
        print(f"\n[{idx}/{len(paths)}] {path.name}")
        try:
            out = run(
                path,
                out_dir,
                tmin=args.tmin,
                tmax=args.tmax,
                use_wavelet=not args.no_wavelet,
                wavelet=args.wavelet,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                nfft=args.nfft,
                eog_regress=not args.no_eog_regress,
                notch_50=not args.no_notch,
                bandpass=bandpass,
                drop_artifact_trials=not args.keep_artifacts,
                mat_dir=mat_dir,
                zscore_per_trial=not args.no_zscore,
                filter_bands=fb,
                log_magnitude=args.log_mag,
            )
            z = np.load(out, allow_pickle=True)
            X = z["X"]
            n_drop = (
                int(z["n_artifact_trials_dropped"])
                if "n_artifact_trials_dropped" in z.files
                else 0
            )
            print(f"  已保存: {out}")
            print(f"  剔除伪迹 trial: {n_drop}")
            print(f"  X: {X.shape}  y: {z['y'].shape}")
        except Exception as e:
            failures.append((path.name, str(e)))
            print(f"  失败: {e}", file=sys.stderr)

    if failures:
        print("\n失败文件:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
