"""读取 npz、按官方 run / session 划分（禁止随机 trial 划分与跨被试混合）。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DATA_DIR, N_CLASSES, SPLITS_DIR, TrainConfig
from .augment import TFAugmentConfig, apply_tf_augment

@dataclass
class TrialRecord:
    global_id: int
    subject: str
    session: str
    local_idx: int
    label: int
    run_id: int
    npz_path: str

@dataclass
class DatasetManifest:
    records: list[TrialRecord]
    class_names: list[str]
    n_channels: int
    freq_bins: int
    time_frames: int
    label_offset: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "n_channels": self.n_channels,
            "freq_bins": self.freq_bins,
            "time_frames": self.time_frames,
            "label_offset": self.label_offset,
            "records": [asdict(r) for r in self.records],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetManifest":
        records = [TrialRecord(**r) for r in d["records"]]
        return cls(
            records=records,
            class_names=d["class_names"],
            n_channels=d["n_channels"],
            freq_bins=d["freq_bins"],
            time_frames=d["time_frames"],
            label_offset=int(d.get("label_offset", 1)),
        )

def _normalize_label(y: int, files_offset: int | None) -> int:
    """npz 内官方 1–4；manifest 存 1–4。"""
    if files_offset == 1:
        if y < 1 or y > 4:
            if 0 <= y <= 3:
                return y + 1
            raise ValueError(f"非法标签 {y}")
        return int(y)
    # 旧 npz 0–3
    if 0 <= y <= 3:
        return y + 1
    return int(y)

def discover_npz_files(data_dir: Path, pattern: str = "*_tfwave.npz") -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 {data_dir / pattern}")
    return files

def build_manifest(data_dir: Path, pattern: str = "*_tfwave.npz") -> DatasetManifest:
    records: list[TrialRecord] = []
    gid = 0
    n_ch = n_f = n_t = None
    label_offset = 1

    for path in discover_npz_files(data_dir, pattern):
        subject = path.stem.replace("_tfwave", "")
        z = np.load(path, allow_pickle=True)
        X, y = z["X"], z["y"]
        off = int(z["label_offset"]) if "label_offset" in z.files else None
        label_offset = off if off is not None else label_offset
        run_id_arr = z["run_id"] if "run_id" in z.files else np.ones(len(y), dtype=int)
        session = str(z["session"]) if "session" in z.files else (
            "E" if subject.upper().endswith("E") else "T"
        )
        if n_ch is None:
            n_ch, n_f, n_t = int(X.shape[1]), int(X.shape[2]), int(X.shape[3])
        for i in range(len(y)):
            lab = _normalize_label(int(y[i]), off)
            records.append(
                TrialRecord(
                    global_id=gid,
                    subject=subject,
                    session=session,
                    local_idx=int(i),
                    label=lab,
                    run_id=int(run_id_arr[i]),
                    npz_path=str(path.resolve()),
                )
            )
            gid += 1

    from .config import CLASS_NAMES

    return DatasetManifest(
        records=records,
        class_names=CLASS_NAMES,
        n_channels=n_ch or 22,
        freq_bins=n_f or 33,
        time_frames=n_t or 64,
        label_offset=label_offset,
    )

def save_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

def load_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

def _indices_by_subject(manifest: DatasetManifest) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in manifest.records:
        out.setdefault(r.subject, []).append(r.global_id)
    return out

def list_subjects(manifest: DatasetManifest, session: str | None = "T") -> list[str]:
    subs = sorted(_indices_by_subject(manifest).keys())
    if session is None:
        return subs
    return sorted(
        s
        for s in subs
        if any(manifest.records[i].session == session for i in _indices_by_subject(manifest)[s])
    )

def eval_subject_for_train(train_subject: str) -> str:
    """A01T -> A01E"""
    s = train_subject.replace("_tfwave", "").upper()
    if s.endswith("T"):
        return s[:-1] + "E"
    if s.endswith("E"):
        return s
    return s + "E"

def _records_for_subject(manifest: DatasetManifest, subject: str) -> list[TrialRecord]:
    subject = subject.replace("_tfwave", "")
    return [r for r in manifest.records if r.subject == subject]

def make_split_run_holdout(
    manifest: DatasetManifest,
    train_subject: str,
    *,
    test_run: int = 6,
    val_run: int = 5,
) -> dict[str, list[int]]:
    """
    单被试 T session：按 MI run 划分（默认 run 1–4 训、5 验、6 测）。
    不跨被试、不随机。
    """
    recs = _records_for_subject(manifest, train_subject)
    recs = [r for r in recs if r.session == "T"]
    if not recs:
        raise ValueError(f"无 T session 数据: {train_subject}")

    train_idx, val_idx, test_idx = [], [], []
    for r in recs:
        if r.run_id == test_run:
            test_idx.append(r.global_id)
        elif r.run_id == val_run:
            val_idx.append(r.global_id)
        elif r.run_id < test_run:
            train_idx.append(r.global_id)
        else:
            raise ValueError(f"未知 run_id={r.run_id}")

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"{train_subject} 划分为空: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"(test_run={test_run}, val_run={val_run})"
        )
    return {"train": train_idx, "val": val_idx, "test": test_idx}

def make_split_session_te(
    manifest: DatasetManifest,
    train_subject: str,
    *,
    val_run: int = 6,
) -> dict[str, list[int]]:
    """
    官方 session：T 上 run 1–5 训练、run 6 验证；E session 全部作测试。
    需已预处理 AxxE.npz（--mat-dir）。
    """
    train_subject = train_subject.replace("_tfwave", "")
    eval_subj = eval_subject_for_train(train_subject)

    train_idx, val_idx, test_idx = [], [], []
    for r in manifest.records:
        if r.subject == train_subject and r.session == "T":
            if r.run_id == val_run:
                val_idx.append(r.global_id)
            elif r.run_id < val_run:
                train_idx.append(r.global_id)
        elif r.subject == eval_subj and r.session == "E":
            test_idx.append(r.global_id)

    if not test_idx:
        raise FileNotFoundError(
            f"缺少评估 npz: {eval_subj}_tfwave.npz，请先 preprocess --all-eval --mat-dir ..."
        )
    if not train_idx or not val_idx:
        raise ValueError(f"{train_subject} T session 划分不完整")
    return {"train": train_idx, "val": val_idx, "test": test_idx}

def make_split_session_te_full(
    manifest: DatasetManifest,
    train_subject: str,
) -> dict[str, list[int]]:
    """T session 全部 trial 训练；E session 全部测试；无 val（对齐竞赛校准集全用）。"""
    train_subject = train_subject.replace("_tfwave", "")
    eval_subj = eval_subject_for_train(train_subject)

    train_idx, test_idx = [], []
    for r in manifest.records:
        if r.subject == train_subject and r.session == "T":
            train_idx.append(r.global_id)
        elif r.subject == eval_subj and r.session == "E":
            test_idx.append(r.global_id)

    if not test_idx:
        raise FileNotFoundError(f"缺少评估 npz: {eval_subj}_tfwave.npz")
    if not train_idx:
        raise ValueError(f"{train_subject} 无 T session 训练 trial")
    return {"train": train_idx, "val": [], "test": test_idx}

def compute_train_norm_stats(
    manifest: DatasetManifest,
    train_indices: list[int],
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """在该被试 train trial 上算 (C,F,T) 的 mean/std。"""
    chunks: list[np.ndarray] = []
    cache: dict[str, np.ndarray] = {}
    for gid in train_indices:
        rec = manifest.records[gid]
        if rec.npz_path not in cache:
            cache[rec.npz_path] = np.load(rec.npz_path, allow_pickle=True)["X"]
        chunks.append(cache[rec.npz_path][rec.local_idx].astype(np.float64))
    stack = np.stack(chunks, axis=0)
    mu = stack.mean(axis=0).astype(np.float32)
    std = (stack.std(axis=0) + eps).astype(np.float32)
    return mu, std

def apply_norm(arr: np.ndarray, mu: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mu is None or std is None:
        return arr
    return ((arr - mu) / std).astype(np.float32)

def make_split_by_subject(
    manifest: DatasetManifest,
    cfg: TrainConfig,
) -> dict[str, dict[str, list[int]]]:
    by_subject: dict[str, dict[str, list[int]]] = {}
    subjects = list_subjects(manifest, session="T")
    for subj in subjects:
        if cfg.split_name == "run_holdout":
            by_subject[subj] = make_split_run_holdout(
                manifest,
                subj,
                test_run=cfg.test_run,
                val_run=cfg.val_run,
            )
        elif cfg.split_name == "session_te":
            by_subject[subj] = make_split_session_te(
                manifest,
                subj,
                val_run=cfg.val_run,
            )
        elif cfg.split_name == "session_te_full":
            by_subject[subj] = make_split_session_te_full(manifest, subj)
        else:
            raise ValueError(
                f"不支持的划分: {cfg.split_name}。"
                "请使用 run_holdout、session_te 或 session_te_full；禁止随机 trial 与跨被试混合。"
            )
    return by_subject

def resolve_split_for_subject(
    split_payload: dict[str, Any],
    subject: str,
) -> dict[str, list[int]]:
    subject = subject.replace(".npz", "").replace("_tfwave", "")
    if not subject.upper().endswith("T"):
        subject = subject[:3] + "T" if len(subject) >= 3 else subject + "T"
    if "by_subject" in split_payload:
        if subject not in split_payload["by_subject"]:
            raise KeyError(f"划分中无被试 {subject}，可选: {list(split_payload['by_subject'])}")
        return split_payload["by_subject"][subject]
    return split_payload

def make_split(
    manifest: DatasetManifest,
    cfg: TrainConfig,
) -> dict[str, Any]:
    return {"by_subject": make_split_by_subject(manifest, cfg)}

def save_split(split: dict[str, Any], path: Path, meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "meta": meta or {}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_split(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["split"], d.get("meta", {})

class BCIC2aTFDataset(Dataset):
    """X (C,H,W)；模型输入标签为 0–3（由官方 1–4 减 1）。"""

    def __init__(
        self,
        manifest: DatasetManifest,
        indices: list[int],
        *,
        augment: bool = False,
        aug_cfg: TFAugmentConfig | None = None,
        seed: int = 42,
        norm_mu: np.ndarray | None = None,
        norm_std: np.ndarray | None = None,
    ):
        self.manifest = manifest
        self.indices = indices
        self.augment = augment
        self.aug_cfg = aug_cfg or TFAugmentConfig()
        self._rng = np.random.default_rng(seed)
        self.norm_mu = norm_mu
        self.norm_std = norm_std
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_subject(self, npz_path: str) -> tuple[np.ndarray, np.ndarray]:
        if npz_path not in self._cache:
            z = np.load(npz_path, allow_pickle=True)
            self._cache[npz_path] = (z["X"], z["y"])
        return self._cache[npz_path]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        rec = self.manifest.records[self.indices[i]]
        X, y_arr = self._load_subject(rec.npz_path)
        arr = X[rec.local_idx].astype(np.float32)
        arr = apply_norm(arr, self.norm_mu, self.norm_std)
        if self.augment:
            arr = apply_tf_augment(arr, self.aug_cfg, self._rng)
        x = torch.from_numpy(arr)
        label_official = rec.label
        label_model = label_official - manifest_label_offset(self.manifest)
        return x, label_model, rec.subject

def manifest_label_offset(manifest: DatasetManifest) -> int:
    return manifest.label_offset

def collate_batch(batch):
    xs, ys, subs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), list(subs)

def _aug_config_from_train(cfg: TrainConfig) -> TFAugmentConfig:
    return TFAugmentConfig(
        noise_std=cfg.aug_noise_std,
        time_mask_max=cfg.aug_time_mask_max,
        freq_mask_max=cfg.aug_freq_mask_max,
        channel_drop_p=cfg.aug_channel_drop_p,
        time_shift_max=cfg.aug_time_shift_max,
        scale_std=cfg.aug_scale_std,
    )

def build_dataloaders(
    manifest: DatasetManifest,
    split: dict[str, list[int]],
    batch_size: int,
    num_workers: int = 0,
    *,
    train_cfg: TrainConfig | None = None,
) -> dict[str, DataLoader]:
    loaders = {}
    aug_cfg = _aug_config_from_train(train_cfg) if train_cfg else TFAugmentConfig()
    use_aug = train_cfg.use_augment if train_cfg else False
    seed = train_cfg.seed if train_cfg else 42
    norm_mu, norm_std = None, None
    if train_cfg and train_cfg.norm_mode == "train" and split.get("train"):
        norm_mu, norm_std = compute_train_norm_stats(manifest, split["train"])

    for name, idx in split.items():
        if not idx:
            continue
        ds = BCIC2aTFDataset(
            manifest,
            idx,
            augment=(name == "train" and use_aug),
            aug_cfg=aug_cfg,
            seed=seed + (1 if name == "train" else 0),
            norm_mu=norm_mu,
            norm_std=norm_std,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )
    return loaders

def build_dataset_artifacts(cfg: TrainConfig) -> tuple[DatasetManifest, dict[str, Any], Path, Path]:
    manifest = build_manifest(cfg.data_dir, cfg.npz_glob)
    manifest_path = cfg.data_dir / "manifest.json"
    save_manifest(manifest, manifest_path)

    split = make_split(manifest, cfg)
    split_path = SPLITS_DIR / f"{cfg.split_name}.json"
    save_split(
        split,
        split_path,
        meta={
            "split_name": cfg.split_name,
            "test_run": cfg.test_run,
            "val_run": cfg.val_run,
            "subjects_T": list_subjects(manifest, session="T"),
            "subjects_E": list_subjects(manifest, session="E"),
            "label_offset": manifest.label_offset,
            "norm_mode": getattr(cfg, "norm_mode", "train"),
            "note": "单被试；按 run 或 session T/E；已剔除 1023；标签官方 1–4",
        },
    )
    return manifest, split, manifest_path, split_path
