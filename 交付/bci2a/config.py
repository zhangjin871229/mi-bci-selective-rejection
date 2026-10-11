"""默认路径与超参数。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 工程根目录 DL/
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
TRUE_LABELS_DIR = PROJECT_ROOT / "true_labels"
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
SPLITS_DIR = DATA_DIR / "splits"

# 与官方 classlabel 1–4 对应
CLASS_NAMES = ["1-左手", "2-右手", "3-双脚", "4-舌头"]
N_CLASSES = 4
N_CHANNELS = 22
FREQ_BINS = 33
TIME_FRAMES = 64

@dataclass
class TrainConfig:
    # 数据
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    npz_glob: str = "*_tfwave.npz"
    # run_holdout: T session 按 run 1–4/5/6 划分
    # session_te: T 训+验，E 测（需 AxxE npz）
    split_name: str = "session_te_full"
    subject: str = "A01T"
    test_run: int = 6
    val_run: int = 5
    # 模型
    vit_model: str = "vit_tiny_patch16_224"
    vit_depth: int = 6  # 方案：6 层 Transformer
    vit_num_heads: int = 8  # 方案：8 头注意力
    pretrained: bool = False  # 改 depth 后预训练权重一般不匹配，慎用 --pretrained
    embed_dim: int = 192
    dropout: float = 0.1
    channel_pool: str = "mean"  # mean | attn
    # 训练
    epochs: int = 100  # 方案：100 轮
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    # GMM（训练 ViT 后可拟合）
    gmm_components_per_class: int = 4  # 方案 3–5，默认 4；可 --gmm-components 3 或 5
    gmm_reduce: str = "pca"  # none | pca | lda，ViT 特征进 GMM 前降维
    gmm_reduce_dim: int = 32
    gmm_covariance: str = "auto"  # auto | full | diag
    use_gmm: bool = False
    # 归一化：train=仅用该被试 train trial 统计量；trial/none=直接用 npz
    norm_mode: str = "train"
    # 训练增强（仅 train loader）
    use_augment: bool = True
    # 时频图尺寸（默认与单带 0–4s 一致；filter-bank 预设由 manifest 覆盖）
    freq_bins: int = FREQ_BINS
    time_frames: int = TIME_FRAMES
    aug_noise_std: float = 0.12
    aug_time_mask_max: int = 10
    aug_freq_mask_max: int = 6
    aug_channel_drop_p: float = 0.12
    aug_time_shift_max: int = 6
    aug_scale_std: float = 0.08

    def img_size(self) -> tuple[int, int]:
        """timm ViT 的 (H, W) = (频率, 时间)."""
        return (self.freq_bins, self.time_frames)
