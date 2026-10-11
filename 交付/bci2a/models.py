"""22 导时频图：共享 ViT + 通道聚合 + 分类 / 特征提取。"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .config import N_CLASSES, TrainConfig

class ChannelAttentionPool(nn.Module):
    """对 22 路 ViT 特征做可学习加权聚合。"""

    def __init__(self, dim: int, n_channels: int = 22):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)  # (B, C)
        return (x * w.unsqueeze(-1)).sum(dim=1)

class ChannelViT(nn.Module):
    """
    输入 (B, C, H, W)，每通道一张时频图，共享 ViT，再聚合为 trial 向量。
    """

    def __init__(self, cfg: TrainConfig, n_classes: int = N_CLASSES, n_channels: int = 22):
        super().__init__()
        self.cfg = cfg
        self.n_channels = n_channels
        img_size = cfg.img_size()
        self.backbone = timm.create_model(
            cfg.vit_model,
            pretrained=cfg.pretrained,
            img_size=img_size,
            in_chans=1,
            num_classes=0,
            drop_rate=cfg.dropout,
            depth=cfg.vit_depth,
            num_heads=cfg.vit_num_heads,
        )
        feat_dim = self.backbone.num_features
        if cfg.channel_pool == "attn":
            self.pool = ChannelAttentionPool(feat_dim, n_channels=n_channels)
        else:
            self.pool = None
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, D)"""
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        feat = self.backbone(x)  # (B*C, D)
        feat = feat.view(b, c, -1)
        if self.pool is not None:
            return self.pool(feat)
        return feat.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.dropout(z)
        return self.classifier(z)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
