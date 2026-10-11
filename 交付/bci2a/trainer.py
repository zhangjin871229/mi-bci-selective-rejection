"""训练与评估循环。"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import CHECKPOINT_DIR, CLASS_NAMES, TrainConfig
from .dataset import DatasetManifest, build_dataloaders, compute_train_norm_stats
from .gmm_head import extract_features, make_gmm_head
from .metrics import compute_metrics, format_metrics_line, save_metrics
from .models import ChannelViT

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total, n = 0.0, 0
    preds, labels = [], []
    criterion = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item() * len(y)
        n += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    if n == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    return total / n, sum(p == l for p, l in zip(np.concatenate(preds), np.concatenate(labels))) / n, np.concatenate(labels), np.concatenate(preds)

def run_train(
    cfg: TrainConfig,
    manifest: DatasetManifest,
    split: dict[str, list[int]],
    run_name: str = "default",
) -> Path:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.freq_bins = manifest.freq_bins
    cfg.time_frames = manifest.time_frames
    loaders = build_dataloaders(manifest, split, cfg.batch_size, cfg.num_workers, train_cfg=cfg)

    model = ChannelViT(cfg, n_channels=manifest.n_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_vit.pt"
    best_val_acc = -1.0
    best_train_loss = float("inf")
    history = []
    has_val = bool(split.get("val")) and "val" in loaders

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        if has_val:
            val_loss, val_acc, _, _ = evaluate(model, loaders["val"], device)
            score = val_acc
            metric_name = "val_acc"
        else:
            val_loss, val_acc = tr_loss, 0.0
            score = -tr_loss
            metric_name = "train_loss"
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(
            f"Epoch {epoch}/{cfg.epochs}  train_loss={tr_loss:.4f}  "
            f"{'val' if has_val else 'train'}_loss={val_loss:.4f}  "
            f"{'val_acc' if has_val else 'train_loss_sel'}="
            f"{val_acc if has_val else tr_loss:.4f}"
        )
        improved = (score > best_val_acc) if has_val else (tr_loss < best_train_loss)
        if improved:
            if has_val:
                best_val_acc = score
            else:
                best_train_loss = tr_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    # 测试集（ViT softmax）
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "test" in loaders:
        _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], device)
        metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
        metrics["head"] = "vit_softmax"
        save_metrics(metrics, run_dir / "test_metrics_vit.json")
        print(f"[ViT] Test {format_metrics_line(metrics)}")

    norm_mu, norm_std = None, None
    if cfg.norm_mode == "train" and split.get("train"):
        norm_mu, norm_std = compute_train_norm_stats(manifest, split["train"])

    # 可选 GMM 头
    if cfg.use_gmm:
        train_idx = split["train"] + split.get("val", [])
        X_tr, y_tr = extract_features(
            model, manifest, train_idx, device, cfg.batch_size,
            norm_mu=norm_mu, norm_std=norm_std,
        )
        gmm = make_gmm_head(cfg)
        gmm.fit(X_tr, y_tr)
        if gmm.reducer is not None:
            print(f"[GMM] feature reduce: {cfg.gmm_reduce} -> dim {gmm.reducer.n_components_}")
        gmm.save(run_dir / "gmm.pkl")
        if "test" in split and split["test"]:
            X_te, y_te = extract_features(
                model, manifest, split["test"], device, cfg.batch_size,
                norm_mu=norm_mu, norm_std=norm_std,
            )
            y_hat = gmm.predict(X_te)
            metrics_g = compute_metrics(y_te, y_hat, CLASS_NAMES)
            metrics_g["head"] = "vit_gmm"
            save_metrics(metrics_g, run_dir / "test_metrics_gmm.json")
            print(f"[GMM] Test {format_metrics_line(metrics_g)}")

    return best_path
