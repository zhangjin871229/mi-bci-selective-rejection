"""Shared paths, protocol loading and small utilities for the BSPC revision pipeline."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parent.parent          # BCI/revision
BCI = REV.parent                                       # BCI
LEGACY = BCI / "交付"                                   # original delivery (read-only)
GDF_DIR = BCI / "BCICIV_2a_gdf"
LABEL_DIR = BCI / "true_labels"
CACHE = REV / "cache"
OUT = REV / "out"
LOGS = REV / "logs"
MODELS = REV / "models"

SUBJECTS = [f"A0{i}" for i in range(1, 10)]
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
SFREQ = 250.0

if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))


def load_protocol() -> dict:
    return json.loads((REV / "protocol.json").read_text(encoding="utf-8"))


def protocol_hash() -> str:
    return hashlib.sha256((REV / "protocol.json").read_bytes()).hexdigest()[:12]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def member_seed(subject: str, member: int, base: int) -> int:
    return base + 1000 * int(subject[1:]) + 10 * member


def member_runs(protocol: dict, member: int) -> dict:
    m = protocol["split"]["members"][str(member)]
    return {"A": m["A"], "A_fit": m["A_fit"], "A_val": m["A_val"], "B": m["B"]}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
