"""Paths and protocol helpers for the BCI IV 2b validation (protocol_2b.json).

Reuses the frozen 2a revision modules in ../../rv unchanged; only the 2b-specific adaptations live here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # revision/bci2b/rv2b
B2 = HERE.parent                                       # revision/bci2b
REV = B2.parent                                        # revision
for p in (HERE, REV / "rv"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common  # noqa: E402,F401  (2a helpers; adds legacy package to sys.path)

import numpy as np  # noqa: E402

DATA = REV.parent / "BCICIV_2b_bnci"
CACHE = B2 / "cache"
MODELS = B2 / "models"
OUT = B2 / "out"
LOGS = B2 / "logs"
SUBJECTS = [f"B0{i}" for i in range(1, 10)]
N_CH, N_CLS = 3, 2
SFREQ = 250.0
BASE_SEED = 2026

# bspc-r1-v1 member split with run k -> D block k
MEMBERS = {
    1: {"A": [3, 4, 5], "A_fit": [3, 4], "A_val": [5], "B": [1, 2]},
    2: {"A": [1, 4, 5], "A_fit": [1, 4], "A_val": [5], "B": [2, 3]},
    3: {"A": [1, 2, 5], "A_fit": [1, 2], "A_val": [5], "B": [3, 4]},
    4: {"A": [1, 2, 3], "A_fit": [1, 2], "A_val": [3], "B": [4, 5]},
    5: {"A": [2, 3, 4], "A_fit": [2, 3], "A_val": [4], "B": [5, 1]},
}
C_BLOCK = 6


def member_seed(subject: str, member: int) -> int:
    return BASE_SEED + 1000 * int(subject[1:]) + 10 * member


def protocol_path() -> Path:
    return B2 / "protocol_2b.json"


def protocol_sha256() -> str:
    return hashlib.sha256(protocol_path().read_bytes()).hexdigest()


def check_frozen() -> dict:
    """Refuse to continue unless the protocol is frozen and unchanged since freezing."""
    P = json.loads(protocol_path().read_text(encoding="utf-8"))
    if not P["status"].startswith("FROZEN"):
        raise SystemExit("protocol_2b.json is not frozen")
    frozen = (B2 / "protocol_2b.sha256").read_text().split()[0]
    if protocol_sha256() != frozen:
        raise SystemExit("protocol_2b.json changed after freezing")
    return P


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
