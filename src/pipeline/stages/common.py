from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def apply_random_seeds(seed_config: dict[str, Any]) -> None:
    py_seed = int(seed_config.get("python", 42))
    np_seed = int(seed_config.get("numpy", py_seed))
    torch_seed = int(seed_config.get("torch", py_seed))
    random.seed(py_seed)
    np.random.seed(np_seed)
    try:
        import torch

        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
