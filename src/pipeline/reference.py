from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "benchmark" / "end_to_end" / "run_final_pipeline.py"

_module: ModuleType | None = None


def e2e() -> ModuleType:
    global _module
    if _module is not None:
        return _module
    spec = importlib.util.spec_from_file_location("absa_final_e2e_reference", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference e2e module from {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _module = module
    return module
