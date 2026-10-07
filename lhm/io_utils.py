"""Serialisation helpers for LHM artefacts.

All artefacts (Hessian traces, LSOS scores, DCCM results, run metadata) are
plain JSON so that they can be inspected, diffed and re-used across
experiments without special tooling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import LHMConfig
from .dccm import DCCMResult
from .hessian import HessianTraceResult
from .lsos import LSOSResult


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(payload: Mapping[str, Any], path: str | os.PathLike) -> str:
    path = str(path)
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: LHMConfig, path: str | os.PathLike) -> str:
    return save_json(config.to_dict(), path)


def load_config(path: str | os.PathLike) -> LHMConfig:
    return LHMConfig.from_dict(load_json(path))


def save_hessian_traces(result: HessianTraceResult, path: str | os.PathLike) -> str:
    return save_json(result.to_dict(), path)


def load_hessian_traces(path: str | os.PathLike) -> HessianTraceResult:
    return HessianTraceResult.from_dict(load_json(path))


def save_lsos(result: LSOSResult, path: str | os.PathLike) -> str:
    return save_json(result.to_dict(), path)


def load_lsos(path: str | os.PathLike) -> LSOSResult:
    return LSOSResult.from_dict(load_json(path))


def save_dccm(result: DCCMResult, path: str | os.PathLike) -> str:
    return save_json(result.to_dict(), path)


def load_dccm(path: str | os.PathLike) -> DCCMResult:
    return DCCMResult.from_dict(load_json(path))


def run_directory(
    output_root: str | os.PathLike,
    scale: str,
    method: str,
    lambda_bar: float,
    alpha: float,
    rank: int,
    tag: Optional[str] = None,
) -> str:
    """Deterministic run directory encoding the key hyper-parameters."""
    name = f"{scale}_{method}_lam{lambda_bar:.2f}_alpha{alpha:.2f}_k{rank}"
    if tag:
        name = f"{name}_{tag}"
    return ensure_dir(os.path.join(str(output_root), name))
