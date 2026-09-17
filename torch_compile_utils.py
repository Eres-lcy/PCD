"""Persistent ``torch.compile`` cache configuration used by Pi0."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys


def _safe_tag(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


def configure_torch_compile_cache(pcd_root: str | Path | None = None) -> Path:
    """Put Inductor/Triton artifacts in a persistent, runtime-specific cache."""
    import torch

    configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if configured:
        cache_dir = Path(configured).expanduser()
    else:
        if pcd_root is None:
            pcd_root = Path(__file__).resolve().parent
        pcd_root = Path(pcd_root)
        checkpoint_root = Path(
            os.environ.get(
                "PCD_CHECKPOINT_ROOT",
                str(pcd_root / "checkpoint"),
            )
        ).expanduser()

        runtime_parts = [
            f"py{sys.version_info.major}.{sys.version_info.minor}",
            f"torch-{_safe_tag(torch.__version__)}",
        ]
        if torch.cuda.is_available():
            cuda_version = _safe_tag(torch.version.cuda or "unknown")
            major, minor = torch.cuda.get_device_capability()
            runtime_parts.extend((f"cuda-{cuda_version}", f"sm-{major}{minor}"))
        else:
            runtime_parts.append("cpu")

        cache_dir = (
            checkpoint_root
            / "torch_compile_cache"
            / "_".join(runtime_parts)
        )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
