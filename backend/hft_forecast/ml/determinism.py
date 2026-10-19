"""Seeding and the environment record (§6.4).

Reproducibility is claimed per hardware class, not across machines: results are reproducible
on the environment recorded in ``environment.json``. Bit-exact agreement between CPU and GPU
is not achievable and is not claimed.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

import numpy as np
import structlog

log = structlog.get_logger(__name__)

RECORDED_PACKAGES: Final = ("polars", "numpy", "scikit-learn", "torch", "pyarrow", "pandas")
HASH_SEED_ENV: Final = "PYTHONHASHSEED"


def seed_everything(seed: int, threads: int | None = None) -> None:
    """Seed ``random``, NumPy's global generator and torch, and pin the thread counts.

    Every generator inside this package is created explicitly with its own seed; the global
    seeds are here for third-party code that does not take one.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if os.environ.get(HASH_SEED_ENV) is None:
        log.warning("pythonhashseed_unset", hint="export PYTHONHASHSEED=0 for exact repeats")
    if threads:
        os.environ.setdefault("POLARS_MAX_THREADS", str(threads))
    _seed_torch(seed, threads)


def _seed_torch(seed: int, threads: int | None) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if threads:
        torch.set_num_threads(threads)
    # on a GPU this also needs CUBLAS_WORKSPACE_CONFIG=:4096:8 and cudnn.deterministic


def environment(seed: int) -> dict[str, Any]:
    """Everything needed to judge whether a result can be reproduced."""
    return {
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {name: _version(name) for name in RECORDED_PACKAGES},
        "env": {
            key: os.environ.get(key)
            for key in (HASH_SEED_ENV, "POLARS_MAX_THREADS", "HFTF_MAX_WORKERS")
        },
    }


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
