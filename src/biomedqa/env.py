"""Process environment setup and version recording for run files."""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from typing import Any

# bm25s imports JAX when it is installed; JAX then preallocates ~75% of GPU memory and the LLM
# runs out of memory. These must be set before JAX is imported anywhere.
SAFE_ENV: dict[str, str] = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TOKENIZERS_PARALLELISM": "false",
}

DEFAULT_PACKAGES = (
    "biomedqa",
    "numpy",
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "scikit-learn",
    "bm25s",
    "PyStemmer",
)


def configure_safe_env() -> None:
    """Set GPU-safety environment variables. Call before importing heavy libraries.

    Values already set by the user are kept, except the JAX ones, which are forced because a
    wrong value silently breaks LLM runs.
    """
    for key, value in SAFE_ENV.items():
        if key.startswith(("JAX_", "XLA_")):
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def collect_versions(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, Any]:
    """Versions of Python, the platform and installed packages (missing ones are skipped)."""
    out: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    pkgs: dict[str, str] = {}
    for name in packages:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    out["packages"] = pkgs
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        out["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    return out
