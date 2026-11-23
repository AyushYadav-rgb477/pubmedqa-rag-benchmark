"""Percentile bootstrap confidence intervals over questions (vectorised)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from biomedqa.scoring.metrics import N_GOLD, N_PRED, encode, f1_per_class

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42
_BATCH = 1_000


def resampled_metrics(
    g: np.ndarray, p: np.ndarray, n_resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy and macro-F1 for ``n_resamples`` bootstrap resamples of the questions."""
    n = len(g)
    codes = g * N_PRED + p
    rng = np.random.default_rng(seed)
    accs: list[np.ndarray] = []
    f1s: list[np.ndarray] = []
    done = 0
    while done < n_resamples:
        b = min(_BATCH, n_resamples - done)
        idx = rng.integers(0, n, size=(b, n))
        flat = codes[idx] + (np.arange(b)[:, None] * N_GOLD * N_PRED)
        cms = np.bincount(flat.ravel(), minlength=b * N_GOLD * N_PRED).reshape(b, N_GOLD, N_PRED)
        accs.append(np.trace(cms[:, :, :N_GOLD], axis1=1, axis2=2) / n)
        f1s.append(f1_per_class(cms).mean(axis=1))
        done += b
    return np.concatenate(accs), np.concatenate(f1s)


def bootstrap_ci(
    golds: Sequence[str],
    preds: Sequence[str],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap CI for accuracy and macro-F1."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    g, p = encode(golds, preds)
    accs, f1s = resampled_metrics(g, p, n_resamples, seed)
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2

    def ci(values: np.ndarray) -> dict[str, float]:
        lo, hi = np.quantile(values, [lo_q, hi_q])
        return {"low": float(lo), "high": float(hi)}

    return {
        "method": "percentile bootstrap over questions",
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence": confidence,
        "accuracy": ci(accs),
        "macro_f1": ci(f1s),
    }
