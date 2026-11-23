"""Point metrics. Parse failures (``invalid``) always count as wrong; they are never dropped.

* accuracy: ``invalid`` is wrong.
* macro-F1: mean F1 over {yes, no, maybe} with ``zero_division=0``. An ``invalid`` prediction is a
  false negative for its gold class and a false positive for no class.
* confusion matrix: gold rows (yes, no, maybe) x predicted columns (yes, no, maybe, invalid).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from biomedqa.labels import INVALID, LABELS, PREDICTION_VALUES, check_gold, check_prediction

N_GOLD = len(LABELS)
N_PRED = len(PREDICTION_VALUES)
GOLD_INDEX = {lab: i for i, lab in enumerate(LABELS)}
PRED_INDEX = {lab: i for i, lab in enumerate(PREDICTION_VALUES)}


def encode(golds: Sequence[str], preds: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(golds) != len(preds):
        raise ValueError(f"{len(golds)} golds vs {len(preds)} predictions")
    if not golds:
        raise ValueError("cannot score an empty set")
    g = np.array([GOLD_INDEX[check_gold(x)] for x in golds], dtype=np.int64)
    p = np.array([PRED_INDEX[check_prediction(x)] for x in preds], dtype=np.int64)
    return g, p


def confusion_from_codes(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(g * N_PRED + p, minlength=N_GOLD * N_PRED).reshape(N_GOLD, N_PRED)


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    """F1 for each real label from confusion matrices of shape (..., 3, 4).

    F1 = 2TP / (2TP + FP + FN), defined as 0 when the denominator is 0. This equals
    2PR / (P + R) with ``zero_division=0`` (scikit-learn semantics).
    """
    tp = np.diagonal(cm[..., :, :N_GOLD], axis1=-2, axis2=-1)
    fp = cm[..., :, :N_GOLD].sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def score(golds: Sequence[str], preds: Sequence[str]) -> dict[str, Any]:
    g, p = encode(golds, preds)
    cm = confusion_from_codes(g, p)
    n = int(cm.sum())
    tp = np.diag(cm[:, :N_GOLD])
    f1 = f1_per_class(cm)
    per_class = {}
    for i, lab in enumerate(LABELS):
        predicted_as = int(cm[:, i].sum())
        support = int(cm[i].sum())
        per_class[lab] = {
            "precision": _safe_div(tp[i], predicted_as),
            "recall": _safe_div(tp[i], support),
            "f1": float(f1[i]),
            "support": support,
            "predicted": predicted_as,
        }
    return {
        "n": n,
        "accuracy": float(tp.sum() / n),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion": {
            "rows_gold": list(LABELS),
            "cols_pred": list(PREDICTION_VALUES),
            "matrix": cm.tolist(),
        },
        "n_parse_failures": int(cm[:, PRED_INDEX[INVALID]].sum()),
    }
