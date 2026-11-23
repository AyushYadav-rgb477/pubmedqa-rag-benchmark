"""Score a prediction file against a split, with the test-set lock and completeness checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biomedqa.data.splits import DEFAULT_SPLITS_FILE, read_splits
from biomedqa.guard import check_ids_not_test, check_split_access
from biomedqa.hashing import sha256_file
from biomedqa.io import read_predictions
from biomedqa.schema import Prediction
from biomedqa.scoring.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci
from biomedqa.scoring.metrics import score


def score_records(
    preds: Sequence[Prediction],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    golds = [p.gold for p in preds]
    predicted = [p.predicted for p in preds]
    return {
        **score(golds, predicted),
        "bootstrap": bootstrap_ci(golds, predicted, n_resamples=n_resamples, seed=seed),
    }


def score_split(
    preds: Sequence[Prediction],
    split: str,
    splits_file: str | Path = DEFAULT_SPLITS_FILE,
    *,
    allow_test: bool = False,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score predictions for exactly the questions of ``split``.

    Raises if the test lock is violated, or if any question of the split is missing or extra:
    missing predictions are never silently dropped.
    """
    check_split_access(split, allow_test)
    splits = read_splits(splits_file)
    ids = [p.id for p in preds]
    check_ids_not_test(ids, splits.test, allow_test)
    expected = set(splits.ids(split))
    got = set(ids)
    if got != expected:
        raise ValueError(
            f"predictions do not cover split {split!r} exactly: "
            f"{len(expected - got)} missing, {len(got - expected)} extra"
        )
    systems = sorted({p.system for p in preds})
    hashes = sorted({p.config_hash for p in preds})
    if len(systems) != 1 or len(hashes) != 1:
        raise ValueError(f"mixed systems {systems} or config hashes {hashes} in one file")
    return {
        "system": systems[0],
        "config_hash": hashes[0],
        "split": split,
        **score_records(preds, n_resamples=n_resamples, seed=seed),
    }


def score_file(path: str | Path, split: str, **kwargs: Any) -> dict[str, Any]:
    report = score_split(read_predictions(path), split, **kwargs)
    report["predictions_file"] = Path(path).as_posix()
    report["predictions_sha256"] = sha256_file(path)
    return report
