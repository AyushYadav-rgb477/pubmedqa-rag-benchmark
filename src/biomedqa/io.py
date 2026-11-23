"""Prediction files: JSONL, one :class:`Prediction` per line, sorted by question id."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from biomedqa.fileio import atomic_write_text
from biomedqa.hashing import sort_ids
from biomedqa.schema import Prediction


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    """Write predictions atomically, sorted by id. Duplicate ids raise ``ValueError``."""
    preds = list(predictions)
    by_id = {p.id: p for p in preds}
    if len(by_id) != len(preds):
        raise ValueError("duplicate prediction ids")
    lines = [
        json.dumps(by_id[i].to_dict(), ensure_ascii=False, sort_keys=True) for i in sort_ids(by_id)
    ]
    atomic_write_text(path, "".join(line + "\n" for line in lines))


def read_predictions(path: str | Path) -> list[Prediction]:
    preds: list[Prediction] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                preds.append(Prediction.from_dict(json.loads(line)))
            except (ValueError, KeyError) as e:
                raise ValueError(f"{path}:{n}: bad prediction record: {e}") from e
    if len({p.id for p in preds}) != len(preds):
        raise ValueError(f"{path}: duplicate prediction ids")
    return preds
