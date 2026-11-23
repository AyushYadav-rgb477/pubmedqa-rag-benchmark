"""Majority-class baseline (protocol section 4.6): the most frequent train label, always."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from biomedqa.data.pubmedqa import Example
from biomedqa.labels import LABELS
from biomedqa.schema import PARSE_OK, Prediction

SYSTEM = "majority"


def majority_label(train: Sequence[Example]) -> str:
    if not train:
        raise ValueError("empty train set")
    counts = Counter(e.label for e in train)
    return max(LABELS, key=lambda lab: (counts.get(lab, 0), -LABELS.index(lab)))


def predict(examples: Sequence[Example], label: str, cfg_hash: str) -> list[Prediction]:
    return [
        Prediction(
            id=e.pmid,
            gold=e.label,
            predicted=label,
            raw_output=label,
            parse_status=PARSE_OK,
            system=SYSTEM,
            config_hash=cfg_hash,
        )
        for e in examples
    ]
