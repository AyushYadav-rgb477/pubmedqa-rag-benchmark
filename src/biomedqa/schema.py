"""One prediction record format shared by every system (classifier, closed-book, oracle, RAG)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from biomedqa.labels import INVALID, check_gold, check_prediction

PARSE_OK = "ok"
PARSE_FAILED = "failed"
PARSE_STATUSES = (PARSE_OK, PARSE_FAILED)


@dataclass(frozen=True)
class Prediction:
    """One system's answer to one question.

    ``predicted`` is ``"invalid"`` exactly when ``parse_status`` is ``"failed"``.
    ``evidence_ids`` are the PMIDs shown to the model, in the order they were numbered [1], [2], ...
    ``citations`` are the 1-based document numbers the model cited.
    """

    id: str
    gold: str
    predicted: str
    raw_output: str
    parse_status: str
    system: str
    config_hash: str
    evidence_ids: tuple[str, ...] = ()
    citations: tuple[int, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_gold(self.gold)
        check_prediction(self.predicted)
        if self.parse_status not in PARSE_STATUSES:
            raise ValueError(f"parse_status must be one of {PARSE_STATUSES}")
        if (self.predicted == INVALID) != (self.parse_status == PARSE_FAILED):
            raise ValueError("predicted is 'invalid' if and only if parse_status is 'failed'")
        if not self.id or not self.system or not self.config_hash:
            raise ValueError("id, system and config_hash must be non-empty")
        # normalise sequence types so records round-trip through JSON unchanged
        object.__setattr__(self, "evidence_ids", tuple(str(e) for e in self.evidence_ids))
        object.__setattr__(self, "citations", tuple(int(c) for c in self.citations))
        object.__setattr__(self, "extra", dict(self.extra))

    @property
    def correct(self) -> bool:
        return self.predicted == self.gold

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gold": self.gold,
            "predicted": self.predicted,
            "raw_output": self.raw_output,
            "parse_status": self.parse_status,
            "evidence_ids": list(self.evidence_ids),
            "citations": list(self.citations),
            "system": self.system,
            "config_hash": self.config_hash,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Prediction:
        return cls(
            id=str(d["id"]),
            gold=d["gold"],
            predicted=d["predicted"],
            raw_output=d.get("raw_output", ""),
            parse_status=d["parse_status"],
            system=d["system"],
            config_hash=d["config_hash"],
            evidence_ids=tuple(d.get("evidence_ids", ())),
            citations=tuple(d.get("citations", ())),
            extra=d.get("extra", {}),
        )
