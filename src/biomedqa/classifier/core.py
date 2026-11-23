"""Pure (CPU, dependency-light) parts of the classifier: inputs, outputs, selection, summary."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from biomedqa.data.pubmedqa import Example
from biomedqa.hashing import config_hash
from biomedqa.labels import ID_TO_LABEL, LABEL_TO_ID, LABELS
from biomedqa.schema import PARSE_OK, Prediction


def input_pair(example: Example, use_long_answer: bool = False) -> tuple[str, str]:
    """(question, context) text pair. ``long_answer`` is refused: it leaks the answer."""
    if use_long_answer:
        raise ValueError("use_long_answer must stay false (protocol section 3, ADR 0004)")
    return example.question, example.context_text


def label_ids(examples: Sequence[Example]) -> list[int]:
    return [LABEL_TO_ID[e.label] for e in examples]


def run_config_hash(cfg: Mapping[str, Any], seed: int) -> str:
    """Hash of everything that defines one training run (the seed list is replaced by the seed)."""
    run = {k: v for k, v in cfg.items() if k != "seeds"}
    run["seed"] = seed
    return config_hash(run)


def probs_to_predictions(
    examples: Sequence[Example],
    probs: Sequence[Sequence[float]],
    *,
    system: str,
    cfg_hash: str,
    extra: Mapping[str, Any] | None = None,
) -> list[Prediction]:
    """Argmax over (yes, no, maybe) probabilities. Classifier outputs never fail to parse."""
    if len(examples) != len(probs):
        raise ValueError("examples and probs differ in length")
    out = []
    for ex, pr in zip(examples, probs, strict=True):
        if len(pr) != len(LABELS):
            raise ValueError(f"expected {len(LABELS)} probabilities, got {len(pr)}")
        best = max(range(len(LABELS)), key=lambda i: (pr[i], -i))  # ties -> label order
        label = ID_TO_LABEL[best]
        out.append(
            Prediction(
                id=ex.pmid,
                gold=ex.label,
                predicted=label,
                raw_output=json.dumps(
                    {lab: round(float(p), 6) for lab, p in zip(LABELS, pr, strict=True)}
                ),
                parse_status=PARSE_OK,
                system=system,
                config_hash=cfg_hash,
                extra={"probs": [float(p) for p in pr], **(extra or {})},
            )
        )
    return out


def select_best_epoch(history: Sequence[Mapping[str, Any]], metric: str = "macro_f1") -> int:
    """Index of the epoch with the highest dev metric; ties go to the earlier epoch."""
    if not history:
        raise ValueError("empty history")
    best = 0
    for i, h in enumerate(history):
        if h[metric] > history[best][metric]:
            best = i
    return best


def summarize_seeds(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and sample std (ddof=1) of dev accuracy and macro-F1 over seeds."""
    if not per_seed:
        raise ValueError("no seeds")
    seeds = sorted(per_seed)
    out: dict[str, Any] = {"seeds": seeds, "n_seeds": len(seeds), "per_seed": {}}
    for m in ("accuracy", "macro_f1"):
        vals = [float(per_seed[s][m]) for s in seeds]
        out[m] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    for s in seeds:
        out["per_seed"][str(s)] = {
            "accuracy": per_seed[s]["accuracy"],
            "macro_f1": per_seed[s]["macro_f1"],
            "best_epoch": per_seed[s].get("best_epoch"),
            "n_parse_failures": per_seed[s].get("n_parse_failures", 0),
        }
    return out
