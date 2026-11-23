"""Run a system on one split and save predictions, metrics and a run record.

Writes ``<out_dir>/<split>_predictions.jsonl``, ``<split>_metrics.json`` and ``<split>_run.json``.
The test split needs ``allow_test=True`` (Phase 4 only).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from biomedqa.data.pubmedqa import Example
from biomedqa.data.splits import DEFAULT_SPLITS_FILE
from biomedqa.env import collect_versions
from biomedqa.fileio import write_json
from biomedqa.guard import check_split_access
from biomedqa.hashing import config_hash, sha256_file
from biomedqa.io import write_predictions
from biomedqa.llm.client import LLMClient
from biomedqa.prompts import v1
from biomedqa.schema import Prediction
from biomedqa.scoring.report import score_split
from biomedqa.systems.llm import EVIDENCE, EvidenceFn, llm_stats, run_system


def llm_run_hash(llm_cfg: Mapping[str, Any], system: str) -> str:
    """Everything that defines an LLM run's outputs (the cache location does not)."""
    cfg = {k: v for k, v in llm_cfg.items() if k != "cache_dir"}
    return config_hash({"llm": cfg, "system": system, "prompt_template": v1.template_hash()})


def save_run(
    preds: Sequence[Prediction],
    split: str,
    out_dir: str | Path,
    run_info: Mapping[str, Any],
    *,
    splits_file: str | Path = DEFAULT_SPLITS_FILE,
    allow_test: bool = False,
    extra_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    pred_path = out / f"{split}_predictions.jsonl"
    write_predictions(pred_path, preds)
    report = score_split(preds, split, splits_file, allow_test=allow_test)
    report["predictions_file"] = pred_path.name
    report["predictions_sha256"] = sha256_file(pred_path)
    if extra_metrics:
        report.update(extra_metrics)
    write_json(out / f"{split}_run.json", {**run_info, "versions": collect_versions()})
    write_json(out / f"{split}_metrics.json", report)  # last: marks the run as complete
    return report


def run_llm_split(
    system: str,
    split: str,
    examples: Sequence[Example],
    client: Any,
    llm_cfg: Mapping[str, Any],
    out_dir: str | Path,
    *,
    splits_file: str | Path = DEFAULT_SPLITS_FILE,
    allow_test: bool = False,
    evidence: EvidenceFn | None = None,
    log_every: int = 10,
) -> dict[str, Any]:
    check_split_access(split, allow_test)
    evidence_fn = evidence or EVIDENCE[system]
    h = llm_run_hash(llm_cfg, system)
    start = time.time()

    def progress(i: int, n: int) -> None:
        if i % log_every == 0 or i == n:
            hits = getattr(client, "hits", "?")
            print(f"{system} {split}: {i}/{n} ({time.time() - start:.0f}s, cache hits {hits})")

    preds = run_system(examples, client, evidence_fn, system=system, cfg_hash=h, progress=progress)
    inner: LLMClient = getattr(client, "inner", client)
    run_info = {
        "system": system,
        "split": split,
        "config": dict(llm_cfg),
        "config_hash": h,
        "prompt_version": v1.PROMPT_VERSION,
        "prompt_template_sha256": v1.template_hash(),
        "model_identity": client.identity,
        "generation": client.generation.to_dict(),
        "effective_generation": getattr(inner, "effective_generation", None),
        "device_map": str(getattr(inner, "device_map", None)),
        "cache_hits": getattr(client, "hits", None),
        "cache_misses": getattr(client, "misses", None),
        "seconds": round(time.time() - start, 1),
    }
    return save_run(
        preds,
        split,
        out_dir,
        run_info,
        splits_file=splits_file,
        allow_test=allow_test,
        extra_metrics={"llm": llm_stats(preds)},
    )
