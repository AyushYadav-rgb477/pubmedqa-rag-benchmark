"""LLM systems: the same prompt v1, only the evidence differs (ADR 0006).

An evidence function maps an example to (evidence PMIDs, evidence texts). Phase 3 adds RAG by
registering an evidence function built from the retriever.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from biomedqa.data.pubmedqa import Example
from biomedqa.llm.client import LLMClient
from biomedqa.parsing import parse_response
from biomedqa.prompts import v1
from biomedqa.schema import Prediction

Evidence = tuple[list[str], list[str]]
EvidenceFn = Callable[[Example], Evidence]


def closed_book(example: Example) -> Evidence:
    return [], []


def oracle(example: Example) -> Evidence:
    """The question's own abstract context (conclusion excluded), as document [1]."""
    return [example.pmid], [example.context_text]


EVIDENCE: dict[str, EvidenceFn] = {"closed_book": closed_book, "oracle": oracle}


def answer_one(
    example: Example,
    client: LLMClient,
    evidence: EvidenceFn,
    *,
    system: str,
    cfg_hash: str,
) -> Prediction:
    ids, docs = evidence(example)
    gen = client.generate(v1.build_messages(example.question, docs))
    parsed = parse_response(gen.text, n_documents=len(docs))
    return Prediction(
        id=example.pmid,
        gold=example.label,
        predicted=parsed.label,
        raw_output=gen.text,
        parse_status=parsed.status,
        evidence_ids=tuple(ids),
        citations=parsed.citations,
        system=system,
        config_hash=cfg_hash,
        extra={
            **parsed.stats(),
            "n_documents": len(docs),
            "n_prompt_tokens": gen.n_prompt_tokens,
            "n_output_tokens": gen.n_output_tokens,
            "hit_max_tokens": gen.hit_max_tokens,
            "cached": gen.cached,
        },
    )


def run_system(
    examples: Sequence[Example],
    client: LLMClient,
    evidence: EvidenceFn,
    *,
    system: str,
    cfg_hash: str,
    progress: Callable[[int, int], None] | None = None,
) -> list[Prediction]:
    preds = []
    for i, ex in enumerate(examples, start=1):
        preds.append(answer_one(ex, client, evidence, system=system, cfg_hash=cfg_hash))
        if progress:
            progress(i, len(examples))
    return preds


def llm_stats(preds: Sequence[Prediction]) -> dict[str, Any]:
    """Parsing, length and citation statistics (protocol section 4.4)."""
    n = len(preds)
    if n == 0:
        raise ValueError("no predictions")
    extras: list[Mapping[str, Any]] = [p.extra for p in preds]
    methods: dict[str, int] = {}
    for e in extras:
        methods[e["parse_method"]] = methods.get(e["parse_method"], 0) + 1
    fabricated = [len(e["fabricated_citations"]) for e in extras]
    out_tokens = [int(e["n_output_tokens"]) for e in extras]
    return {
        "n": n,
        "parse_methods": dict(sorted(methods.items())),
        "n_parse_failures": sum(p.parse_status == "failed" for p in preds),
        "n_regex_fallback": methods.get("regex_fallback", 0),
        "n_hit_max_tokens": sum(bool(e["hit_max_tokens"]) for e in extras),
        "share_with_citation": sum(bool(p.citations) for p in preds) / n,
        "n_fabricated_citations": sum(fabricated),
        "n_questions_with_fabricated": sum(f > 0 for f in fabricated),
        "n_unreadable_citations": sum(int(e["unreadable_citations"]) for e in extras),
        "mean_output_tokens": sum(out_tokens) / n,
        "max_output_tokens": max(out_tokens),
        "n_cached": sum(bool(e["cached"]) for e in extras),
    }
