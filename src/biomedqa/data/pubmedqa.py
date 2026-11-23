"""Load PubMedQA ``pqa_labeled`` from Hugging Face at a pinned revision.

Do not call :func:`load_all_labeled` from experiment code: it returns every question, including
the locked test split. Use :func:`biomedqa.data.splits.get_split_examples`, which applies the
test-set lock.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from biomedqa.labels import check_gold

DATASET_NAME = "qiaojin/PubMedQA"
LABELED_CONFIG = "pqa_labeled"


@dataclass(frozen=True)
class Example:
    pmid: str
    question: str
    contexts: tuple[str, ...]
    long_answer: str
    label: str

    @property
    def context_text(self) -> str:
        """The abstract without its conclusion: context paragraphs joined by a space."""
        return " ".join(c.strip() for c in self.contexts)


def example_from_row(row: Mapping[str, Any]) -> Example:
    """Convert one Hugging Face ``pqa_labeled`` row into an :class:`Example`."""
    context = row["context"]
    return Example(
        pmid=str(row["pubid"]),
        question=str(row["question"]).strip(),
        contexts=tuple(str(c) for c in context["contexts"]),
        long_answer=str(row.get("long_answer", "")),
        label=check_gold(str(row["final_decision"]).strip().lower()),
    )


def examples_by_pmid(rows: Iterable[Mapping[str, Any]]) -> dict[str, Example]:
    out: dict[str, Example] = {}
    for row in rows:
        ex = example_from_row(row)
        if ex.pmid in out:
            raise ValueError(f"duplicate PMID {ex.pmid}")
        out[ex.pmid] = ex
    return out


def load_all_labeled(revision: str, cache_dir: str | None = None) -> dict[str, Example]:
    """All 1000 labelled questions, keyed by PMID. Includes test: see module docstring."""
    from datasets import load_dataset  # heavy import kept lazy

    ds = load_dataset(
        DATASET_NAME, LABELED_CONFIG, revision=revision, split="train", cache_dir=cache_dir
    )
    return examples_by_pmid(ds)
