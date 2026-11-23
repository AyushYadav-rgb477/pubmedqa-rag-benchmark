from __future__ import annotations

from typing import Any

import pytest


def make_row(pmid: int, label: str = "yes", question: str | None = None) -> dict[str, Any]:
    """A fake Hugging Face ``pqa_labeled`` row with the real field layout."""
    return {
        "pubid": pmid,
        "question": question or f"Does treatment {pmid} work?",
        "context": {
            "contexts": [f"Background of {pmid}.", f"Results of {pmid}."],
            "labels": ["BACKGROUND", "RESULTS"],
            "meshes": ["Humans"],
        },
        "long_answer": f"CONCLUSION of {pmid}: the answer is {label}.",
        "final_decision": label,
    }


@pytest.fixture
def row_factory():
    return make_row
