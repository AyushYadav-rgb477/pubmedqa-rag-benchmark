"""Prompt v1 (frozen, protocol section 4.2). Shared by closed-book, oracle and RAG."""

from __future__ import annotations

import re
from collections.abc import Sequence

from biomedqa.hashing import sha256_bytes

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are a biomedical research assistant. You answer research questions about PubMed "
    "studies with yes, no, or maybe."
)

TASK = "Answer the research question with yes, no, or maybe."

INSTRUCTIONS = (
    "Respond with only a JSON object with these keys:\n"
    '"answer": "yes", "no", or "maybe"\n'
    '"rationale": one or two sentences explaining the answer\n'
    '"citations": a list of the document numbers that support the answer, for example [1]; '
    "use [] if no documents are given or none support the answer"
)


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def user_message(question: str, documents: Sequence[str] = ()) -> str:
    parts = [TASK, ""]
    if documents:
        parts.append("Documents:")
        parts += [f"[{i}] {_one_line(d)}" for i, d in enumerate(documents, start=1)]
        parts.append("")
    parts += [f"Question: {_one_line(question)}", "", INSTRUCTIONS]
    return "\n".join(parts)


def build_messages(question: str, documents: Sequence[str] = ()) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(question, documents)},
    ]


def template_hash() -> str:
    """Fingerprint of the fixed prompt text, recorded in every run file."""
    return sha256_bytes("\x00".join([PROMPT_VERSION, SYSTEM, TASK, INSTRUCTIONS]).encode())
