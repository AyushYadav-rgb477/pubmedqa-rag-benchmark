"""Parse an LLM response into an answer and citations (protocol section 4.3, ADR 0006).

1. Strip Markdown code fences; take the first balanced ``{...}`` object; ``json.loads``.
2. Normalise ``answer`` (lowercase, trim spaces / quotes / trailing punctuation).
3. If no JSON object can be read: regex fallback on an ``"answer": "<value>"`` field only.
4. Otherwise ``invalid``. Free-text yes / no / maybe is never used.

Citations: ints or strings like ``"2"`` / ``"[4]"``; numbers outside 1..n_documents are
fabricated; entries that are not numbers are counted as unreadable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from biomedqa.labels import INVALID, LABELS
from biomedqa.schema import PARSE_FAILED, PARSE_OK

METHOD_JSON = "json"
METHOD_REGEX = "regex_fallback"
METHOD_NONE = "none"

_FENCE = re.compile(r"```[A-Za-z0-9_-]*")
_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([^"\n]*)"', re.IGNORECASE)
_CITATIONS_FIELD = re.compile(r'"citations"\s*:\s*\[([^\]]*)\]', re.IGNORECASE)
_STRIP = " \t\r\n\"'`.,;:!?*"


@dataclass(frozen=True)
class ParsedAnswer:
    label: str
    status: str
    method: str
    rationale: str | None
    citations: tuple[int, ...]
    fabricated: tuple[int, ...]
    unreadable_citations: int
    error: str | None = None

    def stats(self) -> dict[str, Any]:
        return {
            "parse_method": self.method,
            "parse_error": self.error,
            "fabricated_citations": list(self.fabricated),
            "unreadable_citations": self.unreadable_citations,
            "rationale": self.rationale,
        }


def normalize_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip(_STRIP).lower()
    return v if v in LABELS else None


def first_json_object(text: str) -> str | None:
    """The first balanced ``{...}`` substring, respecting JSON strings and escapes."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _citation_number(item: Any) -> int | None:
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return item
    if isinstance(item, float) and item.is_integer():
        return int(item)
    if isinstance(item, str):
        s = item.strip().strip("[]()").strip()
        if s.isdigit():
            return int(s)
    return None


def parse_citations(raw: Any, n_documents: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """(citations in order without duplicates, fabricated ones, unreadable count)."""
    if raw is None:
        return (), (), 0
    items = raw if isinstance(raw, list) else [raw]
    seen: list[int] = []
    unreadable = 0
    for item in items:
        n = _citation_number(item)
        if n is None:
            unreadable += 1
        elif n not in seen:
            seen.append(n)
    fabricated = tuple(c for c in seen if not 1 <= c <= n_documents)
    return tuple(seen), fabricated, unreadable


def _regex_citations(text: str) -> list[str] | None:
    m = _CITATIONS_FIELD.search(text)
    if not m:
        return None
    return [p for p in (x.strip().strip('"').strip("'") for x in m.group(1).split(",")) if p]


def parse_response(text: str, n_documents: int) -> ParsedAnswer:
    if n_documents < 0:
        raise ValueError("n_documents must be >= 0")
    cleaned = _FENCE.sub("", text or "")
    obj: dict[str, Any] | None = None
    block = first_json_object(cleaned)
    if block is not None:
        try:
            loaded = json.loads(block)
            obj = loaded if isinstance(loaded, dict) else None
        except ValueError:
            obj = None

    if obj is not None:
        label = normalize_answer(obj.get("answer"))
        cits, fab, unread = parse_citations(obj.get("citations"), n_documents)
        rationale = obj.get("rationale")
        rationale = rationale if isinstance(rationale, str) else None
        if label is None:
            return ParsedAnswer(
                INVALID, PARSE_FAILED, METHOD_JSON, rationale, cits, fab, unread,
                error="answer is not yes / no / maybe",
            )  # fmt: skip
        return ParsedAnswer(label, PARSE_OK, METHOD_JSON, rationale, cits, fab, unread)

    m = _ANSWER_FIELD.search(cleaned)
    label = normalize_answer(m.group(1)) if m else None
    if label is not None:
        cits, fab, unread = parse_citations(_regex_citations(cleaned), n_documents)
        return ParsedAnswer(label, PARSE_OK, METHOD_REGEX, None, cits, fab, unread)
    return ParsedAnswer(
        INVALID, PARSE_FAILED, METHOD_NONE, None, (), (), 0, error="no readable answer field"
    )
