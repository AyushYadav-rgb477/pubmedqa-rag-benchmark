from __future__ import annotations

import pytest

from biomedqa.config import resolve
from biomedqa.parsing import (
    METHOD_JSON,
    METHOD_NONE,
    METHOD_REGEX,
    first_json_object,
    normalize_answer,
    parse_citations,
    parse_response,
)
from biomedqa.prompts import v1

# --- prompt v1 ----------------------------------------------------------------------------------


def test_prompt_text_matches_protocol_exactly() -> None:
    protocol = resolve("docs/experiment-protocol.md").read_text(encoding="utf-8")
    assert v1.SYSTEM in protocol
    assert v1.TASK in protocol
    assert v1.INSTRUCTIONS in protocol


def test_closed_book_prompt_has_no_documents_block() -> None:
    msgs = v1.build_messages("Does X work?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == v1.SYSTEM
    assert "Documents:" not in msgs[1]["content"]
    assert msgs[1]["content"] == f"{v1.TASK}\n\nQuestion: Does X work?\n\n{v1.INSTRUCTIONS}"


def test_documents_are_numbered_and_flattened() -> None:
    user = v1.build_messages("Q?", ["first doc\nline two", "  second  "])[1]["content"]
    assert "Documents:\n[1] first doc line two\n[2] second\n\nQuestion: Q?" in user


def test_only_the_evidence_block_differs() -> None:
    closed = v1.user_message("Q?")
    oracle = v1.user_message("Q?", ["abstract"])
    assert oracle.replace("Documents:\n[1] abstract\n\n", "") == closed


def test_template_hash_is_stable() -> None:
    assert v1.template_hash() == v1.template_hash()
    assert len(v1.template_hash()) == 64
    assert v1.PROMPT_VERSION == "v1"


# --- answer normalisation -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("yes", "yes"), (" No ", "no"), ("MAYBE.", "maybe"), ('"yes"', "yes"), ("**no**", "no"),
     ("yes, probably", None), ("unclear", None), ("", None), (None, None), (1, None)],
)  # fmt: skip
def test_normalize_answer(raw, expected) -> None:
    assert normalize_answer(raw) == expected


# --- JSON extraction ------------------------------------------------------------------------


def test_first_json_object_handles_nesting_and_braces_in_strings() -> None:
    text = (
        'Sure! {"answer": "yes", "rationale": "a {weird} \\"quote\\"", "x": {"y": 1}} trailing {}'
    )
    block = first_json_object(text)
    assert block is not None and block.endswith('"y": 1}}')


def test_first_json_object_none_cases() -> None:
    assert first_json_object("no braces") is None
    assert first_json_object('{"answer": "yes"') is None  # never closed


# --- full parse: happy paths ------------------------------------------------------------------


def test_clean_json() -> None:
    p = parse_response('{"answer": "yes", "rationale": "r", "citations": [1]}', 1)
    assert (p.label, p.status, p.method) == ("yes", "ok", METHOD_JSON)
    assert p.citations == (1,) and p.fabricated == ()
    assert p.rationale == "r"


def test_code_fence_and_surrounding_text() -> None:
    text = (
        "Here is my answer:\n```json\n"
        '{"answer": "No", "rationale": "r", "citations": []}\n```\nHope it helps.'
    )
    p = parse_response(text, 1)
    assert (p.label, p.method) == ("no", METHOD_JSON)


def test_bare_code_fence() -> None:
    p = parse_response('```\n{"answer": "maybe", "citations": []}\n```', 0)
    assert p.label == "maybe"


# --- citations --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "cits", "fab", "unread"),
    [
        ([1, 2], (1, 2), (), 0),
        (["1", "[2]", " 3 "], (1, 2, 3), (), 0),
        ([2, 2, 1], (2, 1), (), 0),  # duplicates removed, order kept
        ([4, 0, -1], (4, 0, -1), (4, 0, -1), 0),  # outside 1..3
        (["doc one", True, None, 1.0, 1.5], (1,), (), 4),
        (2, (2,), (), 0),  # a single number instead of a list
        ("[3]", (3,), (), 0),
        (None, (), (), 0),
    ],
)
def test_parse_citations(raw, cits, fab, unread) -> None:
    assert parse_citations(raw, 3) == (cits, fab, unread)


def test_closed_book_citations_are_all_fabricated() -> None:
    p = parse_response('{"answer": "yes", "citations": [1, "[2]"]}', 0)
    assert p.fabricated == (1, 2)
    assert p.label == "yes"  # citations never change the answer


def test_out_of_range_citation_flagged_in_rag() -> None:
    p = parse_response('{"answer": "no", "citations": ["[4]", 2]}', 3)
    assert p.citations == (4, 2)
    assert p.fabricated == (4,)


# --- fallback and failures ------------------------------------------------------------------


def test_regex_fallback_on_broken_json() -> None:
    text = '{"answer": "yes", "rationale": "trailing comma", "citations": [1, 2],}'
    p = parse_response(text, 2)
    assert (p.label, p.status, p.method) == ("yes", "ok", METHOD_REGEX)
    assert p.citations == (1, 2)


def test_regex_fallback_on_unclosed_json() -> None:
    p = parse_response('{"answer": "Maybe", "rationale": "cut off by the token lim', 1)
    assert (p.label, p.method) == ("maybe", METHOD_REGEX)
    assert p.citations == ()


def test_free_text_answer_is_never_used() -> None:
    p = parse_response("Yes. The study clearly shows the treatment works.", 1)
    assert (p.label, p.status, p.method) == ("invalid", "failed", METHOD_NONE)


def test_json_with_bad_answer_is_invalid_not_fallback() -> None:
    p = parse_response('{"answer": "probably yes", "citations": [1]}', 1)
    assert (p.label, p.status, p.method) == ("invalid", "failed", METHOD_JSON)
    assert p.citations == (1,)


def test_json_without_answer_key_is_invalid() -> None:
    p = parse_response('{"response": "yes"}', 0)
    assert p.label == "invalid"


def test_non_object_json_falls_through() -> None:
    assert parse_response('["yes"]', 0).label == "invalid"


@pytest.mark.parametrize("text", ["", "   ", "{}", "```json\n```", None])
def test_empty_outputs_are_invalid(text) -> None:
    p = parse_response(text, 1)
    assert p.label == "invalid" and p.status == "failed"


def test_first_object_wins_over_later_ones() -> None:
    p = parse_response('{"answer": "no"} then {"answer": "yes"}', 0)
    assert p.label == "no"


def test_regex_only_reads_the_answer_field() -> None:
    # broken JSON whose rationale mentions "answer": "yes" later; the field itself says no
    text = '{"answer": "no", "rationale": "not \\"answer\\": \\"yes\\"", oops'
    assert parse_response(text, 0).label == "no"


def test_negative_documents_rejected() -> None:
    with pytest.raises(ValueError):
        parse_response("{}", -1)


def test_stats_are_json_ready() -> None:
    s = parse_response('{"answer": "yes", "citations": [9, "x"]}', 1).stats()
    assert s == {
        "parse_method": "json",
        "parse_error": None,
        "fabricated_citations": [9],
        "unreadable_citations": 1,
        "rationale": None,
    }
