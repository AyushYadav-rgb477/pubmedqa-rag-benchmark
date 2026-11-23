from __future__ import annotations

from pathlib import Path

import pytest

from biomedqa.io import read_predictions, write_predictions
from biomedqa.schema import PARSE_FAILED, PARSE_OK, Prediction


def pred(i: str = "1", gold: str = "yes", predicted: str = "yes", **kw) -> Prediction:
    status = PARSE_FAILED if predicted == "invalid" else PARSE_OK
    base = {
        "id": i,
        "gold": gold,
        "predicted": predicted,
        "raw_output": '{"answer": "yes"}',
        "parse_status": status,
        "system": "test",
        "config_hash": "abc",
    }
    return Prediction(**{**base, **kw})


def test_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    p = pred(
        "42",
        evidence_ids=[123, "456"],
        citations=["1", 2],
        extra={"probs": [0.7, 0.2, 0.1]},
        raw_output="ünïcode\nline",
    )
    path = tmp_path / "p.jsonl"
    write_predictions(path, [p])
    (back,) = read_predictions(path)
    assert back == p
    assert back.evidence_ids == ("123", "456")
    assert back.citations == (1, 2)


def test_file_is_sorted_numerically_with_lf(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    write_predictions(path, [pred("10"), pred("9"), pred("100")])
    assert [p.id for p in read_predictions(path)] == ["9", "10", "100"]
    assert b"\r\n" not in path.read_bytes()


def test_duplicate_ids_rejected_on_write_and_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        write_predictions(tmp_path / "p.jsonl", [pred("1"), pred("1")])
    path = tmp_path / "dup.jsonl"
    write_predictions(path, [pred("1")])
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_predictions(path)


def test_invalid_iff_failed() -> None:
    pred(predicted="invalid")  # ok
    with pytest.raises(ValueError):
        pred(predicted="invalid", parse_status=PARSE_OK)
    with pytest.raises(ValueError):
        pred(predicted="no", parse_status=PARSE_FAILED)


@pytest.mark.parametrize(
    "kw",
    [{"gold": "invalid"}, {"gold": "Yes"}, {"predicted": "unsure"}, {"system": ""}, {"id": ""}],
)
def test_bad_fields_raise(kw) -> None:
    with pytest.raises(ValueError):
        pred(**kw)


def test_bad_line_reports_location(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"p\.jsonl:1"):
        read_predictions(path)


def test_failed_write_leaves_old_file_intact(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "p.jsonl"
    write_predictions(path, [pred("1")])
    before = path.read_bytes()

    import biomedqa.fileio as fileio

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(fileio.os, "replace", boom)
    with pytest.raises(OSError):
        write_predictions(path, [pred("2")])
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["p.jsonl"]
