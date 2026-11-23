from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomedqa.config import REPO_ROOT
from biomedqa.reporting import END, START, Row, render_table, replace_block


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_missing_file_is_pending(tmp_path: Path) -> None:
    table = render_table([Row("X", "dev", "results/x.json")], root=tmp_path)
    assert "| X | dev | pending | pending | pending | pending |" in table


def test_multi_seed_summary_row(tmp_path: Path) -> None:
    _write(
        tmp_path / "results/s.json",
        {
            "n_seeds": 5,
            "accuracy": {"mean": 0.6123, "std": 0.0211},
            "macro_f1": {"mean": 0.4456, "std": 0.0349},
            "per_seed": {"0": {"n_parse_failures": 0}, "1": {"n_parse_failures": 0}},
        },
    )
    table = render_table([Row("Clf", "dev", "results/s.json")], root=tmp_path)
    assert "| Clf | dev | 61.2 ± 2.1 | 44.6 ± 3.5 | 5 seeds | 0 |" in table


def test_single_run_row_with_ci(tmp_path: Path) -> None:
    _write(
        tmp_path / "results/m.json",
        {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "n_parse_failures": 3,
            "bootstrap": {
                "accuracy": {"low": 0.41, "high": 0.6},
                "macro_f1": {"low": 0.3, "high": 0.5},
            },
        },
    )
    table = render_table([Row("LLM", "dev", "results/m.json")], root=tmp_path)
    assert "| 50.0 [41.0, 60.0] | 40.0 [30.0, 50.0] | 1 run | 3 |" in table


def test_unknown_file_shape_raises(tmp_path: Path) -> None:
    _write(tmp_path / "results/bad.json", {"accuracy": 1})
    with pytest.raises(ValueError):
        render_table([Row("B", "dev", "results/bad.json")], root=tmp_path)


def test_replace_block_is_idempotent() -> None:
    text = f"a\n{START}\nold\n{END}\nb\n"
    once = replace_block(text, "T")
    assert once == f"a\n{START}\nT\n{END}\nb\n"
    assert replace_block(once, "T") == once
    with pytest.raises(ValueError):
        replace_block("no markers", "T")


def test_readme_table_matches_results_files() -> None:
    """Guards rule 5: the README table must equal what the results files say."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert replace_block(readme, render_table()) == readme
