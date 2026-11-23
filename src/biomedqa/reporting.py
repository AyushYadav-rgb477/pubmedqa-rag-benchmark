"""Build the README results table from files in ``results/`` (never typed by hand).

Each row points to one results file. Multi-seed summaries (``summary.json`` with ``n_seeds``)
are shown as mean ± std over seeds; single runs (a metrics JSON with ``bootstrap``) as the point
value with its 95% bootstrap CI. Rows whose file does not exist yet are listed as pending.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biomedqa.config import REPO_ROOT
from biomedqa.fileio import read_json

START = "<!-- results:start -->"
END = "<!-- results:end -->"


@dataclass(frozen=True)
class Row:
    system: str
    split: str
    path: str  # relative to the repo root


# Order of rows in the README. Later phases append closed-book, oracle and RAG rows.
ROWS: tuple[Row, ...] = (
    Row("Majority class (train)", "dev", "results/baselines/majority/dev_metrics.json"),
    Row("PubMedBERT classifier", "dev", "results/classifier/summary.json"),
    Row("Closed-book LLM", "dev", "results/llm/closed_book/dev_metrics.json"),
    Row("Oracle LLM (gold abstract)", "dev", "results/llm/oracle/dev_metrics.json"),
)


def _fmt(x: float) -> str:
    return f"{100 * x:.1f}"


def _cells(obj: dict[str, Any]) -> tuple[str, str, str, str]:
    """(accuracy, macro-F1, runs, parse failures) cells for one results file."""
    if "n_seeds" in obj:
        acc, f1 = obj["accuracy"], obj["macro_f1"]
        fails = sum(int(s.get("n_parse_failures", 0)) for s in obj["per_seed"].values())
        return (
            f"{_fmt(acc['mean'])} ± {_fmt(acc['std'])}",
            f"{_fmt(f1['mean'])} ± {_fmt(f1['std'])}",
            f"{obj['n_seeds']} seeds",
            str(fails),
        )
    if "bootstrap" in obj:
        b = obj["bootstrap"]
        return (
            f"{_fmt(obj['accuracy'])} [{_fmt(b['accuracy']['low'])}, "
            f"{_fmt(b['accuracy']['high'])}]",
            f"{_fmt(obj['macro_f1'])} [{_fmt(b['macro_f1']['low'])}, "
            f"{_fmt(b['macro_f1']['high'])}]",
            "1 run",
            str(obj["n_parse_failures"]),
        )
    raise ValueError("unrecognised results file: needs 'n_seeds' or 'bootstrap'")


def render_table(rows: Sequence[Row] = ROWS, root: Path = REPO_ROOT) -> str:
    lines = [
        "| System | Split | Accuracy (%) | Macro-F1 (%) | Runs | Parse failures | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = root / r.path
        if p.exists():
            acc, f1, runs, fails = _cells(read_json(p))
        else:
            acc = f1 = runs = fails = "pending"
        lines.append(
            f"| {r.system} | {r.split} | {acc} | {f1} | {runs} | {fails} | [`{r.path}`]({r.path}) |"
        )
    note = (
        "Multi-seed rows: mean ± sample std over seeds. Single runs: value [95% bootstrap CI]. "
        "Dev selects settings, so dev numbers are optimistic; test is locked until Phase 4."
    )
    return "\n".join([*lines, "", note])


def replace_block(text: str, table: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README has no {START} ... {END} block")
    return pattern.sub(lambda _: f"{START}\n{table}\n{END}", text, count=1)
