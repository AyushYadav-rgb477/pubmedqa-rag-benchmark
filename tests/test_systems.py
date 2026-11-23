from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomedqa.config import load_config, resolve
from biomedqa.data.pubmedqa import example_from_row
from biomedqa.data.splits import make_splits, verify_or_write
from biomedqa.guard import TestSetLockedError
from biomedqa.labels import LABELS
from biomedqa.llm.cache import CachedClient, ResponseCache
from biomedqa.llm.fake import FakeClient
from biomedqa.systems import majority
from biomedqa.systems.llm import answer_one, closed_book, llm_stats, oracle, run_system
from biomedqa.systems.runner import llm_run_hash, run_llm_split

LLM_CFG = load_config(resolve("configs/llm.yaml"))


@pytest.fixture
def world(tmp_path: Path, row_factory):
    """A tiny dataset with saved splits: 20 train, 10 dev, 3 test."""
    pool = {str(100 + i): LABELS[i % 3] for i in range(30)}
    splits = make_splits(pool, ["1", "2", "3"], n_train=20, n_dev=10, n_test=3, seed=42)
    path = tmp_path / "splits.json"
    verify_or_write(path, splits, {})
    examples = {p: example_from_row(row_factory(int(p), lab)) for p, lab in pool.items()}
    for t in ("1", "2", "3"):
        examples[t] = example_from_row(row_factory(int(t), "yes"))
    return {"path": path, "splits": splits, "examples": examples}


def _dev(world):
    return [world["examples"][i] for i in world["splits"].dev]


def scripted(messages) -> str:
    """Answers 'yes' with a citation when a document is present, else a broken JSON 'no'."""
    user = messages[-1]["content"]
    if "Documents:" in user:
        return '```json\n{"answer": "yes", "rationale": "r", "citations": [1]}\n```'
    return '{"answer": "no", "rationale": "r", "citations": [1],}'  # trailing comma


# --- evidence ---------------------------------------------------------------------------------


def test_closed_book_has_no_evidence(row_factory) -> None:
    assert closed_book(example_from_row(row_factory(5))) == ([], [])


def test_oracle_uses_context_without_conclusion(row_factory) -> None:
    ex = example_from_row(row_factory(5))
    ids, docs = oracle(ex)
    assert ids == ["5"]
    assert docs == [ex.context_text]
    assert "CONCLUSION" not in docs[0]


# --- answering ----------------------------------------------------------------------------------


def test_answer_one_oracle(row_factory) -> None:
    fake = FakeClient(scripted)
    ex = example_from_row(row_factory(7, "yes"))
    p = answer_one(ex, fake, oracle, system="oracle", cfg_hash="h")
    assert (p.predicted, p.parse_status, p.evidence_ids, p.citations) == ("yes", "ok", ("7",), (1,))
    assert p.extra["parse_method"] == "json"
    assert p.extra["fabricated_citations"] == []
    assert "[1] Background of 7." in fake.calls[0][1]["content"]


def test_answer_one_closed_book_flags_fabricated_citation(row_factory) -> None:
    fake = FakeClient(scripted)
    p = answer_one(example_from_row(row_factory(7, "no")), fake, closed_book,
                   system="closed_book", cfg_hash="h")  # fmt: skip
    assert p.predicted == "no"
    assert p.extra["parse_method"] == "regex_fallback"
    assert p.extra["fabricated_citations"] == [1]
    assert "Documents:" not in fake.calls[0][1]["content"]


def test_llm_stats_counts(row_factory) -> None:
    exs = [example_from_row(row_factory(i, "no")) for i in range(1, 5)]
    preds = run_system(exs, FakeClient(scripted), closed_book, system="cb", cfg_hash="h")
    s = llm_stats(preds)
    assert s["n"] == 4
    assert s["n_regex_fallback"] == 4
    assert s["n_fabricated_citations"] == 4
    assert s["n_questions_with_fabricated"] == 4
    assert s["share_with_citation"] == 1.0
    assert s["n_parse_failures"] == 0


def test_run_hash_ignores_cache_dir_but_not_system() -> None:
    other = {**LLM_CFG, "cache_dir": "/somewhere/else"}
    assert llm_run_hash(LLM_CFG, "oracle") == llm_run_hash(other, "oracle")
    assert llm_run_hash(LLM_CFG, "oracle") != llm_run_hash(LLM_CFG, "closed_book")


# --- runner end to end ------------------------------------------------------------------------


def test_run_llm_split_writes_files_and_resumes_from_cache(world, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = ResponseCache(tmp_path / "cache")
    fake = FakeClient(scripted)
    client = CachedClient(fake, cache, "v1")
    rep = run_llm_split("oracle", "dev", _dev(world), client, LLM_CFG, out,
                        splits_file=world["path"])  # fmt: skip
    assert {p.name for p in out.iterdir()} == {
        "dev_predictions.jsonl",
        "dev_metrics.json",
        "dev_run.json",
    }
    assert rep["n"] == 10 and rep["n_parse_failures"] == 0
    assert rep["llm"]["n_cached"] == 0
    run = json.loads((out / "dev_run.json").read_text(encoding="utf-8"))
    assert run["prompt_version"] == "v1"
    assert run["cache_misses"] == 10

    # second run: same answers, all from the cache, the model is never called
    fake2 = FakeClient(lambda m: "should not be called")
    rep2 = run_llm_split("oracle", "dev", _dev(world), CachedClient(fake2, cache, "v1"),
                         LLM_CFG, tmp_path / "out2", splits_file=world["path"])  # fmt: skip
    assert fake2.calls == []
    assert rep2["llm"]["n_cached"] == 10
    assert rep2["macro_f1"] == rep["macro_f1"]


def test_run_llm_split_refuses_test(world, tmp_path: Path) -> None:
    fake = FakeClient(scripted)
    test_examples = [world["examples"][i] for i in world["splits"].test]
    with pytest.raises(TestSetLockedError):
        run_llm_split("oracle", "test", test_examples, fake, LLM_CFG, tmp_path,
                      splits_file=world["path"])  # fmt: skip
    assert fake.calls == []


# --- majority baseline --------------------------------------------------------------------------


def test_majority_label_and_ties(row_factory) -> None:
    def exs(labels):
        return [example_from_row(row_factory(i, lab)) for i, lab in enumerate(labels, 1)]

    assert majority.majority_label(exs(["no", "no", "yes"])) == "no"
    assert majority.majority_label(exs(["maybe", "no"])) == "no"  # tie -> label order
    with pytest.raises(ValueError):
        majority.majority_label([])


def test_majority_predictions_score(world, tmp_path: Path) -> None:
    from biomedqa.systems.runner import save_run

    dev = _dev(world)
    preds = majority.predict(dev, "yes", "h")
    rep = save_run(preds, "dev", tmp_path, {"system": "majority"}, splits_file=world["path"])
    n_yes = sum(e.label == "yes" for e in dev)
    assert rep["accuracy"] == pytest.approx(n_yes / len(dev))
    assert rep["per_class"]["no"]["f1"] == 0.0
