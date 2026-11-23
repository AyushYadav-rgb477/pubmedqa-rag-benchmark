from __future__ import annotations

import importlib
import json

import pytest

from biomedqa.classifier.core import (
    input_pair,
    label_ids,
    probs_to_predictions,
    run_config_hash,
    select_best_epoch,
    summarize_seeds,
)
from biomedqa.config import load_config, resolve
from biomedqa.data.pubmedqa import example_from_row


def test_input_pair_uses_question_and_context_only(row_factory) -> None:
    ex = example_from_row(row_factory(7, "no"))
    q, c = input_pair(ex)
    assert q == ex.question
    assert c == ex.context_text
    assert "CONCLUSION" not in q + c


def test_long_answer_is_refused(row_factory) -> None:
    with pytest.raises(ValueError, match="long_answer"):
        input_pair(example_from_row(row_factory(7)), use_long_answer=True)


def test_label_ids_follow_label_order(row_factory) -> None:
    exs = [example_from_row(row_factory(i, lab)) for i, lab in enumerate(["yes", "no", "maybe"])]
    assert label_ids(exs) == [0, 1, 2]


def test_probs_to_predictions_argmax_and_ties(row_factory) -> None:
    exs = [example_from_row(row_factory(i, "yes")) for i in (1, 2, 3)]
    probs = [[0.1, 0.7, 0.2], [0.4, 0.4, 0.2], [0.2, 0.3, 0.5]]
    preds = probs_to_predictions(exs, probs, system="pubmedbert", cfg_hash="h", extra={"seed": 0})
    assert [p.predicted for p in preds] == ["no", "yes", "maybe"]  # tie -> earlier label
    assert all(p.parse_status == "ok" for p in preds)
    assert preds[0].extra["probs"] == [0.1, 0.7, 0.2]
    assert preds[0].extra["seed"] == 0
    assert json.loads(preds[0].raw_output) == {"yes": 0.1, "no": 0.7, "maybe": 0.2}


def test_probs_to_predictions_validates_shapes(row_factory) -> None:
    exs = [example_from_row(row_factory(1))]
    with pytest.raises(ValueError):
        probs_to_predictions(exs, [], system="s", cfg_hash="h")
    with pytest.raises(ValueError):
        probs_to_predictions(exs, [[0.5, 0.5]], system="s", cfg_hash="h")


def test_select_best_epoch_ties_go_to_earlier() -> None:
    hist = [{"macro_f1": 0.4}, {"macro_f1": 0.6}, {"macro_f1": 0.6}, {"macro_f1": 0.5}]
    assert select_best_epoch(hist) == 1
    with pytest.raises(ValueError):
        select_best_epoch([])


def test_summarize_seeds_mean_and_sample_std() -> None:
    per_seed = {
        0: {"accuracy": 0.5, "macro_f1": 0.4, "best_epoch": 3},
        1: {"accuracy": 0.7, "macro_f1": 0.6, "best_epoch": 5},
    }
    s = summarize_seeds(per_seed)
    assert s["macro_f1"]["mean"] == pytest.approx(0.5)
    assert s["macro_f1"]["std"] == pytest.approx(0.1414213562)  # ddof=1
    assert s["per_seed"]["1"]["best_epoch"] == 5
    assert s["n_seeds"] == 2


def test_run_config_hash_depends_on_seed_not_seed_list() -> None:
    cfg = load_config(resolve("configs/classifier.yaml"))
    assert run_config_hash(cfg, 0) != run_config_hash(cfg, 1)
    assert run_config_hash({**cfg, "seeds": [9]}, 0) == run_config_hash(cfg, 0)


def test_classifier_config_matches_protocol() -> None:
    cfg = load_config(resolve("configs/classifier.yaml"))
    assert cfg["model"]["revision"] == "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
    assert cfg["input"] == {
        "max_length": 512,
        "truncation": "only_second",
        "use_long_answer": False,
    }
    t = cfg["train"]
    assert t["per_device_batch_size"] * t["grad_accum_steps"] == 16
    assert (t["learning_rate"], t["epochs"], t["warmup_ratio"]) == (2e-5, 10, 0.1)
    assert cfg["seeds"] == [0, 1, 2, 3, 4]


def test_train_module_imports_without_torch() -> None:
    # heavy libraries are imported lazily inside functions
    importlib.import_module("biomedqa.classifier.train")
