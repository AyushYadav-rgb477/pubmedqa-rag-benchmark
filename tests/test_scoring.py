from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biomedqa.data.splits import make_splits, verify_or_write
from biomedqa.guard import TestSetLockedError
from biomedqa.io import write_predictions
from biomedqa.labels import LABELS, PREDICTION_VALUES
from biomedqa.schema import PARSE_FAILED, PARSE_OK, Prediction
from biomedqa.scoring.bootstrap import bootstrap_ci, resampled_metrics
from biomedqa.scoring.metrics import encode, score
from biomedqa.scoring.report import score_file, score_split

# --- point metrics: edge cases ------------------------------------------------------------------


def test_perfect_predictions() -> None:
    golds = ["yes", "no", "maybe", "yes"]
    s = score(golds, golds)
    assert s["accuracy"] == 1.0
    assert s["macro_f1"] == 1.0
    assert s["n_parse_failures"] == 0


def test_all_invalid_scores_zero_and_counts_failures() -> None:
    golds = ["yes", "no", "maybe"]
    s = score(golds, ["invalid"] * 3)
    assert s["accuracy"] == 0.0
    assert s["macro_f1"] == 0.0
    assert s["n_parse_failures"] == 3
    assert s["confusion"]["matrix"] == [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]


def test_invalid_is_wrong_not_dropped() -> None:
    # dropping the failure would give accuracy 1.0
    s = score(["yes", "yes"], ["yes", "invalid"])
    assert s["accuracy"] == 0.5
    assert s["n"] == 2


def test_invalid_is_false_negative_not_false_positive() -> None:
    s = score(["yes", "yes", "no"], ["yes", "invalid", "no"])
    yes = s["per_class"]["yes"]
    assert yes["precision"] == 1.0  # invalid did not count as a wrong "yes"
    assert yes["recall"] == 0.5
    assert yes["f1"] == pytest.approx(2 / 3)


def test_absent_class_counts_zero_in_macro_f1() -> None:
    # "maybe" is never gold and never predicted: F1 = 0 (zero_division=0), still averaged
    s = score(["yes", "no"], ["yes", "no"])
    assert s["per_class"]["maybe"]["f1"] == 0.0
    assert s["macro_f1"] == pytest.approx(2 / 3)


def test_predicted_but_never_gold_class() -> None:
    s = score(["yes", "yes"], ["maybe", "yes"])
    assert s["per_class"]["maybe"]["precision"] == 0.0
    assert s["per_class"]["maybe"]["recall"] == 0.0
    assert s["per_class"]["maybe"]["predicted"] == 1


def test_confusion_layout() -> None:
    s = score(["yes", "no", "maybe", "no"], ["no", "no", "invalid", "yes"])
    assert s["confusion"]["rows_gold"] == ["yes", "no", "maybe"]
    assert s["confusion"]["cols_pred"] == ["yes", "no", "maybe", "invalid"]
    assert s["confusion"]["matrix"] == [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]]


@pytest.mark.parametrize(
    ("golds", "preds"),
    [([], []), (["yes"], []), (["invalid"], ["yes"]), (["yes"], ["Yes"])],
)
def test_bad_inputs_raise(golds, preds) -> None:
    with pytest.raises((ValueError, KeyError)):
        score(golds, preds)


@pytest.mark.parametrize("seed", range(20))
def test_matches_scikit_learn(seed: int) -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    golds = list(rng.choice(LABELS, size=n, p=[0.55, 0.34, 0.11]))
    preds = list(rng.choice(PREDICTION_VALUES, size=n, p=[0.4, 0.3, 0.1, 0.2]))
    s = score(golds, preds)
    assert s["accuracy"] == pytest.approx(sk.accuracy_score(golds, preds))
    assert s["macro_f1"] == pytest.approx(
        sk.f1_score(golds, preds, labels=list(LABELS), average="macro", zero_division=0)
    )
    p, r, f, _ = sk.precision_recall_fscore_support(
        golds, preds, labels=list(LABELS), zero_division=0
    )
    for i, lab in enumerate(LABELS):
        assert s["per_class"][lab]["precision"] == pytest.approx(p[i])
        assert s["per_class"][lab]["recall"] == pytest.approx(r[i])
        assert s["per_class"][lab]["f1"] == pytest.approx(f[i])


# --- bootstrap ----------------------------------------------------------------------------------


def _random_case(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    golds = list(rng.choice(LABELS, size=n))
    preds = [g if rng.random() < 0.7 else str(rng.choice(PREDICTION_VALUES)) for g in golds]
    return golds, preds


def test_bootstrap_is_deterministic_and_seeded() -> None:
    golds, preds = _random_case()
    a = bootstrap_ci(golds, preds, n_resamples=2000, seed=42)
    b = bootstrap_ci(golds, preds, n_resamples=2000, seed=42)
    c = bootstrap_ci(golds, preds, n_resamples=2000, seed=1)
    assert a == b
    assert a != c


def test_bootstrap_interval_contains_point_estimate() -> None:
    golds, preds = _random_case(200)
    s = score(golds, preds)
    ci = bootstrap_ci(golds, preds, n_resamples=3000)
    for m in ("accuracy", "macro_f1"):
        assert ci[m]["low"] <= s[m] <= ci[m]["high"]
        assert ci[m]["low"] < ci[m]["high"]


def test_bootstrap_degenerate_when_all_correct() -> None:
    golds = ["yes", "no", "maybe"] * 10
    ci = bootstrap_ci(golds, golds, n_resamples=500)
    assert ci["accuracy"] == {"low": 1.0, "high": 1.0}


def test_vectorised_resamples_match_scalar_metrics() -> None:
    golds, preds = _random_case(37, seed=3)
    g, p = encode(golds, preds)
    accs, f1s = resampled_metrics(g, p, n_resamples=5, seed=11)
    idx = np.random.default_rng(11).integers(0, len(g), size=(5, len(g)))
    for k in range(5):
        s = score([golds[i] for i in idx[k]], [preds[i] for i in idx[k]])
        assert accs[k] == pytest.approx(s["accuracy"])
        assert f1s[k] == pytest.approx(s["macro_f1"])


def test_bootstrap_default_is_10000_resamples_seed_42() -> None:
    ci = bootstrap_ci(["yes", "no"], ["yes", "yes"])
    assert (ci["n_resamples"], ci["seed"]) == (10_000, 42)


# --- scoring a split ----------------------------------------------------------------------------


@pytest.fixture
def split_file(tmp_path: Path) -> Path:
    pool = {str(100 + i): LABELS[i % 3] for i in range(30)}
    splits = make_splits(pool, ["1", "2", "3"], n_train=20, n_dev=10, n_test=3, seed=42)
    path = tmp_path / "splits.json"
    verify_or_write(path, splits, {})
    return path


def _preds_for(ids, system="s", config_hash="h"):
    return [
        Prediction(
            id=i,
            gold="yes",
            predicted="yes" if k % 2 else "invalid",
            raw_output="",
            parse_status=PARSE_OK if k % 2 else PARSE_FAILED,
            system=system,
            config_hash=config_hash,
        )
        for k, i in enumerate(ids)
    ]


def _dev_ids(path: Path) -> list[str]:
    from biomedqa.data.splits import read_splits

    return list(read_splits(path).dev)


def test_score_split_dev(split_file: Path, tmp_path: Path) -> None:
    preds = _preds_for(_dev_ids(split_file))
    out = tmp_path / "p.jsonl"
    write_predictions(out, preds)
    rep = score_file(out, "dev", splits_file=split_file, n_resamples=200)
    assert rep["n"] == 10
    assert rep["n_parse_failures"] == 5
    assert rep["split"] == "dev"
    assert len(rep["predictions_sha256"]) == 64


def test_score_split_refuses_missing_predictions(split_file: Path) -> None:
    with pytest.raises(ValueError, match="1 missing"):
        score_split(_preds_for(_dev_ids(split_file)[:-1]), "dev", split_file, n_resamples=10)


def test_score_split_refuses_test(split_file: Path) -> None:
    with pytest.raises(TestSetLockedError):
        score_split(_preds_for(["1", "2", "3"]), "test", split_file)


def test_score_split_refuses_test_ids_mixed_into_dev(split_file: Path) -> None:
    preds = _preds_for([*_dev_ids(split_file), "1"])
    with pytest.raises(TestSetLockedError):
        score_split(preds, "dev", split_file)


def test_score_split_refuses_mixed_systems(split_file: Path) -> None:
    ids = _dev_ids(split_file)
    preds = _preds_for(ids[:5], system="a") + _preds_for(ids[5:], system="b")
    with pytest.raises(ValueError, match="mixed"):
        score_split(preds, "dev", split_file, n_resamples=10)
