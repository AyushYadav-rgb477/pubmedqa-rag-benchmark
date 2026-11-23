from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from biomedqa.config import load_config, resolve
from biomedqa.data.pubmedqa import examples_by_pmid
from biomedqa.data.splits import (
    SplitIntegrityError,
    SplitMismatchError,
    Splits,
    build_splits_from_config,
    get_split_examples,
    largest_remainder,
    load_split_ids,
    make_splits,
    read_id_list,
    read_splits,
    verify_or_write,
)
from biomedqa.guard import TestSetLockedError, check_ids_not_test, check_split_access
from biomedqa.hashing import ids_checksum

COMMITTED = resolve("splits/pqa_labeled_splits.json")


def _pool(n_yes: int = 55, n_no: int = 34, n_maybe: int = 11, start: int = 1000):
    labels = ["yes"] * n_yes + ["no"] * n_no + ["maybe"] * n_maybe
    return {str(start + i): lab for i, lab in enumerate(labels)}


def _small_splits(seed: int = 42) -> Splits:
    pool = _pool()
    test = [str(i) for i in range(1, 51)]
    return make_splits(pool, test, n_train=80, n_dev=20, n_test=50, seed=seed)


# --- allocation ---------------------------------------------------------------------------


def test_largest_remainder_sums_to_total_and_is_proportional() -> None:
    alloc = largest_remainder({"yes": 276, "no": 169, "maybe": 55}, 100)
    assert sum(alloc.values()) == 100
    # exact quotas 55.2 / 33.8 / 11.0
    assert alloc == {"yes": 55, "no": 34, "maybe": 11}


def test_largest_remainder_breaks_ties_in_label_order() -> None:
    assert largest_remainder({"yes": 1, "no": 1, "maybe": 1}, 2) == {"yes": 1, "no": 1, "maybe": 0}


def test_largest_remainder_rejects_impossible_totals() -> None:
    with pytest.raises(ValueError):
        largest_remainder({"yes": 2}, 3)


# --- split construction ---------------------------------------------------------------------


def test_make_splits_sizes_disjoint_and_stratified() -> None:
    s = _small_splits()
    assert (len(s.train), len(s.dev), len(s.test)) == (80, 20, 50)
    assert not (set(s.train) & set(s.dev))
    pool = _pool()
    dev_counts = Counter(pool[i] for i in s.dev)
    assert dev_counts == {"yes": 11, "no": 7, "maybe": 2}  # 11 / 6.8 / 2.2 -> 11 / 7 / 2


def test_make_splits_is_deterministic_and_seed_dependent() -> None:
    assert _small_splits(42) == _small_splits(42)
    assert _small_splits(42).dev != _small_splits(7).dev


def test_make_splits_is_independent_of_input_order() -> None:
    pool = _pool()
    reversed_pool = dict(reversed(list(pool.items())))
    test = [str(i) for i in range(1, 51)]
    a = make_splits(pool, test, n_train=80, n_dev=20, n_test=50, seed=42)
    b = make_splits(reversed_pool, test[::-1], n_train=80, n_dev=20, n_test=50, seed=42)
    assert a == b


def test_make_splits_refuses_test_labels() -> None:
    pool = _pool()
    test = [*list(pool)[:1], *[str(i) for i in range(1, 50)]]
    with pytest.raises(ValueError, match="test labels"):
        make_splits(pool, test, n_train=80, n_dev=20, n_test=50, seed=42)


def test_make_splits_checks_sizes() -> None:
    with pytest.raises(ValueError):
        make_splits(_pool(), ["1", "2"], n_train=80, n_dev=20, n_test=50, seed=42)
    with pytest.raises(ValueError):
        make_splits(
            _pool(), [str(i) for i in range(1, 51)], n_train=70, n_dev=20, n_test=50, seed=1
        )


# --- save / verify ---------------------------------------------------------------------------


def test_verify_or_write_creates_then_verifies(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    s = _small_splits()
    assert verify_or_write(path, s, {"seed": 42}) == "created"
    assert verify_or_write(path, s, {"seed": 42}) == "verified"
    assert read_splits(path) == s
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["splits"]["dev"]["sha256"] == ids_checksum(s.dev)
    assert b"\r\n" not in path.read_bytes()


def test_verify_or_write_raises_when_recreated_splits_differ(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    verify_or_write(path, _small_splits(42), {})
    with pytest.raises(SplitMismatchError):
        verify_or_write(path, _small_splits(7), {})


def test_tampered_split_file_fails_checksum(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    verify_or_write(path, _small_splits(), {})
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["splits"]["dev"]["ids"][0] = "999999"
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(SplitIntegrityError):
        read_splits(path)


def test_overlapping_splits_are_rejected(tmp_path: Path) -> None:
    s = _small_splits()
    bad = Splits(train=s.train, dev=(*s.dev[:-1], s.train[0]), test=s.test)
    with pytest.raises(SplitIntegrityError):
        bad.check_disjoint()


# --- test-set lock ---------------------------------------------------------------------------


def test_guard_blocks_test_split_by_default() -> None:
    with pytest.raises(TestSetLockedError):
        check_split_access("test")
    check_split_access("dev")
    check_split_access("test", allow_test=True)


def test_guard_blocks_test_ids() -> None:
    with pytest.raises(TestSetLockedError):
        check_ids_not_test(["1", "2"], ["2", "3"])
    check_ids_not_test(["1"], ["2", "3"])
    check_ids_not_test(["2"], ["2"], allow_test=True)


def test_load_split_ids_locks_test(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    s = _small_splits()
    verify_or_write(path, s, {})
    assert load_split_ids("dev", path) == s.dev
    with pytest.raises(TestSetLockedError):
        load_split_ids("test", path)
    assert load_split_ids("test", path, allow_test=True) == s.test


def test_get_split_examples_checks_lock_before_loading(tmp_path: Path, row_factory) -> None:
    path = tmp_path / "splits.json"
    s = _small_splits()
    verify_or_write(path, s, {})

    def loader_must_not_run():
        raise AssertionError("dataset was loaded for a locked split")

    with pytest.raises(TestSetLockedError):
        get_split_examples("test", loader_must_not_run, path)

    rows = [row_factory(int(i)) for i in (*s.train, *s.dev, *s.test)]
    examples = examples_by_pmid(rows)
    dev = get_split_examples("dev", lambda: examples, path)
    assert [e.pmid for e in dev] == list(s.dev)


# --- end to end with fake data -----------------------------------------------------------------


def test_build_splits_from_config_never_reads_test_labels(tmp_path: Path, row_factory) -> None:
    pool = _pool()
    test_ids = [str(i) for i in range(1, 51)]
    rows = [row_factory(int(p), lab) for p, lab in pool.items()]
    rows += [row_factory(int(t), "yes") for t in test_ids]
    (tmp_path / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    cfg = {
        "dataset": {"name": "fake", "config": "pqa_labeled", "revision": "r"},
        "splits": {
            "official_test_pmids": str(tmp_path / "test.txt"),
            "official_test_source": "fake",
            "seed": 42,
            "n_train": 80,
            "n_dev": 20,
            "n_test": 50,
        },
    }
    splits, meta = build_splits_from_config(cfg, lambda: examples_by_pmid(rows))
    assert set(splits.test) == set(test_ids)
    assert set(meta["label_counts"]) == {"train", "dev"}
    assert sum(meta["label_counts"]["train"].values()) == 80


# --- the committed split file ----------------------------------------------------------------


@pytest.mark.skipif(not COMMITTED.exists(), reason="split file not created yet")
def test_committed_split_file_matches_protocol() -> None:
    s = read_splits(COMMITTED)
    assert (len(s.train), len(s.dev), len(s.test)) == (400, 100, 500)
    cfg = load_config(resolve("configs/data.yaml"))
    assert set(s.test) == set(read_id_list(cfg["splits"]["official_test_pmids"]))
