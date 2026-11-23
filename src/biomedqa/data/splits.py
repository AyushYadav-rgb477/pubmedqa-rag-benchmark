"""Deterministic train / dev / test splits of ``pqa_labeled`` with checksums.

Protocol (docs/experiment-protocol.md, section 1):

* test = the official 500 test PMIDs (given as IDs only; their labels are never read here);
* train / dev = the remaining questions, stratified by gold label with seed 42.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biomedqa.config import resolve
from biomedqa.data.pubmedqa import Example
from biomedqa.fileio import read_json, write_json
from biomedqa.guard import TEST_SPLIT, check_split_access
from biomedqa.hashing import ids_checksum, sort_ids
from biomedqa.labels import LABELS, check_gold

SPLIT_NAMES = ("train", "dev", TEST_SPLIT)
DEFAULT_SPLITS_FILE = "splits/pqa_labeled_splits.json"


class SplitIntegrityError(RuntimeError):
    """The saved split file is corrupt (checksum or overlap problem)."""


class SplitMismatchError(RuntimeError):
    """Re-created splits differ from the saved split IDs."""


@dataclass(frozen=True)
class Splits:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]

    def ids(self, name: str) -> tuple[str, ...]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def check_disjoint(self) -> None:
        tr, dv, te = set(self.train), set(self.dev), set(self.test)
        if tr & dv or tr & te or dv & te:
            raise SplitIntegrityError("splits overlap")


def largest_remainder(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` items across classes proportionally to ``counts``.

    Floors the exact quotas, then hands the leftover units to the largest fractional parts.
    Ties go to the class that comes first in ``LABELS`` order.
    """
    n = sum(counts.values())
    if not 0 <= total <= n:
        raise ValueError(f"cannot allocate {total} from {n} items")
    labels = [lab for lab in LABELS if lab in counts]
    quotas = {lab: total * counts[lab] / n for lab in labels}
    alloc = {lab: int(np.floor(quotas[lab])) for lab in labels}
    leftover = total - sum(alloc.values())
    order = sorted(labels, key=lambda lab: (-(quotas[lab] - alloc[lab]), LABELS.index(lab)))
    for lab in order[:leftover]:
        alloc[lab] += 1
    return alloc


def stratified_train_dev(
    pool_labels: Mapping[str, str], n_dev: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``pool_labels`` (PMID -> gold label) into (train, dev), stratified by label."""
    by_label: dict[str, list[str]] = {lab: [] for lab in LABELS}
    for pmid, label in pool_labels.items():
        by_label[check_gold(label)].append(pmid)
    alloc = largest_remainder({k: len(v) for k, v in by_label.items() if v}, n_dev)
    rng = np.random.default_rng(seed)
    train: list[str] = []
    dev: list[str] = []
    for label in LABELS:
        ids = sort_ids(by_label[label])
        if not ids:
            continue
        perm = [ids[i] for i in rng.permutation(len(ids))]
        k = alloc[label]
        dev.extend(perm[:k])
        train.extend(perm[k:])
    return tuple(sort_ids(train)), tuple(sort_ids(dev))


def make_splits(
    pool_labels: Mapping[str, str],
    test_ids: Iterable[str],
    *,
    n_train: int,
    n_dev: int,
    n_test: int,
    seed: int,
) -> Splits:
    """Build the splits. ``pool_labels`` must hold labels for the non-test questions only."""
    test = tuple(sort_ids(set(test_ids)))
    if len(test) != n_test:
        raise ValueError(f"expected {n_test} unique test ids, got {len(test)}")
    if set(test) & set(pool_labels):
        raise ValueError("pool_labels must not contain test ids (test labels are never read)")
    if len(pool_labels) != n_train + n_dev:
        raise ValueError(f"expected {n_train + n_dev} non-test questions, got {len(pool_labels)}")
    train, dev = stratified_train_dev(pool_labels, n_dev=n_dev, seed=seed)
    splits = Splits(train=train, dev=dev, test=test)
    splits.check_disjoint()
    return splits


def splits_to_json(splits: Splits, meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "meta": dict(meta),
        "splits": {
            name: {"n": len(ids), "sha256": ids_checksum(ids), "ids": list(ids)}
            for name in SPLIT_NAMES
            for ids in [splits.ids(name)]
        },
    }


def write_splits(path: str | Path, splits: Splits, meta: Mapping[str, Any]) -> None:
    write_json(path, splits_to_json(splits, meta))


def read_splits(path: str | Path = DEFAULT_SPLITS_FILE) -> Splits:
    """Read the saved split file and verify its checksums and disjointness."""
    obj = read_json(resolve(path))
    parts: dict[str, tuple[str, ...]] = {}
    for name in SPLIT_NAMES:
        entry = obj["splits"][name]
        ids = tuple(str(i) for i in entry["ids"])
        if len(ids) != entry["n"] or ids_checksum(ids) != entry["sha256"]:
            raise SplitIntegrityError(f"checksum or size mismatch in split {name!r}")
        parts[name] = ids
    splits = Splits(**parts)
    splits.check_disjoint()
    return splits


def verify_or_write(path: str | Path, splits: Splits, meta: Mapping[str, Any]) -> str:
    """Write the splits if no file exists; otherwise raise unless the IDs are identical.

    Returns ``"created"`` or ``"verified"``.
    """
    path = resolve(path)
    if not path.exists():
        write_splits(path, splits, meta)
        return "created"
    saved = read_splits(path)
    for name in SPLIT_NAMES:
        if saved.ids(name) != splits.ids(name):
            raise SplitMismatchError(
                f"re-created split {name!r} differs from {path}. The saved split IDs are the "
                "source of truth; investigate instead of overwriting."
            )
    return "verified"


def load_split_ids(
    name: str, path: str | Path = DEFAULT_SPLITS_FILE, *, allow_test: bool = False
) -> tuple[str, ...]:
    check_split_access(name, allow_test)
    return read_splits(path).ids(name)


def get_split_examples(
    name: str,
    load_all: Callable[[], Mapping[str, Example]],
    path: str | Path = DEFAULT_SPLITS_FILE,
    *,
    allow_test: bool = False,
) -> list[Example]:
    """Examples of one split, in sorted PMID order. The lock is checked before any loading."""
    ids = load_split_ids(name, path, allow_test=allow_test)
    examples = load_all()
    missing = [i for i in ids if i not in examples]
    if missing:
        raise KeyError(f"{len(missing)} split ids are missing from the dataset")
    return [examples[i] for i in ids]


def label_counts(ids: Iterable[str], labels: Mapping[str, str]) -> dict[str, int]:
    c = Counter(labels[i] for i in ids)
    return {lab: c.get(lab, 0) for lab in LABELS}


def read_id_list(path: str | Path) -> list[str]:
    with open(resolve(path), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_splits_from_config(
    cfg: Mapping[str, Any], load_all: Callable[[], Mapping[str, Example]]
) -> tuple[Splits, dict[str, Any]]:
    """Build splits as registered in the protocol. Test labels are dropped before splitting."""
    s = cfg["splits"]
    test_ids = read_id_list(s["official_test_pmids"])
    examples = load_all()
    missing = set(test_ids) - set(examples)
    if missing:
        raise KeyError(f"{len(missing)} official test PMIDs are not in the dataset")
    test_set = set(test_ids)
    pool_labels = {p: ex.label for p, ex in examples.items() if p not in test_set}
    splits = make_splits(
        pool_labels,
        test_ids,
        n_train=s["n_train"],
        n_dev=s["n_dev"],
        n_test=s["n_test"],
        seed=s["seed"],
    )
    meta = {
        "dataset": cfg["dataset"]["name"],
        "config": cfg["dataset"]["config"],
        "revision": cfg["dataset"]["revision"],
        "test_source": s["official_test_source"],
        "seed": s["seed"],
        "method": "official test PMIDs; train/dev stratified by label (largest remainder)",
        "label_counts": {
            "train": label_counts(splits.train, pool_labels),
            "dev": label_counts(splits.dev, pool_labels),
        },
    }
    return splits, meta
