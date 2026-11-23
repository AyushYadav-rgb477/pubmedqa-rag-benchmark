"""Majority-class baseline on one split (CPU, seconds).

Usage:
    python scripts/run_majority.py --split dev

Writes results/baselines/majority/<split>_{predictions.jsonl,metrics.json,run.json}.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from biomedqa.config import load_config, resolve
from biomedqa.data.pubmedqa import load_all_labeled
from biomedqa.data.splits import DEFAULT_SPLITS_FILE, get_split_examples
from biomedqa.guard import check_split_access
from biomedqa.hashing import config_hash
from biomedqa.systems import majority
from biomedqa.systems.runner import save_run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", default="results/baselines/majority")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-test", action="store_true", help="Phase 4 only")
    args = ap.parse_args(argv)
    check_split_access(args.split, args.allow_test)

    revision = load_config(resolve(args.data_config))["dataset"]["revision"]
    load_all = functools.cache(lambda: load_all_labeled(revision, cache_dir=args.cache_dir))
    train = get_split_examples("train", load_all, args.splits_file)
    examples = get_split_examples(
        args.split, load_all, args.splits_file, allow_test=args.allow_test
    )
    label = majority.majority_label(train)
    h = config_hash({"system": majority.SYSTEM, "label_from": "train", "revision": revision})
    preds = majority.predict(examples, label, h)
    report = save_run(
        preds,
        args.split,
        Path(args.out),
        {
            "system": majority.SYSTEM,
            "split": args.split,
            "label": label,
            "config_hash": h,
            "dataset_revision": revision,
            "label_source": "most frequent label in train",
        },
        splits_file=args.splits_file,
        allow_test=args.allow_test,
    )
    acc, f1 = report["accuracy"], report["macro_f1"]
    print(f"majority label={label}: accuracy={acc:.4f} macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
