"""Create (first run) or verify (every later run) the PubMedQA splits.

Usage:
    python scripts/make_splits.py --config configs/data.yaml

Raises if the re-created splits differ from the committed split file.
Prints train/dev label counts only; test labels are never read.
"""

from __future__ import annotations

import argparse

from biomedqa.config import load_config, resolve
from biomedqa.data.pubmedqa import load_all_labeled
from biomedqa.data.splits import build_splits_from_config, verify_or_write


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache folder")
    args = ap.parse_args(argv)

    cfg = load_config(resolve(args.config))
    revision = cfg["dataset"]["revision"]
    splits, meta = build_splits_from_config(
        cfg, lambda: load_all_labeled(revision, cache_dir=args.cache_dir)
    )
    status = verify_or_write(cfg["splits"]["file"], splits, meta)
    print(f"splits {status}: {cfg['splits']['file']}")
    print(f"train={len(splits.train)} dev={len(splits.dev)} test={len(splits.test)}")
    print(f"label counts (train/dev only): {meta['label_counts']}")


if __name__ == "__main__":
    main()
