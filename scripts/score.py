"""Score a prediction JSONL file on one split and write a metrics JSON.

Usage:
    python scripts/score.py results/classifier/seed0/dev_predictions.jsonl --split dev \
        --out results/classifier/seed0/dev_metrics.json

The test split is refused unless --allow-test is given (reserved for Phase 4).
"""

from __future__ import annotations

import argparse
import json

from biomedqa.data.splits import DEFAULT_SPLITS_FILE
from biomedqa.fileio import write_json
from biomedqa.scoring.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED
from biomedqa.scoring.report import score_file


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    ap.add_argument("--n-resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--allow-test", action="store_true", help="Phase 4 only")
    args = ap.parse_args(argv)

    report = score_file(
        args.predictions,
        args.split,
        splits_file=args.splits_file,
        allow_test=args.allow_test,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    if args.out:
        write_json(args.out, report)
    summary = {k: report[k] for k in ("system", "split", "n", "accuracy", "macro_f1")}
    summary["n_parse_failures"] = report["n_parse_failures"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
