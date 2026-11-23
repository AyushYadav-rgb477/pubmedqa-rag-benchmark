"""Run a saved PubMedBERT checkpoint on one split and score it (GPU / Colab).

Usage (dev, to check a checkpoint reproduces its training-time predictions):
    python scripts/predict_classifier.py --checkpoint <dir>/seed0 --seed 0 --split dev \
        --out results/classifier/seed0/dev_predictions_reloaded.jsonl

The test split is refused unless --allow-test is given (reserved for Phase 4).
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from biomedqa.env import configure_safe_env

configure_safe_env()

from biomedqa.classifier.core import probs_to_predictions, run_config_hash  # noqa: E402
from biomedqa.config import load_config, resolve  # noqa: E402
from biomedqa.data.pubmedqa import load_all_labeled  # noqa: E402
from biomedqa.data.splits import DEFAULT_SPLITS_FILE, get_split_examples  # noqa: E402
from biomedqa.fileio import write_json  # noqa: E402
from biomedqa.guard import check_split_access  # noqa: E402
from biomedqa.io import write_predictions  # noqa: E402
from biomedqa.scoring.report import score_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, required=True, help="seed the checkpoint was trained with")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", required=True, help="prediction JSONL path")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-test", action="store_true", help="Phase 4 only")
    args = ap.parse_args(argv)
    check_split_access(args.split, args.allow_test)  # fail before any loading

    cfg = load_config(resolve(args.config))
    revision = load_config(resolve(cfg["data_config"]))["dataset"]["revision"]
    load_all = functools.cache(lambda: load_all_labeled(revision, cache_dir=args.cache_dir))
    examples = get_split_examples(
        args.split, load_all, args.splits_file, allow_test=args.allow_test
    )

    import torch

    from biomedqa.classifier.train import load_tokenizer_and_model, predict_probs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_tokenizer_and_model(cfg, path=args.checkpoint)
    model.to(device)
    probs = predict_probs(model, tok, examples, cfg, device)
    preds = probs_to_predictions(
        examples,
        probs,
        system=cfg["system"],
        cfg_hash=run_config_hash(cfg, args.seed),
        extra={"seed": args.seed},
    )
    write_predictions(args.out, preds)
    report = score_file(
        args.out, args.split, splits_file=args.splits_file, allow_test=args.allow_test
    )
    report["predictions_file"] = Path(args.out).name
    write_json(Path(args.out).with_suffix(".metrics.json"), report)
    print(f"{args.split}: accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
