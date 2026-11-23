"""Train PubMedBERT for every registered seed and write dev results (GPU / Colab).

Usage:
    python scripts/train_classifier.py --config configs/classifier.yaml \
        --out results/classifier --checkpoint-dir /content/drive/MyDrive/biomedqa/ckpt/classifier

For each seed writes <out>/seed<k>/{dev_predictions.jsonl, dev_metrics.json, run.json}; finally
<out>/summary.json with mean and std over seeds. Seeds whose dev_metrics.json already exists are
skipped, so the script can be re-run after a Colab disconnect. Only train and dev are touched.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from biomedqa.env import configure_safe_env

configure_safe_env()

from biomedqa.classifier.core import (  # noqa: E402
    probs_to_predictions,
    run_config_hash,
    summarize_seeds,
)
from biomedqa.config import load_config, resolve  # noqa: E402
from biomedqa.data.pubmedqa import load_all_labeled  # noqa: E402
from biomedqa.data.splits import DEFAULT_SPLITS_FILE, get_split_examples  # noqa: E402
from biomedqa.fileio import read_json, write_json  # noqa: E402
from biomedqa.hashing import config_hash, sha256_file  # noqa: E402
from biomedqa.io import write_predictions  # noqa: E402
from biomedqa.scoring.report import score_split  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--out", default="results/classifier")
    ap.add_argument("--checkpoint-dir", default=None, help="where best weights go (not git)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: from config")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(resolve(args.config))
    data_cfg = load_config(resolve(cfg["data_config"]))
    revision = data_cfg["dataset"]["revision"]
    load_all = functools.cache(lambda: load_all_labeled(revision, cache_dir=args.cache_dir))
    train = get_split_examples("train", load_all, args.splits_file)
    dev = get_split_examples("dev", load_all, args.splits_file)
    print(f"train={len(train)} dev={len(dev)}", flush=True)

    from biomedqa.classifier.train import train_one_seed  # heavy imports
    from biomedqa.env import collect_versions

    out = Path(args.out)
    seeds = args.seeds if args.seeds else list(cfg["seeds"])
    per_seed: dict[int, dict] = {}
    for seed in seeds:
        sdir = out / f"seed{seed}"
        metrics_path = sdir / "dev_metrics.json"
        if metrics_path.exists():
            print(f"seed {seed}: results exist, skipping", flush=True)
        else:
            ckpt = Path(args.checkpoint_dir) / f"seed{seed}" if args.checkpoint_dir else None
            res = train_one_seed(cfg, train, dev, seed, checkpoint_dir=ckpt)
            h = run_config_hash(cfg, seed)
            preds = probs_to_predictions(
                dev,
                res["dev_probs"],
                system=cfg["system"],
                cfg_hash=h,
                extra={"seed": seed, "epoch": res["best_epoch"]},
            )
            pred_path = sdir / "dev_predictions.jsonl"
            write_predictions(pred_path, preds)
            report = score_split(preds, "dev", args.splits_file)
            report["predictions_file"] = pred_path.name
            report["predictions_sha256"] = sha256_file(pred_path)
            report["best_epoch"] = res["best_epoch"]
            write_json(
                sdir / "run.json",
                {
                    "system": cfg["system"],
                    "seed": seed,
                    "config": cfg,
                    "config_hash": h,
                    "dataset_revision": revision,
                    "best_epoch": res["best_epoch"],
                    "history": res["history"],
                    "train_seconds": res["train_seconds"],
                    "device": res["device"],
                    "checkpoint_saved": ckpt is not None,
                    "versions": collect_versions(),
                },
            )
            write_json(metrics_path, report)  # written last: marks the seed as complete
        per_seed[seed] = read_json(metrics_path)

    summary = summarize_seeds(per_seed)
    summary.update(
        {
            "system": cfg["system"],
            "split": "dev",
            "config_hash_without_seed": config_hash({k: v for k, v in cfg.items() if k != "seeds"}),
            "note": "Dev selects the epoch and is also reported: optimistic (protocol section 3).",
        }
    )
    write_json(out / "summary.json", summary)
    print(
        f"dev macro-F1 {summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['std']:.4f}; "
        f"accuracy {summary['accuracy']['mean']:.4f} +/- {summary['accuracy']['std']:.4f} "
        f"over {summary['n_seeds']} seeds",
        flush=True,
    )


if __name__ == "__main__":
    main()
