"""Run an LLM system (closed_book or oracle) on one split (GPU / Colab).

Usage:
    python scripts/run_llm.py --system closed_book --split dev \
        --cache-dir /content/drive/MyDrive/biomedqa/llm_cache

    # quick check that the model loads and answers, writes nothing to results/:
    python scripts/run_llm.py --system oracle --split dev --limit 3

Writes results/llm/<system>/<split>_{predictions.jsonl,metrics.json,run.json}. Re-running after a
disconnect re-uses every cached answer. The test split is refused without --allow-test (Phase 4).
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

from biomedqa.env import configure_safe_env

configure_safe_env()  # before anything can import JAX / torch

from biomedqa.config import load_config, resolve  # noqa: E402
from biomedqa.data.pubmedqa import load_all_labeled  # noqa: E402
from biomedqa.data.splits import DEFAULT_SPLITS_FILE, get_split_examples  # noqa: E402
from biomedqa.guard import check_split_access  # noqa: E402
from biomedqa.systems.llm import EVIDENCE, answer_one  # noqa: E402
from biomedqa.systems.runner import llm_run_hash, run_llm_split  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, choices=sorted(EVIDENCE))
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--cache-dir", default=None, help="response cache (default: config)")
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    ap.add_argument("--hf-cache-dir", default=None, help="datasets cache")
    ap.add_argument("--limit", type=int, default=None, help="answer N questions, save nothing")
    ap.add_argument("--allow-test", action="store_true", help="Phase 4 only")
    args = ap.parse_args(argv)
    check_split_access(args.split, args.allow_test)

    cfg = load_config(resolve(args.config))
    revision = load_config(resolve(cfg["data_config"]))["dataset"]["revision"]
    load_all = functools.cache(lambda: load_all_labeled(revision, cache_dir=args.hf_cache_dir))
    examples = get_split_examples(
        args.split, load_all, args.splits_file, allow_test=args.allow_test
    )

    from biomedqa.llm.factory import make_cached_client

    client = make_cached_client(cfg, args.cache_dir)
    print(f"model loaded; device_map={getattr(client.inner, 'device_map', None)}", flush=True)

    if args.limit:
        h = llm_run_hash(cfg, args.system)
        for ex in examples[: args.limit]:
            p = answer_one(ex, client, EVIDENCE[args.system], system=args.system, cfg_hash=h)
            row = {
                "id": p.id,
                "gold": p.gold,
                "pred": p.predicted,
                "method": p.extra["parse_method"],
                "raw": p.raw_output[:300],
            }
            print(json.dumps(row), flush=True)
        print("limit run: nothing saved")
        return

    out_dir = Path(args.out) / args.system
    report = run_llm_split(
        args.system,
        args.split,
        examples,
        client,
        cfg,
        out_dir,
        splits_file=args.splits_file,
        allow_test=args.allow_test,
    )
    s = report["llm"]
    print(
        f"{args.system} {args.split}: accuracy={report['accuracy']:.4f} "
        f"macro_f1={report['macro_f1']:.4f} parse_failures={report['n_parse_failures']} "
        f"regex_fallback={s['n_regex_fallback']} hit_max_tokens={s['n_hit_max_tokens']}"
    )


if __name__ == "__main__":
    main()
