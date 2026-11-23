# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-23

Phase 2: LLM layer, closed-book and oracle.

### Added
- Closed-book and oracle dev results in `results/llm/` (Colab T4, Qwen2.5-7B-Instruct 4-bit).
- Phase 2 protocol (section 4: model, decoding, prompt v1, parser, cache, majority baseline) and
  ADRs 0005-0007, registered before any LLM run.
- LLM client interface, fake client, 4-bit NF4 Hugging Face client with pure greedy decoding,
  `device_map` from config.
- Content-addressed on-disk response cache with atomic writes.
- Frozen prompt v1 and a robust JSON parser (regex fallback on the answer field only;
  fabricated citation counting).
- Closed-book and oracle systems, shared runner, LLM statistics, `scripts/run_llm.py`.
- Majority-class baseline and its dev results (`results/baselines/majority/`).
- Colab notebook `notebooks/02_llm_colab.ipynb`.

### Fixed
- Greedy decoding on transformers 4.51.3: the model's sampling defaults were merged back into
  the passed generation config. The client now replaces the model's generation config and
  checks it before every call (found by the Colab test run, before any LLM result).

## [0.1.0] - 2026-11-23

Phase 1: data, scorer, classifier baseline.

### Added
- 5-seed PubMedBERT dev results in `results/classifier/` (Colab T4).
- Phase 1 protocol (splits, scoring, classifier) and ADRs 0002-0004, registered before any run.
- PubMedQA `pqa_labeled` loader pinned to a dataset revision.
- Splits: official 500 test PMIDs, stratified train 400 / dev 100 (seed 42), SHA-256 checksums;
  re-creating the splits raises on any difference.
- Test-set lock: `TestSetLockedError` unless `allow_test=True` / `--allow-test`.
- Shared `Prediction` record and atomic JSONL prediction files.
- Scorer: accuracy, macro-F1, per-class metrics, confusion matrix with `invalid` column,
  parse failures counted as wrong, percentile bootstrap 95% CI (10,000 resamples, seed 42).
- PubMedBERT classifier: training, prediction and seed summary; Colab notebook.
- `scripts/update_readme_results.py`: README results table generated from `results/` files.
- Data card with split statistics.

### Changed
- Merge policy: merge commits only; no force push, squash merge or rebase merge.

### Fixed
- `.gitignore` rule `data/` also ignored `src/biomedqa/data/`; anchored to `/data/`.

## [0.0.1] - 2026-11-23

### Added
- Project scaffold: src-layout package `biomedqa`, `pyproject.toml` with per-phase extras,
  ruff, pytest, pre-commit, GitHub Actions CI.
- Project docs: README, CLAUDE.md, experiment protocol stub, data card stub, ADR template,
  MIT license.
