# ADR 0007: Content-addressed on-disk response cache

- **Status:** accepted
- **Date:** 2026-11-23

## Context

A 7B model on a free T4 takes several seconds per answer and Colab sessions disconnect. Re-running
a finished question wastes GPU quota; reusing a stale answer after a prompt change would be a
silent bug.

## Decision

- One JSON file per response, named by SHA-256 of a canonical JSON of: model name, revision,
  quantization, generation settings, prompt version and the full chat messages.
- Atomic writes (temp file + rename): a crash never leaves a half-written entry.
- On Colab the cache lives on Google Drive, so runs resume after a disconnect.

## Alternatives considered

- Key on question id only: a prompt or setting change would silently reuse old answers.
- One big cache file (e.g. SQLite or JSONL): a crash mid-write can corrupt it; harder to sync
  through Drive.

## Consequences

Changing anything that affects the output automatically misses the cache. The cache is never
committed (it can be large and is not a result); results files are.
