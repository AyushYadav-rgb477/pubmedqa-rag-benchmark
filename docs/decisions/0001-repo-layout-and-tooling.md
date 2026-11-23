# ADR 0001: Repository layout and tooling

- **Status:** accepted
- **Date:** 2026-11-23

## Context

The project mixes light code (data handling, scoring, parsing) that must be tested on every
change with heavy code (PyTorch, 4-bit LLMs, BM25) that only runs on a Colab GPU.
CI must stay fast and free, and imports must not break when heavy libraries are missing.

## Decision

- `src/` layout with one package, `biomedqa`. Tests import the installed package, not the
  working folder, so packaging mistakes show up in CI.
- Heavy dependencies are optional extras per phase (`classifier`, `llm`, `corpus`, `retrieval`).
  CI installs only `.[dev]`. Heavy imports happen inside functions; tests use fakes.
- ruff for lint and format (one fast tool), pytest, pre-commit with the same ruff version as CI.
- LF line endings via `.gitattributes` so checksums are the same on Windows, Linux and Colab.

## Alternatives considered

- Flat layout: simpler, but tests can pass locally on files that are not actually packaged.
- One big dependency list: CI would install multi-GB GPU wheels for no benefit.
- black + isort + flake8: three tools doing what ruff does alone.

## Consequences

Every module touching torch/transformers/bm25s must keep its imports lazy and offer a seam for a
fake in tests.
