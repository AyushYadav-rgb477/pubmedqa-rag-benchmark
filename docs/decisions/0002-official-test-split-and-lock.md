# ADR 0002: Official test split and a test-set lock in code

- **Status:** accepted
- **Date:** 2026-11-23

## Context

PubMedQA has 1000 expert-labelled questions. Published results use an official set of 500 test
questions. On Hugging Face the 1000 come as one block without that split. Looking at test results
while developing (choosing k, prompts, epochs) would make the final numbers optimistic.

## Decision

- Test = the 500 official test PMIDs from the original PubMedQA GitHub repo, pinned to a commit.
  Only the PMIDs are kept; the labels in that file are discarded unread.
- The other 500 become train 400 / dev 100, stratified by label, seed 42.
- Split IDs and SHA-256 checksums are committed. Re-creating the splits raises on any difference.
- Loading or scoring the test split raises `TestSetLockedError` unless `allow_test=True` /
  `--allow-test` is passed. That flag is reserved for Phase 4.

## Alternatives considered

- Random 500 test split: not comparable with published numbers.
- A rule in the README only: easy to break by accident; code cannot forget.

## Consequences

All development uses dev only. The Phase 4 runner must pass `--allow-test` explicitly.
