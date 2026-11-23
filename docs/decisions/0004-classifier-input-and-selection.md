# ADR 0004: Classifier input and model selection

- **Status:** accepted
- **Date:** 2026-11-23

## Context

Each PubMedQA question comes with the abstract's context paragraphs and a `long_answer`, which is
the abstract's conclusion. Fine-tuning results vary a lot between random seeds on 400 examples.

## Decision

- Input = (question, context paragraphs). `long_answer` is never used, because the conclusion
  nearly states the yes / no / maybe answer and the LLM systems do not get it either.
- Max 512 tokens; only the context is truncated so the question is always seen in full.
- 5 seeds; per seed, keep the epoch with the best dev macro-F1; report mean and std over seeds.

## Alternatives considered

- Include `long_answer` (the "reasoning-required" vs "reasoning-free" setting of the paper):
  much easier task and not comparable with the oracle LLM, which sees only the context.
- One seed: a single lucky or unlucky run could change the conclusion.

## Consequences

Dev scores are optimistic because dev chooses the epoch. Test (Phase 4) gives the clean estimate.
