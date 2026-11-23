# ADR 0003: Parse failures count as wrong answers

- **Status:** accepted
- **Date:** 2026-11-23

## Context

LLM outputs sometimes cannot be parsed into yes / no / maybe. If those questions were dropped,
a model that fails to answer hard questions would score higher than one that tries and is wrong.

## Decision

An unparsable output gets the prediction `invalid`. It is wrong for accuracy. Macro-F1 is averaged
over the three real labels only, so `invalid` lowers recall of the gold class and adds no false
positive. The confusion matrix has an extra `invalid` column and the parse-failure count is
reported next to every score.

## Alternatives considered

- Drop failures: rewards refusing to answer; different systems would be scored on different
  question sets.
- Map failures to the majority class: hides the failure and silently helps on an imbalanced set.

## Consequences

Every system is scored on exactly the same questions. A parser bug shows up as lower scores and a
visible failure count, not as silently missing rows.
