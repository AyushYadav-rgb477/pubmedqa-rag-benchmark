# ADR 0006: One frozen prompt, JSON output, conservative parser

- **Status:** accepted
- **Date:** 2026-11-23

## Context

Closed-book, oracle and RAG must differ only in the evidence they get; otherwise a difference in
scores could come from prompt wording. LLM outputs are sometimes not clean JSON.

## Decision

- One prompt (v1) for every LLM system; only the numbered `Documents` block changes.
- The model returns JSON `{answer, rationale, citations}`.
- Parser: code fences and extra text tolerated; first balanced JSON object; if JSON is broken,
  only an `"answer": "..."` field may be read by regex (counted as a fallback). Free-text
  mentions of yes / no / maybe are never used. Anything else is `invalid` and wrong.
- Citations are parsed and checked (fabricated = outside 1..n documents) but never affect the
  score.
- Prompt changes create a new version; v1 is never edited after it is used.

## Alternatives considered

- Different prompt per system: confounds evidence with wording.
- Strict JSON only: penalises a formatting slip (e.g. a trailing comma) as if it were a wrong
  answer.
- Search the whole output for yes / no / maybe: picks up words from the rationale or the
  question; inflates scores.

## Consequences

Parse failures and fallbacks are visible numbers in every report. Prompt tuning on dev is not
possible without a recorded new version.
