# ADR 0005: Local 4-bit Qwen2.5-7B with pure greedy decoding

- **Status:** accepted
- **Date:** 2026-11-23

## Context

The project must cost $0 and be reproducible by anyone with a free Colab T4 (16 GB). An API model
would cost money and could change behind the same name.

## Decision

- `Qwen/Qwen2.5-7B-Instruct` at a pinned revision, 4-bit NF4 with bitsandbytes, float16 compute.
- `device_map` is a config field, default `{"": 0}`, so all layers stay on the GPU (with `"auto"`
  some layers can land on the CPU on a T4, which is very slow).
- Pure greedy decoding: `do_sample=False` and `repetition_penalty=1.0`, overriding the model's
  default sampling settings. Batch size 1.

## Alternatives considered

- API model (e.g. a hosted LLM): not free, weights not pinned.
- Smaller model (1.5B / 3B): fits easily but is a much weaker reader of abstracts.
- Keep the model's default generation config: sampling makes runs non-reproducible, and its
  repetition penalty silently changes "greedy".

## Consequences

Results are reproducible up to GPU numerical noise. 4-bit weights cost some accuracy compared
with float16, equally for every LLM system, so comparisons between systems stay fair.
