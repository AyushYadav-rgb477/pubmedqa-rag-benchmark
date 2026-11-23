"""A deterministic fake LLM for tests: no GPU, no downloads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from biomedqa.llm.client import Generation, GenerationConfig, Message


class FakeClient:
    """Returns ``respond(messages)``; records every call."""

    def __init__(
        self,
        respond: Callable[[Sequence[Message]], str] | str = '{"answer": "yes"}',
        *,
        name: str = "fake-model",
        generation: GenerationConfig | None = None,
    ) -> None:
        self._respond = respond if callable(respond) else (lambda _m, _r=respond: _r)
        self._name = name
        self._generation = generation or GenerationConfig()
        self.calls: list[list[dict[str, str]]] = []

    @property
    def identity(self) -> dict[str, Any]:
        return {"name": self._name, "revision": "fake", "quantization": None}

    @property
    def generation(self) -> GenerationConfig:
        return self._generation

    def generate(self, messages: Sequence[Message]) -> Generation:
        self.calls.append([dict(m) for m in messages])
        text = self._respond(messages)
        n_in = sum(len(m["content"].split()) for m in messages)
        n_out = len(text.split())
        return Generation(
            text=text,
            n_prompt_tokens=n_in,
            n_output_tokens=n_out,
            hit_max_tokens=n_out >= self._generation.max_new_tokens,
        )
