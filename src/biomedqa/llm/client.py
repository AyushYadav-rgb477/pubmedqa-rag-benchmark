"""The interface every LLM client implements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

Message = Mapping[str, str]  # {"role": "system" | "user" | "assistant", "content": str}


@dataclass(frozen=True)
class GenerationConfig:
    """Pure greedy decoding by default (protocol section 4.1)."""

    max_new_tokens: int = 512
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.do_sample:
            raise ValueError("sampling is not allowed: runs must be reproducible (ADR 0005)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GenerationConfig:
        return cls(**dict(d))


@dataclass(frozen=True)
class Generation:
    text: str
    n_prompt_tokens: int = 0
    n_output_tokens: int = 0
    hit_max_tokens: bool = False
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cached")
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], cached: bool = False) -> Generation:
        return cls(
            text=d["text"],
            n_prompt_tokens=int(d.get("n_prompt_tokens", 0)),
            n_output_tokens=int(d.get("n_output_tokens", 0)),
            hit_max_tokens=bool(d.get("hit_max_tokens", False)),
            cached=cached,
        )


@runtime_checkable
class LLMClient(Protocol):
    """``identity`` names everything about the model that changes its output
    (name, revision, quantization); together with ``generation`` it keys the cache."""

    @property
    def identity(self) -> dict[str, Any]: ...

    @property
    def generation(self) -> GenerationConfig: ...

    def generate(self, messages: Sequence[Message]) -> Generation: ...
