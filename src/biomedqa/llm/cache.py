"""Content-addressed on-disk cache of LLM responses (ADR 0007).

Key = SHA-256 over model identity (name, revision, quantization, dtype), generation settings,
prompt version and the full chat messages. One JSON file per response, written atomically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from biomedqa.fileio import atomic_write_text
from biomedqa.hashing import canonical_json, sha256_bytes
from biomedqa.llm.client import Generation, GenerationConfig, LLMClient, Message


def key_fields(
    identity: Mapping[str, Any],
    generation: GenerationConfig,
    prompt_version: str,
    messages: Sequence[Message],
) -> dict[str, Any]:
    return {
        "model": dict(identity),
        "generation": generation.to_dict(),
        "prompt_version": prompt_version,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }


def cache_key(fields: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(fields).encode("utf-8"))


class ResponseCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Generation | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            entry = json.loads(p.read_text(encoding="utf-8"))
            if entry.get("key") != key:
                return None
            return Generation.from_dict(entry["generation"], cached=True)
        except (ValueError, KeyError):
            return None  # unreadable entry: recompute and overwrite

    def put(self, key: str, fields: Mapping[str, Any], generation: Generation) -> None:
        entry = {"key": key, "key_fields": dict(fields), "generation": generation.to_dict()}
        atomic_write_text(self.path(key), json.dumps(entry, ensure_ascii=False, indent=1) + "\n")


class CachedClient:
    """Wraps any :class:`LLMClient`; identical requests are answered from disk."""

    def __init__(self, inner: LLMClient, cache: ResponseCache, prompt_version: str) -> None:
        self.inner = inner
        self.cache = cache
        self.prompt_version = prompt_version
        self.hits = 0
        self.misses = 0

    @property
    def identity(self) -> dict[str, Any]:
        return self.inner.identity

    @property
    def generation(self) -> GenerationConfig:
        return self.inner.generation

    def generate(self, messages: Sequence[Message]) -> Generation:
        fields = key_fields(self.identity, self.generation, self.prompt_version, messages)
        key = cache_key(fields)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        gen = self.inner.generate(messages)
        self.cache.put(key, fields, gen)
        return gen
