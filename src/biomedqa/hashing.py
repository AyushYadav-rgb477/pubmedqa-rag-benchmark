"""SHA-256 helpers for files, ID lists and configs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sort_ids(ids: Iterable[str]) -> list[str]:
    """Sort PMIDs numerically when they are all digits, else lexicographically."""
    ids = list(ids)
    if all(i.isdigit() for i in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def ids_checksum(ids: Iterable[str]) -> str:
    """SHA-256 of the sorted IDs joined by LF with a trailing LF."""
    return sha256_bytes(("\n".join(sort_ids(ids)) + "\n").encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a config: independent of key order and whitespace."""
    return sha256_bytes(canonical_json(config).encode("utf-8"))
