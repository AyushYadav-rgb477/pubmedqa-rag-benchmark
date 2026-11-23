from __future__ import annotations

from pathlib import Path

import pytest

from biomedqa.llm.cache import CachedClient, ResponseCache, cache_key, key_fields
from biomedqa.llm.client import GenerationConfig
from biomedqa.llm.fake import FakeClient

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "q?"}]


def _key(**over) -> str:
    args = {
        "identity": {"name": "m", "revision": "r", "quantization": {"bits": 4}},
        "generation": GenerationConfig(),
        "prompt_version": "v1",
        "messages": MSGS,
    }
    args.update(over)
    return cache_key(key_fields(**args))


def test_key_is_stable_and_order_independent() -> None:
    a = _key()
    b = _key(identity={"quantization": {"bits": 4}, "revision": "r", "name": "m"})
    assert a == b
    assert len(a) == 64


@pytest.mark.parametrize(
    "over",
    [
        {"identity": {"name": "m2", "revision": "r", "quantization": {"bits": 4}}},
        {"identity": {"name": "m", "revision": "r2", "quantization": {"bits": 4}}},
        {"identity": {"name": "m", "revision": "r", "quantization": None}},
        {"generation": GenerationConfig(max_new_tokens=10)},
        {"generation": GenerationConfig(repetition_penalty=1.05)},
        {"prompt_version": "v2"},
        {"messages": [*MSGS[:1], {"role": "user", "content": "q? "}]},
        {"messages": MSGS[1:]},
    ],
)
def test_every_field_changes_the_key(over) -> None:
    assert _key(**over) != _key()


def test_cached_client_hits_and_misses(tmp_path: Path) -> None:
    fake = FakeClient(lambda m: f"answer to {m[-1]['content']}")
    client = CachedClient(fake, ResponseCache(tmp_path), "v1")
    first = client.generate(MSGS)
    second = client.generate(MSGS)
    assert first.text == second.text == "answer to q?"
    assert not first.cached and second.cached
    assert (client.hits, client.misses) == (1, 1)
    assert len(fake.calls) == 1


def test_cache_persists_across_clients(tmp_path: Path) -> None:
    CachedClient(FakeClient("first"), ResponseCache(tmp_path), "v1").generate(MSGS)
    fake2 = FakeClient("second")
    out = CachedClient(fake2, ResponseCache(tmp_path), "v1").generate(MSGS)
    assert out.text == "first"
    assert fake2.calls == []


def test_prompt_version_change_misses(tmp_path: Path) -> None:
    CachedClient(FakeClient("old"), ResponseCache(tmp_path), "v1").generate(MSGS)
    out = CachedClient(FakeClient("new"), ResponseCache(tmp_path), "v2").generate(MSGS)
    assert out.text == "new"


def test_corrupt_entry_is_recomputed(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    client = CachedClient(FakeClient("good"), cache, "v1")
    client.generate(MSGS)
    (path,) = list(tmp_path.rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    out = CachedClient(FakeClient("again"), cache, "v1").generate(MSGS)
    assert out.text == "again"
    assert "again" in path.read_text(encoding="utf-8")


def test_entry_stores_key_fields_for_audit(tmp_path: Path) -> None:
    import json

    CachedClient(FakeClient("x"), ResponseCache(tmp_path), "v1").generate(MSGS)
    (path,) = list(tmp_path.rglob("*.json"))
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["key_fields"]["prompt_version"] == "v1"
    assert entry["key_fields"]["messages"] == MSGS
    assert path.parent.name == entry["key"][:2]


def test_make_cached_client_uses_config(tmp_path: Path) -> None:
    from biomedqa.config import load_config, resolve
    from biomedqa.llm.factory import make_cached_client

    cfg = load_config(resolve("configs/llm.yaml"))

    def fake_cls(name, revision, **kw):
        return FakeClient("ok", name=name, generation=kw["generation"])

    client = make_cached_client(cfg, tmp_path, client_cls=fake_cls)
    assert client.prompt_version == "v1"
    assert client.cache.root == tmp_path
    assert client.generate(MSGS).text == "ok"


def test_failed_write_leaves_no_entry(tmp_path: Path, monkeypatch) -> None:
    import biomedqa.fileio as fileio

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(fileio.os, "replace", boom)
    with pytest.raises(OSError):
        CachedClient(FakeClient("x"), ResponseCache(tmp_path), "v1").generate(MSGS)
    assert list(tmp_path.rglob("*.json")) == []
    assert list(tmp_path.rglob("*.tmp")) == []
