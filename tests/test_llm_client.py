from __future__ import annotations

import importlib

import pytest

from biomedqa.config import load_config, resolve
from biomedqa.llm.client import Generation, GenerationConfig, LLMClient
from biomedqa.llm.factory import DEFAULT_DEVICE_MAP, device_map_from_config, make_client
from biomedqa.llm.fake import FakeClient


def test_generation_config_defaults_are_pure_greedy() -> None:
    g = GenerationConfig()
    assert g.to_dict() == {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    assert GenerationConfig.from_dict(g.to_dict()) == g


def test_sampling_is_rejected() -> None:
    with pytest.raises(ValueError, match="sampling"):
        GenerationConfig(do_sample=True)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)


def test_generation_roundtrip_drops_cached_flag() -> None:
    g = Generation("hi", 3, 1, False, cached=True)
    d = g.to_dict()
    assert "cached" not in d
    assert Generation.from_dict(d, cached=True) == g


def test_fake_client_implements_protocol_and_records_calls() -> None:
    fake = FakeClient(lambda msgs: msgs[-1]["content"].upper())
    assert isinstance(fake, LLMClient)
    out = fake.generate([{"role": "user", "content": "is it?"}])
    assert out.text == "IS IT?"
    assert fake.calls == [[{"role": "user", "content": "is it?"}]]


class _Recorder:
    def __init__(self, name, revision, **kw):
        self.args = {"name": name, "revision": revision, **kw}


def test_factory_passes_device_map_and_generation() -> None:
    cfg = load_config(resolve("configs/llm.yaml"))
    client = make_client(cfg, client_cls=_Recorder)
    a = client.args
    assert a["device_map"] == {"": 0}
    assert a["generation"] == GenerationConfig()
    assert a["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert a["revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_DEVICE_MAP),
        ({}, DEFAULT_DEVICE_MAP),
        ("auto", "auto"),
        ({"": "cpu"}, {"": "cpu"}),
    ],
)
def test_device_map_from_config(value, expected) -> None:
    model_cfg = {} if value is None else {"device_map": value}
    assert device_map_from_config(model_cfg) == expected


def test_llm_config_matches_protocol() -> None:
    cfg = load_config(resolve("configs/llm.yaml"))
    m = cfg["model"]
    assert m["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert m["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": False,
    }
    assert m["torch_dtype"] == "float16"
    assert cfg["prompt_version"] == "v1"
    GenerationConfig.from_dict(cfg["generation"])  # validates greedy


def test_hf_client_module_imports_without_torch() -> None:
    importlib.import_module("biomedqa.llm.hf_client")
