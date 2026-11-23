"""Build the LLM client from configs/llm.yaml."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from biomedqa.llm.client import GenerationConfig, LLMClient

DEFAULT_DEVICE_MAP: dict[str, int] = {"": 0}


def device_map_from_config(model_cfg: Mapping[str, Any]) -> Any:
    """``device_map`` from the config; default keeps the whole model on GPU 0.

    YAML may load the key ``""`` fine, but guard against an empty or missing value.
    """
    dm = model_cfg.get("device_map")
    return DEFAULT_DEVICE_MAP if dm in (None, {}, "") else dm


def make_client(cfg: Mapping[str, Any], client_cls: Any = None) -> LLMClient:
    """Create the configured client. ``client_cls`` lets tests inject a stand-in."""
    if client_cls is None:
        from biomedqa.llm.hf_client import HFClient as client_cls
    m = cfg["model"]
    return client_cls(
        m["name"],
        m.get("revision"),
        quantization=m.get("quantization"),
        generation=GenerationConfig.from_dict(cfg["generation"]),
        device_map=device_map_from_config(m),
        torch_dtype=m.get("torch_dtype", "float16"),
    )
