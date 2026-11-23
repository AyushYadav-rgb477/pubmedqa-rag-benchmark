"""Local Hugging Face causal LM (4-bit NF4 on a Colab T4). Heavy imports are lazy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from biomedqa.llm.client import Generation, GenerationConfig, Message


def _dtype(name: str | None):
    import torch

    return None if name is None else getattr(torch, name)


def bnb_config(quantization: Mapping[str, Any] | None):
    """BitsAndBytesConfig from the YAML mapping, or None for an unquantized model."""
    if not quantization:
        return None
    from transformers import BitsAndBytesConfig

    q = dict(quantization)
    if "bnb_4bit_compute_dtype" in q:
        q["bnb_4bit_compute_dtype"] = _dtype(q["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**q)


class HFClient:
    def __init__(
        self,
        name: str,
        revision: str | None,
        *,
        quantization: Mapping[str, Any] | None,
        generation: GenerationConfig,
        device_map: Any = None,
        torch_dtype: str | None = "float16",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device_map is None:
            device_map = {"": 0}
        self._identity = {
            "name": name,
            "revision": revision,
            "quantization": dict(quantization) if quantization else None,
            "torch_dtype": torch_dtype,
        }
        self._generation = generation
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            quantization_config=bnb_config(quantization),
            torch_dtype=_dtype(torch_dtype),
            device_map=device_map,
        )
        self.model.eval()
        self.device_map = getattr(self.model, "hf_device_map", device_map)

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    @property
    def generation(self) -> GenerationConfig:
        return self._generation

    def _hf_generation_config(self):
        """A fresh config so none of the model's sampling defaults leak in (ADR 0005)."""
        from transformers import GenerationConfig as HFGenerationConfig

        base = self.model.generation_config
        g = self._generation
        pad = self.tokenizer.pad_token_id
        return HFGenerationConfig(
            max_new_tokens=g.max_new_tokens,
            do_sample=g.do_sample,
            num_beams=g.num_beams,
            repetition_penalty=g.repetition_penalty,
            eos_token_id=base.eos_token_id,
            pad_token_id=pad if pad is not None else base.pad_token_id,
            bos_token_id=base.bos_token_id,
        )

    def generate(self, messages: Sequence[Message]) -> Generation:
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            [dict(m) for m in messages], add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(input_ids, torch.Tensor):  # some versions return a BatchEncoding
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self._hf_generation_config(),
            )
        new = out[0, input_ids.shape[1] :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        ended = bool(new) and new[-1] in eos_ids
        return Generation(
            text=self.tokenizer.decode(new, skip_special_tokens=True).strip(),
            n_prompt_tokens=int(input_ids.shape[1]),
            n_output_tokens=len(new),
            hit_max_tokens=(not ended) and len(new) >= self._generation.max_new_tokens,
        )
