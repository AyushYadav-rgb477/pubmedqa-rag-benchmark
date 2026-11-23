"""Smoke test of HFClient with a tiny random Qwen2 model on CPU (skipped when torch is absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from biomedqa.llm.client import GenerationConfig  # noqa: E402
from biomedqa.llm.hf_client import HFClient  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["<pad>", "<unk>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    words += ["answer", "yes", "no", "maybe", "question", "is", "it", "the", "a", "json"]
    wl = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    wl.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=wl, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>"
    )
    tok.chat_template = CHAT_TEMPLATE
    config = transformers.Qwen2Config(
        vocab_size=len(words),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=None,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config)
    # Sampling defaults like the real Qwen2.5 config: the client must override them.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.repetition_penalty = 1.05
    out = tmp_path_factory.mktemp("tiny_qwen")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def _client(path: Path, max_new_tokens: int = 6) -> HFClient:
    return HFClient(
        str(path),
        None,
        quantization=None,
        generation=GenerationConfig(max_new_tokens=max_new_tokens),
        device_map={"": "cpu"},
        torch_dtype="float32",
    )


MESSAGES = [
    {"role": "system", "content": "answer yes no maybe"},
    {"role": "user", "content": "question is it the answer"},
]


def test_hf_client_overrides_sampling_defaults(tiny_qwen: Path) -> None:
    client = _client(tiny_qwen)
    hf = client._hf_generation_config()
    assert hf.do_sample is False
    assert hf.repetition_penalty == 1.0
    assert hf.num_beams == 1
    assert hf.max_new_tokens == 6


def test_hf_client_generation_is_deterministic(tiny_qwen: Path) -> None:
    client = _client(tiny_qwen)
    a = client.generate(MESSAGES)
    b = client.generate(MESSAGES)
    assert a == b
    assert a.n_prompt_tokens > 0
    assert 0 < a.n_output_tokens <= 6
    assert a.hit_max_tokens == (a.n_output_tokens == 6)


def test_hf_client_matches_manual_greedy(tiny_qwen: Path) -> None:
    client = _client(tiny_qwen, max_new_tokens=4)
    ids = client.tokenizer.apply_chat_template(
        MESSAGES, add_generation_prompt=True, return_tensors="pt"
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    seq = ids
    with torch.no_grad():
        for _ in range(4):
            nxt = client.model(seq).logits[0, -1].argmax().view(1, 1)
            seq = torch.cat([seq, nxt], dim=1)
            if nxt.item() == 2:
                break
    expected = client.tokenizer.decode(seq[0, ids.shape[1] :], skip_special_tokens=True).strip()
    assert client.generate(MESSAGES).text == expected
