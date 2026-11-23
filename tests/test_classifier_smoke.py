"""End-to-end smoke test of the training loop with a tiny random BERT on CPU.

Skipped in CI (no torch). Runs locally / on Colab when torch and transformers are installed.
Nothing is downloaded: the tiny model and its vocabulary are built in a temp folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biomedqa.data.pubmedqa import example_from_row

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def tiny_model_dir(tmp_path: Path) -> Path:
    words = [
        "does",
        "treatment",
        "work",
        "background",
        "results",
        "of",
        "the",
        "answer",
        "is",
        "yes",
        "no",
        "maybe",
    ]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words, *[str(i) for i in range(100)]]
    # Built with the `tokenizers` library so it behaves the same on transformers 4.x and 5.x.
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    wp = Tokenizer(models.WordPiece(vocab={w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    wp.normalizer = normalizers.BertNormalizer(lowercase=True)
    wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    wp.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=wp,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=3,
    )
    model = transformers.BertForSequenceClassification(config)
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def _cfg(model_dir: Path) -> dict:
    return {
        "system": "tiny",
        "model": {"name": str(model_dir), "revision": None},
        "input": {"max_length": 32, "truncation": "only_second", "use_long_answer": False},
        "train": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "per_device_batch_size": 4,
            "grad_accum_steps": 2,
            "epochs": 3,
            "max_grad_norm": 1.0,
            "fp16": True,  # ignored on CPU
            "eval_batch_size": 8,
        },
    }


def test_encode_truncates_context_never_question(tiny_model_dir: Path, row_factory) -> None:
    from biomedqa.classifier.train import encode

    tok = transformers.AutoTokenizer.from_pretrained(str(tiny_model_dir))
    row = row_factory(1)
    row["context"]["contexts"] = ["results " * 500]
    ex = example_from_row(row)
    (feat,) = encode(tok, [ex], {"input": {"max_length": 32, "truncation": "only_second"}})
    assert len(feat["input_ids"]) == 32
    decoded = tok.decode(feat["input_ids"], skip_special_tokens=True)
    assert decoded.startswith("does treatment 1 work")  # the full question survives
    assert feat["labels"] == 0


def test_train_one_seed_runs_and_saves_best(tiny_model_dir: Path, tmp_path: Path, row_factory):
    from biomedqa.classifier.train import (
        load_tokenizer_and_model,
        predict_probs,
        train_one_seed,
    )

    labels = ["yes", "no", "maybe"]
    train = [example_from_row(row_factory(i, labels[i % 3])) for i in range(1, 13)]
    dev = [example_from_row(row_factory(i, labels[i % 3])) for i in range(50, 56)]
    cfg = _cfg(tiny_model_dir)
    ckpt = tmp_path / "ckpt"

    res = train_one_seed(cfg, train, dev, seed=0, checkpoint_dir=ckpt)
    assert len(res["history"]) == 3
    assert 1 <= res["best_epoch"] <= 3
    assert len(res["dev_probs"]) == len(dev)
    assert all(abs(sum(p) - 1) < 1e-4 for p in res["dev_probs"])

    # the saved checkpoint reproduces the selected epoch's dev probabilities
    tok, model = load_tokenizer_and_model(cfg, path=str(ckpt))
    probs = predict_probs(model, tok, dev, cfg, torch.device("cpu"))
    for a, b in zip(probs, res["dev_probs"], strict=True):
        assert a == pytest.approx(b, abs=1e-5)

    # same seed -> same result on CPU
    res2 = train_one_seed(cfg, train, dev, seed=0)
    assert [h["macro_f1"] for h in res2["history"]] == [h["macro_f1"] for h in res["history"]]
