"""Fine-tune and run PubMedBERT (GPU, Colab). Heavy imports happen inside functions."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from biomedqa.classifier.core import input_pair, label_ids, select_best_epoch
from biomedqa.data.pubmedqa import Example
from biomedqa.labels import ID_TO_LABEL, LABEL_TO_ID, LABELS
from biomedqa.scoring.metrics import score


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer_and_model(cfg: Mapping[str, Any], path: str | None = None):
    """From the pinned Hub revision, or from a saved checkpoint folder if ``path`` is given."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = path or cfg["model"]["name"]
    kw = {} if path else {"revision": cfg["model"]["revision"]}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=len(LABELS), id2label=dict(ID_TO_LABEL), label2id=dict(LABEL_TO_ID), **kw
    )
    return tok, model


def encode(tokenizer, examples: Sequence[Example], cfg: Mapping[str, Any]) -> list[dict]:
    inp = cfg["input"]
    pairs = [input_pair(e, inp.get("use_long_answer", False)) for e in examples]
    enc = tokenizer(
        [q for q, _ in pairs],
        [c for _, c in pairs],
        truncation=inp["truncation"],
        max_length=inp["max_length"],
    )
    labels = label_ids(examples)
    return [{**{k: enc[k][i] for k in enc}, "labels": labels[i]} for i in range(len(examples))]


def _loader(features, tokenizer, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        features,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def predict_probs(model, tokenizer, examples, cfg, device) -> list[list[float]]:
    import torch

    feats = encode(tokenizer, examples, cfg)
    loader = _loader(feats, tokenizer, cfg["train"]["eval_batch_size"], shuffle=False, seed=0)
    model.eval()
    out: list[list[float]] = []
    use_amp = bool(cfg["train"].get("fp16")) and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            out.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
    return out


def _param_groups(model, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay, plain = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (plain if any(nd in name for nd in no_decay) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def train_one_seed(
    cfg: Mapping[str, Any],
    train_examples: Sequence[Example],
    dev_examples: Sequence[Example],
    seed: int,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train for the registered number of epochs; keep the epoch with the best dev macro-F1.

    Returns the per-epoch history, the selected epoch and the dev probabilities of that epoch.
    The selected weights are saved to ``checkpoint_dir`` (not to git) for Phase 4.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    t = cfg["train"]
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_tokenizer_and_model(cfg)
    model.to(device)

    train_loader = _loader(
        encode(tok, train_examples, cfg), tok, t["per_device_batch_size"], shuffle=True, seed=seed
    )
    accum = int(t["grad_accum_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * int(t["epochs"])
    optim = torch.optim.AdamW(_param_groups(model, t["weight_decay"]), lr=float(t["learning_rate"]))
    sched = get_linear_schedule_with_warmup(
        optim, round(t["warmup_ratio"] * total_steps), total_steps
    )
    use_amp = bool(t.get("fp16")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dev_gold = [e.label for e in dev_examples]

    history: list[dict[str, Any]] = []
    best_state, best_probs, best_f1 = None, None, -1.0
    start = time.time()
    for epoch in range(1, int(t["epochs"]) + 1):
        model.train()
        losses = []
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * accum)
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["max_grad_norm"]))
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)

        probs = predict_probs(model, tok, dev_examples, cfg, device)
        preds = [ID_TO_LABEL[int(np.argmax(p))] for p in probs]
        s = score(dev_gold, preds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "dev_accuracy": s["accuracy"],
                "macro_f1": s["macro_f1"],
                "elapsed_s": round(time.time() - start, 1),
            }
        )
        print(
            f"seed {seed} epoch {epoch}: loss={history[-1]['train_loss']:.4f} "
            f"dev_acc={s['accuracy']:.4f} dev_macro_f1={s['macro_f1']:.4f}",
            flush=True,
        )
        if s["macro_f1"] > best_f1:  # strict: ties keep the earlier epoch
            best_f1 = s["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = copy.deepcopy(probs)

    best_idx = select_best_epoch(history)
    assert history[best_idx]["macro_f1"] == best_f1
    if checkpoint_dir is not None and best_state is not None:
        model.load_state_dict(best_state)
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)

    return {
        "history": history,
        "best_epoch": history[best_idx]["epoch"],
        "dev_probs": best_probs,
        "train_seconds": round(time.time() - start, 1),
        "device": str(device),
    }
