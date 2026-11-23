# Experiment protocol (pre-registration)

This file records every design choice that could be tuned on results, **before** the run that
uses it. Each section is committed before its run; the git history is the timestamp.
Changes after a run must be added as a dated amendment with the reason, never silently edited.

## 0. Research question and hypotheses

- **RQ.** On PubMedQA (yes / no / maybe), does RAG over PubMed abstracts improve an open LLM
  compared with closed-book answering? How do both compare with an oracle (gold abstract) and a
  fine-tuned PubMedBERT classifier?
- **Primary metric:** macro-F1 over {yes, no, maybe}. **Secondary:** accuracy.

## 1. Data and splits

_Registered 2026-11-23, before any split was created._

- **Questions:** Hugging Face `qiaojin/PubMedQA`, config `pqa_labeled` (1000 expert-labelled
  questions), revision `9001f2853fb87cab8d220904e0de81ac6973b318`. Gold label = `final_decision`.
- **Test (500):** the official PubMedQA test PMIDs, i.e. the keys of `data/test_ground_truth.json`
  in `github.com/pubmedqa/pubmedqa` at commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`
  (downloaded file SHA-256 `939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51`).
  Only the keys were kept (`splits/official_test_pmids.txt`); the labels in that file were
  discarded without being read.
- **Train (400) / dev (100):** the remaining 500 questions, stratified by gold label, seed 42.
  Algorithm (in `biomedqa.data.splits`): dev size per class by largest-remainder proportional
  allocation of 100; within each class (order yes, no, maybe) the PMIDs are sorted numerically and
  permuted with `numpy.random.default_rng(42)`; the first n go to dev, the rest to train.
- **Integrity:** `splits/pqa_labeled_splits.json` stores sorted IDs per split and SHA-256 of
  `"\n".join(sorted_ids) + "\n"`. Re-creating the splits raises if the result differs.
- **Test-set lock:** code refuses to load or score test IDs unless `allow_test=True` /
  `--allow-test` is passed. This flag is used only in Phase 4.

## 2. Scoring

_Registered 2026-11-23, before any system was scored._

- **Labels:** {yes, no, maybe}. Any output that cannot be mapped to one of them gets the
  prediction `invalid` and parse status `failed`.
- **Parse failures count as wrong; they are never dropped.** Accuracy: `invalid` is wrong.
  Macro-F1 is the unweighted mean of F1 over the three real labels with `zero_division=0`;
  an `invalid` prediction is a false negative for the gold class and a false positive for no class.
  The confusion matrix has gold rows (yes, no, maybe) and predicted columns
  (yes, no, maybe, invalid). The parse-failure count is reported separately.
- **Primary metric:** macro-F1 (the classes are imbalanced, `maybe` is rare).
  **Secondary:** accuracy, per-class precision / recall / F1.
- **Uncertainty:** percentile bootstrap 95% CI, 10,000 resamples of questions with replacement,
  `numpy.random.default_rng(42)`, for accuracy and macro-F1.

## 3. Classifier baseline (PubMedBERT)

_Registered 2026-11-23, before any training run._

- **Model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, revision
  `e1354b7a3a09615f6aba48dfad4b7a613eef7062`, sequence-classification head with 3 labels
  (yes=0, no=1, maybe=2).
- **Input:** pair (question, context), context = the abstract's context paragraphs joined by a
  space, in dataset order. `long_answer` (the article's conclusion) is **never** used: it nearly
  states the answer. Max length 512, truncation `only_second` (the context is truncated, never the
  question).
- **Training:** AdamW, learning rate 2e-5, weight decay 0.01, linear schedule with 10% warmup,
  effective batch size 16 (per-device 8 x gradient accumulation 2), 10 epochs, fp16 on GPU,
  gradient clipping at 1.0, no class weighting. Train on the 400 train questions only.
- **Seeds:** 0, 1, 2, 3, 4 (each controls head init, data order and dropout).
- **Model selection:** after every epoch, score dev; keep the epoch with the highest dev macro-F1,
  ties to the earlier epoch. That checkpoint is saved (Google Drive, not git) for Phase 4.
- **Reported:** per seed, dev metrics of the selected epoch with bootstrap CI; across seeds,
  mean and sample standard deviation (ddof=1) of dev accuracy and macro-F1.
- **Known bias:** dev is used both to pick the epoch and to report the score, so dev numbers
  are optimistic. The clean estimate is the locked test set in Phase 4.

## 4. LLM systems: closed-book, oracle and majority baseline

_Registered 2026-11-23, before any LLM run._

### 4.1 Model and decoding

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- **Quantization:** bitsandbytes 4-bit NF4, compute dtype float16 (the T4 has no bfloat16),
  no double quantization; `device_map = {"": 0}` (whole model on the GPU).
- **Decoding:** greedy. The model's own `generation_config.json` (sampling with temperature 0.7,
  top-p 0.8, top-k 20, repetition penalty 1.05) is overridden: `do_sample=False`, `num_beams=1`,
  `repetition_penalty=1.0`, `max_new_tokens=512`. One question per forward pass (batch size 1,
  no padding effects).
- **Chat format:** the tokenizer's chat template with a system message and
  `add_generation_prompt=True`.

### 4.2 Prompt v1 (frozen)

The exact text lives in `src/biomedqa/prompts/v1.py`; a test checks it equals the text below.
Any change makes a new prompt version (v2, ...) recorded as an amendment, never an edit of v1.

System message:

```text
You are a biomedical research assistant. You answer research questions about PubMed studies with yes, no, or maybe.
```

User message (the `Documents` block is present only when documents are given):

```text
Answer the research question with yes, no, or maybe.

Documents:
[1] <document text>
[2] <document text>

Question: <question>

Respond with only a JSON object with these keys:
"answer": "yes", "no", or "maybe"
"rationale": one or two sentences explaining the answer
"citations": a list of the document numbers that support the answer, for example [1]; use [] if no documents are given or none support the answer
```

- **Closed-book:** no `Documents` block.
- **Oracle:** one document, `[1]` = the question's own abstract context (paragraphs joined by a
  space, conclusion / `long_answer` excluded), the same text the classifier sees.
- Phase 3 RAG uses the same prompt with k retrieved documents.

### 4.3 Parsing (decided before seeing any model output)

1. Strip Markdown code fences; take the first balanced `{...}` object in the output; `json.loads`.
2. `answer`: lowercase, strip spaces and trailing punctuation; must be yes / no / maybe.
3. **Regex fallback:** if step 1 fails, read only an `"answer": "<value>"` field with a regex.
   Recorded as `parse_method = regex_fallback` and counted in every report. A yes / no / maybe
   anywhere else in free text is **never** taken as the answer.
4. Otherwise the prediction is `invalid` (counts as wrong, section 2).
5. `citations`: accepted as integers or strings such as `"2"` or `"[4]"`. A citation outside
   1..n_documents is **fabricated** (in closed-book every citation is fabricated). Unreadable
   citation entries are counted separately. Citations never change the answer or the score.

### 4.4 Reported per LLM run

All section 2 metrics, plus: parse failures, regex-fallback count, outputs that hit the
512-token limit, share of answers with at least one citation, number of fabricated citations
and of questions with at least one fabricated citation.

### 4.5 Response cache

Responses are cached on disk keyed by SHA-256 over: model name, revision, quantization settings,
generation settings, prompt version and the full chat messages. Writes are atomic. The cache
only avoids recomputation; with greedy decoding a cached and a fresh response are meant to be
identical. Library versions are recorded in every run file.

### 4.6 Majority-class baseline

Predicts, for every question, the most frequent gold label in the **train** split (ties to the
earlier label in yes, no, maybe order). Scored on dev like every other system; a floor for
interpreting the other numbers. No GPU.

## 5. Retrieval and RAG

_To be written in Phase 3 before the ablation runs._

## 6. Test-set evaluation

_To be written before Phase 4. The test split stays locked until then._

## Amendments

_None._
