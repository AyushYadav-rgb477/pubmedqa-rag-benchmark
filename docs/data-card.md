# Data card

## Source

- **PubMedQA** (Jin et al., 2019), Hugging Face dataset `qiaojin/PubMedQA`,
  revision `9001f2853fb87cab8d220904e0de81ac6973b318`.
- License: MIT, as stated in the dataset card on Hugging Face. The dataset is never redistributed
  in this repository; only question IDs (PubMed PMIDs) are committed.
- Configs: `pqa_labeled` (1000 expert-labelled questions) for evaluation and training;
  `pqa_artificial` and `pqa_unlabeled` will be used only as a retrieval corpus (Phase 3).

## What one example contains

| Field | Used by |
|---|---|
| `pubid` (PMID) | ID everywhere |
| `question` | every system |
| `context.contexts` (abstract paragraphs without the conclusion) | classifier, oracle LLM, retrieval corpus |
| `long_answer` (the abstract's conclusion) | **never used**: it nearly states the answer (ADR 0004) |
| `final_decision` (yes / no / maybe) | gold label |

## Splits

Source of truth: [`splits/pqa_labeled_splits.json`](../splits/pqa_labeled_splits.json)
(IDs + SHA-256 per split). Method: [protocol section 1](experiment-protocol.md#1-data-and-splits).

| Split | n | yes | no | maybe | How |
|---|---|---|---|---|---|
| train | 400 | 221 | 135 | 44 | remaining questions, stratified, seed 42 |
| dev | 100 | 55 | 34 | 11 | remaining questions, stratified, seed 42 |
| test | 500 | locked | locked | locked | official PubMedQA test PMIDs |

Train and dev counts are copied from `meta.label_counts` in the split file. Test label counts are
not shown because test labels are not read before Phase 4.

The official test PMIDs come from `data/test_ground_truth.json` in `github.com/pubmedqa/pubmedqa`
at commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`; only the keys were kept
([`splits/official_test_pmids.txt`](../splits/official_test_pmids.txt)).

## Retrieval corpus

_Phase 3._

## Known limitations

- Small evaluation sets: 100 dev questions means one question is one accuracy point, so dev
  differences of a few points are within noise. Bootstrap CIs are reported for this reason.
- `maybe` is rare (11% of train and dev), which makes macro-F1 sensitive to a handful of questions.
- Labels are expert annotations of what the abstract supports, not clinical truth.
- Dev is used to choose settings (epochs, later RAG k), so dev scores are optimistic.
