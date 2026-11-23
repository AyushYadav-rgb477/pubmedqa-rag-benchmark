# Biomedical QA with Retrieval-Augmented Generation

**Research question.** On [PubMedQA](https://pubmedqa.github.io/) (yes / no / maybe), does
retrieval-augmented generation (RAG) over PubMed abstracts improve an open LLM compared with
closed-book answering? How do both compare with an oracle (gold abstract given) and with a
fine-tuned PubMedBERT classifier?

## Systems

| System | Input to the model | Phase |
|---|---|---|
| PubMedBERT classifier | question + gold abstract context, fine-tuned | 1 |
| Closed-book LLM | question only | 2 |
| Oracle LLM | question + the gold abstract context | 2 |
| RAG (BM25 + LLM) | question + top-k retrieved PubMed abstracts | 3 |

LLM: `Qwen/Qwen2.5-7B-Instruct` in 4-bit on a free Colab T4.

## Results

Generated from the files in `results/` by `scripts/update_readme_results.py`; a test fails if
this table and the files disagree. No number here is typed by hand.

<!-- results:start -->
| System | Split | Accuracy (%) | Macro-F1 (%) | Runs | Parse failures | Source |
|---|---|---|---|---|---|---|
| Majority class (train) | dev | 55.0 [45.0, 65.0] | 23.7 [20.7, 26.3] | 1 run | 0 | [`results/baselines/majority/dev_metrics.json`](results/baselines/majority/dev_metrics.json) |
| PubMedBERT classifier | dev | 57.6 ± 5.1 | 41.0 ± 6.7 | 5 seeds | 0 | [`results/classifier/summary.json`](results/classifier/summary.json) |
| Closed-book LLM | dev | 13.0 [7.0, 20.0] | 9.6 [4.9, 14.8] | 1 run | 0 | [`results/llm/closed_book/dev_metrics.json`](results/llm/closed_book/dev_metrics.json) |
| Oracle LLM (gold abstract) | dev | 44.0 [35.0, 54.0] | 38.5 [29.9, 47.3] | 1 run | 0 | [`results/llm/oracle/dev_metrics.json`](results/llm/oracle/dev_metrics.json) |

Multi-seed rows: mean ± sample std over seeds. Single runs: value [95% bootstrap CI]. Dev selects settings, so dev numbers are optimistic; test is locked until Phase 4.
<!-- results:end -->

**Dev observations so far** (details: confusion matrices in each `dev_metrics.json`):

- The 4-bit Qwen2.5-7B with prompt v1 answers "maybe" for most dev questions, especially
  closed-book, while "maybe" is the rarest gold label. All its outputs were valid JSON, so this
  is the model hedging, not a parsing problem.
- Giving the gold abstract (oracle) raises both accuracy and macro-F1 well above closed-book,
  but the oracle LLM still does not beat the fine-tuned classifier on dev.
- Prompt v1 is frozen (protocol 4.2); it was not changed after seeing these results.

## How it is evaluated

- **Data:** 1000 expert-labelled PubMedQA questions. Test = the 500 official test questions,
  **locked in code** until the final run (Phase 4). The other 500 are split into train 400 and
  dev 100, stratified by label, seed 42. See the [data card](docs/data-card.md).
- **Metrics:** macro-F1 (primary) and accuracy over {yes, no, maybe}. Unparsable answers count as
  wrong, never dropped. 95% CIs by percentile bootstrap (10,000 resamples).
- **Pre-registration:** every choice that could be tuned on results is written in the
  [experiment protocol](docs/experiment-protocol.md) and committed before the run that uses it.

## Reproduce

Local (CPU: tests, splits, scoring):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
pip install -e ".[corpus]"
python scripts/make_splits.py          # must print "splits verified"
```

GPU steps run on Google Colab (T4) with the notebooks in `notebooks/`:

| Phase | Notebook | Writes |
|---|---|---|
| 1 | [`01_classifier_colab.ipynb`](notebooks/01_classifier_colab.ipynb) | `results/classifier/` |
| 2 | [`02_llm_colab.ipynb`](notebooks/02_llm_colab.ipynb) | `results/llm/` |

The majority baseline runs on CPU: `python scripts/run_majority.py --split dev`.

Score any prediction file: `python scripts/score.py <predictions.jsonl> --split dev`.
Rebuild the table: `python scripts/update_readme_results.py`.

## Project docs

- [Experiment protocol (pre-registration)](docs/experiment-protocol.md)
- [Data card](docs/data-card.md)
- [Design decisions (ADRs)](docs/decisions/)
- [Changelog](CHANGELOG.md)

## Credit

Individual re-implementation of a four-person course project (Deep Learning, Sitare University,
2026). All code in this repository was written for it.

## License

MIT, see [LICENSE](LICENSE).
