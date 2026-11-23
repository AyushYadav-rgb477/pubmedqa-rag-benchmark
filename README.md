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
| Closed-book LLM | dev | pending | pending | pending | pending | [`results/llm/closed_book/dev_metrics.json`](results/llm/closed_book/dev_metrics.json) |
| Oracle LLM (gold abstract) | dev | pending | pending | pending | pending | [`results/llm/oracle/dev_metrics.json`](results/llm/oracle/dev_metrics.json) |

Multi-seed rows: mean ± sample std over seeds. Single runs: value [95% bootstrap CI]. Dev selects settings, so dev numbers are optimistic; test is locked until Phase 4.
<!-- results:end -->

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
