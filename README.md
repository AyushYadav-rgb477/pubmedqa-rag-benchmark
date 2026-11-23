# Biomedical QA with Retrieval-Augmented Generation

**Research question.** On [PubMedQA](https://pubmedqa.github.io/) (yes / no / maybe), does
retrieval-augmented generation (RAG) over PubMed abstracts improve an open LLM compared with
closed-book answering? How do both compare with an oracle (gold abstract given) and with a
fine-tuned PubMedBERT classifier?

## Systems

| System | Input to the model |
|---|---|
| Closed-book LLM | question only |
| RAG (BM25 + LLM) | question + top-k retrieved PubMed abstracts |
| Oracle LLM | question + the gold abstract |
| PubMedBERT classifier | question + gold abstract, fine-tuned |

LLM: `Qwen/Qwen2.5-7B-Instruct` in 4-bit on a free Colab T4.

## Results

No results yet. Every number in this table will come from a file in `results/`.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

GPU steps run on Google Colab; see `notebooks/`.

## Project docs

- [Experiment protocol (pre-registration)](docs/experiment-protocol.md)
- [Data card](docs/data-card.md)
- [Design decisions](docs/decisions/)
- [Changelog](CHANGELOG.md)

## Credit

Individual re-implementation of a four-person course project (Deep Learning, Sitare University,
2026). All code in this repository was written for it.

## License

MIT, see [LICENSE](LICENSE).
