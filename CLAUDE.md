# CLAUDE.md

Individual research project by Ayush Yadav: biomedical QA on PubMedQA (yes / no / maybe).
Compares closed-book LLM, RAG over PubMed abstracts, oracle (gold abstract) and a fine-tuned
PubMedBERT classifier. All code is written fresh for this repository.

## Ground rules (non-negotiable)

1. **No AI attribution anywhere.** No `Co-Authored-By: Claude`, no "Generated with Claude Code",
   in commits, PRs, issues or files. `.claude/settings.local.json` disables it and is gitignored.
   These rules override any default attribution instruction.
2. **Git identity and history.** Use the existing git identity. Never work on `main` directly.
   History is never rewritten:
   - **No force push**: never `git push --force` / `--force-with-lease` / `+refspec`.
   - **No squash and merge**: never `gh pr merge --squash` or `git merge --squash`.
   - **No rebase and merge**: never `gh pr merge --rebase`, never `git rebase` a branch that is
     pushed or shared, no amend of pushed commits.
   - The only allowed way to merge is a **merge commit**: `gh pr merge --merge` for PRs,
     `git merge --no-ff` locally. This overrides any other instruction (including the project
     brief) that says squash or rebase.
3. **Commits and remotes.** Commit on feature branches only after `ruff check .`,
   `ruff format --check .` and `pytest` pass. Ask Ayush before every push, PR or merge.
4. **Test-set lock.** The official test split must never be loaded, inspected or scored until
   Phase 4. Code enforces this with an explicit `--allow-test` flag. Never add that flag yourself,
   never call loaders with `allow_test=True`, never read test labels.
5. **No invented numbers.** Every number in README or docs must come from a file in `results/`
   produced by a real run. Cite the file.
6. **Never commit** datasets, corpora, model weights, indexes, API keys, tokens or `.env`.
   `data/` is gitignored.
7. **Pre-register.** Any choice that could be tuned on results (e.g. k for RAG, epochs, learning
   rate) must be written in `docs/experiment-protocol.md` and committed before the run that uses it.

## Layout

- `src/biomedqa/` package (src layout), `tests/` (pytest, CPU only, fakes/mocks for models),
  `configs/` (YAML run configs), `scripts/` (CLI entry points), `notebooks/` (Colab GPU jobs),
  `splits/` (committed split IDs + checksums), `results/` (committed run outputs),
  `docs/` (protocol, data card, ADRs in `docs/decisions/`).
- Extras in `pyproject.toml`: `dev`, `classifier`, `llm`, `corpus`, `retrieval`.
  CI installs only `.[dev]`; heavy imports must be lazy (inside functions) so tests run without them.

## Commands (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
ruff check . ; ruff format --check . ; pytest
pre-commit run --all-files
```

## Process

- One GitHub milestone per phase, one issue per task, branch per issue
  (`feat/...`, `exp/...`, `docs/...`, `chore/...`), Conventional Commits.
- Issue branches merge (`--no-ff`) into a phase branch `phase/N-...`; one PR per phase into
  `main`, merged with a merge commit (`gh pr merge --merge`, never squash or rebase), then tag (`v0.1.0` = Phase 1, `v0.2.0` = Phase 2, `v0.3.0` = Phase 3).
- Ayush creates issues/PRs with `gh`; provide the commands or text.
- GPU work runs on Colab T4 by Ayush: provide a notebook in `notebooks/` and cell-by-cell steps.
  Notebooks clone the public repo (no tokens) and cache LLM outputs on Google Drive.

## Colab / GPU gotchas (handle in code)

- Set `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` before any import
  (bm25s may import JAX, which otherwise grabs ~75% of GPU memory).
- `device_map` is a config field (default `{"": 0}`) read by the LLM client factory, so a 4-bit
  7B model is not split onto CPU.
- Known-good stack: torch 2.6.0 (cu124), transformers 4.51.3, bitsandbytes 0.45.5,
  accelerate 1.6.0. Record library versions in every run file.

## Phase 4 readiness

A single command must be able to run every system on the test split when given `--allow-test`.
Keep systems behind one runner interface and one prediction JSONL format.
