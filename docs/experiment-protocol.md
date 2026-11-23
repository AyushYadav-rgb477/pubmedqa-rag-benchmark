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

_To be written in Phase 1 before splits are created._

## 2. Scoring

_To be written in Phase 1 before any system is scored._

## 3. Classifier baseline (PubMedBERT)

_To be written in Phase 1 before training._

## 4. LLM systems: closed-book and oracle

_To be written in Phase 2 before the dev runs._

## 5. Retrieval and RAG

_To be written in Phase 3 before the ablation runs._

## 6. Test-set evaluation

_To be written before Phase 4. The test split stays locked until then._

## Amendments

_None._
