# Research Paper RAG System

A domain-agnostic Retrieval-Augmented Generation system, demoed on a corpus of ML/AI arXiv papers.
Ask a natural-language question, get a grounded answer with citations drawn from across multiple
papers.

The architecture is general — nothing is hard-coded to machine learning. The ML/AI corpus is demo
data, chosen so answer quality can be rigorously evaluated in a domain the author can verify.

> **Status: in progress.** Built bottom-up, one pipeline stage per sprint. See the table below.

## Pipeline

```
arXiv  ->  extract  ->  chunk  ->  embed  ->  vector store
                                                   |
                                    question  ->  retrieve  ->  re-rank  ->  LLM  ->  cited answer
                                                                                         |
                                                                                    evaluation
```

| # | Stage | Status |
|---|---|---|
| 1 | Data ingestion — fetch from arXiv, PDF → clean structured text | ✅ done |
| 2 | Chunking + embeddings + vector store | next |
| 3 | Retrieval + re-ranking | |
| 4 | Generation with citations | |
| 5 | Evaluation harness (retrieval + faithfulness) | |
| 6 | FastAPI service | |
| 7 | Tests | |
| 8 | React frontend | |
| 9 | Docker Compose | |
| 10 | GitHub Actions CI/CD | |
| 11 | Docs + architecture diagram | |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Unix)
pip install -r requirements.txt

cp .env.example .env              # fill in API keys when Sprint 4 lands

python -m ingestion.fetch_papers  # download the seed corpus from arXiv
python -m ingestion.extract       # PDFs -> data/processed/*.json
```

Current corpus: 10 interlinked ML/AI papers (Transformer → BERT → GPT-3; Sentence-BERT → DPR → RAG),
206 sections, ~492K characters of cleaned text.

`data/` is gitignored — the corpus is reproducible from the two commands above rather than
committed to the repo.

## Layout

```
ingestion/     fetch_papers.py, extract.py  — arXiv -> clean sectioned text
explanations/  per-sprint write-ups: what was built, why, and what to understand
CLAUDE.md      how this project is built
PROJECT_PLAN.md  scope, build order, and what is deliberately out of scope
```

## Explanations

Each sprint has a self-contained write-up in [`explanations/`](explanations/) covering the design
decisions, the concepts involved, and the trade-offs taken.

- [Sprint 1 — Data ingestion](explanations/sprint-01.md)
