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
| 2 | Chunking + embeddings + vector store | ✅ done |
| 3 | Retrieval + re-ranking | ✅ done |
| 4 | Generation with citations | next |
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
python -m ingestion.chunk         # sections -> retrievable passages
python -m rag.store build         # embed + index (~31s on CPU)

python -m rag.retrieve "How do BERT and RoBERTa differ in their pre-training objectives?"
python -m eval.evaluate           # retrieval ablation over the eval question set
```

Current corpus: 10 interlinked ML/AI papers (Transformer → BERT → GPT-3; Sentence-BERT → DPR → RAG),
206 sections → 392 chunks → a 384-dim FAISS index.

Embeddings (`BAAI/bge-small-en-v1.5`) and re-ranking (`ms-marco-MiniLM-L-6-v2`) both run locally —
no API key, no cost.

**Retrieval quality**, measured over 21 labelled questions (`python -m eval.evaluate`):

| configuration | paper recall | distinct papers | precision | s/query |
|---|---|---|---|---|
| dense vector search only | 0.786 | 1.52 | 0.876 | 0.02 |
| + cross-encoder re-rank | 0.921 | 1.86 | 0.886 | 1.32 |
| + relevance-gated diversity | **0.984** | **2.14** | 0.829 | 1.30 |

`data/` is gitignored — raw PDFs, processed text, chunks and the index are all derived state,
rebuilt by the commands above rather than committed to the repo.

## Layout

```
ingestion/     fetch_papers.py, extract.py, chunk.py  — arXiv -> clean sectioned text -> passages
rag/           embed.py, store.py, retrieve.py  — embeddings, FAISS search, re-ranking
eval/          datasets.py, evaluate.py  — eval question set + retrieval metrics
explanations/  per-sprint write-ups: what was built, why, and what to understand
CLAUDE.md      how this project is built
PROJECT_PLAN.md  scope, build order, and what is deliberately out of scope
```

## Explanations

Each sprint has a self-contained write-up in [`explanations/`](explanations/) covering the design
decisions, the concepts involved, and the trade-offs taken.

- [Sprint 1 — Data ingestion](explanations/sprint-01.md)
- [Sprint 2 — Chunking, embeddings and the vector store](explanations/sprint-02.md)
- [Sprint 3 — Retrieval and re-ranking](explanations/sprint-03.md)
