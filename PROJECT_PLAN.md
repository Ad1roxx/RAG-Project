# PROJECT_PLAN.md — Research Paper RAG System (Project 2)

**Domain:** Domain-agnostic RAG, demoed on ML/AI arXiv papers
**Goal:** A production-quality multi-paper RAG system that answers questions across a corpus of
research papers with grounded, cited answers. The retrieval engineering and evaluation rigor are
the stars — not just wiring an LLM to a vector DB.

---

## Scope — What We're Building

Given a natural-language question, the system retrieves the most relevant passages across MANY
papers, re-ranks them, and generates an answer that cites which papers/sections it drew from. It
includes a real evaluation harness so answer quality is measured, not guessed.

**Guiding principle:** the architecture is domain-agnostic; the ML/AI corpus is demo data chosen so
answer quality can be rigorously evaluated in a field the developer understands. "A general system
evaluated on a focused corpus" is the defensible design.

---

## Core Build

| Layer | Includes | Why it's defendable |
|---|---|---|
| **Data ingestion** | Fetch ML/AI papers from arXiv (open/legal); extract + clean text from PDFs; store metadata (title, authors, arXiv id, sections) | Real-world messy-document handling; reproducible fetch, not checked-in data |
| **Chunking** | Split papers into passages with a deliberate strategy (size, overlap, structure-aware); store chunk-to-paper provenance | Chunking strategy directly drives answer quality — a genuine engineering decision to defend |
| **Embeddings + vector store** | Embed chunks with a sentence-embedding model; store + index in a vector DB (FAISS/Chroma) | Core RAG mechanism; understand embeddings + similarity search under the hood |
| **Retrieval + re-ranking** | Vector search for top-k, then a re-ranker to reorder by true relevance; retrieve across multiple papers | Re-ranking is what separates a demo from a real system; hybrid/re-rank is a strong talking point |
| **Generation with citations** | Feed retrieved context to an LLM; generate a grounded answer that cites source papers/sections | Grounding + citation = the "no hallucination" story that makes RAG serious |
| **Evaluation harness** | Measure retrieval quality (are the right chunks retrieved?) and answer faithfulness/groundedness (RAGAS and/or LLM-as-judge) | Rigorous eval is THE thing separating a real RAG project from a toy |
| **Orchestration** | LangChain or LlamaIndex tying the pipeline together (pick one, justify) | Named on every AI-engineer JD; shows you know the ecosystem |
| **API — FastAPI** | `/query` (question -> answer + citations + confidence), `/health`, `/ingest` (add papers), `/info` | Production serving; reuses Project 1 patterns |
| **Frontend — React** | Ask a question -> see answer with inline citations and the source chunks it used | Full-stack integration; citations UI is a strong visual |
| **Docker** | Containerize services with Docker Compose | Proven from Project 1 |
| **CI/CD — GitHub Actions** | Tests + build on every push | Proven from Project 1 |
| **Docs** | README + architecture diagram + API docs | Makes the repo readable fast |

---

## Folder Structure

```
paper-rag/
├── ingestion/
│   ├── fetch_papers.py       # pull ML/AI papers from arXiv
│   ├── extract.py            # PDF -> clean text + metadata
│   └── chunk.py              # chunking strategy
├── rag/
│   ├── embed.py              # embedding model wrapper
│   ├── store.py              # vector DB interface (FAISS/Chroma)
│   ├── retrieve.py           # vector search + re-ranking
│   ├── generate.py           # LLM generation with citation grounding
│   └── pipeline.py           # orchestration (LangChain/LlamaIndex)
├── eval/
│   ├── datasets.py           # eval question set
│   └── evaluate.py           # retrieval + faithfulness metrics
├── api/
│   ├── main.py               # FastAPI: /query /ingest /health /info
│   ├── schemas.py            # Pydantic models
│   └── service.py            # wires API to the pipeline
├── frontend/
│   └── src/                  # React: question -> answer + citations
├── tests/
│   ├── test_retrieval.py
│   ├── test_pipeline.py
│   └── test_api.py
├── explanations/             # per-sprint teaching write-ups (see CLAUDE.md)
├── .github/workflows/
│   └── ci.yml
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── requirements.txt
├── CLAUDE.md
├── PROJECT_PLAN.md
└── README.md
```

---

## Build Order (bottom-up, Git commit after each working piece)

Each step produces something testable before the next is added. Because RAG quality degrades
silently, we build the pipeline stage by stage and check each stage's output before moving on.

1. **Data ingestion** — fetch a handful of ML/AI papers, extract clean text + metadata to disk
2. **Chunking + embeddings + vector store** — chunk papers, embed, index; confirm you can query for similar chunks
3. **Retrieval + re-ranking** — top-k vector search across papers, then re-rank; inspect what comes back
4. **Generation with citations** — feed retrieved context to the LLM, get a grounded, cited answer
5. **Evaluation harness** — build an eval question set; measure retrieval + faithfulness (this is where tuning becomes measurable)
6. **API** — FastAPI serving `/query`, `/ingest`, `/health`, `/info`
7. **Tests** — retrieval, pipeline, API tests
8. **Frontend** — React question box -> answer with inline citations + source chunks
9. **Docker** — containerize with Docker Compose
10. **CI/CD** — GitHub Actions: tests + build on push
11. **Docs** — README, architecture diagram, API docs

**Fallback:** if any single stage (e.g. re-ranking, or LangChain orchestration) fights us for too
long, ship the simpler version first (plain vector retrieval, no re-ranker; or a hand-wired pipeline
instead of the framework) to get an end-to-end answer working, then upgrade. A complete simple
pipeline beats a half-finished sophisticated one — same rule as Project 1.

---

## Future Scope (after a stable version ships — do NOT build now)

Deferred until the core works. Each doubles as an interview talking point ("here's how I'd extend
this") without being built.

**Pull in first, once core is solid:**
- Hybrid search (vector + keyword/BM25 with score fusion) — a clear retrieval-quality upgrade
- Conversational memory (multi-turn follow-up questions)
- A larger corpus + persistent hosted vector DB

**Talk-about-but-don't-build (name as "future work"):**
- User auth + per-user document libraries
- Agentic multi-hop retrieval (the model plans multiple retrieval steps)
- Fine-tuned / domain-adapted embeddings
- GraphRAG (knowledge-graph-backed retrieval)
- Caching, rate limiting, streaming responses

**Why deferred (a mature interview answer):** "I scoped these out to keep the core pipeline focused
and measurable; here's how I'd add each if the product needed it."

---

## The one rule that governs everything

If the developer can't explain a line under questioning, it doesn't go in. RAG especially invites
copy-pasting framework magic — resist it. Understanding WHY each pipeline stage exists and how it
affects answer quality is the whole point, and it's exactly what an interviewer probes.
