# HANDOFF — Research Paper RAG System

**Purpose of this file.** A self-contained snapshot for planning the rest of the project with a
fresh assistant that has never seen this repo. It covers what exists, what the numbers are, what is
already settled, what is genuinely still open, and where the real debt is.

**How to use it.** Read §1–§3 for orientation, §6–§8 to decide what to do next. §5 lists decisions
that are settled with reasons — re-opening them wastes a planning session unless new evidence
appears. §8 is the actual decision surface.

**Verified:** 2026-10-16. All numbers below were re-run against the current commit, not copied from
notes.

---

## 1. TL;DR state

| | |
|---|---|
| **What it is** | Domain-agnostic RAG over research papers. Question in → grounded answer with citations, drawing on multiple papers. Demoed on 10 interlinked ML/AI arXiv papers |
| **Progress** | **3 of 11 stages done** (ingestion, chunking+embeddings+vector store, retrieval+re-ranking) |
| **Next** | Sprint 4 — generation with citations |
| **Repo** | https://github.com/Ad1roxx/RAG-Project · branch `main` · 14 commits · clean tree, all pushed |
| **Code** | ~2,080 lines Python across 11 modules |
| **Docs** | ~1,300 lines: 3 sprint write-ups + README |
| **Cost so far** | £0 / $0. Everything local — no API key has been used yet |
| **Runs on** | Windows, Python 3.11, CPU only. `.venv` in repo root (gitignored) |

**Crucially: there is no LLM in the system yet.** Stages 1–3 are retrieval only. Nothing generates
an answer. That is Sprint 4 and it is the first place an API key is needed.

---

## 2. Architecture as built

```
arXiv API ──► data/raw/pdfs + metadata        ingestion/fetch_papers.py
                      │
                      ▼
            clean sectioned text              ingestion/extract.py
            10 papers → 206 sections
                      │
                      ▼
            overlapping passages              ingestion/chunk.py
            206 sections → 392 chunks
                      │
                      ▼
            384-dim FAISS index               rag/embed.py + rag/store.py
                      │
     question ───────►├─ dense search (k=30)           rag/retrieve.py
                      ├─ cross-encoder re-rank
                      ├─ relevance-gated diversity
                      └─► top 5 passages + citations
                      │
                      ▼
            [ NOT BUILT YET ]  ──► LLM ──► cited answer      Sprint 4
                                             │
                                             ▼
                                        faithfulness eval     Sprint 5
```

### File map

| File | Lines | Role |
|---|---|---|
| `ingestion/fetch_papers.py` | 283 | arXiv Atom API → PDFs + metadata. Idempotent, atomic writes, polite (3s delay) |
| `ingestion/extract.py` | 422 | PDF → clean section-structured JSON. The most defect-prone file in the repo |
| `ingestion/chunk.py` | 327 | Sections → token-budgeted overlapping passages |
| `ingestion/paths.py` | 32 | All on-disk locations in one place |
| `rag/embed.py` | 89 | Embedding wrapper. Holds the query/passage asymmetry |
| `rag/store.py` | 245 | FAISS index + chunk metadata + two anti-corruption guards |
| `rag/retrieve.py` | 395 | The 3-stage retrieval pipeline. Every stage switchable for ablation |
| `eval/datasets.py` | 106 | 21 labelled questions (single / comparison / thematic) |
| `eval/evaluate.py` | 181 | Retrieval ablation harness, 4 metrics |

**Not yet created** (named in PROJECT_PLAN.md): `rag/generate.py`, `rag/pipeline.py`, `api/`,
`frontend/`, `tests/`, `Dockerfile*`, `docker-compose.yml`, `.github/workflows/ci.yml`.

### Models in use (all local, all free)

| Purpose | Model | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | 512-token limit vs MiniLM's 256; same 384 dims |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small, CPU-viable, +28.6pp recall on comparisons |

---

## 3. Hard constraints — any plan must respect these

1. **Strictly zero-cost.** No paid API, ever. Free tiers and local models only. This is a hard
   requirement, not a preference. Verify a free tier's *current* published limits from the
   provider's own docs before wiring it in — they change.
2. **Every line must be defensible in an interview.** This is the project's whole purpose. A
   feature that works but can't be explained is a net negative. It directly rules out
   copy-pasting framework magic.
3. **Sprints, not dumps.** One component at a time, committed and pushed as it works.
4. **Explanations go in `explanations/sprint-XX.md`**, not inline while building, with 2–3
   interview-probe points flagged per sprint.
5. **Scope discipline.** Core scope only. Explicitly deferred as "future work, talking points, do
   not build": user auth, multi-tenant, conversational memory / multi-turn, agentic multi-hop
   retrieval, fine-tuned embeddings, GraphRAG, caching, rate limiting, streaming.
6. **Domain-agnostic architecture.** Nothing hard-coded to ML/AI. The ML corpus is demo data.
7. **Honesty over agreement.** Over-engineered or indefensible suggestions should be pushed back on.

---

## 4. Measured performance — the baseline any change must beat

`python -m eval.evaluate` · 21 labelled questions · k=5 · 392 chunks

| configuration | paper recall | distinct papers | precision | MRR | s/query |
|---|---|---|---|---|---|
| dense only (Sprint 2 baseline) | 0.786 | 1.52 | 0.876 | 0.897 | 0.03 |
| + over-fetch 30 | 0.786 | 1.52 | 0.876 | 0.897 | 0.02 |
| + cross-encoder | 0.921 | 1.86 | 0.886 | 0.849 | 1.19 |
| + MMR (λ=0.6) | 0.921 | 1.86 | 0.867 | 0.841 | 1.06 |
| + hard cap (3) | **0.984** | 2.43 | 0.771 | 0.849 | 1.17 |
| **+ soft cap (3, m=0.2)** ← shipped | **0.984** | 2.14 | 0.829 | 0.849 | 1.24 |
| soft cap, no rerank | 0.786 | 1.67 | 0.848 | 0.897 | 0.02 |

Per question type, shipped config: **single** (10 q) recall 1.000 · **comparison** (7 q) recall
1.000 · **thematic** (4 q) recall 0.917.

**Metric definitions.** *paper recall* = fraction of expected papers appearing in top k (primary).
*distinct papers* = mean different papers in top k (diagnoses the original failure). *precision* =
fraction of returned chunks from an expected paper (the counterweight — catches a diversity change
faking recall by spreading over irrelevant papers). *MRR* = mean reciprocal rank of first
expected-paper chunk.

**Read these numbers with §7.4 in mind** — the ground truth is paper-level, so this measures
retrieval *breadth* well and chunk-level *precision* only loosely.

---

## 5. Settled decisions — don't re-litigate without new evidence

| Decision | Why |
|---|---|
| Raw arXiv Atom API, not the `arxiv` package | Citations depend on this metadata; the arXiv→stored mapping stays visible |
| PyMuPDF for PDF text | Best quality + exposes block coordinates. **Caveat: AGPL** — fine for a portfolio, would force a swap in a closed-source product |
| References section cut entirely | Lexically dense with query terms, informationally empty — it out-competes real passages for prompt slots |
| Blocks filtered by composition (≥50% tokens contain a letter) | Catches flattened table rows, which are *shaped* like prose so shape rules miss them |
| Generic words (`model`, `results`, `method`…) excluded from heading list | They are table column headers. Risk is asymmetric: a missed heading merges two sections, a false heading corrupted 50K chars |
| Chunks never cross section boundaries | A half-Method/half-Results passage describes no single thing; its vector lands between topics |
| Chunks sized in the embedding model's own tokens | Over-limit input truncates *silently*. 400-token budget + headroom for the context header; verified 0 chunks over 512 |
| 60-token overlap | The sentence with the answer and the sentence naming its subject are often neighbours |
| `bge-small-en-v1.5` over `all-MiniLM-L6-v2` | 512 vs 256 token limit at identical dimensions. Chunk size and model limit are one coupled decision |
| BGE query instruction prefix | The model was fine-tuned asymmetrically; omitting it degrades retrieval *silently* |
| Vectors L2-normalised | Makes dot product == cosine, and puts scores on a comparable scale |
| FAISS over Chroma | The index-row→chunk mapping *is* the provenance chain; worth owning, cost ~60 lines |
| `IndexFlatIP` (exact, not approximate) | At 392 vectors, approximate search saves nothing measurable and adds a second suspect when results look wrong |
| Candidate pool = 30 | Measured knee: 20 loses 11pp recall, 50 buys nothing and costs +0.85s |
| Diversity = relevance-gated soft cap (3, margin 0.2) | Strictly dominates a hard cap: identical recall, precision 0.771→0.829 |
| MMR implemented but **not** used | Measured, earned nothing — it penalises vector similarity, not same-document-ness. Kept because a measured null result is worth owning |
| Paper-level eval labels | Chunk-level labels would need re-annotating 392 chunks on every chunking change |

**Two conclusions that were reversed during the build** — worth knowing so they aren't
accidentally re-reversed:

- The **contextual header** was blamed for poor cross-paper diversity. Tested against a no-header
  index: refuted. Diversity was nearly as bad without it (2.0 vs 1.6), recall identical.
- The **cross-encoder** looked nearly worthless when measured under the *hard* cap (one question's
  recall for 55× latency). Under the *soft* cap, removing it costs **19.8pp recall** — because the
  soft cap consults relevance scores and needs them trustworthy, while the hard cap ignores scores
  and had brute-forced away the contribution. Stages interact.

---

## 6. Remaining build (stages 4–11)

Stages 6–11 are the stack already proven in Project 1 (FastAPI, React, Docker, CI/CD) — lower risk,
mostly transferable. Stages 4–5 are the remaining genuinely new RAG work.

| # | Stage | Risk | Notes |
|---|---|---|---|
| **4** | **Generation with citations** | **High** | First LLM in the system. Needs `GEMINI_API_KEY`. Design question in §8.2 |
| **5** | Evaluation harness (faithfulness) | High | Needs `GROQ_API_KEY`. Coupled to 4 — see below |
| 6 | FastAPI: `/query` `/ingest` `/health` `/info` | Low | Latency budget matters (§7.5) |
| 7 | Tests: retrieval, pipeline, API | Low | Note: no tests exist yet at all (§7.1) |
| 8 | React frontend — answer + inline citations + source chunks | Low | Citations UI is the strong visual |
| 9 | Docker + Compose | Low | Model weights in the image vs mounted volume is a real decision |
| 10 | GitHub Actions CI | Low | Tests that need model downloads will be slow/flaky — plan for it |
| 11 | README + architecture diagram + API docs | Low | |

### Sprints 4 and 5 are coupled

Sprint 4's deliverable is a *grounded* answer, and "grounded" is only meaningful if measured.
Sprint 4 can ship with eyeball verification, but the claim that it works is unbacked until Sprint
5's faithfulness scoring exists. Two sane sequencings:

- **4 then 5** (as planned): build generation, eyeball it, then measure and tune. Matches the plan's
  bottom-up order.
- **4 then 5 then revisit 4**: expect to return to the prompt once faithfulness is measurable. This
  is the realistic version — budget for it rather than treating Sprint 4 as closed.

### Decided but not yet verified for Sprint 4/5

- **Generation: Google Gemini**, free AI Studio tier, `GEMINI_API_KEY`.
- **Eval judge: Groq**, free tier, `GROQ_API_KEY` — chosen separately because evaluation makes far
  more requests than generation, so daily request limits bind differently.
- **LLM layer must sit behind a provider interface**, not Gemini calls scattered through the
  pipeline, so the provider is swappable and the choice is defensible.
- **Keys from `.env`** (gitignored); `.env.example` documents the names. Neither key has been
  created or tested yet.
- ⚠️ **Free-tier limits for both are unverified.** Confirm against Google's and Groq's own current
  docs before wiring, and write the rate-limit reasoning into the sprint explanation.

---

## 7. Known debts and defects — ranked by how much they actually matter

### 7.1 No tests exist (structural)
Zero test files. Stage 7 in the plan, but three sprints of logic — extraction heuristics, chunk
packing, soft-cap selection — currently have no regression protection. The heuristics in
`extract.py` are the highest-value targets: they are pure functions with known tricky inputs
(`representa-\ntion` vs `task-\nspecific`, table rows, ToC lines). **Worth considering pulling
earlier than stage 7**, because every later sprint builds on unprotected foundations.

### 7.2 Eval methodology debt (affects every future claim)
- **21 questions is small.** Differences under ~5pp are noise.
- **No held-out split.** `cap_margin=0.2` and `candidate_k=30` are tuned on the same 21 questions
  they're scored on. These values are fitted; expect slightly worse on unseen questions.
- **Self-labelled.** The same person chose the corpus and wrote the labels. Sprint 5's LLM-as-judge
  partly mitigates this by scoring answers without those labels.
- **One question still fails:** *"How is the transformer architecture used across these papers?"*
  misses BERT — and the ground truth for it is honestly debatable.

### 7.3 Tables are dropped entirely
The composition filter removes table rows. Consequence: questions like *"what GLUE score did
BERT-large reach?"* may be unanswerable. Fixing it means table-aware extraction (`pdfplumber`) and
rendering tables as markdown so the numbers keep their column headers. **Medium value** — it opens
a whole question class the system currently can't serve.

### 7.4 Paper-level ground truth can't see chunk quality
The cross-encoder's main job — ordering chunks *within* the right paper — is structurally invisible
to the current metrics, so the ablation **under-measures** it. Sprint 5's faithfulness scoring is
what closes this gap.

### 7.5 Latency: 1.3 s/query
Essentially all of it is 30 cross-encoder forward passes on CPU. Fine for a QA UI, a real budget
item for Sprint 6. Honest mitigations: GPU, a smaller cross-encoder, or caching — *not* a smaller
candidate pool, since 30 is the measured knee.

### 7.6 Unvalidated parameters
- **Contextual header** (paper title + heading prepended before embedding): tested, inconclusive at
  n=13. Kept as a reasoned default.
- **Chunk size 400 / overlap 60**: chosen from the section-length distribution and the model limit,
  never validated against *answer* quality.

Both are Sprint 5 work: faithfulness scoring is the first tool that can settle them.

### 7.7 Minor, measured
- **Footnote splicing:** footnotes extract as page-bottom blocks and land mid-paragraph
  (`...to its left.4 1https://github.com/...`). **3 of 392 chunks (0.8%)**. Fix is font-size
  filtering via `get_text("dict")`. Left deliberately at this defect rate.
- **Corpus imbalance:** GPT-3 is 140 of 392 chunks (36%) — a 75-page paper among 10-page ones. No
  per-paper normalisation in retrieval, so it is structurally over-represented.
- **Appendices after references are lost** (only affects papers ordering them that way).
- **Equations become nonsense** — flattened to stray symbols. Papers explain them in prose, so this
  is survivable.
- **Scanned PDFs would yield nothing** (no OCR). All 10 current papers are born-digital.
- **MRR regressed 0.048** from dense baseline (~one question at n=21). Flagged, not explained.

---

## 8. Open decisions that need a human call

This is the planning surface. Each needs a decision before or during the sprint that depends on it.

### 8.1 LangChain / LlamaIndex — the biggest unmade decision ⚠️

PROJECT_PLAN.md says: *"LangChain or LlamaIndex — pick one, justify the choice."* It is listed as
core scope and "named on every AI-engineer JD."

**Current reality: the entire pipeline is hand-wired.** Ingestion, chunking, embedding, the vector
store, and three-stage retrieval are all bespoke, understood code with no framework.

**The tension is real and worth deciding deliberately rather than by default:**

- *For adopting it:* resume keyword, demonstrates ecosystem familiarity, and RAGAS (Sprint 5)
  integrates more smoothly with framework-shaped objects.
- *Against:* the project's stated purpose is defending every line. Retrofitting a framework over
  working, understood code — purely to name it — directly contradicts CLAUDE.md's instruction to
  resist framework magic. It would also *hide* the exact stages that are this project's strongest
  material (the chunking/model-limit coupling, the bi-encoder/cross-encoder split, the soft cap).

**Recommendation: do not retrofit the pipeline. Instead, make the omission a deliberate, documented
choice** and be able to say precisely what LangChain would have provided, what it would have
abstracted away, and why that abstraction was the wrong trade *for a project whose goal is
understanding*. "I hand-wired the pipeline so I could explain each stage, and here is exactly what
the framework would have given me" is a stronger interview answer than a `RetrievalQA` chain the
candidate can't take apart.

**If the JD-keyword risk feels too high**, the middle path is a thin, clearly-labelled
LangChain-equivalent adapter in a separate module — enough to speak to the API fluently without the
real pipeline depending on it. Decide this before Sprint 5, since it affects how RAGAS is wired.

### 8.2 Sprint 4: how strictly should citations be enforced?

The hard design question in generation. Options, roughly increasing in rigour:

1. **Prompt-only.** Ask for citations, trust the output. Simplest, and silently unreliable.
2. **Verify citation IDs** (recommended floor). Parse emitted citations, check each maps to a real
   retrieved chunk ID, flag or strip inventions. *An invented citation is worse than a refusal,
   because it looks trustworthy.*
3. **Verify claim support.** Check each sentence is actually entailed by its cited chunk. This is
   essentially faithfulness scoring and belongs in Sprint 5.

Also to decide: what happens when retrieval returns nothing relevant — **the system must be able to
say "I don't know."** A RAG system that always answers is a RAG system that hallucinates on
out-of-corpus questions, and that is a likely demo question.

### 8.3 Sprint 5: RAGAS, LLM-as-judge, or both?

The plan says "RAGAS and/or an LLM-as-judge setup." Considerations: RAGAS is the recognisable name
but defaults to OpenAI and needs explicit reconfiguration to run on Groq — and under the zero-cost
constraint that configuration work is mandatory, not optional. A hand-rolled LLM-as-judge is more
transparent and more defensible but less citable. Needs a call; they are not mutually exclusive.

### 8.4 Should the corpus grow?

10 papers / 392 chunks is small enough that retrieval is nearly saturated on single-paper questions
(recall 1.000). Growing to ~50 papers would make the retrieval problem meaningfully harder and the
numbers more credible, at the cost of re-labelling an eval set and a longer index build. **It would
also likely expose weaknesses the current corpus hides.** Trade-off worth an explicit decision —
note that `fetch_papers.py --category cs.CL --max-results N` already supports bulk fetch.

### 8.5 Hybrid search (BM25 + vector)

Listed in PROJECT_PLAN.md as the *first* thing to pull in from future scope once the core works.
Pure dense retrieval has no exact-match path, so a rare literal token (`bert-base-uncased`, a
specific metric name) can be missed where BM25 would nail it. Currently **not built** and correctly
out of scope until the core is done — but it is the highest-value retrieval upgrade remaining, and
the eval harness to prove it works already exists.

---

## 9. How to run and verify everything

```bash
# setup
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# rebuild the whole corpus from scratch (~2 min; data/ is gitignored, derived state)
python -m ingestion.fetch_papers       # arXiv -> data/raw/   (10 PDFs, 3s politeness delay)
python -m ingestion.extract            # PDFs -> data/processed/   (206 sections)
python -m ingestion.chunk              # sections -> data/chunks/  (392 chunks)
python -m rag.store build              # embed + index (~31s, CPU)

# use it
python -m rag.retrieve "How do BERT and RoBERTa differ in their pre-training objectives?"
python -m rag.retrieve "..." --no-rerank --diversity none     # Sprint 2 behaviour, for comparison
python -m eval.evaluate                                        # the full ablation
python -m eval.evaluate --kind comparison --show-failures
python -m ingestion.chunk --stats-only                         # token distribution, writes nothing
```

**Windows note:** prefix with `PYTHONIOENCODING=utf-8` when piping output, or Unicode in paper
titles raises `UnicodeEncodeError` on the cp1252 console.

**Current verified state:** 10 PDFs, 10 processed papers, 392 chunks, FAISS index present
(`data/index/{index.faiss,chunks.jsonl,manifest.json}`).

---

## 10. Where to read more

| Document | Contents |
|---|---|
| `CLAUDE.md` | How the project is built — working agreement, constraints, scope discipline |
| `PROJECT_PLAN.md` | Full scope, build order, folder structure, explicit future scope |
| `README.md` | Public-facing overview, pipeline diagram, stage status, quickstart |
| `explanations/sprint-01.md` | Ingestion: PDF extraction defects, the hyphenation-vocabulary trick, why references are cut |
| `explanations/sprint-02.md` | Chunking/model-limit coupling, overlap, BGE query asymmetry, why the index is exact, the header experiment |
| `explanations/sprint-03.md` | Bi-encoder vs cross-encoder, the full ablation, why MMR failed, hard→soft cap, the reversed conclusion |

Each sprint write-up ends with 2–3 flagged **interview-probe points** — the things most likely to be
pushed on, with the depth to answer them. Those are the highest-value pages in the repo for
interview prep.

---

## 11. One-paragraph brief for a fresh planning session

> A domain-agnostic RAG system over research papers, built bottom-up in sprints with a hard
> zero-cost constraint and a requirement that every line be defensible in an interview. Three of
> eleven stages are done: arXiv ingestion with structure-aware PDF extraction, token-budgeted
> chunking into a local FAISS index, and a three-stage retrieval pipeline (over-fetch →
> cross-encoder re-rank → relevance-gated diversity) that lifted paper recall from 0.786 to 0.984
> on a 21-question labelled set. There is no LLM in the system yet — generation with verified
> citations is Sprint 4 (Gemini free tier, behind a provider interface), and faithfulness
> evaluation is Sprint 5 (Groq free tier). The main open decisions are whether to adopt
> LangChain/LlamaIndex at all (current recommendation: don't retrofit, document the choice), how
> strictly to verify citations, and whether to grow the corpus. The main debts are no tests at all,
> an eval set too small to split into train/held-out, and tables being dropped during extraction.
