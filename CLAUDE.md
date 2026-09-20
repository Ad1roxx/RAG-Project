# CLAUDE.md — Research Paper RAG System (Project 2)

## What this project is
A domain-agnostic Retrieval-Augmented Generation (RAG) system, demoed on a corpus of ML/AI
arXiv papers. Given a natural-language question, it retrieves relevant chunks across MULTIPLE
papers and generates a grounded answer WITH citations. The architecture is general (nothing
hard-coded to ML); the demo corpus is scoped to ML/AI papers so answer quality can be rigorously
evaluated in a field the developer understands.

## Who I am (the developer)
Third-year IT student, strong full-stack background (React, Node, FastAPI, SQL). Just completed
Project 1 (TableTalk — an aspect-based sentiment analysis system: DistilBERT + FastAPI + React +
MLflow + Docker + CI/CD), so I'm comfortable with that production stack. RAG and the LLM/retrieval
pieces are new to me. I am fast-tracking but I MUST understand and be able to defend every line in
interviews — a resume I can't back up has cost me before. Don't let me move past things I don't
understand.

## How we work — READ THIS EVERY SESSION
1. **Sprints, not dumps.** Work in small sprints (one component at a time), like a real agile
   workflow. State the sprint goal in one line at the start; summarize what was built and what's
   next at the end. Do NOT generate large multi-file batches at once.
2. **Write detailed explanations to a separate file, don't explain inline while building.**
   Maintain `explanations/sprint-XX.md` (one per sprint). For each sprint, write a thorough,
   self-contained explanation: what each component does, why it's structured that way, the key
   RAG/ML concepts involved, and the parts most worth understanding. Keep the build moving; put the
   teaching in these files. I read them and discuss separately with Claude (the chat assistant) to
   solidify understanding. Write assuming I'm learning the RAG/LLM concepts as I go.
3. **Flag interview-probe points.** In each sprint's explanation file, explicitly call out the 2-3
   things an interviewer is most likely to probe, so I know where to focus.
4. **Commit after each working piece.** Proper Git workflow — small, meaningful commits with clear
   messages after every component works, pushed to the remote as we go (real incremental history).
   Commit the explanation files too.
5. **Honesty over agreement.** If I suggest something over-engineered, indefensible, or a bad idea,
   tell me directly and explain why. Don't just implement whatever I ask.

## Scope discipline
- Build ONLY the core scope (see PROJECT_PLAN.md). Do not add future-scope features unless I ask.
- RAG quality degrades silently — many parts can each make answers subtly worse. When something's
  off, isolate WHICH stage (chunking, retrieval, re-ranking, generation) rather than guessing.
- Keep the architecture domain-agnostic — nothing hard-coded to ML/AI. The ML corpus is demo data,
  not a baked-in assumption.
- Deliberately OUT of scope for now (name as "future work", don't build): user auth, multi-tenant
  isolation, conversational memory / multi-turn chat, agentic multi-hop retrieval, fine-tuned
  embeddings, GraphRAG. These are talking points, not build items.

## Tech stack
- Reused backbone (proven in Project 1): FastAPI, React, Docker + Docker Compose, GitHub Actions CI/CD
- RAG core (NEW): a vector database (start local — FAISS or Chroma), an embeddings model, a
  retrieval + re-ranking pipeline, an LLM for generation (via API)
- Orchestration (NEW): LangChain or LlamaIndex — pick one, justify the choice in the explanation file
- Evaluation (NEW): a real harness — RAGAS and/or an LLM-as-judge setup, measuring retrieval quality
  and answer faithfulness/groundedness, not just vibes
- Data: arXiv papers (open, legal) — a fetch script, not checked-in PDFs

## The hard part to respect
RAG is a PIPELINE, not a single model: ingestion -> chunking -> embedding -> vector store ->
retrieval -> re-ranking -> generation -> evaluation. Each stage can silently hurt answer quality,
and bad answers are harder to debug than a single model's F1. Expect this to be a BIGGER build than
Project 1. Evaluation is not optional polish — it's how we tell whether a change helped or hurt.

## Definition of done for the core
Fetch + ingest ML/AI papers -> chunk sensibly -> embed + store in a vector DB -> retrieve relevant
chunks across multiple papers -> re-rank -> generate a grounded answer WITH citations -> evaluation
harness reporting retrieval + faithfulness metrics -> FastAPI serving it -> React UI -> Dockerized ->
CI/CD running tests on push -> README with architecture diagram. Every piece committed, every piece
understood.

## Build order (see PROJECT_PLAN.md for detail)
1. Data ingestion  2. Chunking + embeddings + vector store  3. Retrieval + re-ranking
4. Generation with citations  5. Evaluation harness  6. API  7. Tests  8. Frontend
9. Docker  10. CI/CD  11. Docs
