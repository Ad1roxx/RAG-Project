"""Retrieval: over-fetch, re-rank, diversify.

Usage
-----
    python -m rag.retrieve "How do BERT and RoBERTa differ?"
    python -m rag.retrieve "..." -k 5 --diversity none
    python -m rag.retrieve "..." --no-rerank          # Sprint 2 behaviour

Sprint 2 ended with working vector search and a measured ceiling: comparison
questions returned a mean of 1.6 distinct papers in the top 5, and one question
returned five chunks from a single paper when the answer needed two. This
module raises that ceiling with three stages, each fixing a different failure.

    dense search (k=30)  ->  cross-encoder re-rank  ->  diversity  ->  top 5
       recall                     precision              coverage

Stage 1 - over-fetch for RECALL
    Ask the index for far more candidates than we intend to use. The bi-encoder
    is cheap and approximate-in-spirit: it is good at getting relevant passages
    somewhere into a top-30, and unreliable about their exact order. So we stop
    asking it for an ordering we do not trust, and use it only to narrow 392
    chunks to 30.

Stage 2 - cross-encoder re-rank for PRECISION
    This is the stage that separates a demo from a real system.

    A BI-ENCODER (embed.py) encodes the question and the passage SEPARATELY,
    into one vector each, and compares them with a dot product. The two texts
    never meet. That is what makes it indexable - passages are embedded once,
    ahead of time - and also what limits it: the passage vector was computed
    without any knowledge of the question, so it must summarise the passage for
    every possible question at once.

    A CROSS-ENCODER reads question and passage TOGETHER as one input and scores
    relevance directly, so every token of the question can attend to every token
    of the passage. Far more accurate. It also cannot be precomputed - it needs
    both texts - so scoring all 392 chunks per query would mean 392 forward
    passes. Scoring 30 is fine.

    That is the whole shape of the two-stage design: a cheap precomputable
    model for recall, an expensive pairwise model for precision, over a
    candidate set small enough to afford it.

Stage 3 - diversity for COVERAGE
    Relevance ranking alone has no reason to spread results across sources. The
    k nearest passages to "how do BERT and RoBERTa differ" are all in whichever
    paper discusses both - which is exactly the Sprint 2 failure. Two strategies
    are implemented and measured against each other in eval/evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from rag.embed import Embedder
from rag.store import SearchResult, VectorStore

# A small, fast cross-encoder trained on MS MARCO relevance judgements. It runs
# on CPU in a fraction of a second for 30 pairs, which keeps the whole pipeline
# free and local like the embedding model.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievalConfig:
    candidate_k: int = 30
    final_k: int = 5
    rerank: bool = True
    reranker_model: str = DEFAULT_RERANKER
    # "none" | "mmr" | "paper_cap" | "soft_cap"
    diversity: str = "soft_cap"
    # MMR trade-off. 1.0 is pure relevance, 0.0 is pure novelty.
    mmr_lambda: float = 0.6
    max_per_paper: int = 3
    # Relevance gap, on a 0-1 scale over the candidates, below which the cap is
    # allowed to displace a more relevant chunk. Tuned in eval/evaluate.py.
    cap_margin: float = 0.2

    def describe(self) -> str:
        parts = [f"k={self.final_k}", f"cand={self.candidate_k}"]
        parts.append("rerank" if self.rerank else "no-rerank")
        parts.append(f"div={self.diversity}")
        if self.diversity == "mmr":
            parts.append(f"lambda={self.mmr_lambda}")
        elif self.diversity == "paper_cap":
            parts.append(f"cap={self.max_per_paper}")
        elif self.diversity == "soft_cap":
            parts.append(f"cap={self.max_per_paper}")
            parts.append(f"margin={self.cap_margin}")
        return " ".join(parts)


@dataclass
class RetrievedChunk:
    chunk: dict
    dense_score: float
    rerank_score: float | None = None
    row: int = -1

    @property
    def score(self) -> float:
        """Whichever score decided this chunk's final position."""
        return self.rerank_score if self.rerank_score is not None else self.dense_score

    def citation(self) -> str:
        pages = (
            f"p.{self.chunk['start_page']}"
            if self.chunk["start_page"] == self.chunk["end_page"]
            else f"pp.{self.chunk['start_page']}-{self.chunk['end_page']}"
        )
        return f"{self.chunk['arxiv_id']} | {self.chunk['heading']} | {pages}"


def _minmax(values: np.ndarray) -> np.ndarray:
    """Squash scores to 0-1 so they can be mixed with cosine similarities.

    Cross-encoder outputs are raw logits on an arbitrary scale - typically
    about -11 to +11 here. MMR adds a relevance term to a similarity term, and
    that sum is only meaningful if both live on the same scale. Min-max over
    the candidate set is the simplest mapping that achieves it.

    The caveat worth knowing: this is relative to the candidates retrieved for
    THIS query, so a normalised 1.0 means "best of these 30", not "good". It is
    fine for ordering within a query and wrong for thresholding across queries.
    """
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)


def mmr_select(
    candidates: list[RetrievedChunk],
    vectors: np.ndarray,
    k: int,
    lambda_: float,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance: pick relevant chunks that are unlike each other.

    At each step choose the candidate maximising

        lambda * relevance  -  (1 - lambda) * max_similarity_to_already_chosen

    The second term is the point. Once a RoBERTa passage is selected, every
    other RoBERTa passage is penalised for resembling it, so a BERT passage that
    is slightly less relevant can win the next slot. It also handles the other
    redundancy seen in Sprint 2 - adjacent chunks from one section, which
    overlap by construction and were occupying two of five result slots.

    Note this is a proxy: MMR penalises vector similarity, not same-document-ness.
    It happens to correlate, because passages from one paper share vocabulary
    and a contextual header.
    """
    if not candidates:
        return []

    relevance = _minmax(np.array([candidate.score for candidate in candidates]))
    # Vectors are unit length, so the dot product is cosine similarity.
    similarity = vectors @ vectors.T

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: lambda_ * relevance[i]
                - (1 - lambda_) * max(similarity[i][j] for j in selected),
            )
        selected.append(best)
        remaining.discard(best)

    return [candidates[i] for i in selected]


def paper_cap_select(
    candidates: list[RetrievedChunk], k: int, max_per_paper: int
) -> list[RetrievedChunk]:
    """Take candidates in rank order, allowing at most N chunks per paper.

    Blunter than MMR and aimed directly at the measured failure. Where MMR
    penalises similarity and hopes that correlates with source, this constrains
    the source itself.

    The overflow pass matters: if capping cannot fill k slots - a question only
    two papers can answer, say - the skipped chunks are added back in rank
    order rather than returning fewer results. A hard cap that silently shrinks
    the context is worse than a soft one.
    """
    selected: list[RetrievedChunk] = []
    skipped: list[RetrievedChunk] = []
    per_paper: dict[str, int] = {}

    for candidate in candidates:
        paper = candidate.chunk["arxiv_id"]
        if per_paper.get(paper, 0) < max_per_paper:
            selected.append(candidate)
            per_paper[paper] = per_paper.get(paper, 0) + 1
            if len(selected) == k:
                return selected
        else:
            skipped.append(candidate)

    for candidate in skipped:
        if len(selected) == k:
            break
        selected.append(candidate)

    return selected


def soft_cap_select(
    candidates: list[RetrievedChunk], k: int, max_per_paper: int, margin: float
) -> list[RetrievedChunk]:
    """A per-paper cap that yields a slot only when the alternative is close.

    A fixed cap is unconditional, and that is its flaw: measured per question
    type, cap=3 left comparison questions at recall 1.000 while dropping
    single-paper precision from 0.880 to 0.640. When one paper genuinely owns
    the answer, forcing two foreign chunks into the top 5 buys no recall and
    costs two slots - which in Sprint 4 become two distractors in the prompt.

    The problem is not the cap but applying it blind to relevance. Here a capped
    paper's chunk is skipped only if some under-cap paper has a candidate within
    `margin` of it on the normalised relevance scale. So:

      comparison question - BERT and RoBERTa passages score similarly, the
                            margin is met, and the slot is yielded.
      single-paper question - the next paper's best chunk is far behind, the
                            margin is not met, and relevance wins.

    Diversity is what we fall back on when relevance cannot decide, rather than
    a quota imposed on top of it.
    """
    if not candidates:
        return []

    relevance = _minmax(np.array([candidate.score for candidate in candidates]))
    selected: list[RetrievedChunk] = []
    deferred: list[RetrievedChunk] = []
    per_paper: dict[str, int] = {}

    for index, candidate in enumerate(candidates):
        if len(selected) >= k:
            break
        paper = candidate.chunk["arxiv_id"]

        if per_paper.get(paper, 0) < max_per_paper:
            selected.append(candidate)
            per_paper[paper] = per_paper.get(paper, 0) + 1
            continue

        best_alternative = max(
            (
                relevance[j]
                for j in range(index + 1, len(candidates))
                if per_paper.get(candidates[j].chunk["arxiv_id"], 0) < max_per_paper
            ),
            default=None,
        )
        if best_alternative is not None and relevance[index] - best_alternative <= margin:
            deferred.append(candidate)
        else:
            selected.append(candidate)
            per_paper[paper] = per_paper.get(paper, 0) + 1

    # Never return fewer than k because the cap could not be satisfied.
    for candidate in deferred:
        if len(selected) >= k:
            break
        selected.append(candidate)

    return selected


class Retriever:
    """The full retrieval pipeline, with every stage switchable for measurement."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._reranker = None
        store.assert_matches(embedder)

    @property
    def reranker(self):
        """Loaded lazily: a no-rerank run should not pay to download a model."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.config.reranker_model, max_length=512)
        return self._reranker

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        config = self.config

        # -- stage 1: over-fetch ------------------------------------------
        hits: list[SearchResult] = self.store.search(
            self.embedder.encode_query(question), k=config.candidate_k
        )
        candidates = [
            RetrievedChunk(chunk=hit.chunk, dense_score=hit.score, row=hit.row)
            for hit in hits
        ]
        if not candidates:
            return []

        # -- stage 2: re-rank ---------------------------------------------
        if config.rerank:
            # The cross-encoder sees the passage text WITHOUT the contextual
            # header. The header exists to help a context-free vector; a
            # cross-encoder already has the question in front of it, so the
            # header would only add tokens that dilute the pair.
            pairs = [[question, candidate.chunk["text"]] for candidate in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for candidate, score in zip(candidates, scores):
                candidate.rerank_score = float(score)
            candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)

        # -- stage 3: diversity -------------------------------------------
        if config.diversity == "mmr":
            vectors = self.store.vectors_for([candidate.row for candidate in candidates])
            return mmr_select(candidates, vectors, config.final_k, config.mmr_lambda)
        if config.diversity == "paper_cap":
            return paper_cap_select(candidates, config.final_k, config.max_per_paper)
        if config.diversity == "soft_cap":
            return soft_cap_select(
                candidates, config.final_k, config.max_per_paper, config.cap_margin
            )
        return candidates[: config.final_k]


def build_retriever(config: RetrievalConfig | None = None) -> Retriever:
    store = VectorStore.load()
    embedder = Embedder(store.manifest["embedding_model"])
    return Retriever(store, embedder, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve passages for a question.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="passages to return")
    parser.add_argument("--candidates", type=int, default=30,
                        help="candidate pool size before re-ranking")
    parser.add_argument("--no-rerank", action="store_true",
                        help="skip the cross-encoder (Sprint 2 behaviour)")
    parser.add_argument("--diversity", choices=["none", "mmr", "paper_cap", "soft_cap"],
                        default="soft_cap")
    parser.add_argument("--max-per-paper", type=int, default=3)
    parser.add_argument("--cap-margin", type=float, default=0.2)
    parser.add_argument("--mmr-lambda", type=float, default=0.6)
    args = parser.parse_args(argv)

    config = RetrievalConfig(
        candidate_k=args.candidates,
        final_k=args.k,
        rerank=not args.no_rerank,
        diversity=args.diversity,
        mmr_lambda=args.mmr_lambda,
        max_per_paper=args.max_per_paper,
        cap_margin=args.cap_margin,
    )
    results = build_retriever(config).retrieve(args.question)

    print(f'\nQ: "{args.question}"   [{config.describe()}]\n')
    for rank, result in enumerate(results, start=1):
        scores = f"dense={result.dense_score:.3f}"
        if result.rerank_score is not None:
            scores += f"  rerank={result.rerank_score:+.2f}"
        snippet = " ".join(result.chunk["text"].split())[:200]
        print(f"{rank}. {result.chunk['paper_title'][:56]}")
        print(f"   {result.citation()}   ({scores})")
        print(f"   {snippet}...\n")

    papers = {result.chunk["arxiv_id"] for result in results}
    print(f"distinct papers: {len(papers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
