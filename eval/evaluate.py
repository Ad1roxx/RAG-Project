"""Measure retrieval quality across pipeline configurations.

Usage
-----
    python -m eval.evaluate                 # full ablation
    python -m eval.evaluate --k 5
    python -m eval.evaluate --show-failures

This is the Sprint 3 slice of the evaluation harness: retrieval metrics only.
Answer faithfulness and groundedness arrive in Sprint 5, which is a different
question ("did the model use what it retrieved?") needing a different method.

The point of an ablation rather than a single score: when a pipeline has three
stages, a single number tells you the system got better and not WHICH stage
did it. Measuring each configuration separately is what turns "re-ranking
helped" from a belief into a claim with evidence - and occasionally shows a
stage earning nothing, which is worth knowing before it ships.

Metrics
-------
paper recall    of the papers that should answer this question, how many
                appear in the top k. The primary metric: it is what the
                Sprint 2 failure scored badly on.
distinct papers mean number of different papers in the top k. Diagnoses the
                specific Sprint 2 failure - five chunks, one paper.
precision       fraction of returned chunks that come from an expected paper.
                The counterweight: diversity strategies could trivially
                maximise recall by spreading results over irrelevant papers,
                and this is the metric that would catch them doing it.
MRR             mean reciprocal rank of the first chunk from an expected
                paper. Rewards putting a right answer near the top, not just
                somewhere in k.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

from eval.datasets import QUESTIONS, EvalQuestion, summary
from rag.embed import Embedder
from rag.retrieve import RetrievalConfig, RetrievedChunk, Retriever
from rag.store import VectorStore


@dataclass
class Scores:
    paper_recall: float
    distinct_papers: float
    precision: float
    mrr: float
    seconds_per_query: float
    n_questions: int


def score_one(question: EvalQuestion, results: list[RetrievedChunk]) -> tuple[float, int, float, float]:
    papers = [result.chunk["arxiv_id"] for result in results]
    found = set(papers) & question.expected_papers

    recall = len(found) / len(question.expected_papers)
    distinct = len(set(papers))
    precision = (
        sum(1 for paper in papers if paper in question.expected_papers) / len(papers)
        if papers else 0.0
    )
    reciprocal_rank = 0.0
    for rank, paper in enumerate(papers, start=1):
        if paper in question.expected_papers:
            reciprocal_rank = 1 / rank
            break

    return recall, distinct, precision, reciprocal_rank


def evaluate(retriever: Retriever, questions: list[EvalQuestion]) -> tuple[Scores, list]:
    recalls, distincts, precisions, rrs = [], [], [], []
    failures = []

    started = time.perf_counter()
    for question in questions:
        results = retriever.retrieve(question.question)
        recall, distinct, precision, rr = score_one(question, results)
        recalls.append(recall)
        distincts.append(distinct)
        precisions.append(precision)
        rrs.append(rr)
        if recall < 1.0:
            missing = question.expected_papers - {r.chunk["arxiv_id"] for r in results}
            failures.append((question, sorted(missing), results))
    elapsed = time.perf_counter() - started

    return (
        Scores(
            paper_recall=statistics.mean(recalls),
            distinct_papers=statistics.mean(distincts),
            precision=statistics.mean(precisions),
            mrr=statistics.mean(rrs),
            seconds_per_query=elapsed / len(questions),
            n_questions=len(questions),
        ),
        failures,
    )


# The ablation. Each row adds exactly one thing to the row above it, so any
# change in the numbers is attributable to that one change.
def ablation(final_k: int, candidate_k: int) -> list[tuple[str, RetrievalConfig]]:
    return [
        ("dense only (Sprint 2)", RetrievalConfig(
            candidate_k=final_k, final_k=final_k, rerank=False, diversity="none")),
        ("+ over-fetch 30", RetrievalConfig(
            candidate_k=candidate_k, final_k=final_k, rerank=False, diversity="none")),
        ("+ cross-encoder", RetrievalConfig(
            candidate_k=candidate_k, final_k=final_k, rerank=True, diversity="none")),
        ("+ MMR (l=0.6)", RetrievalConfig(
            candidate_k=candidate_k, final_k=final_k, rerank=True,
            diversity="mmr", mmr_lambda=0.6)),
        ("+ hard cap (3)", RetrievalConfig(
            candidate_k=candidate_k, final_k=final_k, rerank=True,
            diversity="paper_cap", max_per_paper=3)),
        ("+ soft cap (3, m=.2) *", RetrievalConfig(
            candidate_k=candidate_k, final_k=final_k, rerank=True,
            diversity="soft_cap", max_per_paper=3, cap_margin=0.2)),
        ("soft cap, no rerank", RetrievalConfig(
            candidate_k=candidate_k, final_k=final_k, rerank=False,
            diversity="soft_cap", max_per_paper=3, cap_margin=0.2)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablate the retrieval pipeline.")
    parser.add_argument("--k", type=int, default=5, help="passages returned per query")
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--kind", choices=["single", "comparison", "thematic"],
                        help="evaluate only one question type")
    parser.add_argument("--show-failures", action="store_true",
                        help="list questions whose expected papers were missed")
    args = parser.parse_args(argv)

    questions = [q for q in QUESTIONS if not args.kind or q.kind == args.kind]

    # Built once and shared: loading the index and the models per configuration
    # would dominate the timing column and measure the wrong thing.
    store = VectorStore.load()
    embedder = Embedder(store.manifest["embedding_model"])
    shared_reranker = None

    print(f"\n{summary()}" + (f"  [filtered to {args.kind}: {len(questions)}]" if args.kind else ""))
    print(f"corpus: {store.index.ntotal} chunks, k={args.k}\n")

    header = f"{'configuration':<24} {'recall':>7} {'papers':>7} {'precis':>7} {'MRR':>6} {'s/query':>8}"
    print(header)
    print("-" * len(header))

    all_failures = {}
    for label, config in ablation(args.k, args.candidates):
        retriever = Retriever(store, embedder, config)
        if config.rerank:
            if shared_reranker is None:
                shared_reranker = retriever.reranker
            retriever._reranker = shared_reranker

        scores, failures = evaluate(retriever, questions)
        all_failures[label] = failures
        print(f"{label:<24} {scores.paper_recall:>7.3f} {scores.distinct_papers:>7.2f} "
              f"{scores.precision:>7.3f} {scores.mrr:>6.3f} {scores.seconds_per_query:>8.3f}")

    if args.show_failures:
        label = "+ soft cap (3, m=.2) *"
        print(f"\nQuestions still missing an expected paper under '{label}':")
        for question, missing, _ in all_failures[label]:
            print(f"  [{question.kind}] {question.question}")
            print(f"     missing: {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
