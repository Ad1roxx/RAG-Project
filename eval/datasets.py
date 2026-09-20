"""The evaluation question set.

This is the Sprint 3 slice: questions labelled with the papers that should
answer them, which is enough to measure RETRIEVAL. Sprint 5 extends the same
set with reference answers so answer faithfulness can be scored too.

Why paper-level labels rather than chunk-level
----------------------------------------------
The honest ground truth for retrieval would name the exact chunks that answer
each question. That means reading all 392 chunks against 21 questions by hand,
and re-doing it whenever the chunking parameters change - which is exactly the
thing we want to be free to tune.

Paper-level labels are stable under re-chunking and still discriminate the
failure we are actually chasing: a comparison question that returns five chunks
from one paper scores badly no matter how its chunks are cut. The cost is real
and worth stating: this metric cannot tell a perfectly-chosen chunk from a
merely same-paper one, so it measures retrieval BREADTH well and retrieval
PRECISION only loosely. Sprint 5's faithfulness scoring is what catches the
second kind of error.

The three question types exist to separate distinct failure modes:

  single      one paper genuinely holds the answer. Tests basic relevance.
  comparison  the answer needs two papers. Tests whether retrieval can spread
              across sources - the failure measured at the end of Sprint 2.
  thematic    a concept several papers touch. Tests breadth without a single
              obviously-correct source.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected_papers: frozenset[str]
    kind: str  # "single" | "comparison" | "thematic"
    note: str = ""


def _q(question: str, papers: list[str], kind: str, note: str = "") -> EvalQuestion:
    return EvalQuestion(question, frozenset(papers), kind, note)


QUESTIONS: list[EvalQuestion] = [
    # ---- single-paper factual ------------------------------------------
    _q("What is the purpose of multi-head attention?", ["1706.03762"], "single"),
    _q("What positional encoding does the Transformer use?", ["1706.03762"], "single"),
    _q("How does BERT's masked language model pre-training objective work?",
       ["1810.04805"], "single"),
    _q("What changes did RoBERTa make to BERT's training setup?", ["1907.11692"], "single"),
    _q("How much smaller and faster is DistilBERT than BERT?", ["1910.01108"], "single"),
    _q("How many parameters does the largest GPT-3 model have?", ["2005.14165"], "single"),
    _q("What pooling strategy does Sentence-BERT use to produce sentence embeddings?",
       ["1908.10084"], "single"),
    _q("How does DPR train its dual-encoder retriever?", ["2004.04906"], "single"),
    _q("What rank is used for the low-rank adaptation matrices in LoRA?",
       ["2106.09685"], "single"),
    _q("What is chain-of-thought prompting?", ["2201.11903"], "single"),

    # ---- cross-paper comparison ----------------------------------------
    _q("How do BERT and RoBERTa differ in their pre-training objectives?",
       ["1810.04805", "1907.11692"], "comparison",
       "The Sprint 2 failure case: returned 5/5 chunks from RoBERTa alone."),
    _q("Compare DistilBERT and BERT in model size and inference speed",
       ["1910.01108", "1810.04805"], "comparison"),
    _q("How does dense passage retrieval compare to sentence embeddings for semantic search?",
       ["2004.04906", "1908.10084"], "comparison"),
    _q("What transformer architecture does GPT-3 build on?",
       ["2005.14165", "1706.03762"], "comparison"),
    _q("How does retrieval-augmented generation use a dense retriever?",
       ["2005.11401", "2004.04906"], "comparison"),
    _q("How do LoRA and DistilBERT each reduce the cost of using large models?",
       ["2106.09685", "1910.01108"], "comparison"),
    _q("Which pre-training objectives do BERT and GPT-3 use, and how do they differ?",
       ["1810.04805", "2005.14165"], "comparison"),

    # ---- thematic / multi-paper ----------------------------------------
    _q("How can retrieval reduce hallucination in language models?",
       ["2005.11401", "2004.04906"], "thematic"),
    _q("What techniques make large language models cheaper to adapt to new tasks?",
       ["2106.09685", "1910.01108", "2005.14165"], "thematic"),
    _q("How is the transformer architecture used across these papers?",
       ["1706.03762", "1810.04805", "2005.14165"], "thematic"),
    _q("What role do attention mechanisms play in encoder models?",
       ["1706.03762", "1810.04805"], "thematic"),
]


def by_kind(kind: str) -> list[EvalQuestion]:
    return [question for question in QUESTIONS if question.kind == kind]


def summary() -> str:
    kinds = sorted({question.kind for question in QUESTIONS})
    parts = [f"{kind}={len(by_kind(kind))}" for kind in kinds]
    return f"{len(QUESTIONS)} questions ({', '.join(parts)})"


if __name__ == "__main__":
    print(summary())
    for question in QUESTIONS:
        print(f"  [{question.kind:<10}] {question.question}")
        print(f"               expects: {', '.join(sorted(question.expected_papers))}")
