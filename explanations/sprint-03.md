# Sprint 3 — Retrieval and Re-ranking

**Goal:** raise the retrieval ceiling measured at the end of Sprint 2, and *prove* it moved.

**Built:** `rag/retrieve.py`, `eval/datasets.py`, `eval/evaluate.py`
**Baseline to beat:** paper recall 0.786, mean 1.52 distinct papers in the top 5.
**Result:** paper recall **0.984**, mean **2.14** distinct papers — at 55× the latency.

---

## 0. Measurement had to come first

Sprint 2 ended with a specific, numbered failure: comparison questions returned a mean of 1.6
distinct papers, and one returned five chunks from a single paper when the answer needed two.

The temptation is to add a re-ranker, eyeball a few queries, and declare victory. That fails on
CLAUDE.md's rule that RAG quality degrades silently — **you cannot see a 5% regression by reading
output**, and every stage added here could cause one.

So `eval/datasets.py` and `eval/evaluate.py` came before any tuning. This is the Sprint 3 slice of
the harness: retrieval metrics only. Answer faithfulness is a different question — *"did the model
use what it retrieved?"* — needing a different method, and it lands in Sprint 5.

### The ground-truth decision, and its cost

Each of 21 questions is labelled with the **papers** that should answer it — not the chunks.

Chunk-level labels would be the ideal ground truth. They'd also mean reading all 392 chunks against
21 questions by hand, and **re-doing it every time a chunking parameter changes** — which is exactly
the freedom we want to keep.

Paper-level labels are stable under re-chunking and still discriminate the failure being chased: a
comparison question returning five chunks from one paper scores badly however its chunks are cut.

**The cost, stated plainly:** this metric measures retrieval *breadth* well and retrieval *precision*
only loosely. It cannot distinguish a perfectly-chosen chunk from a merely same-paper one. That
limitation directly shapes how much I trust each result below — and it bites in §5.

Questions are split into three kinds because they exercise different failures: **single** (10, one
paper genuinely holds the answer), **comparison** (7, needs two papers — the Sprint 2 failure), and
**thematic** (4, a concept several papers touch).

### Four metrics, because one hides things

| Metric | What it catches |
|---|---|
| **paper recall** | The primary metric — how many expected papers reached the top *k* |
| **distinct papers** | Diagnoses the exact Sprint 2 failure: 5 chunks, 1 paper |
| **precision** | **The counterweight.** A diversity strategy could trivially maximise recall by spreading results over irrelevant papers. This is what catches it doing that |
| **MRR** | Rewards putting a right answer *near the top*, not just somewhere in *k* |

Including a metric specifically designed to catch your own improvement cheating is the part worth
copying. Without precision, every result in §4 would look like a straight win.

---

## 1. The three-stage architecture

```
dense search (k=30)  ->  cross-encoder re-rank  ->  diversity  ->  top 5
     recall                    precision             coverage
```

Each stage fixes a different failure, which is why they compose.

### Stage 1 — over-fetch for recall

Ask the index for 30 candidates instead of 5. The bi-encoder is good at getting relevant passages
*somewhere* into a top-30 and unreliable about their exact order — so we stop asking it for an
ordering we don't trust and use it only to narrow 392 chunks down to 30.

**Over-fetching alone does nothing.** The ablation shows it scoring identically to the baseline —
if you take the top 5 by the same ordering, fetching 30 changes nothing. It's only valuable because
a later stage *reorders*. I kept that row in the ablation precisely because it makes this
dependency visible; it also served as a sanity check that the harness wasn't silently mis-wiring
configurations.

### Stage 2 — cross-encoder re-rank for precision

**This is the concept worth understanding deeply.**

A **bi-encoder** (what `embed.py` does) encodes the question and the passage **separately**, into
one vector each, then compares them with a dot product. *The two texts never meet.* That's what
makes it indexable — passages are embedded once, ahead of time, before any question exists. It's
also its fundamental limit: **the passage vector was computed without any knowledge of the
question**, so it has to summarise the passage for every possible question simultaneously.

A **cross-encoder** reads the question and passage **together** as a single input and scores
relevance directly, so every token of the question can attend to every token of the passage. Far
more accurate — and it *cannot be precomputed*, because it needs both texts. Scoring all 392 chunks
per query would mean 392 forward passes.

That trade-off is the entire shape of the two-stage design: **a cheap precomputable model for
recall, an expensive pairwise model for precision, over a candidate set small enough to afford it.**

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — small, CPU-friendly, trained on MS MARCO relevance
judgements, free and local like the embedder.

One detail: the cross-encoder scores the passage text **without** the contextual header from Sprint
2. The header exists to help a context-free vector; a cross-encoder already has the question in
front of it, so the header would only add tokens that dilute the pair.

### Stage 3 — diversity for coverage

Relevance ranking has no reason to spread results across sources. The *k* nearest passages to *"how
do BERT and RoBERTa differ"* all live in whichever paper discusses both — the Sprint 2 failure
exactly. Two strategies were implemented and measured against each other.

---

## 2. The ablation

21 questions, k=5, 392 chunks. Each row adds exactly one thing to the row above, so any change is
attributable to that one change.

| configuration | recall | papers | precision | MRR | s/query |
|---|---|---|---|---|---|
| dense only (Sprint 2) | 0.786 | 1.52 | 0.876 | 0.897 | 0.024 |
| + over-fetch 30 | 0.786 | 1.52 | 0.876 | 0.897 | 0.023 |
| + cross-encoder | 0.921 | 1.86 | 0.886 | 0.849 | 1.315 |
| + MMR (λ=0.6) | 0.921 | 1.86 | 0.867 | 0.841 | 1.308 |
| + hard cap (3) | **0.984** | 2.43 | 0.771 | 0.849 | 1.295 |
| **+ soft cap (3, m=.2)** ← final | **0.984** | 2.14 | **0.829** | 0.849 | 1.301 |
| soft cap, no rerank | 0.786 | 1.67 | 0.848 | 0.897 | 0.023 |

**Headline: recall 0.786 → 0.984, distinct papers 1.52 → 2.14.**

### MMR earned nothing

Maximal Marginal Relevance — pick relevant chunks that are *unlike each other* — is the standard
textbook answer to diversity, and I expected it to work. It produced **identical recall and paper
counts** to no diversity at all, and slightly *worse* precision.

Why: MMR penalises **vector similarity**, not same-document-ness. Those correlate, but loosely —
and Sprint 2's contextual header actively works against it, since every chunk of a paper shares a
title prefix that makes them all somewhat similar to each other *and* to chunks of other papers
about the same topic. MMR was pushing on a proxy when the constraint I actually wanted was on the
source.

I kept the implementation, because "we tried the standard approach and measured it earning nothing"
is a more useful thing to own than quietly deleting it.

---

## 3. The aggregate was hiding the structure

The table above nearly led me to a wrong conclusion. Splitting by question type:

| config | single (10) | comparison (7) | thematic (4) |
|---|---|---|---|
| | recall / precision | recall / precision | recall / precision |
| dense only | **1.000** / 0.880 | 0.571 / 0.886 | 0.625 / 0.850 |
| + cross-encoder | 1.000 / 0.880 | **0.857** / 0.914 | 0.833 / 0.850 |
| + hard cap (3) | 1.000 / **0.640** | **1.000** / 0.914 | 0.917 / 0.850 |

Three things only visible here:

1. **Single-paper questions were already perfect** at recall 1.000 under *every* configuration.
   10 of 21 questions — nearly half the aggregate — had no headroom at all, damping every number in
   the summary table toward "small improvement".
2. **The cross-encoder's real effect is on comparison questions**: +28.6pp recall (0.571 → 0.857).
   The aggregate showed +13.5pp because it was averaged against ten questions that couldn't improve.
3. **The paper cap's damage was concentrated entirely where it had no benefit.** On single-paper
   questions it dropped precision 0.880 → 0.640 while buying exactly zero recall.

**The lesson generalises past this project:** an aggregate over a heterogeneous set reports the
average of "helped a lot", "did nothing", and "actively hurt" as one lukewarm number. If your
evaluation set has structure, measure along it — otherwise you will tune away a real win, or ship a
real regression, without ever seeing either.

---

## 4. From hard cap to soft cap: a fix driven by that breakdown

Finding 3 identifies a genuine design flaw. A fixed cap is **unconditional** — with cap=3 and k=5,
a question that one paper legitimately owns still gets two foreign chunks forced into its results.
They buy no recall, and in Sprint 4 they become **two distractors in the LLM's prompt.**

First attempt, loosening the cap from 2 to 3, was a clean win:

| | recall (all) | precision (all) | comparison precision |
|---|---|---|---|
| cap=2 | 0.984 | 0.657 | 0.800 |
| **cap=3** | 0.984 | **0.771** | **0.914** |
| cap=4 | 0.968 | 0.838 | 0.914 |

cap=3 *strictly dominates* cap=2 — same recall, much better precision. cap=4 starts trading recall
away.

But the real problem wasn't the cap's value, it was applying it **blind to relevance**. So:
`soft_cap_select` skips a capped paper's chunk **only if** some under-cap paper has a candidate
within `margin` of it on the normalised relevance scale.

- *Comparison question* → BERT and RoBERTa passages score similarly → margin met → slot yielded.
- *Single-paper question* → the next paper's best chunk is far behind → margin not met → relevance
  wins.

**Diversity becomes the tie-breaker when relevance can't decide, rather than a quota imposed on top
of it.** Tuning the margin:

| margin | recall (all) | comparison recall | single precision | precision (all) |
|---|---|---|---|---|
| hard cap | 0.984 | 1.000 | 0.640 | 0.771 |
| 0.1 | 0.960 | 0.929 | **0.860** | **0.876** |
| **0.2** ← chosen | **0.984** | **1.000** | 0.760 | 0.829 |
| 0.3 | 0.984 | 1.000 | 0.720 | 0.810 |

**margin=0.2 strictly dominates the hard cap** — identical recall everywhere, precision 0.771 →
0.829. margin=0.1 scores higher precision but starts losing comparison recall, so 0.2 is the largest
precision gain that costs nothing.

---

## 5. The finding that corrected me

My first read of the ablation was that **the cross-encoder was nearly worthless.** With the *hard*
cap, `paper cap, no rerank` scored recall 0.968 against 0.984 with re-ranking — one question's
difference, for 55× the latency. I was ready to write that up as "the expensive stage didn't earn
its place."

The final row refutes it:

| | recall | papers | precision |
|---|---|---|---|
| soft cap **with** cross-encoder | **0.984** | 2.14 | 0.829 |
| soft cap **without** cross-encoder | 0.786 | 1.67 | 0.848 |

**A 19.8pp recall collapse.** The reason is a genuine interaction: **the soft cap depends on
relevance scores being meaningful**, because its margin test compares them. Dense cosine scores are
too flat and too unreliably ordered for that comparison to mean anything — the margin test fires
essentially at random. The cross-encoder is what makes the scores trustworthy enough for a
relevance-gated rule to work.

The hard cap masked this because it never consults scores at all — it imposes a quota regardless.
So my "the cross-encoder adds little" reading was an artifact of measuring it underneath a stage
that had brute-forced away the thing it was contributing.

**The transferable point:** in a multi-stage pipeline, ablating one stage at a time only tells you
about *that configuration*. Stages interact, and a stage can look worthless underneath one
downstream choice and load-bearing underneath another. This is also why I kept every configuration
switchable rather than deleting losers as I went — the `soft cap, no rerank` row that overturned my
conclusion only exists because losing configurations stayed runnable.

---

## 6. Costs, stated honestly

**Latency: 0.024s → 1.30s per query, a 55× increase.** Essentially all of it is 30 cross-encoder
forward passes on CPU. Measuring the candidate pool:

| candidates | recall | precision | s/query |
|---|---|---|---|
| 10 | 0.786 | 0.876 | 0.447 |
| 20 | 0.873 | 0.848 | 0.902 |
| **30** | **0.984** | 0.829 | 1.333 |
| 50 | 0.984 | 0.790 | 2.185 |

**30 is the knee.** 20 gives up 11pp of recall; 50 buys nothing and costs another 0.85s. This
matters for Sprint 6 — 1.3s is acceptable for a question-answering UI but it is a real budget item,
and the honest mitigations are a GPU, a smaller cross-encoder, or caching, not a smaller pool.

**Precision: 0.876 → 0.829.** The deliberate cost of diversity, minimised by the soft cap. The
metric that made this visible is the one I added to catch myself.

**MRR: 0.897 → 0.849.** A real, small regression — about one question moving from rank 1 to rank 2
across 21 questions. At this sample size that's noise, and I'm flagging it rather than explaining it
away; if it persists on a larger question set it's worth chasing.

---

## 7. Known limitations

1. **One question still fails**: *"How is the transformer architecture used across these papers?"*
   misses BERT. It's a vague thematic question, and honestly my ground truth for it is debatable —
   which is itself a limitation of hand-labelled evaluation sets.
2. **21 questions is a small sample.** Differences under ~5pp are noise. Every conclusion here
   should be re-checked when the set grows.
3. **Paper-level ground truth can't see chunk quality** (§0). The cross-encoder's main job —
   ordering chunks *within* the right paper — is structurally invisible to this metric, so the
   ablation *under-measures* it. Sprint 5's faithfulness scoring is what closes that gap.
4. **I labelled my own evaluation set**, having chosen the corpus. That's a real bias risk, and the
   mitigation is Sprint 5's LLM-as-judge, which scores answers without my labels.
5. **`cap_margin=0.2` and `candidate_k=30` are tuned on the same 21 questions they're evaluated on.**
   No held-out split. With 21 questions there isn't enough data to split meaningfully, but it means
   these values are fitted, and the honest expectation is slightly worse performance on unseen
   questions.
6. **No hybrid search.** Still no exact-match path; a rare literal token can be missed where BM25
   would nail it. Future scope.

---

## 8. Interview probe points

### Probe 1: "What's the difference between a bi-encoder and a cross-encoder, and why use both?"

The single most likely retrieval question, and it has a clean answer: **a bi-encoder encodes the
question and passage separately and never lets them interact; a cross-encoder reads them together.**

Then the consequence, which is the part that shows understanding: separate encoding is what makes
the bi-encoder *indexable* — passages are embedded once, before any question exists — and it's also
its ceiling, because the passage vector had to summarise the passage for every possible question at
once. The cross-encoder is more accurate precisely because it can't be precomputed.

So: cheap recall stage over 392 chunks, expensive precision stage over 30. Quantify it — +28.6pp
recall on comparison questions, 55× latency, and 30 candidates is the measured knee.

### Probe 2: "How do you know your re-ranker actually helped?"

Tests whether you measure or assume. The answer is the ablation — one stage at a time, four metrics
including one (precision) specifically there to catch a diversity change cheating on recall.

Then the two findings that make it credible: the aggregate **masked** the effect until I split by
question type (single-paper questions were already at recall 1.000 and were damping everything), and
my first read — *the cross-encoder is nearly worthless* — was **wrong**, overturned by the
`soft cap, no rerank` row. Explain the interaction: the soft cap consults relevance scores, so it
needs those scores to be trustworthy; the hard cap ignores them, which masked the cross-encoder's
contribution entirely.

Reporting a conclusion you reversed is the strongest available evidence that the numbers are driving
the decisions rather than decorating them.

### Probe 3: "Your diversity strategy hurt precision. Why ship it?"

Tests whether you can defend a trade-off rather than pretend one doesn't exist.

Beats: (1) yes — 0.876 → 0.829, and I know because I built the metric that catches it; (2) the
recall it buys is larger — 0.786 → 0.984, and comparison questions went 0.571 → 1.000; (3) the first
version was worse (hard cap, precision 0.771) and the breakdown showed *why* — damage concentrated
on single-paper questions where it bought zero recall; (4) the fix was to make diversity
relevance-gated, a tie-breaker rather than a quota, recovering precision to 0.829 at identical
recall.

If pushed on *"why is recall worth more than precision here?"*: because Sprint 4 feeds these chunks
to an LLM that can ignore an irrelevant passage, but **cannot recover a passage that was never
retrieved**. A missing paper is an unanswerable question; an extra one is a distractor. That
asymmetry is what justifies the trade — and it's a claim Sprint 5 will actually test.

---

## 9. How to run

```bash
python -m rag.retrieve "How do BERT and RoBERTa differ in their pre-training objectives?"
python -m rag.retrieve "..." --no-rerank --diversity none    # Sprint 2 behaviour
python -m eval.evaluate                                       # the full ablation
python -m eval.evaluate --kind comparison --show-failures
```

The Sprint 2 failure case, with final settings:

```
1. RoBERTa  | 7 Conclusion | p.10        (dense=0.793  rerank=+4.76)
2. RoBERTa  | 5 RoBERTa    | pp.6-7      (dense=0.808  rerank=+4.64)
3. RoBERTa  | 5 RoBERTa    | pp.6-7      (dense=0.796  rerank=+4.62)
4. BERT     | 1 Introduction | pp.1-2    (dense=0.711  rerank=+1.98)
5. BERT     | 3 BERT       | pp.3-4      (dense=0.726  rerank=+1.64)
distinct papers: 2
```

Both papers present, where Sprint 2 returned five chunks of RoBERTa and no BERT.

---

## 10. Next: Sprint 4 — Generation with citations

Retrieval now reliably puts the right passages in front of the model. Sprint 4 turns them into a
grounded answer:

- **Gemini** (free AI Studio tier) behind a **provider interface**, so the LLM is swappable without
  touching the pipeline — and so Sprint 5 can plug in Groq for judging.
- A prompt that forces every claim to carry a citation to a retrieved chunk.
- Verification that emitted citations map to real retrieved chunk IDs — a model that invents a
  citation is worse than one that declines to answer, because it looks trustworthy.
- I'll confirm Gemini's current free-tier rate limits against Google's own docs before wiring
  anything, rather than assuming numbers that change.

The metric that matters then is faithfulness, not recall: *is every claim in the answer actually
supported by the retrieved text?* That needs Sprint 5's harness to answer properly — which is why
generation and evaluation are adjacent sprints.
