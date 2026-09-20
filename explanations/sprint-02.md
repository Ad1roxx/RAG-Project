# Sprint 2 — Chunking, Embeddings and the Vector Store

**Goal:** split the 206 sections into retrievable passages, embed them locally, index them, and
confirm we can find semantically similar chunks across papers.

**Built:** `ingestion/chunk.py`, `rag/embed.py`, `rag/store.py`
**Result:** 392 chunks → 392 × 384-dim vectors → a FAISS index that answers a query in under a
millisecond. Plus one measured limitation that defines what Sprint 3 has to fix.

---

## 0. First: three Sprint 1 bugs that only became visible now

Before any chunking, I measured the section-length distribution to pick a chunk size. That
immediately surfaced an impossible number: a section titled **`Model` containing 50,237
characters**.

Chasing it found a cascade in the GPT-3 paper:

1. GPT-3 puts its **References on page 68**, *after* a 25-page appendix — not before it, as the
   other nine papers do. So the "stop at references" rule fired 25 pages too late.
2. The appendix contains large results tables. A flattened table row
   (`T5-Small 2.08E+00 1.80E+20 60 1,000 3 3 1 0.5`) is long and has many tokens per line, so it
   **passed the words-per-line filter** from Sprint 1 cleanly.
3. A table's `Model` column header is a one-word block — and `"model"` was in my unnumbered-heading
   list. It became a heading and swallowed everything after it.

### The fixes, and why they're the ones I chose

**Table rows: filter on composition, not shape.** Sprint 1's filter tested a block's *shape*
(words per line). That cannot catch table rows, which are shaped like prose. What distinguishes
them is what they're made of: prose is overwhelmingly alphabetic, table rows are mostly numeric. A
rule of "at least 50% of tokens must contain a letter" removes them wherever they appear in the
document.

Note the pattern — Sprint 1's noise problem also needed a *ratio* rather than a magnitude. Different
ratio, same lesson: **when a threshold on size fails, the signal is usually in composition or
proportion.**

**Generic headings: respect the asymmetry of the risk.** I removed `model`, `method`, `results`,
`approach`, `evaluation`, `analysis` from the unnumbered-heading set. They are exactly the words
that appear as table column headers. The reasoning is about *asymmetric damage*:

- A **missed** heading merges a section into its neighbour → slightly worse citation precision.
- A **false** heading corrupts document structure from that point onward → a 50,000-character
  phantom section.

When one error is recoverable and the other is catastrophic, the threshold should not sit in the
middle. Papers that section on these words nearly always number them (`3 Model`), which the
numbered pattern catches anyway — so the cost is close to zero.

**Lettered appendix headings.** `A Details of Common Crawl Filtering`, `H.5 Results for Scramble
Tasks`. I added these because appendix *prose* is often genuinely answer-bearing — method details
that didn't fit the page limit. It was only the appendix *tables* that were poisoning retrieval, and
those are now handled by composition.

The catch: `A Details of Common Crawl Filtering` and `A Transformer uses attention` have identical
shape. Title case separates them — real headings capitalise most non-stopwords, sentences don't. So
a lettered heading must pass a title-case test at 60%.

**Result:** GPT-3 went from 51 sections / 203,652 chars (including 66K of table soup under a
phantom heading) to 51 sections / 187,714 chars with a properly structured appendix.

**Why this matters more than the bug count:** none of these threw an exception. Sprint 1 reported
"done" and the output looked plausible. They surfaced because I computed a distribution and one
number was absurd. In a pipeline this is the *normal* way defects are found.

---

## 1. Chunking — the first decision that directly moves answer quality

### Why not just embed whole sections?

Sections run from 58 to 50,000 characters. Two independent reasons that fails:

**The hard reason:** the embedding model accepts 512 tokens. Longer input is **truncated, not
rejected** — no error, no warning. A 4,000-token section would be indexed as its first 512 tokens,
and the other 87% would be unfindable while appearing to be in the index.

**The soft reason:** an embedding is a single point in space. Encode a 3,000-word section covering
architecture, training setup and three ablations, and you get a vector that averages all of it —
close to the centroid of several topics and therefore strongly similar to *none of them*. This is
worth internalising: **embeddings do not "contain" text, they position it.** One vector can only be
in one place, so a passage should be about one thing.

### The trade-off, stated honestly

| | Cost |
|---|---|
| **Chunks too large** | Vector averages several ideas; each retrieved chunk burns prompt space on text the question didn't ask about |
| **Chunks too small** | Passage loses the context that made it meaningful. *"It improves by 2.1 points"* is both unretrievable and unusable |

There's no universally correct size — it depends on corpus, model and question style. What's
defensible is the *reasoning*, and measuring it later (Sprint 5).

### Three commitments

**1. Chunks never cross a section boundary.** A passage half-Method and half-Results describes no
single thing, so its vector sits between two topics and is a near-miss for queries about both.
Section boundaries are semantic boundaries the authors already drew for us — Sprint 1's structure
extraction is what makes this possible.

**2. Chunks are measured in the embedding model's own tokens.** Not characters, not words.
`chunk.py` loads the *same tokenizer* `bge-small-en-v1.5` uses. Sizing with a different tokenizer
makes the budget a guess, and guessing over the limit truncates silently.

Budget: **400 tokens**, leaving headroom under the 512 limit for the contextual header (§2). The
pipeline verifies this rather than trusting it — `chunk.py` reports how many chunks would exceed
512 after the header is added. It prints **0**.

**3. Chunk boundaries land on sentence boundaries.** No chunk starts or ends mid-thought.

### Overlap: why 60 tokens of duplication is worth it

Consecutive chunks share ~60 tokens. This sounds wasteful — the same sentences indexed twice.

The reason: **the sentence that answers a question and the sentence that names its subject are
often neighbours.** Split between them with no overlap and *neither* chunk is retrievable — one has
the answer without the subject, the other the subject without the answer. Overlap keeps the pair
intact in at least one chunk.

Verified across the corpus: 166 of 185 chunk seams carry verified overlap. The other 19 are seams
where the final sentence was longer than the 60-token budget, so nothing could be carried — correct
behaviour, not a bug.

### Sentence splitting in academic prose

Naive splitting on `[.!?]` fails badly on papers: `et al.`, `e.g.`, `i.e.`, `Fig. 3`, `Eq. 2`, and
author initials (`J. Devlin`) are all periods that don't end sentences. `split_sentences` splits
then **re-joins** where the previous fragment ended on a known abbreviation or a single-capital
initial.

There's also a hard-split fallback for a single sentence exceeding the whole chunk budget (rare, but
real — a sentence listing 40 benchmark names). Without it, that chunk overflows and gets silently
truncated.

### Small-chunk handling

A trailing fragment under 32 tokens merges back into the previous chunk of the same section. If
there's nothing to merge into, it's dropped — those are whole sections that are one short sentence,
which in practice means a preamble sitting above its subsections:

> *"In this section, we present BERT fine-tuning results on 11 NLP tasks."*

That **announces** content rather than containing any. Indexed, it only competes for the queries its
own subsections should win. 7 such chunks were dropped.

**Final: 392 chunks.** Median 342 tokens, p25 198, max 400, min 33. Zero exceed the model's limit.

---

## 2. The contextual header — and an experiment that didn't go my way

Each chunk is embedded with its paper title and section heading prepended:

```
BERT: Pre-training of Deep Bidirectional Transformers... | 3.1 Pre-training BERT

BERTBASE was chosen to have the same model size as OpenAI GPT...
```

**The motivation:** a chunk reading *"We use 12 layers, 768 hidden units and 12 attention heads"* is
nearly content-free alone — it could be any transformer paper ever written, and its vector sits in a
crowded, undiscriminating region. The header anchors it to a specific paper and section.

**The cost I documented up front:** the header repeats in every chunk of a section, nudging them all
toward each other and diluting each passage's own signal.

### So I measured it

Two indexes, identical except for the header, on two question sets:

| | Distinct papers in top-5 (5 comparison questions) | Correct paper @1 / @3 (8 paper-specific questions) |
|---|---|---|
| **With header** | 1.6 | 6/8 · **8/8** |
| **Without header** | **2.0** | 6/8 · 7/8 |

**The honest read: this is inconclusive.** The header helps paper-specific recall by one question
and costs diversity — both differences well within noise at n=5 and n=8. Expected-paper recall was
*identical* (7/10).

I'm keeping the header, because its motivation for orphan chunks is sound and it gives citations
context. But I'm recording that this is **a reasoned default, not a validated one**, and it's on the
list for Sprint 5 when there's a real eval harness to settle it.

Resisting the urge to declare a winner from 13 questions is the point. A 0.4-paper difference on
5 questions is the kind of number that turns into a confident wrong belief if you let it.

---

## 3. Embeddings

### What an embedding actually is

A model that maps text to a fixed-length vector (384 numbers here) positioned so semantically
similar texts land near each other. That's the whole mechanism behind "search by meaning": embed
every passage once, embed the question at query time, find nearest vectors. **No keyword needs to
match** — *"how do you stop a model hallucinating"* can retrieve a passage about *"grounding
generation in retrieved evidence"*.

### Model choice: `bge-small-en-v1.5` over `all-MiniLM-L6-v2`

MiniLM-L6-v2 is the default in most RAG tutorials. I measured both:

| Model | Max tokens | Dim |
|---|---|---|
| `all-MiniLM-L6-v2` | **256** | 384 |
| `bge-small-en-v1.5` | **512** | 384 |

**That limit is a hard ceiling on chunk size.** With a 400-token budget, MiniLM would have silently
discarded roughly the back half of every full chunk — and nothing downstream would have looked
wrong. Same dimensionality, same rough size, double the context, better MTEB retrieval scores. The
context limit is the deciding argument.

**The general lesson:** the embedding model's `max_seq_length` and your chunk size are *one coupled
decision*, not two. Most tutorials never mention this, which is why they pick 512-token chunks and a
256-token model.

### Query/passage asymmetry (easy to get wrong, silent when wrong)

BGE models are trained **asymmetrically**: passages are embedded bare, but queries get an
instruction prefix — `"Represent this sentence for searching relevant passages: "`.

This isn't decoration, it's how the model was fine-tuned. It exists because **a question and the
passage answering it are not paraphrases.** *"What optimiser did BERT use?"* shares little surface
form with *"We train with Adam using a learning rate of 1e-4"*. The prefix tells the model to encode
the text as a *search intent* rather than a statement, pulling it toward answering passages instead
of toward other questions.

Omit it and nothing breaks — results still come back, just measurably worse. It's one of the most
common mistakes made with BGE models, which is why it lives in `embed.py` where it can't be
forgotten, rather than at call sites.

### Normalisation

Every vector is normalised to unit length, which makes the dot product of two vectors **exactly
their cosine similarity**. Two consequences: the index can use fast inner-product search and still
rank by cosine; and every score lands on a comparable ~0–1 scale, so a threshold means something.

---

## 4. The vector store

### FAISS over Chroma

Chroma bundles index, metadata and persistence, returning documents directly. FAISS does one thing:
given vectors, find nearest. The mapping from *"index row 247"* → *"chunk 1810.04805:3:1"* is ours
to maintain.

That's the reason to choose it here. **That mapping is the provenance chain every citation depends
on** — worth understanding rather than delegating. It cost about 60 lines.

### `IndexFlatIP` — exact, not approximate

"Flat" means exhaustive: every query compares against all 392 vectors, so results are **exact**. The
alternatives (IVF, HNSW) are approximate — they trade recall for speed and need tuning plus a
training pass.

At this size that trade would be a mistake. 392 × 384 floats ≈ 600 KB; search takes under a
millisecond. An approximate index would save nothing measurable while **introducing a second
possible cause for a bad result**. When retrieval looks wrong, the first question is always "is it
retrieval or generation?" — an exact index permanently removes one branch of that search.

*The threshold worth knowing:* exhaustive search stays comfortable into the low hundreds of
thousands of vectors. Past that, HNSW is the usual next step.

### Two guards against silent corruption

Both defend the same invariant — index row `i` and `chunks[i]` are the same passage:

1. **On load**, `index.ntotal != len(chunks)` raises. A mismatch wouldn't crash, it would return
   confidently **wrong citations** — row *i* pointing at another passage's metadata. That's worse
   than a crash: the system would cite a real paper for a claim it never made.
2. **Before every query**, the index's manifest records which embedding model built it, and
   `assert_matches` refuses a query from a different one. Two models place text in two unrelated
   spaces; querying one with the other returns *k* results ranked by a similarity that means
   nothing. It is the single easiest way to get a quietly broken RAG system.

---

## 5. Does it work? Yes — and here's exactly where it doesn't

### It works for direct questions

> **"What is the purpose of multi-head attention?"**
> 1. `[0.743]` Attention Is All You Need — **§3.2.2 Multi-Head Attention**, pp.4–5
> 2. `[0.733]` Attention Is All You Need — §3.2 Attention, pp.3–4

Top hit is precisely the right subsection, with page-level provenance. This is the payoff from
Sprint 1's structure extraction: we can cite *§3.2.2, pp.4–5*, not just *"the Transformer paper"*.

### It fails on comparison questions — the sprint's most important finding

> **"How do BERT and RoBERTa differ in their pretraining objectives?"**
> 1. `[0.816]` RoBERTa — §5 RoBERTa
> 2. `[0.803]` RoBERTa — §5 RoBERTa
> 3. `[0.802]` RoBERTa — §2 Background
> 4. `[0.800]` RoBERTa — §7 Conclusion
> 5. `[0.798]` RoBERTa — §1 Introduction

**All five from RoBERTa. Zero from BERT.** A question explicitly about two papers retrieved one.
Worse, results 1 and 2 are overlapping neighbours from the same section — near-duplicates wasting
slots.

Across 5 comparison questions, mean distinct papers in top-5 was **1.6**.

My first hypothesis was the contextual header — RoBERTa's title contains both "BERT" and
"Pretraining", so it would lift every RoBERTa chunk. **The experiment in §2 refuted that**:
diversity was nearly as bad without headers (2.0), and expected-paper recall identical.

So the cause is more fundamental: **a single query vector can only point at one place.** The query
embeds to one point; its nearest neighbours are whichever region best matches the *whole* query.
RoBERTa's paper discusses both BERT and RoBERTa, so its chunks are uniformly closest. Nothing in
pure top-*k* dense retrieval encourages spreading results across sources — *k*-nearest-neighbours
does exactly what it says, and "nearest" says nothing about "diverse".

**This is not a bug. It's the known ceiling of naive dense retrieval, and it's precisely what
Sprint 3 exists to fix.** Finding it with a number attached, before building generation on top, is
the sprint working as intended — an ungrounded answer to a comparison question would have been
*much* harder to debug from the far end of the pipeline.

---

## 6. Known limitations

1. **Comparison questions return one paper** (§5). Sprint 3's target.
2. **Footnote splicing.** Footnotes extract as page-bottom blocks and land mid-paragraph in reading
   order: `...to its left.4 1https://github.com/... 3In all cases we set...`. **Measured: 3 of 392
   chunks** show the splice pattern, 11 contain a URL. Real but rare — the fix (font-size filtering
   via `get_text("dict")`, since footnotes are set smaller) is a Sprint 1 change I chose not to make
   on a 0.8% defect rate. Revisit if eval implicates it.
3. **Corpus imbalance.** GPT-3 is 140 of 392 chunks (36%) — it's a 75-page paper among 10-page ones.
   Retrieval has no per-paper normalisation, so it's structurally over-represented.
4. **Chunk size is a reasoned guess.** 400/60 was chosen from the distribution and the model limit,
   not measured against answer quality. Sprint 5.
5. **The header is unvalidated** (§2).
6. **No hybrid search.** Pure dense retrieval has no exact-match path, so a rare literal token (a
   specific metric name, `bert-base-uncased`) can be missed where BM25 would nail it. Named as
   future scope in PROJECT_PLAN.md.

---

## 7. Interview probe points

### Probe 1: "How did you choose your chunk size?"

The trap is answering "512 with 50 overlap" — the tutorial default, which signals you copied it.

The strong answer makes the **coupling** explicit: chunk size is not an independent parameter, it's
bounded by the embedding model's `max_seq_length`, and exceeding it **truncates silently**. So the
order of reasoning is: pick the model → learn its limit (512) → reserve headroom for the contextual
header → budget 400 → verify zero chunks exceed the limit after headers.

Then the trade-off (too large averages topics, too small loses context), the section-boundary
constraint, and finally the honest part: **the size is a reasoned starting point, not a validated
optimum** — validating it is what the eval harness is for. That last sentence is what separates
someone who tuned a RAG system from someone who configured one.

### Probe 2: "Your system returns 5 chunks from one paper for a comparison question. Why?"

This is the deepest thing in the sprint and I found it in my own system.

Beats: (1) the observation, with numbers — 5/5 from RoBERTa, mean 1.6 distinct papers; (2) the
hypothesis I actually tested and **refuted** (contextual header — diversity was 2.0 without it,
recall identical); (3) the real cause — *a single query vector can only point at one region, and
k-NN optimises nearness, which is silent about diversity*; (4) the fixes, in order of cost: a larger
candidate pool re-ranked by a cross-encoder, MMR or per-document caps for diversity, and hybrid
search; query decomposition is the agentic approach and is deliberately out of scope.

What makes this strong is beat (2). Reporting a hypothesis you disproved shows you measure instead
of rationalise.

### Probe 3: "What breaks if you swap the embedding model?"

Tests whether you understand that embeddings define a *space*, not a *format*.

The answer: **everything, silently.** Two models place text in unrelated vector spaces. Querying an
index built by model A with a vector from model B returns *k* results, ranked, with plausible
scores, all meaningless. Nothing raises.

So the index stores its model in a manifest and `assert_matches` refuses the query. Follow-ups worth
being ready for: re-embedding the whole corpus is the only migration path (vectors can't be
converted); dimensionality changes force an index rebuild; and this is why `requirements.txt` pins
model-related versions — a silent upgrade that changes embeddings invalidates the index without
changing a line of code.

---

## 8. How to run

```bash
python -m ingestion.chunk                 # 206 sections -> 392 chunks
python -m rag.store build                 # embed + index (~31s, CPU)
python -m rag.store query "What is the purpose of multi-head attention?" -k 5
```

`python -m ingestion.chunk --stats-only` reports the token distribution without writing, for
experimenting with `--chunk-tokens` / `--overlap-tokens`.

---

## 9. Next: Sprint 3 — Retrieval and re-ranking

Sprint 2 produced working vector search **and a measured ceiling**. Sprint 3 raises it:

- Retrieve a **larger candidate pool** (k≈20–30) instead of trusting the top 5.
- **Re-rank with a cross-encoder.** The key difference: a bi-encoder (what we have) embeds query and
  passage *separately* — fast, index-able, but the two never interact. A cross-encoder reads them
  *together* and scores relevance directly. Far more accurate, far too slow to run over 392 chunks,
  perfect for re-ordering 25 candidates.
- **Diversity** — MMR or per-paper caps — aimed squarely at the comparison-question failure.

Then I'll re-run the same measurements from §5. That number, 1.6 distinct papers, is now the
baseline Sprint 3 has to beat.
