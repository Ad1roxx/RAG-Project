# Sprint 1 — Data Ingestion

**Goal:** fetch ML/AI papers from arXiv and turn their PDFs into clean, section-structured text
on disk — reproducibly, with no PDFs committed to the repo.

**Built:** `ingestion/paths.py`, `ingestion/fetch_papers.py`, `ingestion/extract.py`
**Result:** 10 papers → 206 sections → ~492,000 characters of clean text in `data/processed/`

---

## 1. Why ingestion is its own sprint (and why it matters more than it looks)

It is tempting to treat "get the text out of the PDF" as plumbing you rush through to reach the
interesting part. Resist that. Here is the causal chain:

```
PDF text → chunks → embeddings → vector index → retrieved chunks → LLM prompt → answer
```

Every arrow is lossy and *silent*. If a figure's axis labels end up in the text, they become part
of a chunk. That chunk gets embedded. The embedding sits in the index looking like any other
vector. One day a user's question lands near it, it gets retrieved, it occupies a slot in the
prompt that a real passage should have had, and the answer is slightly worse. Nothing errors.
No test fails. You just have a system that is quietly a bit dumber than it should be.

This is the defining difficulty of RAG that PROJECT_PLAN.md warns about: **quality degrades
silently**. In Project 1 (TableTalk), a broken preprocessing step showed up as a visibly worse F1
score. Here there is no F1 until we build the eval harness in Sprint 5. Until then, the only
defence is inspecting output at each stage — which is exactly what we did in this sprint, and why
we found three real bugs before writing a single line of chunking code.

**The principle:** in a pipeline where errors are invisible, you buy correctness by looking at the
data after every stage, not by writing more careful code up front.

---

## 2. `fetch_papers.py` — getting the corpus

### What it does
Calls the arXiv API for a list of paper IDs, saves the metadata as JSON, and downloads each PDF.

```
python -m ingestion.fetch_papers                     # the 10-paper seed corpus
python -m ingestion.fetch_papers --ids 1706.03762
python -m ingestion.fetch_papers --category cs.CL --max-results 15
```

### Why the raw API instead of the `arxiv` PyPI package

The `arxiv` package would have saved ~60 lines. I skipped it deliberately.

The arXiv API returns an **Atom feed** — XML, one `<entry>` per paper, with arXiv-specific fields
in their own namespace. Parsing it with `xml.etree` means the mapping from "what arXiv said" to
"what we stored" is visible in our code, not inside someone else's library.

That matters *specifically for RAG*, more than it would in a normal project. The end product of
this system is an answer with **citations**. A citation is a promise: "this claim came from this
paper." That promise is only as good as the provenance chain, and this file is link one of that
chain. If the citation metadata is wrong, the system's core value proposition is a lie — and
"I don't know, the library handled that" is not an answer you want to give when an interviewer
pulls on it.

This is the CLAUDE.md rule applied concretely: *if you can't explain a line, it doesn't go in.*

### Why this specific seed corpus

The 10 papers are not random:

```
Attention Is All You Need  →  BERT  →  RoBERTa  →  DistilBERT
                           →  GPT-3  →  Chain-of-Thought
Sentence-BERT  →  DPR  →  RAG
LoRA
```

They **cite and build on each other**. That is the whole point. The system's claim is
*multi-paper* retrieval — the ability to answer a question whose evidence is spread across several
documents. A corpus of 10 unrelated papers physically cannot demonstrate that: every question
would be answerable from exactly one paper, and a single-document QA system would score
identically.

With this corpus, questions like *"how do BERT, RoBERTa and DistilBERT differ in their pre-training
objectives?"* require pulling passages from three papers and synthesising. That is the capability
we will measure in Sprint 5. **Choosing a corpus that can actually exercise the feature you're
building is part of the engineering, not a detail.**

Bonus: two of these (Sentence-BERT, DPR) are the papers behind the embedding and retrieval
techniques we're about to implement. The system will be able to explain its own architecture,
which makes for a good demo.

### Three production habits worth noticing

**Idempotency.** `download_pdf` skips files already on disk. Re-running the script tops up the
corpus instead of re-downloading everything. This is what makes "reproducible fetch, not
checked-in data" actually workable — otherwise adding an 11th paper means a 10-paper round trip.

**Atomic writes.** PDFs are written to `.pdf.part` and then `.replace()`d into position. Without
this, a crash mid-download leaves a truncated `.pdf` on disk — and because the script skips files
that exist, that corrupt file is treated as valid *forever*. You'd get a mysterious extraction
failure weeks later with no obvious cause. One extra line prevents a genuinely nasty bug class.

**Politeness.** 3-second delay between downloads, descriptive `User-Agent`, bounded retries with
backoff. arXiv is a free service run by a university library. A scraper that hammers it gets
IP-banned, and then your "reproducible" corpus isn't reproducible. The `User-Agent` exists so they
can email you instead of silently blocking you.

---

## 3. `extract.py` — PDF to clean text (the hard part)

### Why PDF extraction is genuinely hard

A PDF does not contain paragraphs. It contains instructions of the form *"draw glyph 'e' at
coordinates (x=142.7, y=388.2) in font Times-Roman-10"*. There is no "this is a heading", no "this
paragraph continues on the next page", no "this text is a figure label, ignore it". All structure
you get out is structure you **reconstructed** from geometry and text patterns.

Research papers are the hard case of the hard case: two-column layouts, figures with text inside
them, equations, footnotes, margin stamps, and a references section that is 20% of the document.

### Library choice: PyMuPDF

| Option | Verdict |
|---|---|
| `pypdf` | Pure Python, no dependencies, but weak text extraction — mangles multi-column layouts |
| `pdfplumber` | MIT licensed, great for tables, but slow and built on the aging `pdfminer.six` |
| **PyMuPDF** | **Chosen** — fastest, best text quality, and exposes *block* structure with coordinates |

**The honest caveat:** PyMuPDF is AGPL-licensed. For a portfolio project and internal research this
is fine. For a commercial closed-source product it would force a paid license or a switch to
`pdfplumber`. Know this — it is exactly the kind of thing a senior engineer asks about, and
"I hadn't considered the license" is a bad answer. The extraction interface is small enough that
swapping libraries would be a contained change.

The deciding factor was **blocks**. `page.get_text("blocks")` returns visually grouped runs of text
with their bounding-box coordinates, rather than one flat string. That structure turned out to be
the key to the noise problem below.

### Problem 1: figure internals pollute the text

Page 3 of the BERT paper produced this before filtering:

```
BERT | E[CLS] | E1 | E[SEP] | ... | EN | C | T1 | T[SEP] | ... | TN |
[CLS] | Tok 1 | [SEP] | ... | Tok N | Masked Sentence A | Pre-training | NSP
```

That is the inside of Figure 1 — node labels from a diagram. As flat text it is meaningless, but
it would embed happily and pollute retrieval for any BERT-related query.

**The obvious fix that doesn't work:** "drop short blocks." I checked the real numbers — those
figure blocks run up to 11 words, and papers contain legitimate one-line paragraphs. Any threshold
that kills the noise also kills real content.

**What actually works — words *per line*:**

| Block type | Words | Lines | Words/line |
|---|---|---|---|
| Body paragraph | 108 | 9 | **12.0** |
| Figure labels | 11 | 7 | **1.6** |
| Heading ("1 Introduction") | 2 | 2 | 1.0 |

Prose wraps at ~8–12 words per line because that's how much fits in a column. Figure labels are one
short label per line by construction. The ratio separates them almost perfectly, where raw length
does not. Headings score low too — so they're rescued by an explicit heading check that runs first.

**This is the transferable lesson:** when a simple threshold fails, look for a *ratio* or a
*shape* feature instead of a magnitude one. I found this by printing the actual blocks and reading
them, not by reasoning about it in the abstract.

### Problem 2: hyphenation — the genuinely ambiguous one

This is the most interesting problem in the sprint. In a justified two-column layout, LaTeX breaks
words across lines with a hyphen:

```
...a new language representa-
tion model called BERT...        →  "representation"   (soft hyphen — remove it)

...without substantial task-
specific architecture...          →  "task-specific"    (real hyphen — keep it)
```

**The characters on the page are identical.** `word-\nword` in both cases. There is no local rule
— no regex, no heuristic about the surrounding characters — that can tell them apart, because the
information genuinely isn't there. Getting it wrong produces `representa tion` (splitting a real
word) or `taskspecific` (fusing a real compound). Both corrupt retrieval.

**The solution: ask the document itself.**

Papers repeat their own terminology constantly. So before cleaning anything, we scan the whole
document and build two vocabularies: words seen unbroken, and hyphenated compounds seen intact.
Then for each line-break hyphen:

1. Does the joined form (`representation`) appear elsewhere in this paper? → soft hyphen, join it.
2. Does the hyphenated form (`task-specific`) appear elsewhere intact? → real hyphen, keep it.
3. Neither? → join. Soft hyphenation is far more common in justified text, so it's the better bet.

Verified on the corpus: **zero** line-break hyphens remain, `representation` is correctly fused,
and `fine-tuned`, `pre-train` and `state-of-the-art` correctly kept their hyphens.

**Why this is worth understanding deeply:** the shape of this solution — *when local information is
insufficient, use the broader context you already have* — is the same shape as RAG itself. A
language model asked a factual question has insufficient local information (its weights). RAG's
answer is to go get the relevant context and decide with it. Same move, different scale. If you
understand why the vocabulary trick works here, you understand why retrieval augmentation works.

### Problem 3: the references section is retrieval poison

We cut everything from the `References` heading onward. This deserves justification, because
deleting 20% of every document sounds reckless.

A references entry reads: *"Devlin, J., Chang, M., Lee, K., Toutanova, K. 2019. BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding. In NAACL."*

Now consider the question *"What is BERT?"*. That reference chunk is **lexically dense with exactly
the right words** — BERT, pre-training, bidirectional, transformers — so it embeds very close to
the query. It will be retrieved. And it answers nothing. It is a pointer, not content.

So references are the worst possible case for a vector store: maximally attractive to relevant
queries, minimally useful as answers. Dropping them is not losing data, it's removing a systematic
source of false positives.

**Known cost, honestly stated:** papers that place an appendix *after* the references lose that
appendix. For this corpus the appendices are hyperparameter tables and extra figures, so the loss
is small. If it mattered, the fix is to resume collecting at an `Appendix` heading rather than
stopping outright.

### Problem 4: Unicode debris

LaTeX PDFs are full of **ligatures** — `fi`, `fl`, `ffi` rendered as single glyphs (U+FB01 etc.).
`"fine-tuned"` extracts as `"ﬁne-tuned"` with one character where you expect two. To a tokenizer
that is a *different word*, so a query about "fine-tuning" gets a weaker match than it should.

`unicodedata.normalize("NFKC", text)` expands all of them in one line. NFKC is "compatibility
composition" — it folds characters to their canonical equivalents. It leaves curly quotes and
en/em dashes alone, so those get an explicit mapping table. Verified: **zero** ligatures remain
across all 492K characters.

### Section detection and why we track page numbers

Headings are found with two patterns: numbered (`3.1 Model Architecture`) and a known list of
unnumbered ones (`Abstract`, `Related Work`, `Conclusion`). Each section records `start_page` and
`end_page`.

Those page numbers exist for **citation granularity**. There is a real difference between a system
that says *"according to the BERT paper"* and one that says *"BERT (Devlin et al.), §3.1
Pre-training BERT, p.4"*. The second is verifiable by the reader in seconds. That is the whole
credibility story of RAG, and it is only possible if provenance is captured here, at ingestion —
you cannot reconstruct it later from an embedding.

**Alternative worth knowing:** font size is a more robust heading signal than regex (headings are
bold/larger), and PyMuPDF exposes it via `get_text("dict")`. I chose regex because it is verifiable
by eye against the real corpus and doesn't vary across LaTeX template families. If we add papers
from a venue with unusual formatting and headings start getting missed, font size is the upgrade.

---

## 4. The three bugs we caught by inspecting output

Worth recording, because "how do you debug a pipeline where nothing throws an exception" is a real
interview question, and this is the concrete answer.

**Bug 1 — phantom date headings.** BERT produced a section called `'24 May 2019'` holding 1,306
characters of the real Introduction. Cause: arXiv prints a vertical stamp down the margin of page 1
— `arXiv:1810.04805v2 [cs.CL] 24 May 2019`. My regex stripped up to `[cs.CL]` and left the date,
which then matched the numbered-heading pattern (`24` + `May 2019`) perfectly. Fix: make the date
part of the stamp pattern.

**Bug 2 — table-of-contents phantom sections.** GPT-3 (a 40-page paper) has a ToC, and every entry
— `2 Approach 6`, `3 Results 10` — parsed as a heading, creating a duplicate of every real section.
Fix: reject heading titles ending in a bare number, since that's a page reference.

**Bug 3 — silently missing sections.** BERT was missing `4.2 SQuAD v1.1` and `4.3`; their content
had been absorbed into `4.1 GLUE`, which ballooned to 7,252 characters. Cause: my title pattern was
`[A-Z][^.]{2,80}` — it *excluded periods*, and `SQuAD v1.1` contains one. Fix: allow periods,
guard against false positives with "no trailing period" and "no trailing page number" instead.

Bug 3 is the one to pay attention to. Nothing crashed. The output looked plausible. The only tell
was a section with an anomalous character count — and I only saw it because I printed every section
with its size instead of trusting that it worked. **That is the RAG debugging discipline in
miniature: print the intermediate output and read it.**

---

## 5. The data contract

`data/processed/<arxiv_id>.json`, consumed by the chunker in Sprint 2:

```json
{
  "arxiv_id": "1810.04805",
  "title": "BERT: Pre-training of Deep Bidirectional Transformers...",
  "authors": ["Jacob Devlin", "..."],
  "abstract": "We introduce a new language representation model...",
  "abs_url": "http://arxiv.org/abs/1810.04805v2",
  "n_pages": 16,
  "sections": [
    {
      "index": 1,
      "heading": "1 Introduction",
      "start_page": 1,
      "end_page": 2,
      "text": "Language model pre-training has been shown...",
      "char_count": 2573
    }
  ],
  "char_count": 35170
}
```

Two deliberate choices:

- **No `full_text` field.** It would double every file for data already present in `sections`. The
  chunker joins sections when it needs continuous text.
- **Metadata is copied in, not referenced.** A processed paper is self-contained, so downstream
  stages never need to open two files to build a citation.

`data/` is gitignored in full — it is *derived* state, rebuildable by running two commands. The
repo stays small and the corpus stays reproducible.

---

## 6. Known limitations (say these before an interviewer finds them)

1. **Tables are dropped.** Table cells are short lines, so the words-per-line filter removes them.
   For results tables this is mostly fine — they're garbled by flat extraction anyway — but a
   question like *"what GLUE score did BERT-large reach?"* may be unanswerable. Fixing it properly
   means table-aware extraction (`pdfplumber`) and rendering tables as markdown.
2. **Appendices after the references are lost** (see §3, Problem 3).
3. **Equations become nonsense.** `page.get_text` flattens math to stray symbols. Papers explain
   their equations in prose, so this is survivable.
4. **Scanned PDFs would produce nothing.** Every one of these is a born-digital LaTeX PDF with a
   real text layer. A scanned paper would need OCR — out of scope, worth naming.
5. **Heading detection is English- and arXiv-shaped.** A differently formatted venue could fall
   through to the single-`Body`-section fallback.

---

## 7. Interview probe points

These are the three things most likely to get pulled on. Be ready to go deep.

### Probe 1: "How did you handle PDF extraction quality, and how do you know it worked?"

This is the real question behind any RAG interview — it tests whether you built a pipeline or
followed a tutorial. Tutorials call `PyPDFLoader` and move on.

Answer with the **specific defects and the evidence**: figure-label noise (solved with a
words-per-line ratio after a length threshold provably couldn't work), ligatures (NFKC), hyphenation
(the vocabulary heuristic), references (cut, because they're lexically attractive but
informationally empty). Then the verification: zero ligatures, zero line-break hyphens, zero stamps
across 492K characters, and section structure eyeballed paper by paper.

The follow-up is usually *"how do you know you didn't remove real content?"* — answer: the
words-per-line filter has an unconditional keep above 25 words, headings are checked before
filtering, and the known cost (tables) is documented rather than hidden.

### Probe 2: "Walk me through the hyphenation problem."

This one rewards depth because most candidates have never noticed it exists. The key beats:

1. The ambiguity is **real** — identical characters, two different correct outputs.
2. Therefore no local rule can work. State this explicitly; it shows you can recognise when a
   problem is underdetermined rather than flailing at heuristics.
3. The document's own vocabulary is the extra information that resolves it.
4. The fallback (join) is chosen on base rates, not arbitrarily.

If they push on failure modes: a word appearing *only* in hyphenated-broken form has no evidence
either way and falls back to joining. Rare, and the damage is one malformed token in one chunk.

### Probe 3: "Why did you cut the references section?"

A trap question — it sounds like you threw away data. The strong answer inverts it: references are
the **highest-risk** content in a paper for a vector store, because they are maximally similar to
relevant queries and minimally useful as answers. A reference to the BERT paper will out-compete a
real passage for the query "what is BERT?" while containing no information.

This shows you reason about retrieval as *competition for limited prompt slots*, not just
similarity scores — which is the mental model that separates people who've tuned a RAG system from
people who've built one that technically runs.

---

## 8. How to run and verify

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m ingestion.fetch_papers     # ~10 PDFs, ~1 min (3s politeness delay)
python -m ingestion.extract          # PDFs -> data/processed/*.json
```

Current output: 10 papers, 206 sections, 491,866 characters.

To sanity-check after any change, print each section with its size and read the headings — an
anomalous `char_count` is how Bug 3 was caught.

---

## 9. Decisions recorded but not yet built

- **Generation LLM: Google Gemini** (free AI Studio tier), wired in Sprint 4 behind a provider
  interface so it can be swapped without touching the pipeline.
- **Evaluation LLM-as-judge: Groq** (free tier), Sprint 5 — chosen separately because evaluation
  runs many more requests than generation does, so daily request limits bind differently. The
  rate-limit reasoning gets written up in `sprint-04.md`/`sprint-05.md` once verified against
  Google's and Groq's current published limits.
- Keys load from `.env` (gitignored); `.env.example` documents the variable names.

---

## 10. Next: Sprint 2 — Chunking + embeddings + vector store

Take these 206 sections and split them into passages, embed them with a local sentence-transformer,
and index them so we can query for similar chunks. Chunking strategy is the first decision that
directly moves answer quality — chunks too large dilute the embedding, too small lose the context
that makes a passage meaningful. That is where the retrieval engineering really begins.
