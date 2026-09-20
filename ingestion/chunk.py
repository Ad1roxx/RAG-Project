"""Split section-structured papers into retrievable passages.

Usage
-----
    python -m ingestion.chunk
    python -m ingestion.chunk --chunk-tokens 320 --overlap-tokens 48
    python -m ingestion.chunk --stats-only          # report, write nothing

Reads : data/processed/<id>.json
Writes: data/chunks/chunks.jsonl   (one JSON object per line)

Chunking is the first decision in this pipeline that directly moves answer
quality, and it is a genuine trade-off rather than a setting to copy:

  too large   the embedding averages several ideas into one vector, so it is
              strongly similar to nothing in particular, and each retrieved
              chunk burns prompt space on text the question did not ask about.
  too small   the passage loses the context that made it meaningful. "It
              improves by 2.1 points" is unretrievable and unusable alone.

Three commitments make the trade-off defensible here:

1. Chunks never cross a section boundary. A passage that is half Method and
   half Results describes no single thing, so its embedding points somewhere
   between two topics - near-miss retrieval for both.
2. Chunks are measured in the EMBEDDING MODEL'S OWN TOKENS, not characters or
   words, because that model is what silently truncates oversized input.
3. Chunks carry their paper title and section heading into the embedded text,
   so a passage stays interpretable outside its document.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Iterator

from transformers import AutoTokenizer

from ingestion.paths import CHUNKS_FILE, PROCESSED_DIR, ensure_dirs

# The tokenizer must be the embedding model's own. Sizing chunks with a
# different tokenizer means the budget is a guess, and going over the model's
# limit does not raise - it silently truncates, so the tail of every oversized
# chunk would be indexed as if it did not exist.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# bge-small-en-v1.5 accepts 512 tokens. The budget below leaves headroom for
# the contextual header (title + heading, ~30-60 tokens) that gets prepended
# before embedding, so a full chunk plus its header still fits under the limit.
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60

# A passage this short is a heading fragment or a stray caption, not an answer.
MIN_CHUNK_TOKENS = 32

_tokenizer = None


def tokenizer():
    """Load the tokenizer once. Cheap, but not free, and this runs per chunk."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def token_count(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------
# Chunk boundaries land on sentence boundaries, so no chunk starts or ends
# mid-thought. That means splitting sentences correctly, and academic prose is
# full of periods that do not end sentences.
ABBREVIATIONS = {
    "et al.", "e.g.", "i.e.", "cf.", "etc.", "vs.", "resp.", "approx.",
    "fig.", "eq.", "sec.", "tab.", "no.", "pp.", "al.", "dr.", "prof.",
    "inc.", "ltd.", "st.", "ca.", "ref.", "refs.", "eqs.", "figs.",
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# A single capital followed by a period is an initial ("J. Devlin"), not an end.
INITIAL = re.compile(r"\b[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, re-joining false splits."""
    pieces = SENTENCE_BOUNDARY.split(text)
    sentences: list[str] = []

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if sentences:
            previous = sentences[-1]
            tail = previous.split()[-1].lower() if previous.split() else ""
            # Re-attach when the previous "sentence" ended on an abbreviation
            # or an initial - it was never a real boundary.
            if tail in ABBREVIATIONS or INITIAL.search(previous):
                sentences[-1] = f"{previous} {piece}"
                continue
        sentences.append(piece)

    return sentences


def split_oversized_sentence(sentence: str, limit: int) -> list[str]:
    """Hard-split a single sentence that exceeds the whole chunk budget.

    Rare, but real: a sentence listing 40 benchmark names, or prose where the
    splitter found no boundary. Without this the chunk would overflow the
    model's limit and be truncated invisibly.
    """
    words = sentence.split()
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if token_count(" ".join(current)) >= limit:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts


# --------------------------------------------------------------------------
# Chunk assembly
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    paper_title: str
    heading: str
    section_index: int
    chunk_index: int
    start_page: int
    end_page: int
    text: str
    embed_text: str
    token_count: int


def contextual_text(paper_title: str, heading: str, text: str) -> str:
    """What actually gets embedded: the passage plus where it came from.

    A chunk reading "We use 12 layers, 768 hidden units and 12 attention heads"
    is nearly content-free on its own - it could be any transformer paper ever
    written, and its embedding sits in a crowded, undiscriminating region.
    Prefixing "BERT ... | 3.1 Pre-training BERT" anchors it to a specific paper
    and a specific part of that paper.

    The cost is real and worth stating: the header is repeated in every chunk of
    a section, which nudges all of them toward each other and slightly dilutes
    the passage's own signal. The trade is worth it for orphan passages like the
    one above, and this is the kind of thing Sprint 5's eval harness exists to
    settle with numbers rather than intuition.
    """
    return f"{paper_title} | {heading}\n\n{text}"


def pack_sentences(sentences: list[str], chunk_tokens: int, overlap_tokens: int) -> Iterator[str]:
    """Greedily fill chunks with whole sentences, overlapping at the seams.

    Overlap exists because the sentence that answers a question and the sentence
    that names its subject are often neighbours. Split between them with no
    overlap and neither chunk is retrievable: one has the answer without the
    subject, the other the subject without the answer. Overlap duplicates the
    boundary sentences so the pair survives in at least one chunk.
    """
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_count(sentence)

        if sentence_tokens > chunk_tokens:
            if current:
                yield " ".join(current)
                current, current_tokens = [], 0
            for part in split_oversized_sentence(sentence, chunk_tokens):
                yield part
            continue

        if current_tokens + sentence_tokens > chunk_tokens and current:
            yield " ".join(current)
            # Carry the tail of the finished chunk into the next one.
            carried: list[str] = []
            carried_tokens = 0
            for previous in reversed(current):
                previous_tokens = token_count(previous)
                if carried_tokens + previous_tokens > overlap_tokens:
                    break
                carried.insert(0, previous)
                carried_tokens += previous_tokens
            current, current_tokens = carried, carried_tokens

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        yield " ".join(current)


def chunk_paper(paper: dict, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []

    for section in paper["sections"]:
        sentences = split_sentences(section["text"])
        if not sentences:
            continue

        for chunk_index, text in enumerate(pack_sentences(sentences, chunk_tokens, overlap_tokens)):
            count = token_count(text)
            if count < MIN_CHUNK_TOKENS:
                # Tiny trailing remainder: fold it into the previous chunk of
                # the same section rather than indexing a fragment. Dropping it
                # outright would lose real text.
                if chunks and chunks[-1].section_index == section["index"]:
                    merged = f"{chunks[-1].text} {text}"
                    chunks[-1].text = merged
                    chunks[-1].embed_text = contextual_text(
                        paper["title"], section["heading"], merged
                    )
                    chunks[-1].token_count = token_count(merged)
                    continue
                # Nothing to merge into: this is a whole section that is one
                # short sentence, which in practice means a preamble sitting
                # above its subsections ("In this section, we present BERT
                # fine-tuning results on 11 NLP tasks."). It announces content
                # rather than containing any, so indexing it only adds a weak
                # competitor for the queries its own subsections should win.
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{paper['arxiv_id']}:{section['index']}:{chunk_index}",
                    arxiv_id=paper["arxiv_id"],
                    paper_title=paper["title"],
                    heading=section["heading"],
                    section_index=section["index"],
                    chunk_index=chunk_index,
                    start_page=section["start_page"],
                    end_page=section["end_page"],
                    text=text,
                    embed_text=contextual_text(paper["title"], section["heading"], text),
                    token_count=count,
                )
            )

    return chunks


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def load_papers() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PROCESSED_DIR.glob("*.json"))
    ]


def report(chunks: list[Chunk], chunk_tokens: int) -> None:
    if not chunks:
        return
    counts = sorted(chunk.token_count for chunk in chunks)
    over_limit = sum(1 for chunk in chunks if token_count(chunk.embed_text) > 512)

    def percentile(fraction: float) -> int:
        return counts[min(int(len(counts) * fraction), len(counts) - 1)]

    papers = {chunk.arxiv_id for chunk in chunks}
    print(f"\n{len(chunks):,} chunks from {len(papers)} papers")
    print(f"tokens  min={counts[0]}  p25={percentile(.25)}  median={percentile(.5)}  "
          f"p75={percentile(.75)}  max={counts[-1]}  (budget {chunk_tokens})")
    print(f"mean tokens/chunk: {sum(counts) / len(counts):.0f}")
    # The number that matters: anything here is silently truncated at embed time.
    print(f"chunks whose embed_text exceeds the model's 512-token limit: {over_limit}")


def run(chunk_tokens: int, overlap_tokens: int, *, write: bool = True) -> list[Chunk]:
    ensure_dirs()
    papers = load_papers()
    if not papers:
        print("No processed papers. Run: python -m ingestion.extract")
        return []

    all_chunks: list[Chunk] = []
    for paper in papers:
        chunks = chunk_paper(paper, chunk_tokens, overlap_tokens)
        all_chunks.extend(chunks)
        print(f"{paper['arxiv_id']}: {len(paper['sections']):3d} sections -> {len(chunks):4d} chunks")

    if write:
        with CHUNKS_FILE.open("w", encoding="utf-8") as handle:
            for chunk in all_chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        print(f"\nWrote {CHUNKS_FILE}")

    report(all_chunks, chunk_tokens)
    return all_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split processed papers into retrievable chunks.")
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--stats-only", action="store_true",
                        help="report chunk statistics without writing the output file")
    args = parser.parse_args(argv)

    if args.overlap_tokens >= args.chunk_tokens:
        parser.error("--overlap-tokens must be smaller than --chunk-tokens")

    run(args.chunk_tokens, args.overlap_tokens, write=not args.stats_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
