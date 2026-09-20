"""PDF -> clean, section-structured text.

Usage
-----
    python -m ingestion.extract                  # every paper in data/raw
    python -m ingestion.extract --ids 1810.04805
    python -m ingestion.extract --force          # re-extract existing output

Reads : data/raw/pdfs/<id>.pdf + data/raw/metadata/<id>.json
Writes: data/processed/<id>.json

The job here is narrow but load-bearing: everything downstream (chunking,
embedding, retrieval, the citations the user finally sees) inherits whatever
text this module produces. Garbage that survives this stage is invisible later -
it just quietly makes answers worse. So each cleanup step below exists because
of a specific defect observed in real arXiv PDFs, and is documented as such.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

from ingestion.paths import METADATA_DIR, PDF_DIR, PROCESSED_DIR, ensure_dirs

# --------------------------------------------------------------------------
# Block filtering
# --------------------------------------------------------------------------
# PyMuPDF returns a page as "blocks" - roughly, visually grouped runs of text.
# Body paragraphs come back as one block of 80-150 words whose lines each hold
# ~8-12 words. The internals of a figure (axis labels, node captions, "E[CLS]",
# "Tok 1") come back as blocks with a similar word count but ~1-2 words PER
# LINE, because every label is its own line. Words-per-line separates the two
# far more reliably than raw length does.
MIN_WORDS_PER_LINE = 3.0
# Above this word count a block is prose regardless of its line shape.
ALWAYS_KEEP_WORD_COUNT = 25

# The vertical stamp arXiv prints down the margin of page 1, e.g.
# "arXiv:1810.04805v2  [cs.CL]  24 May 2019". The trailing date has to be part
# of the pattern: stripping only up to "[cs.CL]" leaves "24 May 2019" behind,
# which then looks exactly like a numbered heading and swallows the section
# that follows it.
ARXIV_STAMP = re.compile(
    r"arXiv:\s*\d{4}\.\d{4,5}v?\d*\s*\[[^\]]+\]\s*(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})?",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Heading detection
# --------------------------------------------------------------------------
# Two patterns cover the overwhelming majority of arXiv/ACL formatting:
# numbered headings ("3.1 Model Architecture") and a known set of unnumbered
# ones ("Abstract", "References"). Font size would be more robust and is the
# natural upgrade, but it is also more brittle across template families - this
# is the simple version that we can verify by eye on the real corpus.
NUMBERED_HEADING = re.compile(r"^(?P<number>\d+(?:\.\d+){0,3})\.?\s+(?P<title>[A-Z].{1,80})$")
# A heading title ending in a bare number is a table-of-contents line, not a
# heading: "2 Approach 6" is the ToC entry pointing at page 6. Long papers
# (GPT-3) have a ToC, and treating its entries as headings creates a phantom
# duplicate of every real section.
TOC_PAGE_NUMBER = re.compile(r"\s\d{1,3}$")
UNNUMBERED_HEADINGS = {
    "abstract", "introduction", "background", "related work", "prior work",
    "method", "methods", "methodology", "approach", "model", "model architecture",
    "experiments", "experimental setup", "experiments and results", "results",
    "evaluation", "analysis", "ablation study", "discussion", "limitations",
    "conclusion", "conclusions", "conclusion and future work", "future work",
    "broader impact", "ethics statement", "acknowledgments", "acknowledgements",
    "references", "bibliography", "appendix",
}
MAX_HEADING_WORDS = 10

# Everything from here on is a citation list, not prose. Retrieving "Devlin et
# al., 2019. BERT..." never answers a question, but it embeds close to every
# query that mentions BERT - so it is pure retrieval poison. We cut it.
STOP_SECTIONS = {"references", "bibliography"}

WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
HYPHEN_LINEBREAK = re.compile(r"([A-Za-z]+)-\n([A-Za-z]+)")


@dataclass
class Section:
    heading: str
    start_page: int
    end_page: int
    lines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
def normalise_unicode(text: str) -> str:
    """Fold typographic characters down to plain ASCII-ish equivalents.

    NFKC does the heavy lifting: it expands the ligatures that litter LaTeX
    PDFs, so 'ﬁne-tuned' becomes 'fine-tuned'. That matters because a query for
    "fine-tuning" and a chunk containing the ligature form are, to a tokenizer,
    different strings. Quotes and dashes NFKC leaves alone, so we map those.
    """
    text = unicodedata.normalize("NFKC", text)
    for source, target in {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-",
        " ": " ", "​": "",
    }.items():
        text = text.replace(source, target)
    return text


def build_hyphen_vocabularies(text: str) -> tuple[set[str], set[str]]:
    """Collect which word forms this document actually uses.

    Used to resolve the one genuinely ambiguous case in PDF text extraction:
    a hyphen at a line break is either a soft hyphen inserted by LaTeX
    ('representa-\\ntion' -> 'representation') or a real hyphen in a compound
    that happened to break there ('task-\\nspecific' -> 'task-specific'). The
    characters on the page are identical, so no local rule can tell them apart.

    The document itself can: papers repeat their own terminology. If
    'representation' appears elsewhere unbroken, the hyphen was soft; if
    'task-specific' appears elsewhere intact, it was real.
    """
    plain: set[str] = set()
    hyphenated: set[str] = set()
    for line in text.split("\n"):
        # Drop a trailing hyphenated fragment - it is the evidence we are
        # trying to interpret, not evidence about the vocabulary.
        line = re.sub(r"[A-Za-z]+-$", "", line.strip())
        for match in WORD.finditer(line):
            word = match.group(0).lower()
            (hyphenated if "-" in word else plain).add(word)
    return plain, hyphenated


def dehyphenate(text: str, plain: set[str], hyphenated: set[str]) -> str:
    def resolve(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if (left + right).lower() in plain:
            return left + right
        if f"{left}-{right}".lower() in hyphenated:
            return f"{left}-{right}"
        # No evidence either way: soft hyphenation is far more common in
        # justified two-column text, so joining is the better default.
        return left + right

    return HYPHEN_LINEBREAK.sub(resolve, text)


def flatten(text: str) -> str:
    """Line breaks inside a paragraph carry no meaning once hyphens are fixed."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Structure detection
# --------------------------------------------------------------------------
def heading_of(block_text: str) -> str | None:
    """Return the normalised heading this block is, or None if it is body text.

    PyMuPDF often splits a heading across lines ('1\\nIntroduction'), so we
    compare on the flattened form.
    """
    # Headings are normalised here rather than with the body, because they are
    # user-visible: every citation the system emits names one.
    candidate = flatten(normalise_unicode(block_text))
    if not candidate or len(candidate.split()) > MAX_HEADING_WORDS:
        return None

    stripped = candidate.rstrip(".").strip()
    if stripped.lower() in UNNUMBERED_HEADINGS:
        return stripped

    match = NUMBERED_HEADING.match(stripped)
    if match:
        title = match.group("title").strip()
        # Titles must be allowed to contain periods ("4.2 SQuAD v1.1"), so the
        # guards against false positives are: no sentence-ending period, and no
        # trailing page number.
        if title.endswith(".") or TOC_PAGE_NUMBER.search(title):
            return None
        return f"{match.group('number')} {title}"
    return None


def is_content_block(block_text: str) -> bool:
    """Keep prose; drop figure internals, page numbers and stray glyphs."""
    lines = [line for line in block_text.split("\n") if line.strip()]
    if not lines:
        return False
    words = block_text.split()
    if len(words) >= ALWAYS_KEEP_WORD_COUNT:
        return True
    return len(words) / len(lines) >= MIN_WORDS_PER_LINE


def section_key(heading: str) -> str:
    """'3.1 Model Architecture' -> 'model architecture'; used for stop checks."""
    without_number = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading)
    return without_number.strip().rstrip(".").lower()


# --------------------------------------------------------------------------
# Main extraction
# --------------------------------------------------------------------------
def collect_sections(document: fitz.Document) -> tuple[list[Section], str]:
    """Walk the PDF page by page, splitting blocks into sections.

    Also returns the concatenated raw text, which the de-hyphenation step needs
    in order to learn the document's own vocabulary before any cleaning.
    """
    sections: list[Section] = []
    current: Section | None = None
    raw_parts: list[str] = []
    stop = False

    for page_number, page in enumerate(document, start=1):
        if stop:
            break
        for block in page.get_text("blocks"):
            block_text = ARXIV_STAMP.sub(" ", block[4]).strip()
            if not block_text:
                continue

            heading = heading_of(block_text)
            if heading is not None:
                if section_key(heading) in STOP_SECTIONS:
                    stop = True
                    break
                current = Section(heading=heading, start_page=page_number, end_page=page_number)
                sections.append(current)
                continue

            if not is_content_block(block_text):
                continue
            if current is None:
                # Title block, author list, affiliations - already captured in
                # the arXiv metadata, so there is nothing to gain by keeping a
                # garbled PDF copy of it.
                continue

            current.lines.append(block_text)
            current.end_page = page_number
            raw_parts.append(block_text)

    return sections, "\n".join(raw_parts)


def extract_paper(arxiv_id: str, metadata: dict) -> dict:
    pdf_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
    document = fitz.open(pdf_path)

    sections, raw_text = collect_sections(document)
    plain_vocab, hyphen_vocab = build_hyphen_vocabularies(normalise_unicode(raw_text))

    cleaned: list[dict] = []
    for index, section in enumerate(sections):
        body = normalise_unicode("\n".join(section.lines))
        body = dehyphenate(body, plain_vocab, hyphen_vocab)
        body = flatten(body)
        if not body:
            # Headings with no surviving body (often a figure-only subsection).
            continue
        cleaned.append(
            {
                "index": index,
                "heading": section.heading,
                "start_page": section.start_page,
                "end_page": section.end_page,
                "text": body,
                "char_count": len(body),
            }
        )

    if not cleaned:
        # Fallback for a paper whose headings we failed to detect: better one
        # undifferentiated section than an empty document.
        body = flatten(dehyphenate(normalise_unicode(raw_text), plain_vocab, hyphen_vocab))
        if body:
            cleaned = [{
                "index": 0, "heading": "Body", "start_page": 1,
                "end_page": document.page_count, "text": body, "char_count": len(body),
            }]

    return {
        "arxiv_id": metadata["arxiv_id"],
        "title": metadata["title"],
        "authors": metadata["authors"],
        "abstract": metadata["abstract"],
        "published": metadata["published"],
        "primary_category": metadata["primary_category"],
        "categories": metadata["categories"],
        "abs_url": metadata["abs_url"],
        "pdf_url": metadata["pdf_url"],
        "n_pages": document.page_count,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "sections": cleaned,
        "char_count": sum(section["char_count"] for section in cleaned),
    }


def available_ids() -> list[str]:
    return sorted(path.stem for path in METADATA_DIR.glob("*.json"))


def run(ids: list[str] | None = None, *, force: bool = False) -> list[dict]:
    ensure_dirs()
    targets = ids or available_ids()
    if not targets:
        print("No metadata found. Run: python -m ingestion.fetch_papers")
        return []

    results: list[dict] = []
    for arxiv_id in targets:
        safe = arxiv_id.replace("/", "_")
        output_path = PROCESSED_DIR / f"{safe}.json"
        if output_path.exists() and not force:
            print(f"{arxiv_id}: already extracted, skipping")
            continue

        metadata_path = METADATA_DIR / f"{safe}.json"
        pdf_path = PDF_DIR / f"{safe}.pdf"
        if not metadata_path.exists() or not pdf_path.exists():
            print(f"{arxiv_id}: missing PDF or metadata, skipping")
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paper = extract_paper(arxiv_id, metadata)
        output_path.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")

        headings = ", ".join(s["heading"] for s in paper["sections"][:4])
        print(
            f"{arxiv_id}: {len(paper['sections']):2d} sections, "
            f"{paper['char_count']:6,d} chars  [{headings}...]"
        )
        results.append(paper)

    print(f"\nDone. Processed papers -> {PROCESSED_DIR}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract clean text from fetched arXiv PDFs.")
    parser.add_argument("--ids", nargs="+", metavar="ARXIV_ID", help="limit to these papers")
    parser.add_argument("--force", action="store_true", help="re-extract papers already processed")
    args = parser.parse_args(argv)
    run(args.ids, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
