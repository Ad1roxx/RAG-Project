"""Fetch ML/AI papers from arXiv: metadata via the arXiv API, plus the PDF.

Usage
-----
    python -m ingestion.fetch_papers                          # the seed corpus
    python -m ingestion.fetch_papers --ids 1706.03762 2005.11401
    python -m ingestion.fetch_papers --category cs.CL --max-results 15

Writes:
    data/raw/metadata/<arxiv_id>.json
    data/raw/pdfs/<arxiv_id>.pdf

Why the raw arXiv API instead of a wrapper library: the API returns an Atom
feed, and parsing it ourselves keeps the data contract visible. There is no
hidden mapping between "what arXiv said" and "what we stored" - which matters
later, because every citation the RAG system emits traces back to this metadata.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from typing import Iterable
from xml.etree import ElementTree

import requests

from ingestion.paths import METADATA_DIR, PDF_DIR, ensure_dirs

ARXIV_API = "http://export.arxiv.org/api/query"

# arXiv's API terms ask for one request at a time and ~3s between them. We are a
# guest on a free service; being impolite is how scripts get blocked.
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

# arXiv asks for a descriptive User-Agent so they can contact abusers rather
# than silently banning them.
USER_AGENT = "paper-rag/0.1 (research prototype; contact via repository)"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# A deliberately *interlinked* seed corpus. These papers cite and build on each
# other (transformer -> BERT/RoBERTa/DistilBERT -> GPT-3 -> CoT; DPR ->
# Sentence-BERT -> RAG). That matters: the system's selling point is retrieval
# ACROSS papers, so the demo corpus must contain questions whose answer lives in
# more than one paper. A pile of unrelated papers could not demonstrate that.
SEED_CORPUS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "1907.11692",  # RoBERTa
    "1910.01108",  # DistilBERT
    "2005.14165",  # GPT-3: Language Models are Few-Shot Learners
    "1908.10084",  # Sentence-BERT
    "2004.04906",  # Dense Passage Retrieval
    "2005.11401",  # RAG (Lewis et al.)
    "2106.09685",  # LoRA
    "2201.11903",  # Chain-of-Thought prompting
]

# arXiv ids look like 1706.03762, 1706.03762v7, or the old style cs/0112017.
ID_WITH_VERSION = re.compile(r"^(?P<base>.+?)(?P<version>v\d+)?$")


def collapse_whitespace(text: str) -> str:
    """arXiv hard-wraps titles and abstracts; we want single-line values."""
    return re.sub(r"\s+", " ", text or "").strip()


def split_version(raw_id: str) -> tuple[str, str | None]:
    """'1706.03762v7' -> ('1706.03762', 'v7'). Version-less ids pass through."""
    match = ID_WITH_VERSION.match(raw_id.strip())
    return match.group("base"), match.group("version")


def safe_filename(arxiv_id: str) -> str:
    """Old-style ids contain '/', which is not a legal filename character."""
    return arxiv_id.replace("/", "_")


def _get(url: str, *, params: dict | None = None, stream: bool = False) -> requests.Response:
    """One HTTP GET with bounded retries and linear backoff.

    Calls against a free public API fail intermittently. Retrying a handful of
    times is the difference between a reproducible corpus and a script that
    half-works and leaves you guessing which papers are missing.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                stream=stream,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            backoff = REQUEST_DELAY_SECONDS * attempt
            print(f"    request failed ({error}); retry {attempt}/{MAX_RETRIES} in {backoff:.0f}s")
            time.sleep(backoff)
    raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts") from last_error


def query_arxiv(
    *,
    ids: list[str] | None = None,
    search_query: str | None = None,
    max_results: int = 10,
) -> str:
    """Call the arXiv API once and return the raw Atom XML.

    Exactly one of `ids` / `search_query` is used: `id_list` fetches specific
    papers, `search_query` does a field search (e.g. 'cat:cs.CL').
    """
    params: dict[str, str | int] = {"max_results": max_results, "start": 0}
    if ids:
        params["id_list"] = ",".join(ids)
    elif search_query:
        params["search_query"] = search_query
        # Newest first, so re-running with a larger max_results is predictable.
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"
    else:
        raise ValueError("query_arxiv needs either ids or search_query")

    return _get(ARXIV_API, params=params).text


def parse_entries(atom_xml: str) -> list[dict]:
    """Turn the Atom feed into plain metadata dicts, one per paper."""
    root = ElementTree.fromstring(atom_xml)
    entries: list[dict] = []

    for entry in root.findall("atom:entry", NS):
        abs_url = entry.findtext("atom:id", default="", namespaces=NS).strip()
        # abs_url looks like http://arxiv.org/abs/1706.03762v7
        raw_id = abs_url.rsplit("/abs/", 1)[-1]
        arxiv_id, version = split_version(raw_id)

        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{raw_id}"

        primary = entry.find("arxiv:primary_category", NS)

        entries.append(
            {
                "arxiv_id": arxiv_id,
                "version": version,
                "title": collapse_whitespace(entry.findtext("atom:title", "", NS)),
                "authors": [
                    collapse_whitespace(name.text or "")
                    for name in entry.findall("atom:author/atom:name", NS)
                ],
                "abstract": collapse_whitespace(entry.findtext("atom:summary", "", NS)),
                "published": entry.findtext("atom:published", "", NS).strip(),
                "updated": entry.findtext("atom:updated", "", NS).strip(),
                "primary_category": primary.get("term") if primary is not None else None,
                "categories": [
                    c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")
                ],
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return entries


def download_pdf(entry: dict, *, force: bool = False) -> bool:
    """Download one PDF. Returns True only if a network download happened.

    Skipping files already on disk makes the script idempotent: you can re-run
    it to top up the corpus without re-fetching everything.
    """
    destination = PDF_DIR / f"{safe_filename(entry['arxiv_id'])}.pdf"
    if destination.exists() and not force:
        print(f"    pdf already on disk, skipping ({destination.name})")
        return False

    response = _get(entry["pdf_url"], stream=True)

    # Write to a temp file, then move into place. A half-written .pdf left by a
    # crash would otherwise be silently treated as "already downloaded".
    temp = destination.with_suffix(".pdf.part")
    with temp.open("wb") as handle:
        for block in response.iter_content(chunk_size=64 * 1024):
            handle.write(block)
    temp.replace(destination)

    print(f"    downloaded {destination.name} ({destination.stat().st_size / 1024:.0f} KB)")
    return True


def save_metadata(entry: dict) -> None:
    path = METADATA_DIR / f"{safe_filename(entry['arxiv_id'])}.json"
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch(
    ids: Iterable[str] | None = None,
    *,
    search_query: str | None = None,
    max_results: int = 10,
    force: bool = False,
) -> list[dict]:
    ensure_dirs()

    id_list = [split_version(i)[0] for i in ids] if ids else None
    entries = parse_entries(
        query_arxiv(ids=id_list, search_query=search_query, max_results=max_results)
    )

    if not entries:
        print("arXiv returned no entries - check the ids or the query string.")
        return []

    print(f"arXiv returned {len(entries)} entrie(s).\n")
    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] {entry['arxiv_id']}  {entry['title'][:70]}")
        save_metadata(entry)
        downloaded = download_pdf(entry, force=force)
        # Only sleep when we actually hit the network.
        if downloaded and index < len(entries):
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Metadata -> {METADATA_DIR}\n      PDFs     -> {PDF_DIR}")
    return entries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv papers (metadata + PDF) into data/raw/."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ids", nargs="+", metavar="ARXIV_ID",
        help="specific arXiv ids, e.g. 1706.03762 2005.11401",
    )
    group.add_argument(
        "--category", metavar="CAT",
        help="fetch the newest papers in an arXiv category, e.g. cs.CL",
    )
    parser.add_argument(
        "--max-results", type=int, default=10,
        help="cap on results for --category (default: 10)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download PDFs that are already on disk",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.category:
        fetch(search_query=f"cat:{args.category}", max_results=args.max_results, force=args.force)
    else:
        ids = args.ids or SEED_CORPUS
        fetch(ids=ids, max_results=len(ids), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
