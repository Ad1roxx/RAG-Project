"""Vector store: build, persist and search a FAISS index over chunk embeddings.

Usage
-----
    python -m rag.store build                       # embed chunks, write index
    python -m rag.store query "how does BERT differ from GPT?"
    python -m rag.store query "..." -k 10

Why FAISS and not Chroma
------------------------
Chroma would bundle the index, the metadata and the persistence together and
hand back documents directly. FAISS does one thing: given vectors, find the
nearest. The mapping from "index row 247" back to "chunk 1810.04805:3:1" is
ours to maintain, which is exactly why it is worth using here - that mapping is
the provenance chain a citation depends on, and it is better understood than
delegated. It is about 60 lines.

Why IndexFlatIP
---------------
"Flat" means exhaustive: every query is compared against all 392 vectors, so
results are EXACT. The alternatives (IVF, HNSW) are approximate - they trade
some recall for speed, and need tuning and a training pass.

At this corpus size that trade would be a mistake. 392 x 384 floats is about
600 KB and a search takes under a millisecond; an approximate index would save
nothing measurable while introducing a second possible cause for a bad result.
When retrieval quality is wrong, the first question is always "is it retrieval
or generation?" - and an exact index removes one branch of that search
permanently.

The threshold worth knowing: exhaustive search stays comfortable into the
low hundreds of thousands of vectors. Past that, HNSW is the usual next step.

"IP" is inner product. Because `embed.py` normalises every vector to unit
length, the inner product of two vectors IS their cosine similarity, so this
index ranks by cosine while using the faster primitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from ingestion.paths import CHUNKS_FILE, INDEX_DIR, ensure_dirs
from rag.embed import DEFAULT_MODEL, Embedder

INDEX_FILE = "index.faiss"
CHUNKS_SIDECAR = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class SearchResult:
    score: float
    chunk: dict

    def citation(self) -> str:
        pages = (
            f"p.{self.chunk['start_page']}"
            if self.chunk["start_page"] == self.chunk["end_page"]
            else f"pp.{self.chunk['start_page']}-{self.chunk['end_page']}"
        )
        return f"{self.chunk['arxiv_id']} | {self.chunk['heading']} | {pages}"


class VectorStore:
    """A FAISS index plus the chunk metadata its rows correspond to.

    The invariant that makes this work: `index` row i and `chunks[i]` describe
    the same passage. Everything else here exists to keep that true across a
    save/load cycle.
    """

    def __init__(self, index: faiss.Index, chunks: list[dict], manifest: dict) -> None:
        self.index = index
        self.chunks = chunks
        self.manifest = manifest

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict], embedder: Embedder) -> "VectorStore":
        texts = [chunk["embed_text"] for chunk in chunks]
        vectors = embedder.encode_passages(texts)

        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(vectors)

        manifest = {
            "embedding_model": embedder.model_name,
            "dimension": embedder.dimension,
            "max_seq_length": embedder.max_seq_length,
            "n_chunks": len(chunks),
            "normalized": True,
            "index_type": "IndexFlatIP",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        return cls(index, chunks, manifest)

    # -- persistence -------------------------------------------------------
    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with (directory / CHUNKS_SIDECAR).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        (directory / MANIFEST_FILE).write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "VectorStore":
        index_path = directory / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index at {index_path}. Build one with: python -m rag.store build"
            )
        index = faiss.read_index(str(index_path))
        chunks = [
            json.loads(line)
            for line in (directory / CHUNKS_SIDECAR).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))

        # The invariant, checked rather than assumed: a mismatch here would not
        # crash, it would return confidently wrong citations - row i pointing at
        # some other passage's metadata.
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs {len(chunks)} chunks. "
                "Rebuild the index."
            )
        return cls(index, chunks, manifest)

    # -- search ------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchResult]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results: list[SearchResult] = []
        for score, row in zip(scores[0], indices[0]):
            # FAISS returns -1 when fewer than k vectors exist.
            if row == -1:
                continue
            results.append(SearchResult(score=float(score), chunk=self.chunks[row]))
        return results

    def assert_matches(self, embedder: Embedder) -> None:
        """Refuse to search an index built by a different embedding model.

        Two models place text in two unrelated vector spaces. Querying one
        model's index with another model's vector does not fail - it returns
        k results, ranked by a similarity that means nothing. This is the
        single easiest way to get a RAG system that is quietly broken, so it
        gets an explicit guard rather than a comment.
        """
        if self.manifest["embedding_model"] != embedder.model_name:
            raise ValueError(
                f"Index was built with {self.manifest['embedding_model']!r} but the query "
                f"uses {embedder.model_name!r}. Rebuild the index or switch models."
            )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"No chunks at {CHUNKS_FILE}. Run: python -m ingestion.chunk"
        )
    return [
        json.loads(line)
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def command_build(args: argparse.Namespace) -> None:
    ensure_dirs()
    chunks = load_chunks()
    embedder = Embedder(args.model)

    print(f"Embedding {len(chunks):,} chunks with {embedder.model_name} "
          f"(dim={embedder.dimension}, max_seq={embedder.max_seq_length})")
    store = VectorStore.build(chunks, embedder)
    store.save()

    print(f"\nIndexed {store.index.ntotal:,} vectors -> {INDEX_DIR}")
    print(json.dumps(store.manifest, indent=2))


def command_query(args: argparse.Namespace) -> None:
    store = VectorStore.load()
    embedder = Embedder(store.manifest["embedding_model"])
    store.assert_matches(embedder)

    results = store.search(embedder.encode_query(args.question), k=args.k)

    print(f'\nQ: "{args.question}"\n')
    for rank, result in enumerate(results, start=1):
        snippet = " ".join(result.chunk["text"].split())[:240]
        print(f"{rank}. [{result.score:.3f}] {result.chunk['paper_title'][:58]}")
        print(f"   {result.citation()}")
        print(f"   {snippet}...\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and query the chunk vector index.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="embed all chunks and write the FAISS index")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.set_defaults(func=command_build)

    query = sub.add_parser("query", help="search the index")
    query.add_argument("question")
    query.add_argument("-k", type=int, default=5, help="how many chunks to return")
    query.set_defaults(func=command_query)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
