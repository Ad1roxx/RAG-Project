"""Embedding model wrapper.

An embedding turns a piece of text into a fixed-length vector positioned so
that semantically similar texts land near each other. That is the whole
mechanism behind "search by meaning instead of by keyword": we embed every
passage once, embed the question at query time, and look for the nearest
vectors. No keyword needs to match - "how do you stop a model hallucinating"
can retrieve a passage about "grounding generation in retrieved evidence".

Model: BAAI/bge-small-en-v1.5 (local, free, CPU-friendly)
-------------------------------------------------------
Chosen over sentence-transformers/all-MiniLM-L6-v2, the usual tutorial default,
for one concrete reason: MiniLM-L6-v2 accepts **256 tokens** and bge-small
accepts **512**, at the same 384 dimensions and a comparable model size.

That limit is a hard ceiling on chunk size, and exceeding it does not raise an
error - the model truncates and embeds the surviving prefix as though the rest
of the passage did not exist. With a 400-token chunking budget, MiniLM would
have silently discarded roughly the back half of every full chunk, and nothing
downstream would have looked wrong. bge-small also scores better on MTEB
retrieval benchmarks, but the context limit is the deciding argument.

Why local embeddings rather than an API
---------------------------------------
Embedding runs over the entire corpus on every rebuild, and again for every
query. A local model makes that free and offline, which suits this project's
zero-cost constraint. It also keeps the interesting part inspectable: the
vectors are ours, and we can look at them.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models are trained ASYMMETRICALLY: passages are embedded bare, while
# queries get an instruction prefix. This is not decoration - it is how the
# model was fine-tuned, and it exists because a question and the passage that
# answers it are not paraphrases of each other. "What optimiser did BERT use?"
# shares little surface form with "We train with Adam using a learning rate of
# 1e-4". The prefix tells the model to encode the text as a search intent
# rather than as a statement, which pulls it toward answering passages instead
# of toward other questions.
#
# Getting this wrong degrades retrieval quietly: everything still runs, results
# are still returned, they are just measurably worse. It is one of the most
# common mistakes made with BGE models.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Thin wrapper that keeps the passage/query asymmetry in one place."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.max_seq_length = self.model.max_seq_length

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: list[str], *, batch_size: int = 32,
                        show_progress: bool = True) -> np.ndarray:
        """Embed corpus passages. No instruction prefix - see above."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            # Normalising to unit length makes the dot product of two vectors
            # exactly their cosine similarity, which lets the index use fast
            # inner-product search and still rank by cosine. It also puts every
            # score on a comparable 0-1 scale, so a threshold means something.
            normalize_embeddings=True,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single search query, with the BGE instruction prefix."""
        return self.model.encode(
            QUERY_INSTRUCTION + query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def count_tokens(self, text: str) -> int:
        """Token length as this model sees it - the number that decides truncation."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))
