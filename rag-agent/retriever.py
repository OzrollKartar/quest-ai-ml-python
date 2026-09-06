"""
Retrieval backends for the RAG agent.
=====================================
Two ways to score chunk relevance against a query:

  TfidfRetriever      -- pure numpy, no model, no download. Lexical overlap.
                         The default: runs immediately with what you have.
  EmbeddingRetriever  -- semantic, via an Ollama embedding model
                         (e.g. `ollama pull nomic-embed-text`). Better recall,
                         understands synonyms. Used automatically if available.

Both expose the same tiny API:
    r = Retriever(chunks)
    hits = r.top(query, k=4)   # -> list[(index, score)]
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class TfidfRetriever:
    """Classic TF-IDF + cosine similarity, implemented in a few lines of numpy."""

    name = "tfidf"

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        docs_tokens = [_tokenize(c) for c in chunks]

        # vocabulary + inverse document frequency
        df: Counter = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        n = len(chunks)
        self.vocab = {w: i for i, w in enumerate(df)}
        self.idf = np.zeros(len(self.vocab))
        for w, c in df.items():
            self.idf[self.vocab[w]] = math.log((1 + n) / (1 + c)) + 1

        self.matrix = np.vstack([self._vec(toks) for toks in docs_tokens])

    def _vec(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        if not tokens:
            return v
        counts = Counter(tokens)
        for w, c in counts.items():
            j = self.vocab.get(w)
            if j is not None:
                v[j] = (c / len(tokens)) * self.idf[j]
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def top(self, query: str, k: int = 4) -> list[tuple[int, float]]:
        q = self._vec(_tokenize(query))
        scores = self.matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


class EmbeddingRetriever:
    """Semantic retrieval using an Ollama embedding model."""

    name = "embeddings"

    def __init__(self, chunks: list[str], model: str = "nomic-embed-text"):
        self.chunks = chunks
        self.model = model
        self.matrix = self._embed(chunks)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=120,
            )
            r.raise_for_status()
            e = np.array(r.json().get("embedding", []), dtype=float)
            if e.size == 0:
                raise RuntimeError(f"model '{self.model}' returned no embedding")
            n = np.linalg.norm(e)
            vecs.append(e / n if n else e)
        return np.vstack(vecs)

    def top(self, query: str, k: int = 4) -> list[tuple[int, float]]:
        q = self._embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]


def embedding_model_available(model: str = "nomic-embed-text") -> bool:
    """True if an Ollama embedding model is pulled AND actually embeds."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": "test"},
            timeout=20,
        )
        return r.ok and len(r.json().get("embedding", [])) > 0
    except requests.RequestException:
        return False


def build_retriever(chunks: list[str], embed_model: str = "nomic-embed-text"):
    """Pick the best available retriever for these chunks."""
    if embedding_model_available(embed_model):
        return EmbeddingRetriever(chunks, model=embed_model)
    return TfidfRetriever(chunks)
