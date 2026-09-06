"""
MongoDB as the RAG knowledge base.
==================================
Instead of fetching from an API on every question, we do it once, offline:
fetch -> chunk -> embed -> STORE in MongoDB. Then queries retrieve straight
from Mongo. That's how real RAG systems are built: a slow ingestion pass, then
fast lookups.

Each stored document (one per chunk) looks like:
    { "text": "...", "title": "...", "url": "...", "embedding": [floats] }

Retrieval picks the best backend automatically:
  * If an Ollama embed model is available AND you've created an Atlas Vector
    Search index -> server-side $vectorSearch (scales to millions of chunks).
  * Else if embeddings exist -> cosine similarity computed in Python.
  * Else (no embed model) -> TF-IDF over the chunk text pulled from Mongo.

Local mongod works for everything except $vectorSearch (that needs Atlas).
"""

from __future__ import annotations

import numpy as np
import requests
from pymongo import MongoClient

from retriever import TfidfRetriever, embedding_model_available
from sources import Document

OLLAMA_URL = "http://localhost:11434"


def _embed(text: str, model: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("embedding", [])


class MongoStore:
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db: str = "rag_demo",
        collection: str = "chunks",
        embed_model: str = "nomic-embed-text",
        vector_index: str = "vector_index",
    ):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.col = self.client[db][collection]
        self.embed_model = embed_model
        self.vector_index = vector_index
        self.use_embeddings = embedding_model_available(embed_model)

    # --------------------------------------------------------------- ingestion
    def ingest(self, docs: list[Document], chunker) -> int:
        """Chunk + (optionally) embed docs and REPLACE the collection contents."""
        self.col.delete_many({})
        rows = []
        for d in docs:
            for chunk in chunker(d.text):
                row = {"text": chunk, "title": d.title, "url": d.url}
                if self.use_embeddings:
                    row["embedding"] = _embed(chunk, self.embed_model)
                rows.append(row)
        if rows:
            self.col.insert_many(rows)
        return len(rows)

    def count(self) -> int:
        return self.col.count_documents({})

    # -------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        if self.use_embeddings:
            hits = self._vector_search(query, k)
            if hits is None:  # no Atlas index -> cosine in Python
                hits = self._cosine_search(query, k)
        else:
            hits = self._tfidf_search(query, k)
        return hits

    def _vector_search(self, query: str, k: int) -> list[Document] | None:
        """Atlas $vectorSearch. Returns None if the index/feature isn't there."""
        qvec = _embed(query, self.embed_model)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": qvec,
                    "numCandidates": 100,
                    "limit": k,
                }
            },
            {"$project": {"text": 1, "title": 1, "url": 1, "_id": 0}},
        ]
        try:
            docs = list(self.col.aggregate(pipeline))
        except Exception:
            return None  # local mongod / no vector index -> caller falls back
        return [Document(d["title"], d["text"], d["url"]) for d in docs]

    def _cosine_search(self, query: str, k: int) -> list[Document]:
        qvec = np.array(_embed(query, self.embed_model), dtype=float)
        qn = np.linalg.norm(qvec) or 1.0
        qvec = qvec / qn
        rows = list(self.col.find({}, {"text": 1, "title": 1, "url": 1, "embedding": 1}))
        scored = []
        for r in rows:
            v = np.array(r.get("embedding", []), dtype=float)
            if v.size == 0:
                continue
            vn = np.linalg.norm(v) or 1.0
            scored.append((float(qvec @ (v / vn)), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Document(r["title"], r["text"], r["url"]) for _, r in scored[:k]]

    def _tfidf_search(self, query: str, k: int) -> list[Document]:
        rows = list(self.col.find({}, {"text": 1, "title": 1, "url": 1}))
        chunks = [r["text"] for r in rows]
        if not chunks:
            return []
        ret = TfidfRetriever(chunks)
        return [Document(rows[i]["title"], rows[i]["text"], rows[i]["url"])
                for i, _ in ret.top(query, k)]
