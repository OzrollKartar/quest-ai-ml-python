"""
Ingest documents from an API into MongoDB (the offline half of RAG).
====================================================================
Fetch from a source, chunk, embed, and store in MongoDB so the agent can
retrieve from Mongo instead of hitting the API on every question.

    py ingest_to_mongo.py "model context protocol"
    py ingest_to_mongo.py --source hackernews "gemma 3" --limit 8

Then query it:
    py rag_agent.py --store mongo "how does MCP work?"
"""

import argparse

from mongo_store import MongoStore
from rag_agent import chunk_text
from sources import SOURCES


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch from an API and load MongoDB for RAG.")
    p.add_argument("query", nargs="+", help="what to fetch and store")
    p.add_argument("--source", default="wikipedia", choices=list(SOURCES))
    p.add_argument("--limit", type=int, default=3, help="docs to fetch")
    p.add_argument("--db", default="rag_demo")
    p.add_argument("--collection", default="chunks")
    args = p.parse_args()

    query = " ".join(args.query)
    docs = SOURCES[args.source].fetch(query, limit=args.limit)
    print(f"fetched {len(docs)} docs from {args.source} for {query!r}")

    store = MongoStore(db=args.db, collection=args.collection)
    print(f"embeddings: {'ON (nomic-embed-text)' if store.use_embeddings else 'OFF (TF-IDF at query time)'}")

    n = store.ingest(docs, chunker=chunk_text)
    print(f"stored {n} chunks in {args.db}.{args.collection}")
    print("done. Query with:  py rag_agent.py --store mongo \"<your question>\"")


if __name__ == "__main__":
    main()
