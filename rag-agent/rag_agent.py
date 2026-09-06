"""
RAG Agent  -- APIs as the knowledge base, Gemma 3 (via Ollama) as the brain.
============================================================================
Flow (a small agentic RAG loop):

    question
      -> [LLM] rewrite into good API search terms      (agent step)
      -> [API] fetch live documents (Wikipedia / HN)   (retrieval source)
      -> chunk + index
      -> retrieve top-k chunks for the question        (TF-IDF or embeddings)
      -> [LLM] answer grounded ONLY in those chunks, cite sources
      -> answer + sources

Why it's an "agent" and not just a search box: the model chooses what to search
for, then reasons over what came back and refuses to answer if the context
doesn't support it.

Usage:
    py rag_agent.py "how does the Model Context Protocol work?"
    py rag_agent.py --source hackernews "what do people think of gemma 3?"
    py rag_agent.py --model gemma3:4b "explain retrieval augmented generation"
    py rag_agent.py                       # interactive loop

Requires: a running Ollama with a gemma3 model pulled.
    ollama pull gemma3:1b        (or gemma3:4b for better answers)
Optional (better retrieval):
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import requests

from retriever import build_retriever
from sources import SOURCES

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_WORDS = 120
CHUNK_OVERLAP = 25
TOP_K = 4


# ---------------------------------------------------------------- Ollama calls
def ollama_generate(prompt: str, model: str, system: str | None = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"[ollama] request failed: {e}\nIs Ollama running and '{model}' pulled?")
    return r.json().get("response", "").strip()


# --------------------------------------------------------------------- chunking
def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size])
        if piece:
            chunks.append(piece)
        if start + size >= len(words):
            break
    return chunks


# ----------------------------------------------------------------- agent steps
def rewrite_query(question: str, model: str) -> str:
    """Ask the model for concise search keywords (the 'agent decides' step)."""
    out = ollama_generate(
        prompt=(
            "Rewrite the user's question into a short web-search query "
            "(3-6 keywords, no punctuation, no quotes). Return ONLY the query.\n\n"
            f"Question: {question}"
        ),
        model=model,
        system="You turn questions into effective search queries.",
    )
    first_line = out.splitlines()[0].strip().strip('"') if out else ""
    return first_line or question


def answer_with_context(question: str, contexts: list[tuple[str, str]], model: str) -> str:
    """Generate a grounded answer. contexts = list of (source_label, text)."""
    context_block = "\n\n".join(
        f"[{i + 1}] ({label})\n{text}" for i, (label, text) in enumerate(contexts)
    )
    system = (
        "You are a careful research assistant. Answer ONLY using the provided "
        "context. Cite the sources you used with their [number]. If the context "
        "does not contain the answer, say so plainly -- do not invent facts."
    )
    prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nGrounded answer:"
    return ollama_generate(prompt, model=model, system=system)


# ------------------------------------------------------- retrieval strategies
def _retrieve_from_api(question: str, source_name: str, model: str, verbose: bool):
    """Live path: rewrite -> fetch from API -> chunk -> retrieve top-k."""
    source = SOURCES[source_name]
    search_query = rewrite_query(question, model)
    if verbose:
        print(f"  search query : {search_query!r}  (via {model})")

    docs = source.fetch(search_query)
    if not docs:
        return []
    if verbose:
        print(f"  fetched      : {len(docs)} docs from {source.name}")

    chunks, owners = [], []
    for di, d in enumerate(docs):
        for c in chunk_text(d.text):
            chunks.append(c)
            owners.append(di)

    retriever = build_retriever(chunks, EMBED_MODEL)
    hits = retriever.top(question, k=TOP_K)
    if verbose:
        print(f"  retriever    : {retriever.name}, {len(chunks)} chunks -> top {len(hits)}")
    return [(docs[owners[i]].title, chunks[i], docs[owners[i]].url) for i, _ in hits]


def _retrieve_from_mongo(question: str, verbose: bool):
    """Stored path: retrieve pre-ingested chunks straight from MongoDB."""
    from mongo_store import MongoStore  # optional dependency

    store = MongoStore()
    total = store.count()
    if total == 0:
        raise SystemExit(
            "MongoDB collection is empty. Ingest first, e.g.:\n"
            '  py ingest_to_mongo.py "model context protocol"'
        )
    docs = store.retrieve(question, k=TOP_K)
    if verbose:
        mode = "vector/cosine" if store.use_embeddings else "tfidf"
        print(f"  mongo        : {total} chunks stored, {mode} -> top {len(docs)}")
    return [(d.title, d.text, d.url) for d in docs]


# ------------------------------------------------------------------- RAG round
def run(question: str, source_name: str, model: str, store: str = "api", verbose: bool = True) -> str:
    if store == "mongo":
        results = _retrieve_from_mongo(question, verbose)
    else:
        results = _retrieve_from_api(question, source_name, model, verbose)

    if not results:
        return "No documents available for that query."

    contexts = [(title, text) for title, text, _ in results]
    answer = answer_with_context(question, contexts, model)

    used, sources_out = set(), []
    for title, _, url in results:
        if url not in used:
            used.add(url)
            sources_out.append(f"  - {title}: {url}")

    return answer + "\n\nSources:\n" + "\n".join(sources_out)


# ------------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description="RAG agent over live APIs, powered by Gemma 3.")
    p.add_argument("question", nargs="*", help="your question (omit for interactive mode)")
    p.add_argument("--source", default="wikipedia", choices=list(SOURCES), help="API data source")
    p.add_argument("--store", default="api", choices=["api", "mongo"],
                   help="'api' = fetch live; 'mongo' = retrieve from pre-ingested MongoDB")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama gen model (gemma3:1b / gemma3:4b)")
    args = p.parse_args()

    where = "mongo" if args.store == "mongo" else f"source={args.source}"
    print(f"RAG agent | model={args.model} | {where}\n")

    if args.question:
        print(run(" ".join(args.question), args.source, args.model, store=args.store))
        return

    print("Interactive mode. Ask a question (blank line or Ctrl+C to quit).\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            break
        print("\n" + textwrap.indent(run(q, args.source, args.model, store=args.store), "  ") + "\n")


if __name__ == "__main__":
    main()
