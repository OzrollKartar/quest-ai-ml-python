# Lab 23 — RAG Agent (APIs as data source, Gemma 3 as the model)

A retrieval-augmented **agent** that answers questions using **live web APIs** as
its knowledge base and a local **Gemma 3** model (via [Ollama](https://ollama.com))
as its reasoning engine. No vector DB, no API keys.

```
question
  → [LLM] rewrite into search keywords        (agent decides what to look up)
  → [API] fetch live docs (Wikipedia / HN)    (the "R" — retrieval source)
  → chunk + index
  → retrieve top-k relevant chunks            (TF-IDF, or Ollama embeddings)
  → [LLM] answer grounded in those chunks, cite sources, refuse if unsupported
```

## Files

| File | Role |
|------|------|
| `rag_agent.py` | The agent: query rewrite → fetch → retrieve → grounded answer |
| `sources.py`   | Pluggable API data sources (`WikipediaSource`, `HackerNewsSource`) |
| `retriever.py` | Two retrieval backends: TF-IDF (numpy) and Ollama embeddings |
| `mongo_store.py` | MongoDB as the knowledge base (ingest + vector / cosine / TF-IDF retrieval) |
| `ingest_to_mongo.py` | Offline step: fetch from an API, chunk, embed, load into MongoDB |

Read-along notes: [`../Lab23_RAG_Agent.py`](../Lab23_RAG_Agent.py)

## Setup

```bash
pip install -r requirements.txt
ollama pull gemma3:1b          # or: ollama pull gemma3:4b  (better answers)
# optional — upgrades retrieval from lexical to semantic:
ollama pull nomic-embed-text
```

The agent auto-detects `nomic-embed-text`; if it's absent it falls back to the
built-in TF-IDF retriever, so it runs with just `gemma3:1b`.

## Run

```bash
py rag_agent.py "how does the Model Context Protocol work?"
py rag_agent.py --source hackernews "what do people think of gemma 3?"
py rag_agent.py --model gemma3:4b "explain retrieval augmented generation"
py rag_agent.py                       # interactive loop
```

## Extend it

**Add a new API source** — subclass `Source` in `sources.py` and register it:

```python
class GitHubSource(Source):
    name = "github"
    def fetch(self, query, limit=5):
        data = requests.get("https://api.github.com/search/repositories",
                            params={"q": query, "per_page": limit}).json()
        return [Document(r["full_name"], r["description"] or "", r["html_url"])
                for r in data["items"]]

SOURCES["github"] = GitHubSource()
```

Then: `py rag_agent.py --source github "python vector database"`.

## Connecting RAG to MongoDB

Real RAG systems split into two phases: a slow **offline ingestion** that fills a
store, and fast **online queries** against it. Here MongoDB is that store.

```
                 offline (once)                        online (per question)
  API ──fetch──► chunk ──embed──► MongoDB   ◄──retrieve── question ──► Gemma 3
```

Prereqs: a running MongoDB (local `mongod` on `:27017` is fine) and
`pip install pymongo`. Embeddings are stored if `nomic-embed-text` is pulled.

```bash
# 1) ingest: fetch from an API and load Mongo (db=rag_demo, collection=chunks)
py ingest_to_mongo.py "model context protocol"
py ingest_to_mongo.py --source hackernews "gemma 3" --limit 8

# 2) query: retrieve from Mongo instead of hitting the API live
py rag_agent.py --store mongo "how does the model context protocol work?"
```

**Retrieval backend is chosen automatically:**
1. `$vectorSearch` if you're on **Atlas** with a vector index (scales to millions);
2. otherwise **cosine similarity in Python** over the stored embeddings (works on local `mongod`);
3. otherwise **TF-IDF** over the stored chunk text (no embed model needed).

**Atlas vector index** (for path 1) — create a search index named `vector_index`
on the `embedding` field:

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine" }
  ]
}
```

(`numDimensions` must match your embed model — `nomic-embed-text` is 768.)
The code tries `$vectorSearch` first and silently falls back to Python cosine if
the index isn't there, so the same code runs on local mongod and on Atlas.

## Design notes

- **Why an agent, not just search?** The model chooses the search query and is
  instructed to answer *only* from retrieved context and cite it — it declines
  when the API didn't return the answer, instead of hallucinating.
- **Why TF-IDF by default?** Zero downloads, pure numpy, and it makes the
  retrieval step transparent. Swap in `nomic-embed-text` for semantic recall.
- **gemma3:1b vs 4b** — `1b` is fast and fine for extraction/grounding; `4b`
  writes noticeably better synthesized answers. Pass `--model gemma3:4b`.
