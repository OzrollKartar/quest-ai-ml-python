# ============================================
# Lab 23: RAG Agent  (APIs as data source, Gemma 3 as the model)
# ============================================
# Agenda (~55 min):
#   1. What RAG is and why we need it            (~8 min)
#   2. RAG vs. a plain chatbot vs. an agent      (~7 min)
#   3. The pipeline, stage by stage              (~15 min)
#   4. Using a LIVE API as the knowledge base    (~8 min)
#   5. Retrieval: TF-IDF vs. embeddings          (~10 min)
#   6. Running it with Gemma 3 on Ollama         (~7 min)
#
# The runnable agent lives in ./rag-agent/  -- this file is the read-along.
# Setup:  ollama pull gemma3:1b   (and: pip install -r rag-agent/requirements.txt)

# ============================================
# 1. WHAT IS RAG?
# ============================================
# RAG = "Retrieval-Augmented Generation".
# A language model only knows what was in its training data (frozen, and often
# stale). RAG fixes that at question time:
#
#   RETRIEVE relevant text from an outside source, then let the model GENERATE
#   its answer while READING that text.
#
# So instead of "answer from memory" you get "answer from these specific
# documents I just handed you". Benefits: fresh facts, citable sources, far
# fewer hallucinations, and no expensive re-training.


# ============================================
# 2. RAG vs CHATBOT vs AGENT
# ============================================
#   Plain chatbot -> model answers from memory. No sources. Can hallucinate.
#   RAG           -> model answers from retrieved docs. Grounded + citable.
#   Agent         -> model also DECIDES and ACTS: what to search, whether the
#                    evidence is enough, when to refuse.
#
# Our program is a small AGENT: it (a) rewrites your question into a search
# query itself, and (b) is instructed to answer ONLY from what came back and to
# say "not in the sources" rather than invent facts.


# ============================================
# 3. THE PIPELINE  (see rag-agent/rag_agent.py -> run())
# ============================================
#   question
#     |
#     |  [LLM] rewrite_query()   -> "gemma 3 benchmarks android"     (agent step)
#     v
#   [API] source.fetch()         -> live documents (Wikipedia / HN)
#     |
#     |  chunk_text()            -> break long docs into ~120-word pieces
#     v
#   build_retriever() + .top()   -> the k most relevant chunks for the question
#     |
#     |  [LLM] answer_with_context()  -> grounded answer that cites [1], [2]...
#     v
#   answer + sources
#
# Why chunk? Retrieval works best on small passages, and the model has a limited
# context window. We attach a bit of OVERLAP so a sentence split across a
# boundary still survives in one piece.


# ============================================
# 4. THE DATA SOURCE IS A LIVE API
# ============================================
# The "knowledge base" is not a folder of files -- it's whatever an API returns
# RIGHT NOW. See rag-agent/sources.py:
#
#   WikipediaSource  -> MediaWiki API: search for titles, then pull plain-text
#                       article extracts. No key.
#   HackerNewsSource -> Algolia HN search: live discussion + points/comments.
#
# The interface is deliberately tiny so any API drops in:
#
#   class Source:
#       def fetch(self, query, limit) -> list[Document]:  ...
#
#   @dataclass
#   class Document:
#       title: str; text: str; url: str
#
# Add GitHub, arXiv, your company's REST API, etc. -- subclass, register in
# SOURCES, done. That's the whole point of "APIs as the data source": the model
# stays fixed while the knowledge is always current and swappable.


# ============================================
# 5. RETRIEVAL: TWO BACKENDS  (see rag-agent/retriever.py)
# ============================================
# "Which chunks are relevant?" can be answered two ways:
#
#   TF-IDF (lexical)     -> score by important word overlap. Pure numpy, no
#                           model, no download. Great, transparent default.
#                           Weakness: "car" and "automobile" look unrelated.
#
#   Embeddings (semantic)-> turn each chunk + the query into a vector with an
#                           embedding model, rank by cosine similarity.
#                           Understands meaning/synonyms. Needs a real embedder:
#                               ollama pull nomic-embed-text
#
# Note: gemma3:1b is a GENERATION model -- Ollama returns an EMPTY embedding for
# it, so it can't be the embedder. That's why we use a dedicated embed model,
# and fall back to TF-IDF when it isn't installed. build_retriever() auto-picks.
#
# Cosine similarity in one line, once vectors are unit-normalized:
#   scores = matrix @ query_vector      # dot product == cosine here


# ============================================
# 6. GENERATION WITH GEMMA 3 (via OLLAMA)
# ============================================
# Ollama runs the model locally and exposes an HTTP API on :11434.
# We POST to /api/generate with a SYSTEM prompt that enforces grounding:
#
#   system = ("Answer ONLY using the provided context. Cite sources with their "
#             "[number]. If the context lacks the answer, say so -- don't invent.")
#   prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nGrounded answer:"
#
# gemma3:1b  -> tiny + fast; fine for extracting/grounding. Runs on a laptop CPU.
# gemma3:4b  -> better written, more coherent synthesis. Pass --model gemma3:4b.


# ============================================
# TRY IT
# ============================================
#   cd rag-agent
#   pip install -r requirements.txt
#   ollama pull gemma3:1b                 # or gemma3:4b
#
#   py rag_agent.py "what is retrieval augmented generation"
#   py rag_agent.py --source hackernews "what do people think of gemma 3?"
#   py rag_agent.py --model gemma3:4b "how does the Model Context Protocol work?"
#   py rag_agent.py                       # interactive loop
#
# See rag-agent/README.md for extending it with your own API source.
print("Lab 23 is a read-along. The runnable RAG agent is in ./rag-agent/ -- see README.md")
