"""
API data sources for the RAG agent.
===================================
Each source turns a natural-language topic into a list of `Document`s pulled
LIVE from a web API. Add your own by subclassing `Source`.

The bundled sources need no API key:
  * WikipediaSource -- MediaWiki search + plain-text article extracts
  * HackerNewsSource -- Algolia HN search (great for "what's the buzz on X")
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

TIMEOUT = 20


@dataclass
class Document:
    title: str
    text: str
    url: str


class Source:
    """Interface: given a topic, return live documents from an API."""

    name = "base"

    def fetch(self, query: str, limit: int = 3) -> list[Document]:
        raise NotImplementedError


class WikipediaSource(Source):
    name = "wikipedia"
    API = "https://en.wikipedia.org/w/api.php"

    def fetch(self, query: str, limit: int = 3) -> list[Document]:
        # 1) search for relevant page titles
        search = requests.get(
            self.API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            },
            headers={"User-Agent": "quest-rag-agent/1.0"},
            timeout=TIMEOUT,
        ).json()
        titles = [hit["title"] for hit in search.get("query", {}).get("search", [])]
        if not titles:
            return []

        # 2) pull plain-text extracts for those titles in one call
        extracts = requests.get(
            self.API,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": "|".join(titles),
                "format": "json",
            },
            headers={"User-Agent": "quest-rag-agent/1.0"},
            timeout=TIMEOUT,
        ).json()

        docs: list[Document] = []
        for page in extracts.get("query", {}).get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            if not text:
                continue
            title = page.get("title", "")
            docs.append(
                Document(
                    title=title,
                    text=text,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                )
            )
        return docs


class HackerNewsSource(Source):
    name = "hackernews"
    API = "https://hn.algolia.com/api/v1/search"

    def fetch(self, query: str, limit: int = 5) -> list[Document]:
        data = requests.get(
            self.API,
            params={"query": query, "tags": "story", "hitsPerPage": limit},
            timeout=TIMEOUT,
        ).json()
        docs: list[Document] = []
        for hit in data.get("hits", []):
            title = hit.get("title") or hit.get("story_title") or "(untitled)"
            body = hit.get("story_text") or ""
            points = hit.get("points", 0)
            comments = hit.get("num_comments", 0)
            text = f"{title}\n({points} points, {comments} comments)\n{body}".strip()
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            docs.append(Document(title=title, text=text, url=url))
        return docs


SOURCES: dict[str, Source] = {
    "wikipedia": WikipediaSource(),
    "hackernews": HackerNewsSource(),
}
