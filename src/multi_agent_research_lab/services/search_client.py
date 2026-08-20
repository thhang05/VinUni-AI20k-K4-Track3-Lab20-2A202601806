"""Search client abstraction for ResearcherAgent.

Implemented as a local, offline retriever over `ai_agent_offline_research_corpus_v2/`
instead of a paid web-search API (no TAVILY_API_KEY is configured for this lab).
Retrieval quality is intentionally simple (keyword overlap) — the point of the lab
is agent orchestration, not IR.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with", "is",
    "are", "be", "how", "what", "why", "does", "do", "when", "into", "about",
    "write", "summary", "words", "compare", "research", "explain", "summarize",
}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if tok.lower() not in _STOPWORDS}


def _find_corpus_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "ai_agent_offline_research_corpus_v2" / "topics"
        if candidate.is_dir():
            return candidate
    raise AgentExecutionError(
        "Could not locate ai_agent_offline_research_corpus_v2/topics/ from the repo tree."
    )


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict[str, Any]]:
    topics_dir = _find_corpus_root()
    topics = []
    for path in sorted(topics_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            topics.append(json.load(handle))
    return topics


class SearchClient:
    """Retrieves source documents from the offline research corpus."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Find the most relevant topic in the corpus, then the most relevant
        knowledge articles / source documents within it.
        """

        query_tokens = _tokenize(query)
        topics = _load_corpus()
        if not topics:
            raise AgentExecutionError("Offline research corpus is empty.")

        best_topic = max(topics, key=lambda t: self._score_topic(t, query_tokens))
        candidates = self._candidate_documents(best_topic)
        scored = sorted(
            candidates,
            key=lambda item: self._score_text(item["text"], query_tokens),
            reverse=True,
        )
        top = scored[:max_results] if any(
            self._score_text(item["text"], query_tokens) > 0 for item in scored
        ) else candidates[:max_results]

        return [
            SourceDocument(
                title=item["title"],
                url=item.get("url"),
                snippet=item["text"][:600],
                metadata={
                    "source_id": item["source_id"],
                    "topic": best_topic["topic"]["name"],
                    "doc_type": item["doc_type"],
                },
            )
            for item in top
        ]

    @staticmethod
    def _score_topic(topic: dict[str, Any], query_tokens: set[str]) -> int:
        meta = topic["topic"]
        haystack = " ".join(
            [
                meta.get("name", ""),
                " ".join(meta.get("tags", [])),
                meta.get("research_question", ""),
            ]
        )
        return len(query_tokens & _tokenize(haystack))

    @staticmethod
    def _score_text(text: str, query_tokens: set[str]) -> int:
        if not query_tokens:
            return 0
        return len(query_tokens & _tokenize(text))

    @staticmethod
    def _candidate_documents(topic: dict[str, Any]) -> list[dict[str, Any]]:
        kb = topic["knowledge_base"]
        candidates: list[dict[str, Any]] = []
        for article in kb.get("knowledge_articles", []):
            candidates.append(
                {
                    "source_id": article["article_id"],
                    "title": article["title"],
                    "url": None,
                    "text": article["content"],
                    "doc_type": "knowledge_article",
                }
            )
        for doc in kb.get("source_documents", []):
            candidates.append(
                {
                    "source_id": doc["document_id"],
                    "title": doc["title"],
                    "url": doc.get("provenance_url"),
                    "text": doc["full_text"],
                    "doc_type": "source_document",
                }
            )
        return candidates
