"""
Minimal retrieval-augmented generation (RAG) layer.

Design choice: this ships with a dependency-free TF-IDF retriever (scikit-learn)
so the service runs and is testable with zero API keys and zero network calls.
If OPENAI_API_KEY is set, `embed_and_retrieve` can be swapped for a real
embedding-based retriever (see `OpenAIRetriever` below) without touching the
FastAPI route — that's the point of hiding retrieval behind the `Retriever`
interface: the LLM call in main.py doesn't know or care which one is active.

This mirrors a real decision I make in client work (ShopHelp / NeverBlast):
start with the cheapest retriever that's correct, only pay for embeddings once
the knowledge base is big enough that TF-IDF's lexical matching stops being
good enough.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SUFFIXES = ("ing", "edly", "ed", "es", "s")


def _lightly_stem(token: str) -> str:
    """
    Cheap suffix-stripping so 'returns'/'returning'/'returned' and
    'ships'/'shipping' collapse to a shared root. Not a real stemmer
    (no Porter/Snowball dependency) — good enough for short-document
    lexical matching, which is the only thing TfidfRetriever is for.
    """
    if len(token) <= 4:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def _stemming_analyzer(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [_lightly_stem(w) for w in words]


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float


class Retriever(Protocol):
    def index(self, docs: dict[str, str]) -> None: ...
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]: ...


class TfidfRetriever:
    """Lexical retriever. No API key required, deterministic, fast to test."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(analyzer=_stemming_analyzer)
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._matrix = None

    def index(self, docs: dict[str, str]) -> None:
        self._doc_ids = list(docs.keys())
        self._doc_texts = list(docs.values())
        if not self._doc_texts:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(self._doc_texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        if self._matrix is None or not self._doc_ids:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            RetrievedDoc(doc_id=self._doc_ids[i], text=self._doc_texts[i], score=float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


class OpenAIRetriever:
    """
    Drop-in replacement for TfidfRetriever using OpenAI embeddings.
    Not wired up by default — no API key is bundled with this repo.
    Kept here to show the intended upgrade path, not as dead code.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set — OpenAIRetriever requires it.")
        # Embedding client construction intentionally omitted from this
        # public template; wire up `openai.OpenAI(api_key=self.api_key)`
        # and cache embeddings per doc_id in your own deployment.
        self._docs: dict[str, str] = {}

    def index(self, docs: dict[str, str]) -> None:
        self._docs = docs

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        raise NotImplementedError(
            "Wire this up to your embeddings provider of choice in a private deployment."
        )


def get_default_retriever() -> Retriever:
    """Factory: picks the retriever based on environment, defaulting to the free one."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIRetriever()
    return TfidfRetriever()
