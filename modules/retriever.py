"""
Retriever module — FAISS semantic search + hybrid reranking.

Flow:
  1. Embed all paper abstracts → build FAISS index
  2. Embed query → cosine similarity search (top_k * 2 candidates)
  3. Rerank candidates with hybrid scorer (semantic + keyword + length)
  4. Return final top_k papers with their normalised relevance scores
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from db.schemas import Paper
from db.vector_store import VectorStore
from modules.reranker import rerank
from services.embedding_service import EmbeddingService
from utils.logger import get_logger
from utils.text_cleaning import truncate

logger = get_logger(__name__)

_EMBED_MAX_CHARS = 512
# Fetch more candidates than needed so reranker has room to work
_CANDIDATE_MULTIPLIER = 2


def retrieve(
    query: str,
    papers: List[Paper],
    embedding_service: EmbeddingService,
    top_k: int,
) -> List[Paper]:
    """Backward-compatible wrapper — returns only Paper list."""
    results = retrieve_with_scores(query, papers, embedding_service, top_k)
    return [p for p, _ in results]


def retrieve_with_scores(
    query: str,
    papers: List[Paper],
    embedding_service: EmbeddingService,
    top_k: int,
) -> List[Tuple[Paper, float]]:
    """
    Returns list of (Paper, relevance_score) sorted by hybrid rerank score.
    relevance_score is normalised to [0, 1].
    """
    if not papers:
        return []

    doc_texts = [
        truncate(f"{p.title}. {p.abstract}", _EMBED_MAX_CHARS)
        for p in papers
    ]

    doc_vectors: np.ndarray = embedding_service.embed_batch(doc_texts)

    store = VectorStore(dim=embedding_service.dim)
    store.add(doc_vectors)

    query_vector = embedding_service.embed(truncate(query, _EMBED_MAX_CHARS))

    # Fetch extra candidates for reranker
    candidate_k = min(top_k * _CANDIDATE_MULTIPLIER, len(papers))
    hits = store.search(query_vector, top_k=candidate_k)

    if not hits:
        return []

    candidate_papers = [papers[idx] for idx, _ in hits]
    semantic_scores  = [float(dist) for _, dist in hits]

    # Hybrid rerank → returns sorted (Paper, final_score)
    reranked = rerank(query, candidate_papers, semantic_scores)

    final = reranked[:top_k]
    logger.info(f"Retriever: {len(final)} papers after reranking (from {len(hits)} candidates)")
    return final
