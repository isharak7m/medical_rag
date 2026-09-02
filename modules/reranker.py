"""
Reranker module — hybrid scoring on top of FAISS semantic retrieval.

Combines three signals:
  1. Semantic score   — cosine similarity from FAISS (passed in)
  2. Keyword overlap  — BM25-style term frequency overlap between query and doc
  3. Length penalty   — discounts very short abstracts (low information)

Final score = 0.6 * semantic + 0.3 * keyword + 0.1 * length_norm

No external calls. Pure scoring logic.
"""

from __future__ import annotations
import math
import re
from typing import List, Tuple

from db.schemas import Paper
from utils.logger import get_logger

logger = get_logger(__name__)

# Scoring weights — must sum to 1.0
_W_SEMANTIC = 0.6
_W_KEYWORD  = 0.3
_W_LENGTH   = 0.1

# Abstracts shorter than this are penalised
_MIN_ABSTRACT_CHARS = 200
_MAX_ABSTRACT_CHARS = 2000

# Common biomedical stopwords to exclude from keyword matching
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "with", "that", "this", "it", "as", "at", "by", "from",
    "on", "be", "been", "has", "have", "had", "not", "but", "we", "our",
    "their", "which", "who", "also", "may", "can", "could", "would", "should",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _keyword_score(query_tokens: List[str], doc_text: str) -> float:
    """
    BM25-inspired overlap: fraction of unique query terms found in document,
    weighted by log term frequency in the document.
    Returns value in [0, 1].
    """
    if not query_tokens:
        return 0.0

    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0

    doc_freq: dict[str, int] = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1

    score = 0.0
    for qt in set(query_tokens):
        if qt in doc_freq:
            # log-dampened term frequency
            score += math.log(1 + doc_freq[qt])

    # Normalise by number of unique query terms
    return min(score / (len(set(query_tokens)) * math.log(2 + len(doc_tokens))), 1.0)


def _length_score(abstract: str) -> float:
    """
    Normalised length score. Short abstracts score lower.
    Returns value in [0, 1].
    """
    length = len(abstract)
    if length >= _MAX_ABSTRACT_CHARS:
        return 1.0
    if length <= _MIN_ABSTRACT_CHARS:
        return length / _MIN_ABSTRACT_CHARS * 0.5
    return 0.5 + 0.5 * (length - _MIN_ABSTRACT_CHARS) / (_MAX_ABSTRACT_CHARS - _MIN_ABSTRACT_CHARS)


def rerank(
    query: str,
    papers: List[Paper],
    semantic_scores: List[float],
) -> List[Tuple[Paper, float]]:
    """
    Re-rank papers using hybrid scoring.

    Args:
        query:           Normalised query string.
        papers:          Papers in FAISS retrieval order.
        semantic_scores: Cosine similarity scores from FAISS (same order as papers).

    Returns:
        List of (Paper, final_score) sorted descending by final_score.
    """
    if not papers:
        return []

    query_tokens = _tokenize(query)

    # Normalise semantic scores to [0, 1]
    max_sem = max(semantic_scores) if semantic_scores else 1.0
    norm_semantic = [s / max_sem if max_sem > 0 else 0.0 for s in semantic_scores]

    results: List[Tuple[Paper, float]] = []
    for paper, sem_score in zip(papers, norm_semantic):
        doc_text = f"{paper.title} {paper.abstract}"
        kw   = _keyword_score(query_tokens, doc_text)
        leng = _length_score(paper.abstract)
        final = _W_SEMANTIC * sem_score + _W_KEYWORD * kw + _W_LENGTH * leng
        results.append((paper, round(final, 4)))
        logger.debug(
            f"Rerank PMID {paper.pmid}: sem={sem_score:.3f} kw={kw:.3f} "
            f"len={leng:.3f} → final={final:.3f}"
        )

    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Reranker: top score={results[0][1]:.3f} (PMID {results[0][0].pmid})")
    return results
