"""
Evaluation module — lightweight, self-contained scoring.

Metrics (no external datasets required):
  - retrieval_score  : keyword overlap between query and retrieved docs (avg)
  - faithfulness     : fraction of answer tokens grounded in retrieved docs
  - coverage         : fraction of query concepts present in the final answer

Logging:
  - Appends one JSON record per query to eval_log.jsonl
"""

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import List

from db.schemas import EvalLog, Paper, RichQueryResponse
from utils.logger import get_logger

logger = get_logger(__name__)

_LOG_PATH = Path("eval_log.jsonl")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "with", "that", "this", "it", "as", "at", "by", "from",
    "on", "be", "been", "has", "have", "had", "not", "but", "we", "our",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


# ---------------------------------------------------------------------------
# Retrieval relevance score
# ---------------------------------------------------------------------------

def score_retrieval(query: str, papers: List[Paper], core_keywords: List[str] = None) -> float:
    """
    Fraction of core topic keywords found in retrieved papers.
    Uses core_keywords (not the full expanded PubMed query) for fair scoring.
    Returns float in [0, 1].
    """
    if not papers:
        return 0.0

    # Use core keywords if provided, else tokenize raw query
    if core_keywords:
        query_tokens = set(k.lower() for k in core_keywords if len(k) > 2)
    else:
        query_tokens = set(_tokenize(query))

    if not query_tokens:
        return 0.0

    # Score = fraction of papers that contain at least one core keyword
    hits = 0
    for paper in papers:
        doc_text = f"{paper.title} {paper.abstract}".lower()
        if any(kw in doc_text for kw in query_tokens):
            hits += 1

    return round(hits / len(papers), 4)


# ---------------------------------------------------------------------------
# Faithfulness score
# ---------------------------------------------------------------------------

def score_faithfulness(answer: str, papers: List[Paper]) -> float:
    """
    Fraction of unique answer tokens that appear in at least one retrieved paper.
    Measures whether the answer is grounded in the evidence.
    Returns float in [0, 1].
    """
    if not answer or not papers:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return 0.0

    # Build corpus of all tokens across all retrieved papers
    corpus_tokens: set[str] = set()
    for paper in papers:
        corpus_tokens.update(_tokenize(f"{paper.title} {paper.abstract}"))

    grounded = answer_tokens & corpus_tokens
    return round(len(grounded) / len(answer_tokens), 4)


# ---------------------------------------------------------------------------
# Coverage score
# ---------------------------------------------------------------------------

def score_coverage(query: str, answer: str) -> float:
    """
    Fraction of unique query terms present in the answer.
    Measures whether the answer addresses the full query.
    Returns float in [0, 1].
    """
    if not query or not answer:
        return 0.0

    query_tokens = set(_tokenize(query))
    answer_tokens = set(_tokenize(answer))

    if not query_tokens:
        return 0.0

    covered = query_tokens & answer_tokens
    return round(len(covered) / len(query_tokens), 4)


# ---------------------------------------------------------------------------
# Evaluate and log
# ---------------------------------------------------------------------------

def evaluate_and_log(
    query: str,
    papers: List[Paper],
    response: RichQueryResponse,
    core_keywords: List[str] = None,
) -> RichQueryResponse:
    """
    Compute all three scores, attach them to the response, and append to log.
    Returns the response with scores filled in (mutated copy).
    """
    retrieval  = score_retrieval(query, papers, core_keywords)
    faithful   = score_faithfulness(response.final_answer + " " + response.summary, papers)
    coverage   = score_coverage(query, response.final_answer)

    response.retrieval_score  = retrieval
    response.faithfulness_score = faithful
    response.coverage_score   = coverage

    log_entry = EvalLog(
        query=query,
        retrieved_pmids=[c.pmid for c in response.evidence_cards],
        final_answer=response.final_answer,
        faithfulness_score=faithful,
        coverage_score=coverage,
        retrieval_score=retrieval,
        verdict=response.verdict.value,
        confidence_score=response.confidence.score,
        diagnostics=response.diagnostics,
    )

    _append_log(log_entry)

    logger.info(
        f"Eval | retrieval={retrieval:.2f} faithful={faithful:.2f} coverage={coverage:.2f}"
    )
    return response


def _append_log(entry: EvalLog) -> None:
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
    except Exception as exc:
        logger.warning(f"Eval log write failed: {exc}")


# ---------------------------------------------------------------------------
# Read logs (for dashboard)
# ---------------------------------------------------------------------------

def load_eval_logs() -> List[EvalLog]:
    """Load all evaluation logs from disk. Returns empty list if none exist."""
    if not _LOG_PATH.exists():
        return []
    logs = []
    try:
        with _LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(EvalLog(**json.loads(line)))
    except Exception as exc:
        logger.warning(f"Eval log read failed: {exc}")
    return logs
