"""
Query processor — intent classification + precise PubMed query building.

Key fixes:
  - Extracts ONLY the core topic keywords (no noise words like "or", "good", "bad")
  - Builds proper PubMed AND queries — never uses loose OR
  - Returns both the PubMed search query and the core topic terms for domain filtering
"""

from __future__ import annotations
import re
from typing import Tuple
from utils.text_cleaning import normalize
from utils.logger import get_logger

logger = get_logger(__name__)

# Intent detection patterns
_INTENT_PATTERNS = {
    "supplement": re.compile(
        r"\b(creatine|protein|whey|bcaa|omega.?3|vitamin|zinc|magnesium|caffeine|"
        r"collagen|glutamine|pre.?workout|supplement|probiotic|fish.?oil|melatonin)\b",
        re.IGNORECASE,
    ),
    "drug": re.compile(
        r"\b(metformin|aspirin|ibuprofen|statin|antibiotic|vaccine|medication|"
        r"drug|pharmaceutical|dosage|prescription)\b",
        re.IGNORECASE,
    ),
    "clinical": re.compile(
        r"\b(cancer|diabetes|hypertension|alzheimer|depression|anxiety|obesity|"
        r"cardiovascular|stroke|disease|disorder|syndrome|therapy|treatment)\b",
        re.IGNORECASE,
    ),
    "general_health": re.compile(
        r"\b(exercise|muscle|strength|weight|fat|sleep|diet|nutrition|fitness|"
        r"performance|recovery|endurance|aging|longevity)\b",
        re.IGNORECASE,
    ),
}

# Words to strip before extracting topic — question words, filler, sentiment
_STRIP_RE = re.compile(
    r"\b(is|are|does|do|can|will|should|what|how|why|when|where|"
    r"good|bad|safe|dangerous|effective|worth|better|worse|"
    r"it|this|that|the|a|an|for|me|you|us|or|and|but|"
    r"really|actually|basically|generally|typically)\b",
    re.IGNORECASE,
)

# PubMed filter terms per intent — appended with AND
_PUBMED_FILTERS = {
    "supplement":    '("humans"[MeSH] OR "clinical trial"[pt] OR "randomized controlled trial"[pt])',
    "drug":          '("humans"[MeSH] OR "clinical trial"[pt] OR "adverse effects"[sh])',
    "clinical":      '("humans"[MeSH] OR "systematic review"[pt] OR "meta-analysis"[pt])',
    "general_health": '("humans"[MeSH] OR "review"[pt])',
    "other":         '"humans"[MeSH]',
}


def _classify_intent(text: str) -> str:
    for intent, pattern in _INTENT_PATTERNS.items():
        if pattern.search(text):
            return intent
    return "other"


def _extract_core_keywords(text: str) -> list[str]:
    """Strip noise, return meaningful content words only."""
    cleaned = _STRIP_RE.sub(" ", text)
    cleaned = re.sub(r"[?!.,;:\"']+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Return unique words longer than 2 chars
    words = [w for w in cleaned.split() if len(w) > 2]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    return unique


def process_query(raw_query: str) -> Tuple[str, list[str]]:
    """
    Returns (pubmed_query, core_keywords).
    pubmed_query  — precise AND-based query for PubMed ESearch
    core_keywords — topic words used for domain filtering after fetch
    """
    normalized = normalize(raw_query)
    intent = _classify_intent(normalized)
    keywords = _extract_core_keywords(normalized)

    if not keywords:
        keywords = [normalized]

    # Build PubMed query: join keywords with AND, append domain filter
    keyword_str = " AND ".join(keywords)
    pubmed_filter = _PUBMED_FILTERS.get(intent, _PUBMED_FILTERS["other"])
    pubmed_query = f"({keyword_str}) AND {pubmed_filter}"

    logger.info(
        f"Query rewrite | intent={intent} | '{raw_query}' → '{pubmed_query}' "
        f"| keywords={keywords}"
    )
    return pubmed_query, keywords


def get_query_intent(raw_query: str) -> str:
    return _classify_intent(normalize(raw_query))
