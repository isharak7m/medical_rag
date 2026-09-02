"""
Generate multiple PubMed queries from one normalized query while preserving intent.
"""

from __future__ import annotations

import ast
import re
from typing import List

from services.query_intent import build_query_intent
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_QUERIES = 8
_MIN_QUERIES = 2

_GENERIC_TERMS = {
    "humans", "human", "adults", "adult", "clinical", "trial", "trials",
    "study", "studies", "supplementation", "supplement", "effects", "effect",
    "outcomes", "outcome", "efficacy", "evidence", "randomized", "controlled",
    "double", "blind", "placebo", "review", "systematic", "meta", "analysis",
}

_EXPAND_PROMPT = """You are a biomedical PubMed search expert.

Generate {n} alternative PubMed search queries for the topic below.

Intent metadata:
- Subject terms: {subject_terms}
- Polarity: {polarity}
- Direction: {direction}
- Focus terms: {focus_terms}

Rules:
- Preserve the same intervention/exposure/entity
- Keep the same user intent
- For safety or risk questions, keep harm/safety/risk framing in the query
- Do not drift into indirect behavioral or policy topics unless the original query is about them
- Keep each query concise: 4-10 words
- Return ONLY a Python list of strings

Original query: {query}

Alternative queries:"""


def _extract_anchor_terms(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9-]+", query.lower())
    anchors = [token for token in tokens if len(token) > 3 and token not in _GENERIC_TERMS]
    return anchors[:3]


def _parse_query_list(raw: str) -> List[str]:
    raw = re.sub(r"```(?:python)?", "", raw).strip().rstrip("`").strip()
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return [str(item).strip() for item in result if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    quoted = re.findall(r'"([^"]{8,})"', raw)
    if len(quoted) >= _MIN_QUERIES:
        return quoted

    quoted_single = re.findall(r"'([^']{8,})'", raw)
    if len(quoted_single) >= _MIN_QUERIES:
        return quoted_single

    numbered = re.findall(r"^\s*\d+[.)]\s*(.+)$", raw, re.MULTILINE)
    if len(numbered) >= _MIN_QUERIES:
        return [item.strip() for item in numbered]

    bullets = re.findall(r"^\s*[-*]\s*(.+)$", raw, re.MULTILINE)
    if len(bullets) >= _MIN_QUERIES:
        return [item.strip() for item in bullets]

    logger.warning(f"Could not parse query list from: {raw[:200]}")
    return []


def _keep_query(
    candidate: str,
    anchors: list[str],
    focus_terms: set[str],
    polarity: str,
    subject_aliases: set[str],
) -> bool:
    lower = candidate.lower()
    if len(candidate.split()) < 2:
        return False
    if subject_aliases and not any(alias in lower for alias in subject_aliases):
        return False
    if anchors and not any(anchor in lower for anchor in anchors[:2]):
        return False
    if polarity in {"safety", "overall_value"}:
        safety_terms = {term for term in focus_terms if term in {"safety", "risk", "risks", "harm", "harms", "adverse", "toxicity", "side", "effects"}}
        value_terms = {term for term in focus_terms if term in {"benefits", "risks", "adverse", "safety", "health", "outcomes", "harms", "performance", "value"}}
        required_terms = safety_terms or value_terms
        if required_terms and not any(term in lower for term in required_terms):
            return False
    return True


def generate_multiple_queries(
    normalized_query: str,
    llm,
    n: int = 4,
) -> List[str]:
    n = min(n, _MAX_QUERIES)
    intent = build_query_intent(normalized_query)
    anchors = _extract_anchor_terms(normalized_query)
    primary_subject = intent.subject_terms[0] if intent.subject_terms else ""
    primary_aliases = {primary_subject, f"{primary_subject}s"} if primary_subject else set()
    primary_aliases.update(
        {
            alias
            for alias in intent.subject_aliases
            if primary_subject and (primary_subject in alias or alias in {primary_subject, f"{primary_subject}s"})
        }
    )

    try:
        prompt = _EXPAND_PROMPT.format(
            query=normalized_query,
            n=n,
            subject_terms=", ".join(intent.subject_terms) or "unknown",
            polarity=intent.polarity,
            direction=intent.hypothesis_direction,
            focus_terms=", ".join(sorted(intent.focus_terms)) or "none",
        )
        raw = llm.generate(prompt).strip()
        queries = _parse_query_list(raw)

        seen = set()
        final = []
        for query in [normalized_query] + queries:
            cleaned = " ".join(query.split()).strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            if not _keep_query(cleaned, anchors, intent.focus_terms, intent.polarity, primary_aliases or intent.subject_aliases):
                logger.info(f"Dropping off-topic expanded query: {cleaned}")
                continue
            seen.add(key)
            final.append(cleaned)

        if len(final) < _MIN_QUERIES:
            logger.warning("Query expansion did not produce enough on-topic variants; using original only")
            return [normalized_query]

        final = final[: _MAX_QUERIES + 1]
        logger.info(f"Query expander: {len(final)} queries from '{normalized_query}'")
        for index, query in enumerate(final, start=1):
            logger.info(f"  Query {index}: {query}")
        return final

    except Exception as exc:
        logger.warning(f"Query expansion failed: {exc} -> using original query only")
        return [normalized_query]
