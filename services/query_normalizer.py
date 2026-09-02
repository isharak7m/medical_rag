"""
Normalize user queries into PubMed-friendly search text.
Returns structured QueryIntent alongside the PubMed query.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional

from services.query_intent import build_query_intent, QueryIntent
from utils.logger import get_logger

logger = get_logger(__name__)

_SLANG_MAP = {
    r"\bgains\b": "muscle hypertrophy",
    r"\bbulk(ing)?\b": "muscle mass gain",
    r"\bcut(ting)?\b": "fat loss body composition",
    r"\bbro split\b": "resistance training frequency",
    r"\bpump\b": "muscle blood flow",
    r"\bshredded\b": "low body fat",
    r"\bswole\b": "muscle hypertrophy",
    r"\bpre-?workout\b": "pre-exercise supplementation",
    r"\bpost-?workout\b": "post-exercise recovery supplementation",
    r"\bnatty\b": "natural training without supplements",
    r"\bnootropic\b": "cognitive enhancement",
    r"\bsuperfood\b": "nutrient-dense food",
    r"\bdetox\b": "detoxification",
    r"\bcleanse\b": "dietary intervention",
}

_NORMALIZE_PROMPT = """You are a biomedical query normalizer for PubMed literature search.

Rewrite the user query into a precise PubMed-style search query.

Intent metadata:
- Intervention/supplement: {intervention}
- Outcome/measurement: {outcome}
- Population: {population}
- Question type: {question_type}

Rules:
- Keep the same subject matter and user intent
- Preserve the intervention/exposure/entity; do not switch to a different substance or treatment class
- Make the search query scientifically precise
- Include both the intervention AND outcome terms from the original query
- Add humans if appropriate
- Output only the rewritten query
- Keep it concise: 4-14 words

User query: {query}

Rewritten query:"""

_STOPWORDS = {
    "and", "or", "the", "for", "in", "of", "to", "a", "an", "with",
    "humans", "human", "clinical", "trial", "study", "studies", "evidence",
    "safety", "efficacy", "effects", "effect", "impact", "role", "use",
    "randomized", "controlled", "systematic", "review", "analysis",
    "meta", "placebo", "double", "blind", "based", "versus", "vs",
    "does", "work", "good", "bad", "best", "better", "worse",
}


def _apply_slang_map(text: str) -> str:
    result = text.lower().strip()
    for pattern, replacement in _SLANG_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _basic_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[?!.]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _looks_like_bad_rewrite(result: str) -> bool:
    lower = result.lower()
    return (
        result.startswith("{")
        or result.startswith("[")
        or any(marker in lower for marker in ("final_answer", "key_claims", "summary"))
    )


def _build_fallback_query(intent: QueryIntent) -> str:
    parts: list[str] = []

    if intent.intervention_terms:
        parts.extend(intent.intervention_terms[:3])
    elif intent.subject_terms:
        parts.extend(intent.subject_terms[:2])

    if intent.outcome_terms:
        parts.extend(intent.outcome_terms[:3])
    elif intent.polarity == "safety":
        parts.extend(["safety", "adverse", "effects"])
    elif intent.polarity == "efficacy":
        parts.extend(sorted(term for term in intent.focus_terms if len(term) > 3)[:3])

    parts.append("humans")

    final: list[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key and key not in seen:
            seen.add(key)
            final.append(part)
    return " ".join(final) if final else intent.cleaned_query


def _llm_normalize(query: str, llm, intent: QueryIntent) -> str:
    try:
        prompt = _NORMALIZE_PROMPT.format(
            query=query,
            intervention=", ".join(intent.intervention_terms) or "unknown",
            outcome=", ".join(intent.outcome_terms) or "unknown",
            population=intent.population or "humans",
            question_type=intent.question_type,
        )
        result = llm.generate(prompt).strip()
        if _looks_like_bad_rewrite(result):
            logger.warning(f"LLM normalization returned structured output; using fallback for '{query}'")
            return _build_fallback_query(intent)

        result = re.sub(r'^["\']|["\']$', "", result).strip()
        result = re.sub(r"```.*?```", "", result, flags=re.DOTALL).strip()
        result = re.sub(r"(?i)^rewritten query:\s*", "", result).strip()
        result = re.sub(r"(?i)^->?\s*", "", result).strip()

        words = result.split()
        if len(words) < 2:
            logger.warning(f"LLM normalization too short: '{result}' -> using fallback")
            return _build_fallback_query(intent)
        if len(words) > 20:
            result = " ".join(words[:15])

        logger.info(f"LLM normalized: '{query}' -> '{result}'")
        return result
    except Exception as exc:
        logger.warning(f"LLM normalization failed: {exc} -> using fallback")
        return _build_fallback_query(intent)


def _extract_core_keywords(pubmed_query: str, intent: QueryIntent) -> List[str]:
    keywords: list[str] = []

    if intent.intervention_terms:
        keywords.extend(intent.intervention_terms[:3])
    if intent.outcome_terms:
        keywords.extend(intent.outcome_terms[:3])

    if not keywords:
        words = re.findall(r"[a-z0-9-]+", pubmed_query.lower())
        keywords = [word for word in words if word not in _STOPWORDS and len(word) > 3]
        keywords.sort(key=len, reverse=True)

    seen = set()
    unique: list[str] = []
    for k in keywords:
        k_lower = k.lower()
        if k_lower not in seen:
            seen.add(k_lower)
            unique.append(k)
    return unique[:6]


def normalize_query(
    raw_query: str,
    embedding_service,
    llm,
) -> Tuple[str, str, List[str], Optional[QueryIntent]]:
    cleaned = _basic_clean(raw_query)
    cleaned = _apply_slang_map(cleaned)
    logger.info(f"After basic clean: '{raw_query}' -> '{cleaned}'")

    intent = build_query_intent(cleaned)

    pubmed_query = _llm_normalize(cleaned, llm, intent)
    core_keywords = _extract_core_keywords(pubmed_query, intent)
    if not core_keywords:
        fallback = [word for word in pubmed_query.lower().split() if len(word) > 3]
        core_keywords = fallback[:1] if fallback else []

    logger.info(f"Core keywords for domain filter: {core_keywords}")
    return pubmed_query, pubmed_query, core_keywords, intent
