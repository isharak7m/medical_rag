from __future__ import annotations

from dataclasses import dataclass, field
import re


_STOPWORDS = {
    "is", "are", "does", "do", "can", "could", "should", "would", "will",
    "the", "a", "an", "for", "of", "in", "to", "on", "with", "and", "or",
    "humans", "human", "adults", "adult", "people", "person", "use", "using",
}

_POSITIVE_QUERY_CUES = {
    "healthy", "safe", "good", "beneficial", "benefit", "benefits",
    "helpful", "effective", "works", "work", "improve", "improves",
    "better", "worth",
}

_NEGATIVE_QUERY_CUES = {
    "harmful", "unsafe", "bad", "dangerous", "risky", "risk", "risks",
    "toxic", "toxicity", "harm", "harms", "worse", "side", "effects",
}

_VALUE_JUDGMENT_CUES = {
    "good", "bad", "healthy", "harmful", "beneficial", "dangerous", "safe", "unsafe",
}

_SAFETY_FOCUS = {
    "safety", "safe", "health", "healthy", "risk", "risks", "harm", "harms",
    "adverse", "toxicity", "mortality", "complication", "complications",
    "side", "effects", "tolerated", "unsafe", "dangerous",
}

_EFFICACY_FOCUS = {
    "improve", "improves", "improved", "benefit", "benefits", "effective",
    "efficacy", "work", "works", "response", "outcomes", "performance",
    "strength", "recovery", "growth", "mass",
}

_OVERALL_VALUE_FOCUS = {
    "benefits", "risks", "adverse", "safety", "effectiveness", "performance",
    "health", "outcomes", "harms", "value",
}

_CONCEPT_REGISTRY = {
    "steroid": {
        "aliases": {"steroid", "steroids", "anabolic", "androgenic", "aas", "testosterone", "nandrolone"},
        "exclude": {"corticosteroid", "corticosteroids", "dexamethasone", "prednisone", "prednisolone", "nsaid", "nsaids"},
    },
    "vaping": {
        "aliases": {"vaping", "vape", "e-cigarette", "ecigarette", "electronic cigarette", "electronic cigarettes"},
        "exclude": set(),
    },
    "alcohol": {
        "aliases": {"alcohol", "ethanol", "drinking", "drinkers"},
        "exclude": set(),
    },
    "trt": {
        "aliases": {"trt", "testosterone replacement", "testosterone therapy", "testosterone"},
        "exclude": set(),
    },
    "gym": {
        "aliases": {"gym", "exercise", "training", "workout", "resistance training", "physical activity", "fitness"},
        "exclude": set(),
    },
    "creatine": {
        "aliases": {"creatine", "creatine monohydrate"},
        "exclude": set(),
    },
}


@dataclass
class QueryIntent:
    raw_query: str
    cleaned_query: str
    polarity: str
    hypothesis_direction: str
    subject_terms: list[str] = field(default_factory=list)
    subject_aliases: set[str] = field(default_factory=set)
    exclusion_terms: set[str] = field(default_factory=set)
    focus_terms: set[str] = field(default_factory=set)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


def _normalize_subject(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def build_query_intent(query: str) -> QueryIntent:
    cleaned = " ".join(_tokenize(query))
    tokens = _tokenize(query)
    content_tokens = [token for token in tokens if token not in _STOPWORDS]

    positive_hits = [token for token in content_tokens if token in _POSITIVE_QUERY_CUES]
    negative_hits = [token for token in content_tokens if token in _NEGATIVE_QUERY_CUES]
    has_value_judgment = any(token in _VALUE_JUDGMENT_CUES for token in content_tokens)

    if any(token in _SAFETY_FOCUS for token in content_tokens if token not in {"good", "bad"}):
        polarity = "safety"
    elif any(token in _EFFICACY_FOCUS for token in content_tokens if token not in {"good", "bad"}):
        polarity = "efficacy"
    elif has_value_judgment:
        polarity = "overall_value"
    else:
        polarity = "general"

    hypothesis_direction = "positive"
    if negative_hits and not positive_hits:
        hypothesis_direction = "negative"

    subject_terms = [
        token for token in content_tokens
        if token not in _POSITIVE_QUERY_CUES
        and token not in _NEGATIVE_QUERY_CUES
        and token not in _SAFETY_FOCUS
        and token not in _EFFICACY_FOCUS
        and token not in _OVERALL_VALUE_FOCUS
    ]
    subject_terms = [_normalize_subject(token) for token in subject_terms]
    subject_terms = list(dict.fromkeys(subject_terms))

    primary_subject = subject_terms[0] if subject_terms else ""
    registry = _CONCEPT_REGISTRY.get(primary_subject, {})

    aliases = set(subject_terms)
    if primary_subject:
        aliases.add(primary_subject)
        aliases.add(primary_subject + "s")
    aliases.update(registry.get("aliases", set()))

    exclusions = set(registry.get("exclude", set()))

    focus_terms = set()
    if polarity == "safety":
        focus_terms.update(_SAFETY_FOCUS)
    elif polarity == "efficacy":
        focus_terms.update(_EFFICACY_FOCUS)
    elif polarity == "overall_value":
        focus_terms.update(_OVERALL_VALUE_FOCUS)

    residual_terms = [
        token for token in content_tokens
        if token not in aliases and token not in _POSITIVE_QUERY_CUES and token not in _NEGATIVE_QUERY_CUES
    ]
    focus_terms.update(residual_terms[:4])

    return QueryIntent(
        raw_query=query,
        cleaned_query=cleaned,
        polarity=polarity,
        hypothesis_direction=hypothesis_direction,
        subject_terms=subject_terms,
        subject_aliases=aliases,
        exclusion_terms=exclusions,
        focus_terms=focus_terms,
    )
