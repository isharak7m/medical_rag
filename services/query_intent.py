"""
Extract structured query components from natural language biomedical questions.
Identifies intervention, outcome, population, and question type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


_STOPWORDS = {
    "is", "are", "does", "do", "can", "could", "should", "would", "will",
    "the", "a", "an", "for", "of", "in", "to", "on", "with", "and", "or",
    "humans", "human", "adults", "adult", "people", "person", "use", "using",
    "it", "this", "that", "what", "how", "why", "when", "where", "which",
    "mandatory", "necessary",
    "effective", "helpful", "useful", "important", "significant",
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

_SAFETY_FOCUS = {
    "safety", "safe", "health", "healthy", "risk", "risks", "harm", "harms",
    "adverse", "toxicity", "mortality", "complication", "complications",
    "side", "effects", "tolerated", "unsafe", "dangerous",
}

_EFFICACY_FOCUS = {
    "improve", "improves", "improved", "benefit", "benefits", "effective",
    "efficacy", "work", "works", "response", "outcomes", "performance",
    "strength", "recovery", "growth", "mass", "mandatory", "necessary",
    "needed", "required", "increase", "decrease", "reduce", "enhance",
}

_OVERALL_VALUE_FOCUS = {
    "benefits", "risks", "adverse", "safety", "effectiveness", "performance",
    "health", "outcomes", "harms", "value",
}

_OUTCOME_CUES = {
    "hypertrophy", "strength", "mass", "lean", "muscle", "growth",
    "testosterone", "estrogen", "hormone", "cortisol", "insulin",
    "fat", "loss", "weight", "bmi", "body", "composition",
    "endurance", "performance", "power", "speed", "recovery",
    "depression", "anxiety", "mood", "cognitive", "memory", "focus",
    "cardiovascular", "heart", "blood", "pressure", "cholesterol",
    "diabetes", "glucose", "insulin", "metabolic",
    "pain", "inflammation", "joint", "bone", "density",
    "immune", "immunity", "infection",
    "fatigue", "energy", "tiredness",
    "mortality", "death", "survival",
    "grip", "press", "bench", "squat", "deadlift", "chest",
    "contraction", "force", "torque", "activation",
    "safety", "adverse", "side",
    "bioavailability", "absorption", "metabolism",
}

_POPULATION_CUES = {
    "human", "humans", "adult", "adults", "male", "female", "men", "women",
    "young", "elderly", "old", "athlete", "athletes", "trained", "untrained",
    "patient", "patients", "healthy", "clinical", "overweight", "obese",
    "lean", "normal", "type 2", "type 1", "older", "younger",
}

_QUESTION_TYPE_CUES = {
    "safety": ["safe", "safety", "risk", "dangerous", "toxic", "harm", "side effect"],
    "efficacy": ["effective", "efficacy", "improve", "help", "benefit", "work"],
    "necessity": ["mandatory", "necessary", "needed", "required", "essential"],
    "comparison": ["best", "better", "worse", "vs", "compared", "versus"],
    "association": ["association", "correlation", "relationship", "link", "associated"],
    "mechanism": ["mechanism", "how", "why", "pathway", "biological"],
}


@dataclass
class QueryIntent:
    raw_query: str
    cleaned_query: str
    intervention: str = ""
    outcome: str = ""
    population: str = ""
    question_type: str = "general"
    polarity: str = "general"
    hypothesis_direction: str = "positive"
    subject_terms: list[str] = field(default_factory=list)
    subject_aliases: set[str] = field(default_factory=set)
    exclusion_terms: set[str] = field(default_factory=set)
    focus_terms: set[str] = field(default_factory=set)
    intervention_terms: list[str] = field(default_factory=list)
    outcome_terms: list[str] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


def _normalize_subject(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _extract_intervention(tokens: list[str]) -> list[str]:
    """Extract likely intervention/supplement/drug terms."""
    intervention_stop = {
        "is", "are", "does", "do", "can", "the", "a", "an", "for", "of",
        "in", "to", "on", "with", "and", "or", "good", "bad", "best",
        "mandatory", "necessary", "effective", "helpful", "safe", "risk",
        "healthy", "improve", "increase", "decrease", "reduce",
    }
    result = []
    for t in tokens:
        if t not in intervention_stop and len(t) > 2:
            result.append(t)
    return result[:4]


def _extract_outcome(tokens: list[str]) -> list[str]:
    """Extract likely outcome/measurement terms."""
    result = []
    for t in tokens:
        if t in _OUTCOME_CUES or any(t.startswith(cue[:6]) for cue in _OUTCOME_CUES if len(cue) > 5):
            result.append(t)
    return result[:4]


def _extract_population(text: str) -> str:
    lower = text.lower()
    for pop in ["in athletes", "in trained", "in untrained", "in elderly",
                 "in young", "in men", "in women", "in patients",
                 "in healthy adults", "in adults", "in humans"]:
        if pop in lower:
            return pop.replace("in ", "")
    if any(t in lower for t in ["human", "humans", "adult", "adults"]):
        return "humans"
    return ""


def _detect_question_type(tokens: list[str]) -> str:
    for qtype, cues in _QUESTION_TYPE_CUES.items():
        if any(cue in " ".join(tokens) for cue in cues):
            return qtype
    return "general"


def _detect_polarity_and_direction(tokens: list[str]) -> tuple[str, str]:
    content_tokens = [t for t in tokens if t not in _STOPWORDS]
    positive_hits = [t for t in content_tokens if t in _POSITIVE_QUERY_CUES]
    negative_hits = [t for t in content_tokens if t in _NEGATIVE_QUERY_CUES]

    has_value_cues = any(t in _OVERALL_VALUE_FOCUS for t in content_tokens)
    has_safety_cues = any(t in _SAFETY_FOCUS for t in content_tokens if t not in {"good", "bad"})
    has_efficacy_cues = any(t in _EFFICACY_FOCUS for t in content_tokens if t not in {"good", "bad"})

    has_good_bad = any(t in {"good", "bad"} for t in content_tokens)

    if has_value_cues or (positive_hits and negative_hits):
        polarity = "overall_value"
    elif has_safety_cues:
        polarity = "safety"
    elif has_efficacy_cues:
        polarity = "efficacy"
    elif any(t in _OUTCOME_CUES for t in content_tokens):
        polarity = "efficacy"
    elif has_good_bad and not has_safety_cues and not has_efficacy_cues:
        polarity = "overall_value"
    else:
        polarity = "general"

    direction = "positive"
    if negative_hits and not positive_hits:
        direction = "negative"

    return polarity, direction


def build_query_intent(query: str) -> QueryIntent:
    cleaned = " ".join(_tokenize(query))
    tokens = _tokenize(query)
    content_tokens = [token for token in tokens if token not in _STOPWORDS]

    intervention_terms = _extract_intervention(content_tokens)
    outcome_terms = _extract_outcome(content_tokens)
    population = _extract_population(query)
    question_type = _detect_question_type(content_tokens)
    polarity, direction = _detect_polarity_and_direction(content_tokens)

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

    intervention_str = " ".join(intervention_terms)
    outcome_str = " ".join(outcome_terms)

    focus_terms = set()
    if polarity == "safety":
        focus_terms.update(_SAFETY_FOCUS)
    elif polarity == "efficacy":
        focus_terms.update(_EFFICACY_FOCUS)
    else:
        focus_terms.update(_OUTCOME_CUES)

    residual_terms = [
        token for token in content_tokens
        if token not in subject_terms and token not in _POSITIVE_QUERY_CUES
        and token not in _NEGATIVE_QUERY_CUES
    ]
    focus_terms.update(residual_terms[:4])

    return QueryIntent(
        raw_query=query,
        cleaned_query=cleaned,
        intervention=intervention_str,
        outcome=outcome_str,
        population=population,
        question_type=question_type,
        polarity=polarity,
        hypothesis_direction=direction,
        subject_terms=subject_terms,
        subject_aliases=set(subject_terms),
        exclusion_terms=set(),
        focus_terms=focus_terms,
        intervention_terms=intervention_terms,
        outcome_terms=outcome_terms,
    )
