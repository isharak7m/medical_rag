"""
Extract one query-aware claim per paper and classify stance.
"""

from __future__ import annotations

import json
import re
from typing import List

from db.schemas import Claim, Paper, Stance, RelevanceLabel
from services.query_intent import QueryIntent, build_query_intent
from utils.logger import get_logger
from utils.text_cleaning import truncate

logger = get_logger(__name__)

_BATCH_STANCE_PROMPT = """You are a biomedical evidence classifier.

USER QUERY: {query}
QUERY POLARITY: {polarity}
QUERY HYPOTHESIS DIRECTION: {direction}
PRIMARY SUBJECT TERMS: {subject_terms}

Classify each paper for its stance relative to the user query.

Rules:
- SUPPORT: directly relevant paper and the findings support the query hypothesis
- OPPOSE: directly relevant paper and the findings contradict the query hypothesis
- NEUTRAL: indirect paper, wrong intervention/population/outcome, or insufficient directional evidence
- Distinguish the paper's raw finding polarity from the query's hypothesis polarity
- For safety/healthiness questions, harm and risk findings oppose a positive hypothesis and support a negative hypothesis
- Do not mark directly relevant reviews or meta-analyses neutral if they clearly summarize outcome evidence

Return ONLY a JSON array in the same order:
[{{"id": 1, "stance": "support|oppose|neutral", "reason": "one sentence"}}, ...]

PAPERS:
{papers_block}
"""

_POSITIVE_FINDING_PATTERNS = re.compile(
    r"\b("
    r"significant(ly)?|effective(ly)?|improve[sd]?|benefit(ed|s|ial)?|"
    r"increased?|enhanced?|favourable?|favorable|superior|beneficial|"
    r"ergogenic|accelerat(es?|ed|ing)|greater|improved performance|"
    r"improves maximal strength|increased repetitions|improved 1rm|"
    r"safe\b|well tolerated|no serious adverse events|no major harm|reduced risk|"
    r"lower mortality|protective|prevention"
    r")\b",
    re.IGNORECASE,
)

_NEGATIVE_FINDING_PATTERNS = re.compile(
    r"\b("
    r"no significant|not effective|ineffective|no benefit|no improvement|"
    r"no difference|no effect|failed? to|did not (show|demonstrate|improve|increase)|"
    r"worsened?|harmful|adverse|negative(ly)?|inferior|detrimental|unsafe|"
    r"risk|mortality|side effect|side effects|toxicity|complication|complications|"
    r"concern|concerns|disease|injury|fertility consequences|health consequences|"
    r"contraindicated|damage|abuse"
    r")\b",
    re.IGNORECASE,
)

_SENT_RE = re.compile(r"[^.!?]*[.!?]")
_CONCLUSION_INDICATORS = re.compile(
    r"\b(conclud|suggest|demonstrat|indicat|show|reveal|find|report|confirm|establish|support)\b",
    re.IGNORECASE,
)


def _extract_key_sentence(abstract: str) -> str:
    sentences = [match.group().strip() for match in _SENT_RE.finditer(abstract)]
    if not sentences:
        return truncate(abstract, 260)
    filtered = [sentence for sentence in sentences if len(sentence) > 25] or sentences
    conclusions = [sentence for sentence in filtered if _CONCLUSION_INDICATORS.search(sentence)]
    return conclusions[-1] if conclusions else filtered[-1]


def _raw_finding_polarity(text: str) -> Stance:
    negative_hits = len(_NEGATIVE_FINDING_PATTERNS.findall(text))
    stripped = re.sub(
        r"\b(no|not|never|without|fail(ed|s)?|lack(s|ed)?)\s+\w+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    positive_hits = len(_POSITIVE_FINDING_PATTERNS.findall(stripped))
    if positive_hits > 0 and negative_hits > 0 and abs(positive_hits - negative_hits) <= 1:
        return Stance.NEUTRAL
    if negative_hits > positive_hits:
        return Stance.OPPOSE
    if positive_hits > negative_hits:
        return Stance.SUPPORT
    return Stance.NEUTRAL


def _is_direct_match(intent: QueryIntent, paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract}".lower()
    if intent.exclusion_terms and any(term in text for term in intent.exclusion_terms):
        return False

    if intent.subject_aliases:
        alias_match = any(alias in text for alias in intent.subject_aliases)
        if not alias_match:
            return False

    focus_hits = sum(1 for term in intent.focus_terms if len(term) > 3 and term in text)
    if intent.polarity == "safety":
        return focus_hits >= 1
    if intent.focus_terms:
        return focus_hits >= 1 or len(intent.subject_terms) == 1
    return True


def _map_raw_to_query_stance(intent: QueryIntent, raw_finding: Stance) -> Stance:
    if raw_finding == Stance.NEUTRAL:
        return Stance.NEUTRAL
    if intent.hypothesis_direction == "positive":
        return raw_finding
    return Stance.SUPPORT if raw_finding == Stance.OPPOSE else Stance.OPPOSE


def _heuristic_query_stance(query: str, paper: Paper, intent: QueryIntent | None = None) -> Stance:
    intent = intent or build_query_intent(query)
    if not _is_direct_match(intent, paper):
        return Stance.NEUTRAL
    raw = _raw_finding_polarity(f"{paper.title} {paper.abstract}")
    return _map_raw_to_query_stance(intent, raw)


def _check_intervention_match(intent: QueryIntent, paper: Paper) -> bool:
    if not intent or not intent.intervention_terms:
        return False
    text = f"{paper.title} {paper.abstract}".lower()
    return any(term.lower() in text for term in intent.intervention_terms)


def _check_outcome_match(intent: QueryIntent, paper: Paper) -> bool:
    if not intent or not intent.outcome_terms:
        return False
    text = f"{paper.title} {paper.abstract}".lower()
    return any(term.lower() in text for term in intent.outcome_terms)


def _classify_relevance(intent: QueryIntent | None, paper: Paper, stance: Stance) -> RelevanceLabel:
    if not intent:
        return RelevanceLabel.DIRECTLY_RELEVANT

    intervention_match = _check_intervention_match(intent, paper)
    outcome_match = _check_outcome_match(intent, paper)

    if intervention_match and outcome_match:
        return RelevanceLabel.DIRECTLY_RELEVANT
    elif intervention_match or outcome_match:
        return RelevanceLabel.INDIRECTLY_RELEVANT
    else:
        return RelevanceLabel.CONTEXTUAL


def _llm_classify_stances_batch(query: str, papers: list, llm_generate) -> list[Stance]:
    intent = build_query_intent(query)
    papers_block = "\n\n".join(
        f"Paper {index}:\nTITLE: {paper.title}\nABSTRACT: {truncate(paper.abstract, 400)}"
        for index, paper in enumerate(papers, start=1)
    )
    prompt = _BATCH_STANCE_PROMPT.format(
        query=query,
        polarity=intent.polarity,
        direction=intent.hypothesis_direction,
        subject_terms=", ".join(intent.subject_terms) or "unknown",
        papers_block=papers_block,
    )

    try:
        raw = llm_generate(prompt)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        results = json.loads(raw)
        stances: list[Stance] = []
        for index, paper in enumerate(papers, start=1):
            entry = next((row for row in results if row.get("id") == index), None)
            llm_stance = Stance(str(entry.get("stance", "neutral")).lower()) if entry else Stance.NEUTRAL
            heuristic = _heuristic_query_stance(query, paper, intent)

            if heuristic == Stance.OPPOSE and llm_stance != Stance.OPPOSE:
                stance = heuristic
            elif heuristic == Stance.SUPPORT and llm_stance == Stance.NEUTRAL:
                stance = heuristic
            else:
                stance = llm_stance
            stances.append(stance)
        return stances
    except Exception as exc:
        logger.warning(f"Batch LLM stance failed: {exc} -> heuristic fallback for all papers")
        return [_heuristic_query_stance(query, paper, intent) for paper in papers]


def extract_claims(
    papers: List[Paper],
    query: str = "",
    llm_generate=None,
) -> List[Claim]:
    use_llm = bool(query and llm_generate)
    intent = build_query_intent(query) if query else None
    stances = (
        _llm_classify_stances_batch(query, papers, llm_generate)
        if use_llm
        else [_map_raw_to_query_stance(intent, _raw_finding_polarity(f"{paper.title} {paper.abstract}")) if intent else _raw_finding_polarity(f"{paper.title} {paper.abstract}") for paper in papers]
    )

    return [
        Claim(
            paper_pmid=paper.pmid,
            paper_title=paper.title,
            claim_text=_extract_key_sentence(paper.abstract),
            stance=stance,
            relevance_label=_classify_relevance(intent, paper, stance),
            intervention_match=_check_intervention_match(intent, paper) if intent else False,
            outcome_match=_check_outcome_match(intent, paper) if intent else False,
        )
        for paper, stance in zip(papers, stances)
    ]


def extract_claims_llm(papers: List[Paper], llm_generate) -> List[Claim]:
    claims: List[Claim] = []
    for paper in papers:
        prompt = (
            "Extract the main scientific claim from this abstract and classify "
            "its stance as 'support', 'oppose', or 'neutral'.\n\n"
            f"ABSTRACT:\n{truncate(paper.abstract, 800)}\n\n"
            'Respond ONLY in JSON: {"claim": "...", "stance": "support|oppose|neutral"}'
        )
        try:
            raw = llm_generate(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
            claims.append(
                Claim(
                    paper_pmid=paper.pmid,
                    paper_title=paper.title,
                    claim_text=data["claim"],
                    stance=Stance(data.get("stance", "neutral")),
                )
            )
        except Exception:
            claims.extend(extract_claims([paper]))
    return claims
