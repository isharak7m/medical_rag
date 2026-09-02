"""
Format pipeline outputs into a RichQueryResponse.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from db.schemas import (
    Claim,
    ClaimLink,
    Confidence,
    ConfidenceDetail,
    ContradictionReport,
    ContradictionSummary,
    DecisionResult,
    EvidenceCard,
    Paper,
    PipelineDiagnostics,
    RichQueryResponse,
    Stance,
    Verdict,
    RelevanceLabel,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _confidence_score(decision: DecisionResult) -> int:
    if decision.confidence == Confidence.LOW:
        return max(10, round(decision.support_ratio * 40))
    if decision.confidence == Confidence.MEDIUM:
        return max(41, min(69, round(decision.support_ratio * 100)))
    return max(70, min(99, round(decision.support_ratio * 100)))


_CONFIDENCE_EXPLANATION = {
    Confidence.HIGH: "Strong majority of retrieved studies support this finding.",
    Confidence.MEDIUM: "Moderate evidence with some variability across studies.",
    Confidence.LOW: "Limited or conflicting evidence. Interpret with caution.",
}

_VERDICT_EVIDENCE_STRENGTH = {
    Verdict.STRONG_SUPPORT: "Strong",
    Verdict.MODERATE_SUPPORT: "Moderate",
    Verdict.CONFLICTED: "Conflicted",
    Verdict.WEAK: "Weak",
}


def _parse_llm_output(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("LLM output could not be parsed as JSON; using raw text as summary")
    return {"final_answer": "", "key_claims": [], "summary": raw.strip()}


def _build_evidence_cards(
    papers_with_scores: List[Tuple[Paper, float]],
    claims: List[Claim],
) -> List[EvidenceCard]:
    claim_map = {claim.paper_pmid: claim for claim in claims}
    scores = [score for _, score in papers_with_scores]
    max_score = max(scores) if scores else 1.0

    cards: List[EvidenceCard] = []
    for paper, score in papers_with_scores:
        claim = claim_map.get(paper.pmid)
        cards.append(
            EvidenceCard(
                pmid=paper.pmid,
                title=paper.title,
                claim_text=claim.claim_text if claim else paper.abstract[:200],
                stance=claim.stance if claim else Stance.NEUTRAL,
                relevance_score=round(score / max_score, 3) if max_score > 0 else 0.0,
                sample_size=paper.sample_size,
                relevance_label=claim.relevance_label if claim else RelevanceLabel.DIRECTLY_RELEVANT,
            )
        )
    return cards


def _build_contradiction_summary(report: ContradictionReport) -> ContradictionSummary:
    if report.has_conflict:
        explanation = (
            f"{report.support_count} studies support and {report.oppose_count} oppose this claim. "
            "The evidence is divided across study conditions."
        )
    elif report.oppose_count == 0 and report.support_count == 0:
        explanation = "No evidence retrieved; contradiction analysis unavailable."
    elif report.oppose_count > 0:
        explanation = (
            f"Predominantly supportive evidence ({report.support_count} studies) with "
            f"{report.oppose_count} opposing study/studies."
        )
    else:
        explanation = f"{report.support_count} supporting, {report.neutral_count} neutral studies retrieved."

    return ContradictionSummary(
        has_conflict=report.has_conflict,
        support_count=report.support_count,
        oppose_count=report.oppose_count,
        neutral_count=report.neutral_count,
        explanation=explanation,
        scope_note=report.scope_note,
    )


def format_response(
    query: str,
    raw_llm_output: str,
    claims: List[Claim],
    papers_with_scores: List[Tuple[Paper, float]],
    decision: DecisionResult,
    contradiction: ContradictionReport,
    diagnostics: PipelineDiagnostics | None = None,
) -> RichQueryResponse:
    parsed = _parse_llm_output(raw_llm_output)

    final_answer = parsed.get("final_answer", "").strip()
    key_claims_raw: List[str] = parsed.get("key_claims", [])
    summary = parsed.get("summary", raw_llm_output).strip()

    if not final_answer and claims:
        final_answer = claims[0].claim_text
    elif not final_answer:
        final_answer = summary[:200]

    evidence_cards = _build_evidence_cards(papers_with_scores, claims)
    claim_links: List[ClaimLink] = [
        ClaimLink(
            claim_text=claim.claim_text,
            stance=claim.stance,
            paper_title=claim.paper_title,
            pmid=claim.paper_pmid,
        )
        for claim in claims
    ]

    for key_claim in key_claims_raw:
        if key_claim.strip():
            claim_links.append(
                ClaimLink(
                    claim_text=key_claim.strip(),
                    stance=Stance.NEUTRAL,
                    paper_title="",
                    pmid="",
                )
            )

    return RichQueryResponse(
        query=query,
        final_answer=final_answer,
        summary=summary,
        confidence=ConfidenceDetail(
            label=decision.confidence,
            score=_confidence_score(decision),
            explanation=_CONFIDENCE_EXPLANATION.get(decision.confidence, ""),
        ),
        verdict=decision.verdict,
        evidence_strength=_VERDICT_EVIDENCE_STRENGTH.get(decision.verdict, "Unknown"),
        evidence_cards=evidence_cards,
        claim_links=claim_links,
        contradiction=_build_contradiction_summary(contradiction),
        diagnostics=diagnostics or PipelineDiagnostics(),
    )
