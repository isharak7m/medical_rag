"""
Decision engine for verdict and confidence.

Direct evidence is scored separately from neutral/indirect evidence so
neutral papers do not automatically drag a supportive answer to WEAK.
Confidence is based on relevance quality and directness.
"""

from __future__ import annotations

from typing import List

from app.config import get_settings
from db.schemas import Confidence, ContradictionReport, DecisionResult, RankedEvidence, Stance, Verdict, RelevanceLabel
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_RETRIEVAL_QUALITY_THRESHOLD = 0.10


def run(
    ranked: List[RankedEvidence],
    contradiction: ContradictionReport,
    retrieval_score: float = 1.0,
) -> DecisionResult:
    if not ranked:
        return DecisionResult(
            verdict=Verdict.WEAK,
            confidence=Confidence.LOW,
            support_score=0.0,
            oppose_score=0.0,
            total_score=0.0,
            support_ratio=0.0,
            oppose_ratio=0.0,
        )

    support_items = [item for item in ranked if item.claim.stance == Stance.SUPPORT]
    oppose_items = [item for item in ranked if item.claim.stance == Stance.OPPOSE]
    neutral_items = [item for item in ranked if item.claim.stance == Stance.NEUTRAL]

    support_score = sum(item.score for item in support_items)
    oppose_score = sum(item.score for item in oppose_items)
    neutral_score = sum(item.score for item in neutral_items)
    total_score = support_score + oppose_score + neutral_score
    direct_score = support_score + oppose_score

    support_ratio = support_score / direct_score if direct_score > 0 else 0.0
    oppose_ratio = oppose_score / direct_score if direct_score > 0 else 0.0
    neutral_ratio = neutral_score / total_score if total_score > 0 else 1.0

    support_count = len(support_items)
    oppose_count = len(oppose_items)
    direct_count = support_count + oppose_count

    directly_relevant_items = [
        item for item in ranked
        if getattr(item.claim, 'relevance_label', None) == RelevanceLabel.DIRECTLY_RELEVANT
    ]
    indirectly_relevant_items = [
        item for item in ranked
        if getattr(item.claim, 'relevance_label', None) == RelevanceLabel.INDIRECTLY_RELEVANT
    ]
    directly_relevant_count = len(directly_relevant_items)
    indirectly_relevant_count = len(indirectly_relevant_items)
    total_relevant = directly_relevant_count + indirectly_relevant_count
    direct_relevance_ratio = directly_relevant_count / total_relevant if total_relevant > 0 else 0.0

    avg_relevance_score = 0.0
    if ranked:
        relevance_scores = [
            getattr(item.claim, 'relevance_score', item.score)
            for item in ranked
        ]
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores)

    if contradiction.has_conflict:
        verdict = Verdict.CONFLICTED
    elif direct_count == 0:
        verdict = Verdict.WEAK
    elif support_ratio >= 0.80 and support_count >= 2:
        verdict = Verdict.STRONG_SUPPORT
    elif support_ratio >= 0.60 and support_count >= 1:
        verdict = Verdict.MODERATE_SUPPORT
    else:
        verdict = Verdict.WEAK

    if retrieval_score < _RETRIEVAL_QUALITY_THRESHOLD:
        confidence = Confidence.LOW
    elif verdict == Verdict.CONFLICTED:
        confidence = Confidence.LOW
    elif verdict == Verdict.STRONG_SUPPORT and support_count >= 3 and neutral_ratio < 0.60:
        if direct_relevance_ratio < 0.30:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.HIGH
    elif verdict in (Verdict.STRONG_SUPPORT, Verdict.MODERATE_SUPPORT) and support_count >= 1:
        if directly_relevant_count == 0:
            confidence = Confidence.LOW
        elif direct_relevance_ratio < 0.50:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    if directly_relevant_count == 0 and indirectly_relevant_count > 0:
        confidence = Confidence.LOW
        logger.info("Capping confidence to LOW: no directly relevant evidence")

    result = DecisionResult(
        verdict=verdict,
        confidence=confidence,
        support_score=round(support_score, 4),
        oppose_score=round(oppose_score, 4),
        total_score=round(total_score, 4),
        support_ratio=round(support_ratio, 4),
        oppose_ratio=round(oppose_ratio, 4),
    )

    logger.info(
        f"Decision: verdict={verdict.value} | confidence={confidence.value} | "
        f"support_ratio={support_ratio:.2%} | direct_count={direct_count} | "
        f"directly_relevant={directly_relevant_count} | indirectly_relevant={indirectly_relevant_count} | "
        f"retrieval_score={retrieval_score:.2%}"
    )
    return result
