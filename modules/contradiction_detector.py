"""
Detect whether evidence is genuinely mixed rather than merely containing a minority dissent.
"""

from __future__ import annotations

from typing import List

from db.schemas import ContradictionReport, RankedEvidence, Stance
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_PAPERS_FOR_CERTAINTY = 4
_MEANINGFUL_MINORITY_RATIO = 0.25


def detect_contradictions(
    ranked: List[RankedEvidence],
    retrieval_score: float = 1.0,
) -> ContradictionReport:
    if not ranked:
        return ContradictionReport(
            support_count=0,
            oppose_count=0,
            neutral_count=0,
            has_conflict=False,
            scope_note="No evidence retrieved; contradiction analysis unavailable.",
        )

    support = sum(1 for item in ranked if item.claim.stance == Stance.SUPPORT)
    oppose = sum(1 for item in ranked if item.claim.stance == Stance.OPPOSE)
    neutral = sum(1 for item in ranked if item.claim.stance == Stance.NEUTRAL)
    total = len(ranked)
    direct_total = support + oppose

    minority_ratio = min(support, oppose) / direct_total if direct_total else 0.0
    has_conflict = support >= 1 and oppose >= 1 and minority_ratio >= _MEANINGFUL_MINORITY_RATIO

    low_retrieval = retrieval_score < 0.10
    small_sample = total < _MIN_PAPERS_FOR_CERTAINTY

    if has_conflict:
        scope_note = (
            f"{support} studies support and {oppose} oppose this claim. "
            "The evidence is genuinely mixed across study conditions."
        )
        logger.info(f"Contradiction detected: support={support}, oppose={oppose}, neutral={neutral}")
    elif oppose >= 1 and support == 0:
        scope_note = (
            f"{oppose} studies oppose this claim and {neutral} are neutral. "
            "No direct supporting evidence was found in this search."
        )
    elif support >= 1 and oppose == 0:
        scope_note = (
            f"{support} studies support this claim and {neutral} are neutral or indirect. "
            "No directly opposing evidence was found in this search."
        )
    elif support >= 1 and oppose >= 1:
        scope_note = (
            f"Evidence is directionally leaning one way ({support} support, {oppose} oppose), "
            "but a minority of studies disagree."
        )
    elif low_retrieval or small_sample:
        scope_note = (
            f"No contradictions found in the {total} retrieved studies. "
            "Retrieval scope was limited; contradicting evidence may exist outside this search."
        )
    else:
        scope_note = (
            f"No contradictions detected across {total} retrieved studies. "
            "Evidence is broadly consistent within this search scope."
        )

    return ContradictionReport(
        support_count=support,
        oppose_count=oppose,
        neutral_count=neutral,
        has_conflict=has_conflict,
        scope_note=scope_note,
    )
