"""
Evidence ranker — pure scoring logic.
No external calls. Takes Claims + Paper metadata, returns RankedEvidence list.

Scoring formula:
    score = base_weight + log(sample_size + 1)

Where base_weight is stance-dependent:
    SUPPORT  → 1.0
    OPPOSE   → 1.0
    NEUTRAL  → 0.5

If sample_size is unknown, a fallback base_weight of 0.75 is used.
"""

from __future__ import annotations
import math
from typing import Dict, List

from db.schemas import Claim, Paper, RankedEvidence, Stance
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE_WEIGHT: Dict[Stance, float] = {
    Stance.SUPPORT: 1.0,
    Stance.OPPOSE: 1.0,
    Stance.NEUTRAL: 0.5,
}
_FALLBACK_BASE = 0.75


def _score(claim: Claim, paper_map: Dict[str, Paper]) -> float:
    paper = paper_map.get(claim.paper_pmid)
    sample_size = paper.sample_size if paper else None
    base = _BASE_WEIGHT.get(claim.stance, _FALLBACK_BASE)

    if sample_size is not None and sample_size > 0:
        return base + math.log(sample_size + 1)
    return _FALLBACK_BASE


def rank_evidence(claims: List[Claim], papers: List[Paper]) -> List[RankedEvidence]:
    """Score every claim and return list sorted descending by score."""
    paper_map: Dict[str, Paper] = {p.pmid: p for p in papers}

    ranked = [
        RankedEvidence(claim=claim, score=_score(claim, paper_map))
        for claim in claims
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)

    if ranked:
        logger.debug(
            f"Evidence ranked. Top score={ranked[0].score:.2f} "
            f"(PMID {ranked[0].claim.paper_pmid})"
        )
    else:
        logger.debug("No evidence to rank.")

    return ranked
