from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.decision_engine import run as current_decision
from db.schemas import Claim, Confidence, ContradictionReport, RankedEvidence, Stance, Verdict


@dataclass
class Case:
    name: str
    ranked: list[RankedEvidence]
    contradiction: ContradictionReport
    expected: Verdict


def _legacy_decision(ranked: list[RankedEvidence], contradiction: ContradictionReport) -> Verdict:
    support_score = sum(item.score for item in ranked if item.claim.stance == Stance.SUPPORT)
    oppose_score = sum(item.score for item in ranked if item.claim.stance == Stance.OPPOSE)
    total_score = sum(item.score for item in ranked)
    support_ratio = support_score / total_score if total_score else 0.0

    if contradiction.has_conflict:
        return Verdict.CONFLICTED
    if support_ratio >= 0.70:
        return Verdict.STRONG_SUPPORT
    if support_ratio >= 0.40:
        return Verdict.MODERATE_SUPPORT
    return Verdict.WEAK


def _claim(pmid: str, stance: Stance) -> Claim:
    return Claim(paper_pmid=pmid, paper_title=pmid, claim_text=pmid, stance=stance)


def _case_support_with_neutrals() -> Case:
    ranked = [
        RankedEvidence(claim=_claim("1", Stance.SUPPORT), score=3.0),
        RankedEvidence(claim=_claim("2", Stance.NEUTRAL), score=2.0),
        RankedEvidence(claim=_claim("3", Stance.NEUTRAL), score=2.0),
        RankedEvidence(claim=_claim("4", Stance.NEUTRAL), score=1.5),
    ]
    contradiction = ContradictionReport(
        support_count=1, oppose_count=0, neutral_count=3, has_conflict=False, scope_note=""
    )
    return Case("support_with_many_neutrals", ranked, contradiction, Verdict.MODERATE_SUPPORT)


def _case_strong_support() -> Case:
    ranked = [
        RankedEvidence(claim=_claim("1", Stance.SUPPORT), score=4.0),
        RankedEvidence(claim=_claim("2", Stance.SUPPORT), score=3.5),
        RankedEvidence(claim=_claim("3", Stance.SUPPORT), score=2.8),
        RankedEvidence(claim=_claim("4", Stance.NEUTRAL), score=1.0),
    ]
    contradiction = ContradictionReport(
        support_count=3, oppose_count=0, neutral_count=1, has_conflict=False, scope_note=""
    )
    return Case("clear_strong_support", ranked, contradiction, Verdict.STRONG_SUPPORT)


def _case_conflicted() -> Case:
    ranked = [
        RankedEvidence(claim=_claim("1", Stance.SUPPORT), score=3.0),
        RankedEvidence(claim=_claim("2", Stance.OPPOSE), score=2.9),
        RankedEvidence(claim=_claim("3", Stance.NEUTRAL), score=1.0),
    ]
    contradiction = ContradictionReport(
        support_count=1, oppose_count=1, neutral_count=1, has_conflict=True, scope_note=""
    )
    return Case("conflicted_evidence", ranked, contradiction, Verdict.CONFLICTED)


def _case_no_direct_evidence() -> Case:
    ranked = [
        RankedEvidence(claim=_claim("1", Stance.NEUTRAL), score=2.5),
        RankedEvidence(claim=_claim("2", Stance.NEUTRAL), score=2.2),
    ]
    contradiction = ContradictionReport(
        support_count=0, oppose_count=0, neutral_count=2, has_conflict=False, scope_note=""
    )
    return Case("all_neutral", ranked, contradiction, Verdict.WEAK)


def _case_negative_hypothesis_supported_by_harm() -> Case:
    ranked = [
        RankedEvidence(claim=_claim("1", Stance.SUPPORT), score=3.2),
        RankedEvidence(claim=_claim("2", Stance.SUPPORT), score=2.8),
        RankedEvidence(claim=_claim("3", Stance.OPPOSE), score=1.0),
    ]
    contradiction = ContradictionReport(
        support_count=2, oppose_count=1, neutral_count=0, has_conflict=True, scope_note=""
    )
    return Case("negative_hypothesis_harm_signal", ranked, contradiction, Verdict.CONFLICTED)


def _score(verdict: Verdict, expected: Verdict) -> int:
    return int(verdict == expected)


def main() -> None:
    cases = [
        _case_support_with_neutrals(),
        _case_strong_support(),
        _case_conflicted(),
        _case_no_direct_evidence(),
        _case_negative_hypothesis_supported_by_harm(),
    ]

    legacy_hits = 0
    current_hits = 0

    for case in cases:
        legacy = _legacy_decision(case.ranked, case.contradiction)
        current = current_decision(case.ranked, case.contradiction, retrieval_score=1.0).verdict
        legacy_hits += _score(legacy, case.expected)
        current_hits += _score(current, case.expected)
        print(
            f"{case.name}: expected={case.expected.value} | legacy={legacy.value} | current={current.value}"
        )

    print("---")
    print(f"legacy_score={legacy_hits}/{len(cases)}")
    print(f"current_score={current_hits}/{len(cases)}")
    print(f"improvement={current_hits - legacy_hits}")


if __name__ == "__main__":
    main()
