"""
Prompt builder — pure string construction.

Instructs the LLM to return a structured JSON response covering:
  - final_answer  : concise 1-2 sentence direct answer to the query
  - key_claims    : 3-5 bullet findings from evidence
  - summary       : full prose synthesis (3-5 sentences)

Domain-agnostic — works for any biomedical topic.
No I/O. No external calls.
"""

from __future__ import annotations
from typing import List

from db.schemas import Claim, DecisionResult, RankedEvidence


def build_prompt(
    query: str,
    claims: List[Claim],
    ranked: List[RankedEvidence],
    decision: DecisionResult,
) -> str:
    claims_block    = _format_claims(claims)
    evidence_summary = _format_evidence_summary(ranked, decision)
    support_count = sum(1 for item in ranked if item.claim.stance.value == "support")
    oppose_count = sum(1 for item in ranked if item.claim.stance.value == "oppose")
    neutral_count = sum(1 for item in ranked if item.claim.stance.value == "neutral")
    conflict_note = "Yes" if oppose_count > 0 and support_count > 0 else "No"

    prompt = f"""You are a senior biomedical research analyst. Your task is to synthesize scientific evidence and answer the user query.

QUERY: {query}

EVIDENCE CONTEXT:
- Verdict: {decision.verdict.value}
- Overall confidence: {decision.confidence.value}
- Direct evidence support ratio: {decision.support_ratio:.0%}
- Supporting papers: {support_count}
- Opposing papers: {oppose_count}
- Neutral or indirect papers: {neutral_count}
- Conflict detected: {conflict_note}

CLAIMS EXTRACTED FROM PAPERS:
{claims_block}

QUANTITATIVE EVIDENCE SUMMARY:
{evidence_summary}

YOUR TASK:
Analyze the evidence above and respond ONLY with this exact JSON structure.
Do NOT include markdown, code fences, or any text outside the JSON.
Do not understate clearly supportive evidence when direct support is strong and opposing evidence is absent.
Do not overstate certainty when most papers are neutral or indirect.

{{
  "final_answer": "A direct 1-2 sentence answer to the user query based strictly on the evidence provided.",
  "key_claims": [
    "Finding 1 from the evidence",
    "Finding 2 from the evidence",
    "Finding 3 from the evidence"
  ],
  "summary": "A 3-5 sentence flowing prose synthesis of the evidence. Acknowledge any conflicts or limitations. Base everything on the retrieved papers. Do not speculate."
}}"""

    return prompt


def _format_claims(claims: List[Claim]) -> str:
    if not claims:
        return "  (no claims extracted)"
    return "\n".join(
        f"  {i}. [{c.stance.value.upper()}] {c.claim_text} (PMID: {c.paper_pmid})"
        for i, c in enumerate(claims, 1)
    )


def _format_evidence_summary(ranked: List[RankedEvidence], decision: DecisionResult) -> str:
    return (
        f"  Total evidence score: {decision.total_score:.2f}\n"
        f"  Supporting score: {decision.support_score:.2f} ({decision.support_ratio:.0%})\n"
        f"  Opposing score:   {decision.oppose_score:.2f} ({decision.oppose_ratio:.0%})\n"
        f"  Papers analysed:  {len(ranked)}"
    )
