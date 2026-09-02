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
    intent=None,
) -> str:
    claims_block    = _format_claims(claims)
    evidence_summary = _format_evidence_summary(ranked, decision)
    support_count = sum(1 for item in ranked if item.claim.stance.value == "support")
    oppose_count = sum(1 for item in ranked if item.claim.stance.value == "oppose")
    neutral_count = sum(1 for item in ranked if item.claim.stance.value == "neutral")
    conflict_note = "Yes" if oppose_count > 0 and support_count > 0 else "No"

    direct_evidence = [
        item for item in ranked
        if getattr(item.claim, 'relevance_label', None) and item.claim.relevance_label.value == "directly_relevant"
    ]
    indirect_evidence = [
        item for item in ranked
        if getattr(item.claim, 'relevance_label', None) and item.claim.relevance_label.value == "indirectly_relevant"
    ]

    intervention_str = ""
    outcome_str = ""
    if intent:
        intervention_str = getattr(intent, 'intervention', '') or ""
        outcome_str = getattr(intent, 'outcome', '') or ""

    context_section = ""
    if intervention_str or outcome_str:
        context_section = f"""
QUERY DECOMPOSITION:
- Intervention/Exposure: {intervention_str or "not specified"}
- Outcome/Measurement: {outcome_str or "not specified"}
- Question type: {getattr(intent, 'question_type', 'general') if intent else 'general'}
"""

    direct_count = len(direct_evidence)
    indirect_count = len(indirect_evidence)

    direct_claims_block = ""
    if direct_evidence:
        direct_claims_block = "\n".join(
            f"  {i}. [{item.claim.stance.value.upper()}] {item.claim.claim_text} (PMID: {item.claim.paper_pmid}) [DIRECTLY RELEVANT]"
            for i, item in enumerate(direct_evidence[:5], 1)
        )
    else:
        direct_claims_block = "  (no directly relevant claims found)"

    indirect_claims_block = ""
    if indirect_evidence:
        indirect_claims_block = "\n".join(
            f"  {i}. [{item.claim.stance.value.upper()}] {item.claim.claim_text} (PMID: {item.claim.paper_pmid}) [INDIRECTLY RELEVANT]"
            for i, item in enumerate(indirect_evidence[:3], 1)
        )
    else:
        indirect_claims_block = "  (no indirectly relevant claims found)"

    prompt = f"""You are a senior biomedical research analyst. Your task is to synthesize scientific evidence and answer the user query.

QUERY: {query}
{context_section}
EVIDENCE CONTEXT:
- Verdict: {decision.verdict.value}
- Overall confidence: {decision.confidence.value}
- Direct evidence support ratio: {decision.support_ratio:.0%}
- Supporting papers: {support_count}
- Opposing papers: {oppose_count}
- Neutral or indirect papers: {neutral_count}
- Conflict detected: {conflict_note}
- Directly relevant evidence: {direct_count} papers
- Indirectly relevant evidence: {indirect_count} papers

DIRECTLY RELEVANT CLAIMS (papers that directly address the query):
{direct_claims_block}

INDIRECTLY RELEVANT CLAIMS (papers that provide context but do not directly address the query):
{indirect_claims_block}

ALL CLAIMS EXTRACTED FROM PAPERS:
{claims_block}

QUANTITATIVE EVIDENCE SUMMARY:
{evidence_summary}

YOUR TASK:
Analyze the evidence above and respond ONLY with this exact JSON structure.
Do NOT include markdown, code fences, or any text outside the JSON.
Do not understate clearly supportive evidence when direct support is strong and opposing evidence is absent.
Do not overstate certainty when most papers are neutral or indirect.

CRITICAL INSTRUCTIONS:
- Answer the EXACT question asked. Do not shift to a related but different topic.
- If the user asked about a specific intervention and outcome, your answer MUST address BOTH.
- Prioritize DIRECTLY RELEVANT evidence over indirect evidence.
- If there is insufficient directly relevant evidence, say so explicitly.
- Do not use indirect evidence to make strong claims about the specific question asked.
- Each claim in key_claims must be supported by at least one paper citation (PMID).

{f"If the user asked about {{intervention_str}} and {{outcome_str}}, your answer MUST specifically address whether {{intervention_str}} affects {{outcome_str}}." if intervention_str and outcome_str else ""}

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
