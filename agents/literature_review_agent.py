"""
Literature Review Agent — synthesizes multi-document reviews.
Retrieves papers, extracts consensus and disagreements, generates structured reviews.
"""

from __future__ import annotations

import json
from typing import Callable, List

from agents.orchestrator import AgentResult, AgentRole, AgentTask, BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class LiteratureReviewAgent(BaseAgent):
    role = AgentRole.LITERATURE_REVIEW

    def __init__(self, llm_generate: Callable = None, **kwargs):
        super().__init__(llm_generate, **kwargs)

    async def execute(self, task: AgentTask) -> AgentResult:
        try:
            query = task.input_data.get("query", "")
            papers = task.input_data.get("papers", [])
            claims = task.input_data.get("claims", [])
            evidence = task.input_data.get("evidence", {})

            if not papers:
                return AgentResult(
                    agent_role=self.role,
                    task_id=task.task_id,
                    success=False,
                    error="No papers available for literature review",
                )

            review = self._generate_review(query, papers, claims, evidence)

            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "review": review,
                    "query": query,
                    "papers_reviewed": len(papers),
                    "sections": self._extract_sections(review),
                },
            )
        except Exception as exc:
            logger.error(f"LiteratureReviewAgent failed: {exc}")
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )

    def _generate_review(
        self,
        query: str,
        papers: List[dict],
        claims: List[dict],
        evidence: dict,
    ) -> str:
        paper_summaries = []
        for i, p in enumerate(papers[:20], 1):
            sample = p.get("sample_size")
            sample_note = f" (n={sample})" if sample else ""
            paper_summaries.append(
                f"{i}. {p.get('title', 'Unknown')}{sample_note} "
                f"[PMID: {p.get('pmid', 'N/A')}]\n"
                f"   Abstract: {p.get('abstract', 'N/A')[:250]}"
            )

        support_claims = [c for c in claims if c.get("stance") == "support"]
        oppose_claims = [c for c in claims if c.get("stance") == "oppose"]
        neutral_claims = [c for c in claims if c.get("stance") == "neutral"]

        if self._llm:
            prompt = (
                f"Write a comprehensive literature review on: {query}\n\n"
                f"PAPERS REVIEWED ({len(papers)} total):\n"
                + "\n".join(paper_summaries[:15])
                + f"\n\nSUPPORTING EVIDENCE ({len(support_claims)} claims):\n"
                + "\n".join(f"- {c.get('claim_text', '')}" for c in support_claims[:8])
                + f"\n\nCONTRADICTING EVIDENCE ({len(oppose_claims)} claims):\n"
                + "\n".join(f"- {c.get('claim_text', '')}" for c in oppose_claims[:8])
                + f"\n\nNEUTRAL/INCONCLUSIVE ({len(neutral_claims)} claims):\n"
                + "\n".join(f"- {c.get('claim_text', '')}" for c in neutral_claims[:5])
                + "\n\n"
                "Structure the review with these sections:\n"
                "## Executive Summary\n"
                "## Background\n"
                "## Key Findings\n"
                "### Supporting Evidence\n"
                "### Contradicting Evidence\n"
                "### Inconclusive Findings\n"
                "## Research Gaps\n"
                "## Clinical Implications\n"
                "## Conclusion\n\n"
                "Be specific about evidence quality. Note sample sizes and study types where possible."
            )
            try:
                return self._llm(prompt)
            except Exception as exc:
                logger.warning(f"LLM review generation failed: {exc}")

        return self._fallback_review(query, papers, support_claims, oppose_claims, neutral_claims)

    def _fallback_review(
        self,
        query: str,
        papers: List[dict],
        support: List[dict],
        oppose: List[dict],
        neutral: List[dict],
    ) -> str:
        sections = []
        sections.append(f"# Literature Review: {query}\n")
        sections.append(f"**Papers reviewed:** {len(papers)}\n")

        sections.append("## Executive Summary\n")
        sections.append(
            f"This review examines {len(papers)} publications on {query}. "
            f"Of the extracted claims, {len(support)} support, {len(oppose)} contradict, "
            f"and {len(neutral)} are neutral regarding the research question.\n"
        )

        if support:
            sections.append("## Supporting Evidence\n")
            for c in support[:5]:
                sections.append(f"- **{c.get('paper_title', 'Unknown')}**: {c.get('claim_text', '')}\n")

        if oppose:
            sections.append("## Contradicting Evidence\n")
            for c in oppose[:5]:
                sections.append(f"- **{c.get('paper_title', 'Unknown')}**: {c.get('claim_text', '')}\n")

        if neutral:
            sections.append("## Inconclusive Findings\n")
            for c in neutral[:3]:
                sections.append(f"- **{c.get('paper_title', 'Unknown')}**: {c.get('claim_text', '')}\n")

        sections.append("## Research Gaps\n")
        sections.append(
            "- Further large-scale randomized controlled trials are needed\n"
            "- Long-term follow-up studies are lacking\n"
            "- Population-specific effects require more investigation\n"
        )

        sections.append("## Conclusion\n")
        sections.append(
            f"The evidence base for {query} includes {len(papers)} studies with "
            f"mixed findings. While {len(support)} claims support the hypothesis, "
            f"{len(oppose)} claims present contradicting evidence. "
            f"More rigorous research is needed to resolve these discrepancies.\n"
        )

        return "\n".join(sections)

    def _extract_sections(self, review: str) -> List[str]:
        import re
        headings = re.findall(r"^##\s+(.+)$", review, re.MULTILINE)
        return headings
