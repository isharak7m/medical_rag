"""
Writing Agent — generates structured text: literature reviews, summaries, drafts.
Uses LLM for multi-document synthesis with structured prompts.
"""

from __future__ import annotations

import json
from typing import Callable, List

from agents.orchestrator import AgentResult, AgentRole, AgentTask, BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class WritingAgent(BaseAgent):
    role = AgentRole.WRITING

    def __init__(self, llm_generate: Callable = None, **kwargs):
        super().__init__(llm_generate, **kwargs)

    async def execute(self, task: AgentTask) -> AgentResult:
        try:
            task_type = task.task_type.value
            if task_type == "writing_draft":
                return await self._generate_draft(task)
            return await self._generate_summary(task)
        except Exception as exc:
            logger.error(f"WritingAgent failed: {exc}")
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )

    async def _generate_draft(self, task: AgentTask) -> AgentResult:
        query = task.input_data.get("query", "")
        papers = task.input_data.get("papers", [])
        claims = task.input_data.get("claims", [])
        evidence = task.input_data.get("evidence", {})
        draft_type = task.input_data.get("draft_type", "literature_review")

        if not self._llm:
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error="No LLM available for writing",
            )

        paper_summaries = []
        for p in papers[:20]:
            paper_summaries.append(
                f"- {p.get('title', 'Unknown')} (PMID: {p.get('pmid', 'N/A')}): "
                f"{p.get('abstract', 'No abstract')[:200]}"
            )

        claim_text = ""
        for c in claims[:15]:
            stance = c.get("stance", "neutral")
            claim_text += f"  [{stance.upper()}] {c.get('claim_text', '')}\n"

        if draft_type == "literature_review":
            prompt = (
                f"You are a biomedical research writer. Write a literature review on: {query}\n\n"
                f"PAPERS RETRIEVED ({len(papers)} total):\n"
                + "\n".join(paper_summaries[:15])
                + f"\n\nEXTRACTED CLAIMS:\n{claim_text}\n"
                f"EVIDENCE SUMMARY: {json.dumps(evidence.get('summary', {}))}\n\n"
                "Write a structured literature review with:\n"
                "1. Introduction (1-2 sentences)\n"
                "2. Key Findings (3-5 bullet points)\n"
                "3. Conflicting Evidence (if any)\n"
                "4. Gaps in Literature\n"
                "5. Conclusion (2-3 sentences)\n\n"
                "Be precise, cite PMIDs where possible, and note when evidence is limited."
            )
        elif draft_type == "research_note":
            prompt = (
                f"Write a concise research note on: {query}\n\n"
                "Key evidence:\n" + "\n".join(
                    f"- {c.get('claim_text', '')} ({c.get('stance', 'neutral')})"
                    for c in claims[:10]
                )
                + "\n\nFormat as a structured note with sections: Overview, Evidence, Implications, Open Questions."
            )
        else:
            prompt = (
                f"Write a brief summary on: {query}\n\n"
                "Based on:\n" + "\n".join(paper_summaries[:10])
                + "\n\nProvide a clear, evidence-based summary in 3-5 paragraphs."
            )

        try:
            result = self._llm(prompt)
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "draft": result,
                    "draft_type": draft_type,
                    "query": query,
                    "papers_used": len(papers[:15]),
                },
            )
        except Exception as exc:
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=f"LLM writing failed: {exc}",
            )

    async def _generate_summary(self, task: AgentTask) -> AgentResult:
        papers = task.input_data.get("papers", [])
        query = task.input_data.get("query", "")

        if not self._llm:
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error="No LLM available",
            )

        paper_list = "\n".join(
            f"- {p.get('title', 'Unknown')}: {p.get('abstract', '')[:150]}"
            for p in papers[:15]
        )

        prompt = (
            f"Summarize the research consensus on: {query}\n\n"
            f"Papers:\n{paper_list}\n\n"
            "Provide a 2-3 paragraph synthesis covering consensus views and disagreements."
        )

        try:
            result = self._llm(prompt)
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={"summary": result, "query": query},
            )
        except Exception as exc:
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )
