"""
Evidence Agent — classifies evidence stance using NLI model + LLM.
Handles support/neutral/contradiction classification.
"""

from __future__ import annotations

from typing import Callable, List

from agents.orchestrator import AgentResult, AgentRole, AgentTask, BaseAgent
from services.nli_classifier import NLIClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceAgent(BaseAgent):
    role = AgentRole.EVIDENCE

    def __init__(self, llm_generate: Callable = None, **kwargs):
        super().__init__(llm_generate, **kwargs)
        self._nli = NLIClassifier(llm_generate=llm_generate)

    async def execute(self, task: AgentTask) -> AgentResult:
        try:
            papers = task.input_data.get("papers", [])
            query = task.input_data.get("query", "")
            claims = task.input_data.get("claims", [])

            if not claims and papers:
                claims = [
                    {
                        "paper_pmid": p.get("pmid", ""),
                        "paper_title": p.get("title", ""),
                        "claim_text": p.get("abstract", "")[:300],
                    }
                    for p in papers
                ]

            classified = []
            for claim in claims:
                claim_text = claim.get("claim_text", "")
                if not claim_text:
                    continue
                stance, confidence = self._nli.classify_pair(query, claim_text)
                classified.append(
                    {
                        **claim,
                        "stance": stance.value,
                        "nli_confidence": confidence,
                    }
                )

            support_count = sum(1 for c in classified if c["stance"] == "support")
            oppose_count = sum(1 for c in classified if c["stance"] == "oppose")
            neutral_count = sum(1 for c in classified if c["stance"] == "neutral")

            logger.info(
                f"EvidenceAgent: classified {len(classified)} claims — "
                f"S:{support_count} O:{oppose_count} N:{neutral_count}"
            )

            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "classified_claims": classified,
                    "summary": {
                        "support": support_count,
                        "oppose": oppose_count,
                        "neutral": neutral_count,
                        "total": len(classified),
                    },
                },
                metadata={"nli_model": self._nli._model_name if self._nli._model else "llm_fallback"},
            )
        except Exception as exc:
            logger.error(f"EvidenceAgent failed: {exc}")
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )
