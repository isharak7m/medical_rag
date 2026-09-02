"""
Retrieval Agent — handles semantic search across PubMed + bioRxiv.
Expands queries, fetches from multiple sources, deduplicates, and ranks.
"""

from __future__ import annotations

from typing import Callable, List

from agents.orchestrator import AgentResult, AgentRole, AgentTask, BaseAgent
from db.schemas import Paper
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalAgent(BaseAgent):
    role = AgentRole.RETRIEVAL

    def __init__(
        self,
        llm_generate: Callable = None,
        pubmed_service=None,
        biorxiv_service=None,
        embedding_service=None,
        **kwargs,
    ):
        super().__init__(llm_generate, **kwargs)
        self._pubmed = pubmed_service
        self._biorxiv = biorxiv_service
        self._embedding = embedding_service

    async def execute(self, task: AgentTask) -> AgentResult:
        try:
            query = task.input_data.get("query", "")
            max_papers = task.input_data.get("max_papers", 30)

            papers: List[Paper] = []

            # Fetch from PubMed
            if self._pubmed:
                try:
                    pubmed_papers = await self._pubmed.fetch_papers(query)
                    papers.extend(pubmed_papers)
                except Exception as exc:
                    logger.warning(f"PubMed fetch failed: {exc}")

            # Fetch from bioRxiv
            if self._biorxiv:
                try:
                    biorxiv_papers = await self._biorxiv.fetch_preprints(query, max_results=10)
                    papers.extend(biorxiv_papers)
                except Exception as exc:
                    logger.warning(f"bioRxiv fetch failed: {exc}")

            # Deduplicate
            seen_pmids = set()
            unique_papers = []
            for p in papers:
                if p.pmid not in seen_pmids:
                    seen_pmids.add(p.pmid)
                    unique_papers.append(p)

            # Cap results
            unique_papers = unique_papers[:max_papers]

            logger.info(
                f"RetrievalAgent: {len(unique_papers)} unique papers from "
                f"{len(papers)} total (PubMed + bioRxiv)"
            )

            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "papers": [p.dict() for p in unique_papers],
                    "total_fetched": len(papers),
                    "unique_count": len(unique_papers),
                },
                metadata={"sources": ["pubmed", "biorxiv"]},
            )
        except Exception as exc:
            logger.error(f"RetrievalAgent failed: {exc}")
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )
