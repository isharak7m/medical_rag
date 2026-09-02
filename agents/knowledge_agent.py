"""
Knowledge Graph Agent — builds and queries biomedical knowledge graphs.
Uses NER for entity extraction and relationship inference.
"""

from __future__ import annotations

from typing import Callable

from agents.orchestrator import AgentResult, AgentRole, AgentTask, BaseAgent
from services.knowledge_graph import KnowledgeGraphService
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeAgent(BaseAgent):
    role = AgentRole.KNOWLEDGE

    def __init__(self, llm_generate: Callable = None, **kwargs):
        super().__init__(llm_generate, **kwargs)
        self._kg_service = KnowledgeGraphService(llm_generate=llm_generate)

    async def execute(self, task: AgentTask) -> AgentResult:
        try:
            papers = task.input_data.get("papers", [])
            query = task.input_data.get("query", "")

            # Build knowledge graph from papers
            texts = [
                (p.get("pmid", ""), f"{p.get('title', '')} {p.get('abstract', '')}")
                for p in papers
                if p.get("abstract")
            ]

            if not texts:
                return AgentResult(
                    agent_role=self.role,
                    task_id=task.task_id,
                    success=True,
                    data={
                        "graph": {"nodes": [], "edges": []},
                        "entity_count": 0,
                        "relation_count": 0,
                        "message": "No abstracts available for knowledge graph extraction",
                    },
                )

            kg = self._kg_service.build_graph(texts, query)
            graph_dict = kg.to_dict()

            # Extract insights
            entity_types = {}
            for e in kg.entities.values():
                entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1

            relation_types = {}
            for r in kg.relations:
                relation_types[r.relation_type] = relation_types.get(r.relation_type, 0) + 1

            logger.info(
                f"KnowledgeAgent: {len(kg.entities)} entities, {len(kg.relations)} relations"
            )

            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "graph": graph_dict,
                    "entity_count": len(kg.entities),
                    "relation_count": len(kg.relations),
                    "entity_types": entity_types,
                    "relation_types": relation_types,
                    "top_entities": [
                        {
                            "id": e.id,
                            "name": e.name,
                            "type": e.entity_type,
                            "sources": len(e.source_pmids),
                        }
                        for e in sorted(
                            kg.entities.values(),
                            key=lambda x: len(x.source_pmids),
                            reverse=True,
                        )[:20]
                    ],
                },
                metadata={"model": "regex_ner" if not self._kg_service._nlp else "spacy_ner"},
            )
        except Exception as exc:
            logger.error(f"KnowledgeAgent failed: {exc}")
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )
