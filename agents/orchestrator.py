"""
Multi-agent orchestrator — coordinates specialized agents for complex tasks.
Uses a graph-based flow pattern where the orchestrator routes to the appropriate agent(s).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class AgentRole(str, Enum):
    RETRIEVAL = "retrieval"
    EVIDENCE = "evidence"
    WRITING = "writing"
    CITATION = "citation"
    KNOWLEDGE = "knowledge"
    LITERATURE_REVIEW = "literature_review"
    ORCHESTRATOR = "orchestrator"


class TaskType(str, Enum):
    QUERY = "query"
    LITERATURE_REVIEW = "literature_review"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    WRITING_DRAFT = "writing_draft"
    CITATION_FORMAT = "citation_format"
    KNOWLEDGE_GRAPH_BUILD = "knowledge_graph_build"
    CONTRADICTION_ANALYSIS = "contradiction_analysis"


@dataclass
class AgentTask:
    task_id: str
    task_type: TaskType
    input_data: dict
    assigned_agent: Optional[AgentRole] = None
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    agent_role: AgentRole
    task_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base class for all specialized agents."""

    role: AgentRole = AgentRole.ORCHESTRATOR

    def __init__(self, llm_generate: Callable = None, **services):
        self._llm = llm_generate
        self._services = services

    async def execute(self, task: AgentTask) -> AgentResult:
        raise NotImplementedError

    def _parse_json_response(self, text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return text


class Orchestrator:
    """
    Multi-agent orchestrator that coordinates specialized agents.
    Routes tasks to agents based on task type, manages dependencies,
    and aggregates results.
    """

    def __init__(self, llm_generate: Callable = None, **services):
        self._llm = llm_generate
        self._services = services
        self._agents: Dict[AgentRole, BaseAgent] = {}
        self._task_queue: List[AgentTask] = []
        self._results: Dict[str, AgentResult] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.role] = agent
        logger.info(f"Registered agent: {agent.role.value}")

    async def route(self, task_type: TaskType, input_data: dict) -> dict:
        """
        Route a high-level task to the appropriate agent(s).
        Returns aggregated results.
        """
        logger.info(f"Orchestrator routing task: {task_type.value}")

        if task_type == TaskType.QUERY:
            return await self._handle_query(input_data)
        elif task_type == TaskType.LITERATURE_REVIEW:
            return await self._handle_literature_review(input_data)
        elif task_type == TaskType.EVIDENCE_ANALYSIS:
            return await self._handle_evidence_analysis(input_data)
        elif task_type == TaskType.WRITING_DRAFT:
            return await self._handle_writing_draft(input_data)
        elif task_type == TaskType.KNOWLEDGE_GRAPH_BUILD:
            return await self._handle_knowledge_graph(input_data)
        elif task_type == TaskType.CONTRADICTION_ANALYSIS:
            return await self._handle_contradiction_analysis(input_data)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    async def _handle_query(self, data: dict) -> dict:
        pipeline = self._services.get("pipeline")
        if pipeline:
            result = await pipeline.run_rich(data.get("query", ""))
            if hasattr(result, "dict"):
                return result.dict()
            return result if isinstance(result, dict) else {"result": str(result)}
        return {"error": "Pipeline service not available"}

    async def _handle_literature_review(self, data: dict) -> dict:
        agent = self._agents.get(AgentRole.LITERATURE_REVIEW)
        if not agent:
            return {"error": "Literature review agent not registered"}

        task = AgentTask(
            task_id="lit_review_001",
            task_type=TaskType.LITERATURE_REVIEW,
            input_data=data,
        )
        result = await agent.execute(task)
        return result.data if result.success else {"error": result.error}

    async def _handle_evidence_analysis(self, data: dict) -> dict:
        agent = self._agents.get(AgentRole.EVIDENCE)
        if not agent:
            return {"error": "Evidence agent not registered"}

        task = AgentTask(
            task_id="evidence_001",
            task_type=TaskType.EVIDENCE_ANALYSIS,
            input_data=data,
        )
        result = await agent.execute(task)
        return result.data if result.success else {"error": result.error}

    async def _handle_writing_draft(self, data: dict) -> dict:
        agent = self._agents.get(AgentRole.WRITING)
        if not agent:
            return {"error": "Writing agent not registered"}

        task = AgentTask(
            task_id="writing_001",
            task_type=TaskType.WRITING_DRAFT,
            input_data=data,
        )
        result = await agent.execute(task)
        return result.data if result.success else {"error": result.error}

    async def _handle_knowledge_graph(self, data: dict) -> dict:
        agent = self._agents.get(AgentRole.KNOWLEDGE)
        if not agent:
            return {"error": "Knowledge graph agent not registered"}

        task = AgentTask(
            task_id="kg_001",
            task_type=TaskType.KNOWLEDGE_GRAPH_BUILD,
            input_data=data,
        )
        result = await agent.execute(task)
        return result.data if result.success else {"error": result.error}

    async def _handle_contradiction_analysis(self, data: dict) -> dict:
        pipeline = self._services.get("pipeline")
        if not pipeline:
            return {"error": "Pipeline not available"}

        query = data.get("query", "")
        result = await pipeline.run_rich(query)
        if hasattr(result, "dict"):
            result = result.dict()
        return {
            "contradiction": result.get("contradiction", {}),
            "evidence_cards": result.get("evidence_cards", []),
            "claim_links": result.get("claim_links", []),
        }

    def get_registered_agents(self) -> List[str]:
        return [role.value for role in self._agents.keys()]
