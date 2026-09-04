"""
API routes — thin request/response layer only.
No business logic. No direct service calls.
All work is delegated to the injected pipeline, orchestrator, or stores.

Endpoints:
  POST /query              → QueryResponse        (backward-compatible)
  POST /query/rich         → RichQueryResponse    (structured, used by dashboard)
  GET  /paper/{pmid}       → dict                 (full paper metadata)
  GET  /health             → {"status": "ok"}
  POST /agent/query        → AgentQueryResponse   (multi-agent routing)
  POST /review             → LiteratureReviewResponse
  POST /kg/build           → KnowledgeGraphResponse
  GET  /kg/graph/{query}   → KnowledgeGraphResponse
  GET  /artifacts          → List[ArtifactResponse]
  POST /artifacts          → ArtifactResponse
  GET  /artifacts/{id}     → ArtifactResponse
  PUT  /artifacts/{id}     → ArtifactResponse
  GET  /artifacts/{id}/history → List[VersionResponse]
  POST /artifacts/{id}/comment → dict
  GET  /workspaces         → List[WorkspaceResponse]
  POST /workspaces         → WorkspaceResponse
  GET  /workspaces/{id}    → WorkspaceResponse
  POST /workspaces/{id}/comment → dict
  GET  /workspaces/{id}/comments → List[dict]
  POST /citations/format   → dict
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from core.pipeline import Pipeline
from db.schemas import (
    Confidence, ConfidenceDetail, ContradictionSummary,
    QueryRequest, QueryResponse, RichQueryResponse, Verdict,
    LiteratureReviewRequest, LiteratureReviewResponse,
    KnowledgeGraphResponse, ArtifactRequest, ArtifactUpdateRequest,
    ArtifactResponse, VersionResponse, WorkspaceRequest,
    CommentRequest, WorkspaceResponse, AgentQueryRequest, AgentQueryResponse,
)
from utils.logger import get_logger
from agents.orchestrator import AgentTask, TaskType

logger = get_logger(__name__)

router = APIRouter()


def _get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _get_orchestrator(request: Request):
    return request.app.state.orchestrator


def _get_kg(request: Request):
    return request.app.state.kg_service


def _get_version_store(request: Request):
    return request.app.state.version_store


def _get_collab_store(request: Request):
    return request.app.state.collab_store


def _wrap_fallback(qr: QueryResponse) -> RichQueryResponse:
    """Convert a flat QueryResponse fallback into a minimal RichQueryResponse for the dashboard."""
    return RichQueryResponse(
        query="",
        interpreted_query="",
        final_answer=qr.claim,
        summary=qr.summary,
        confidence=ConfidenceDetail(
            label=Confidence.LOW,
            score=10,
            explanation=qr.summary,
        ),
        verdict=Verdict.WEAK,
        evidence_strength=qr.evidence_strength,
        evidence_cards=[],
        claim_links=[],
        contradiction=ContradictionSummary(
            has_conflict=False,
            support_count=0,
            oppose_count=0,
            neutral_count=0,
            explanation=qr.summary,
            scope_note="No evidence retrieved.",
        ),
    )


# ── Original endpoints ──────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a biomedical research query",
)
async def query_endpoint(
    body: QueryRequest,
    pipeline: Pipeline = Depends(_get_pipeline),
) -> QueryResponse:
    logger.info(f"POST /query | query={body.query!r}")
    try:
        return await pipeline.run(body.query)
    except Exception as exc:
        logger.error(f"Unhandled pipeline error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your query.",
        )


@router.post(
    "/query/rich",
    response_model=RichQueryResponse,
    summary="Submit a query — returns full structured response",
)
async def query_rich_endpoint(
    body: QueryRequest,
    pipeline: Pipeline = Depends(_get_pipeline),
) -> RichQueryResponse:
    logger.info(f"POST /query/rich | query={body.query!r}")
    try:
        result = await pipeline.run_rich(body.query)
        if not isinstance(result, RichQueryResponse):
            return _wrap_fallback(result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unhandled pipeline error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your query.",
        )


@router.get("/paper/{pmid}", summary="Fetch full paper abstract by PMID")
async def get_paper(pmid: str) -> dict:
    import httpx
    from xml.etree import ElementTree as ET
    from app.config import get_settings
    settings = get_settings()
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(EFETCH_URL, params={
                "db": "pubmed", "id": pmid,
                "retmode": "xml", "rettype": "abstract",
                "email": settings.PUBMED_EMAIL,
            })
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
        article = root.find(".//PubmedArticle")
        if article is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        def _text(path):
            el = article.find(path)
            return (el.text or "").strip() if el is not None else ""

        title = _text(".//ArticleTitle")
        abstract_parts = [el.text or "" for el in article.findall(".//AbstractText")]
        abstract = " ".join(abstract_parts).strip()
        journal = _text(".//Journal/Title")
        year = _text(".//PubDate/Year") or _text(".//PubDate/MedlineDate")[:4]
        authors = []
        for a in article.findall(".//Author"):
            ln = a.find("LastName")
            fn = a.find("ForeName")
            if ln is not None:
                authors.append(f"{ln.text or ''} {fn.text if fn is not None else ''}".strip())
        doi_el = article.find(".//ArticleId[@IdType='doi']")
        doi = doi_el.text.strip() if doi_el is not None else ""
        return {
            "pmid": pmid, "title": title, "abstract": abstract,
            "journal": journal, "year": year,
            "authors": authors[:8], "doi": doi,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Paper fetch failed for PMID {pmid}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}


# ── Agent endpoints ─────────────────────────────────────────

@router.post(
    "/agent/query",
    response_model=AgentQueryResponse,
    summary="Route a query through the multi-agent system",
)
async def agent_query(
    body: AgentQueryRequest,
    request: Request,
) -> AgentQueryResponse:
    orchestrator = _get_orchestrator(request)
    task_map = {
        "query": TaskType.QUERY,
        "literature_review": TaskType.LITERATURE_REVIEW,
        "evidence_analysis": TaskType.EVIDENCE_ANALYSIS,
        "writing_draft": TaskType.WRITING_DRAFT,
        "knowledge_graph": TaskType.KNOWLEDGE_GRAPH_BUILD,
        "contradiction": TaskType.CONTRADICTION_ANALYSIS,
    }
    task_type = task_map.get(body.task_type, TaskType.QUERY)
    result = await orchestrator.route(task_type, {
        "query": body.query,
        "style": body.style,
        "draft_type": body.draft_type,
    })
    return AgentQueryResponse(
        task_type=body.task_type,
        result=result if isinstance(result, dict) else {"output": str(result)},
        success="error" not in result,
    )


# ── Literature review endpoints ─────────────────────────────

@router.post(
    "/review",
    response_model=LiteratureReviewResponse,
    summary="Generate a literature review for a query",
)
async def literature_review(
    body: LiteratureReviewRequest,
    request: Request,
) -> LiteratureReviewResponse:
    orchestrator = _get_orchestrator(request)
    pipeline = _get_pipeline(request)

    papers = []
    claims = []
    evidence = {}
    try:
        result = await pipeline.run_rich(body.query)
        if hasattr(result, "evidence_cards"):
            for card in result.evidence_cards:
                papers.append({
                    "pmid": card.pmid,
                    "title": card.title,
                    "abstract": card.claim_text,
                    "sample_size": getattr(card, "sample_size", None),
                })
        if hasattr(result, "claim_links"):
            for cl in result.claim_links:
                stance_val = cl.stance.value if hasattr(cl.stance, "value") else str(cl.stance)
                claims.append({
                    "paper_title": cl.paper_title,
                    "claim_text": cl.claim_text,
                    "stance": stance_val,
                })
        evidence = {
            "verdict": result.verdict.value if hasattr(result, "verdict") else "",
            "confidence": result.confidence.label.value if hasattr(result, "confidence") else "",
            "support_count": result.contradiction.support_count if hasattr(result, "contradiction") else 0,
            "oppose_count": result.contradiction.oppose_count if hasattr(result, "contradiction") else 0,
            "neutral_count": result.contradiction.neutral_count if hasattr(result, "contradiction") else 0,
        }
    except Exception as exc:
        logger.warning(f"Pipeline failed during literature review: {exc}")

    agent_result = await orchestrator.route(
        TaskType.LITERATURE_REVIEW,
        {"query": body.query, "papers": papers, "claims": claims, "evidence": evidence},
    )

    review_text = agent_result.get("review", "")
    if not review_text or review_text.startswith("{"):
        review_text = _build_fallback_review(body.query, papers, claims, evidence)

    return LiteratureReviewResponse(
        query=body.query,
        review=review_text,
        papers_reviewed=len(papers),
        sections=_extract_review_sections(review_text),
    )


def _extract_review_sections(review: str) -> list:
    import re
    return re.findall(r"^##\s+(.+)$", review, re.MULTILINE)


def _build_fallback_review(query: str, papers: list, claims: list, evidence: dict) -> str:
    support = [c for c in claims if c.get("stance") == "support"]
    oppose = [c for c in claims if c.get("stance") == "oppose"]
    neutral = [c for c in claims if c.get("stance") == "neutral"]
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


# ── Knowledge graph endpoints ───────────────────────────────

@router.post(
    "/kg/build",
    response_model=KnowledgeGraphResponse,
    summary="Build a knowledge graph from a query",
)
async def build_knowledge_graph(
    body: QueryRequest,
    request: Request,
) -> KnowledgeGraphResponse:
    orchestrator = _get_orchestrator(request)
    pipeline = _get_pipeline(request)
    try:
        result = await pipeline.run_rich(body.query)
        papers = []
        if hasattr(result, "evidence_cards"):
            for card in result.evidence_cards:
                papers.append({"pmid": card.pmid, "title": card.title, "abstract": card.claim_text})
    except Exception as exc:
        logger.warning(f"Pipeline failed during KG build: {exc}")
        papers = []

    try:
        kg_result = await orchestrator.route(
            TaskType.KNOWLEDGE_GRAPH_BUILD,
            {"query": body.query, "papers": papers},
        )
    except Exception as exc:
        logger.error(f"KG orchestrator failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge graph build failed: {exc}",
        )
    return KnowledgeGraphResponse(
        nodes=kg_result.get("graph", {}).get("nodes", []),
        edges=kg_result.get("graph", {}).get("edges", []),
        entity_count=kg_result.get("entity_count", 0),
        relation_count=kg_result.get("relation_count", 0),
        entity_types=kg_result.get("entity_types", {}),
        top_entities=kg_result.get("top_entities", []),
    )


@router.get(
    "/kg/graph/{query}",
    response_model=KnowledgeGraphResponse,
    summary="Build knowledge graph from a search query",
)
async def get_knowledge_graph(
    query: str,
    request: Request,
) -> KnowledgeGraphResponse:
    orchestrator = _get_orchestrator(request)
    pipeline = _get_pipeline(request)
    try:
        result = await pipeline.run_rich(query)
        papers = []
        if hasattr(result, "evidence_cards"):
            for card in result.evidence_cards:
                papers.append({"pmid": card.pmid, "title": card.title, "abstract": card.claim_text})
    except Exception as exc:
        logger.warning(f"Pipeline failed during KG build: {exc}")
        papers = []

    try:
        kg_result = await orchestrator.route(
            TaskType.KNOWLEDGE_GRAPH_BUILD,
            {"query": query, "papers": papers},
        )
    except Exception as exc:
        logger.error(f"KG orchestrator failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge graph build failed: {exc}",
        )
    return KnowledgeGraphResponse(
        nodes=kg_result.get("graph", {}).get("nodes", []),
        edges=kg_result.get("graph", {}).get("edges", []),
        entity_count=kg_result.get("entity_count", 0),
        relation_count=kg_result.get("relation_count", 0),
        entity_types=kg_result.get("entity_types", {}),
        top_entities=kg_result.get("top_entities", []),
    )


# ── Version control / artifacts endpoints ───────────────────

@router.get("/artifacts", summary="List all research artifacts")
async def list_artifacts(
    request: Request,
    artifact_type: str = None,
    tag: str = None,
) -> list:
    store = _get_version_store(request)
    artifacts = store.list_artifacts(artifact_type=artifact_type, tag=tag)
    return [a.dict() for a in artifacts]


@router.post("/artifacts", response_model=ArtifactResponse, summary="Create a research artifact")
async def create_artifact(
    body: ArtifactRequest,
    request: Request,
) -> ArtifactResponse:
    store = _get_version_store(request)
    artifact = store.create_artifact(
        artifact_type=body.artifact_type,
        title=body.title,
        content=body.content,
        author=body.author,
        tags=body.tags,
        message=body.message,
    )
    return ArtifactResponse(**artifact.dict())


@router.get("/artifacts/{artifact_id}", response_model=ArtifactResponse, summary="Get artifact by ID")
async def get_artifact(artifact_id: str, request: Request) -> ArtifactResponse:
    store = _get_version_store(request)
    artifact = store.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return ArtifactResponse(**artifact.dict())


@router.put("/artifacts/{artifact_id}", response_model=ArtifactResponse, summary="Update artifact")
async def update_artifact(
    artifact_id: str,
    body: ArtifactUpdateRequest,
    request: Request,
) -> ArtifactResponse:
    store = _get_version_store(request)
    version = store.update_artifact(
        artifact_id=artifact_id,
        content=body.content,
        message=body.message,
        author=body.author,
        tags=body.tags,
    )
    if not version:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact = store.get_artifact(artifact_id)
    return ArtifactResponse(**artifact.dict())


@router.get("/artifacts/{artifact_id}/history", summary="Get version history")
async def get_artifact_history(artifact_id: str, request: Request) -> list:
    store = _get_version_store(request)
    versions = store.get_history(artifact_id)
    return [v.dict() for v in versions]


@router.post("/artifacts/{artifact_id}/comment", summary="Add comment to artifact")
async def add_artifact_comment(
    artifact_id: str,
    body: CommentRequest,
    request: Request,
) -> dict:
    store = _get_version_store(request)
    comment = store.add_comment(
        artifact_id=artifact_id,
        author=body.author,
        content=body.content,
        parent_comment_id=body.parent_comment_id,
    )
    return comment


@router.get("/artifacts/{artifact_id}/comments", summary="Get artifact comments")
async def get_artifact_comments(artifact_id: str, request: Request) -> list:
    store = _get_version_store(request)
    return store.get_comments(artifact_id)


# ── Collaboration / workspace endpoints ─────────────────────

@router.get("/workspaces", summary="List workspaces")
async def list_workspaces(
    request: Request,
    username: str = None,
) -> list:
    store = _get_collab_store(request)
    workspaces = store.list_workspaces(username=username)
    return [w.dict() for w in workspaces]


@router.post("/workspaces", response_model=WorkspaceResponse, summary="Create workspace")
async def create_workspace(
    body: WorkspaceRequest,
    request: Request,
) -> WorkspaceResponse:
    store = _get_collab_store(request)
    ws = store.create_workspace(
        name=body.name,
        description=body.description,
        created_by=body.created_by,
    )
    return WorkspaceResponse(**ws.dict())


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse, summary="Get workspace")
async def get_workspace(workspace_id: str, request: Request) -> WorkspaceResponse:
    store = _get_collab_store(request)
    ws = store.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return WorkspaceResponse(**ws.dict())


@router.post("/workspaces/{workspace_id}/comment", summary="Add comment to workspace")
async def add_workspace_comment(
    workspace_id: str,
    body: CommentRequest,
    request: Request,
) -> dict:
    store = _get_collab_store(request)
    comment = store.add_comment(
        workspace_id=workspace_id,
        author=body.author,
        content=body.content,
        artifact_id=body.artifact_id,
        parent_comment_id=body.parent_comment_id,
    )
    return comment.dict()


@router.get("/workspaces/{workspace_id}/comments", summary="Get workspace comments")
async def get_workspace_comments(
    workspace_id: str,
    request: Request,
    artifact_id: str = None,
) -> list:
    store = _get_collab_store(request)
    comments = store.get_comments(workspace_id, artifact_id=artifact_id)
    return [c.dict() for c in comments]


# ── Citation formatting endpoint ────────────────────────────

@router.post("/citations/format", summary="Format citations in specified style")
async def format_citations(
    body: dict,
    request: Request,
) -> dict:
    from agents.citation_agent import CitationAgent
    papers = body.get("papers", [])
    style = body.get("style", "apa")
    text = body.get("text")
    agent = CitationAgent()
    task = AgentTask(
        task_id="cite_001",
        task_type=TaskType.CITATION_FORMAT,
        input_data={"papers": papers, "style": style, "text": text},
    )
    result = await agent.execute(task)
    return result.data if result.success else {"error": result.error}
