"""
Application entry point.
Initialises all services (once, at startup) and wires them into the pipeline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from api.routes import router
from core.pipeline import Pipeline
from db.cache_store import build_cache_store
from services.cache_service import CacheService
from services.embedding_service import EmbeddingService
from services.llm_service import build_llm
from services.pubmed_service import PubMedService
from services.biorxiv_service import BioRxivService
from services.knowledge_graph import KnowledgeGraphService
from services.version_control import VersionControlStore
from services.collaboration import CollaborationStore
from services.auth import AuthStore
from agents.orchestrator import Orchestrator
from agents.retrieval_agent import RetrievalAgent
from agents.evidence_agent import EvidenceAgent
from agents.writing_agent import WritingAgent
from agents.citation_agent import CitationAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.literature_review_agent import LiteratureReviewAgent
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────
# Lifespan: startup + shutdown
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    pubmed_service = PubMedService()
    biorxiv_service = BioRxivService()
    embedding_service = EmbeddingService()
    llm = build_llm()
    cache_store = build_cache_store(settings.CACHE_BACKEND, settings.CACHE_DB_PATH)
    cache_service = CacheService(store=cache_store)

    # Version control and collaboration stores
    version_store = VersionControlStore(db_path=settings.ARTIFACTS_DB_PATH)
    collab_store = CollaborationStore(db_path=settings.COLLAB_DB_PATH)
    auth_store = AuthStore(db_path=settings.USERS_DB_PATH)

    # Knowledge graph service
    kg_service = KnowledgeGraphService(llm_generate=llm.generate)

    # Build pipeline
    pipeline = Pipeline(
        pubmed_service=pubmed_service,
        embedding_service=embedding_service,
        llm=llm,
        cache_service=cache_service,
    )

    # Build multi-agent orchestrator
    orchestrator = Orchestrator(llm_generate=llm.generate, pipeline=pipeline)
    orchestrator.register_agent(RetrievalAgent(
        llm_generate=llm.generate,
        pubmed_service=pubmed_service,
        biorxiv_service=biorxiv_service,
        embedding_service=embedding_service,
    ))
    orchestrator.register_agent(EvidenceAgent(llm_generate=llm.generate))
    orchestrator.register_agent(WritingAgent(llm_generate=llm.generate))
    orchestrator.register_agent(CitationAgent(llm_generate=llm.generate))
    orchestrator.register_agent(KnowledgeAgent(llm_generate=llm.generate))
    orchestrator.register_agent(LiteratureReviewAgent(llm_generate=llm.generate))

    # Attach to app state
    app.state.pipeline = pipeline
    app.state.orchestrator = orchestrator
    app.state.kg_service = kg_service
    app.state.version_store = version_store
    app.state.collab_store = collab_store
    app.state.auth_store = auth_store
    app.state.biorxiv_service = biorxiv_service

    logger.info("All services initialised. MyoCortex is ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────
    await pubmed_service.close()
    await biorxiv_service.close()
    logger.info("MyoCortex shut down cleanly.")


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Biomedical RAG research engine with multi-agent orchestration, "
            "knowledge graphs, version control, and collaboration. "
            "Powered by PubMed + bioRxiv + FAISS + LLM."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
