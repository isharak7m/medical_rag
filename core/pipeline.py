"""
Core pipeline orchestration for the biomedical RAG flow.
"""

from __future__ import annotations

from typing import List

from app.config import get_settings
from core.decision_engine import run as run_decision
from core.evaluation import evaluate_and_log, score_retrieval
from core.prompt_builder import build_prompt
from core.response_formatter import format_response
from db.schemas import Confidence, PipelineDiagnostics, QueryResponse, RichQueryResponse, SourceItem, RelevanceLabel
from modules.claim_extractor import extract_claims
from modules.contradiction_detector import detect_contradictions
from modules.evidence_ranker import rank_evidence
from modules.query_expander import generate_multiple_queries
from modules.retriever import retrieve_with_scores
from services.cache_service import CacheService
from services.embedding_service import EmbeddingService
from services.llm_service import BaseLLM, FallbackLLM
from services.pubmed_service import PubMedService
from services.query_normalizer import normalize_query
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_MIN_PAPERS_FOR_ANSWER = 2

_INSUFFICIENT_EVIDENCE = QueryResponse(
    claim="Insufficient evidence found for this query.",
    confidence=Confidence.LOW,
    evidence_strength="No papers retrieved",
    supporting_studies=0,
    opposing_studies=0,
    summary=(
        "No relevant research papers were found for the given query. "
        "Please try a more specific biomedical term or rephrase your question."
    ),
    sources=[],
)

_PUBMED_FAILURE = QueryResponse(
    claim="Unable to retrieve research data at this time.",
    confidence=Confidence.LOW,
    evidence_strength="Service unavailable",
    supporting_studies=0,
    opposing_studies=0,
    summary=(
        "PubMed retrieval failed. The system was unable to connect to the "
        "literature database. Please try again shortly."
    ),
    sources=[],
)

_LOW_QUALITY_RETRIEVAL = QueryResponse(
    claim="Insufficient high-quality evidence retrieved for this query.",
    confidence=Confidence.LOW,
    evidence_strength="Insufficient",
    supporting_studies=0,
    opposing_studies=0,
    summary=(
        "The retrieved papers do not sufficiently match your query topic. "
        "Please refine your query with more specific medical or scientific terms."
    ),
    sources=[],
)


def _get_active_llm_name(llm: BaseLLM) -> str:
    if isinstance(llm, FallbackLLM):
        return "Groq (primary) -> HuggingFace (fallback)"
    return type(llm).__name__.replace("LLM", "").replace("API", " API")


def _domain_filter(papers, core_keywords: List[str], intent=None):
    if not core_keywords:
        return papers

    intervention_terms = getattr(intent, 'intervention_terms', []) if intent else []
    outcome_terms = getattr(intent, 'outcome_terms', []) if intent else []
    keywords = [keyword.lower() for keyword in core_keywords]

    if intervention_terms and outcome_terms:
        intervention_kws = [t.lower() for t in intervention_terms]
        outcome_kws = [t.lower() for t in outcome_terms]
        strict_filtered = [
            paper
            for paper in papers
            if any(kw in paper.title.lower() or kw in paper.abstract.lower() for kw in intervention_kws)
            and any(kw in paper.title.lower() or kw in paper.abstract.lower() for kw in outcome_kws)
        ]
        if len(strict_filtered) >= 2:
            removed = len(papers) - len(strict_filtered)
            if removed:
                logger.info(f"Strict domain filter: removed {removed} papers (need intervention AND outcome), kept {len(strict_filtered)}")
            return strict_filtered
        logger.info(f"Strict filter only kept {len(strict_filtered)} papers; relaxing to keyword-only filter")

    filtered = [
        paper
        for paper in papers
        if any(keyword in paper.title.lower() or keyword in paper.abstract.lower() for keyword in keywords)
    ]

    if not filtered:
        logger.warning(
            f"Domain filter would remove all {len(papers)} papers; skipping filter (keywords: {core_keywords})"
        )
        return papers

    removed = len(papers) - len(filtered)
    if removed:
        logger.info(f"Domain filter: removed {removed} off-topic papers, kept {len(filtered)}")
    return filtered


class Pipeline:
    def __init__(
        self,
        pubmed_service: PubMedService,
        embedding_service: EmbeddingService,
        llm: BaseLLM,
        cache_service: CacheService,
    ) -> None:
        self._pubmed = pubmed_service
        self._embedder = embedding_service
        self._llm = llm
        self._cache = cache_service

    async def run(self, raw_query: str) -> QueryResponse:
        rich = await self.run_rich(raw_query)
        if isinstance(rich, QueryResponse):
            return rich
        return QueryResponse(
            claim=rich.final_answer,
            confidence=rich.confidence.label,
            evidence_strength=rich.evidence_strength,
            supporting_studies=rich.contradiction.support_count,
            opposing_studies=rich.contradiction.oppose_count,
            summary=rich.summary,
            sources=[SourceItem(title=card.title, pmid=card.pmid) for card in rich.evidence_cards],
        )

    async def run_rich(self, raw_query: str) -> RichQueryResponse | QueryResponse:
        diagnostics = PipelineDiagnostics()

        cached = self._cache.get(raw_query)
        if cached:
            logger.info("Returning cached response")
            if isinstance(cached, RichQueryResponse):
                cached.diagnostics.cache_hit = True
            return cached

        pubmed_query, interpreted_query, core_keywords, intent = normalize_query(
            raw_query,
            embedding_service=self._embedder,
            llm=self._llm,
        )
        logger.info(f"Pipeline start | original='{raw_query}' | normalized='{pubmed_query}'")

        expanded_queries = generate_multiple_queries(
            normalized_query=pubmed_query,
            llm=self._llm,
            n=settings.QUERY_EXPANSION_N,
        )
        diagnostics.search_queries = expanded_queries
        logger.info(f"Expanded to {len(expanded_queries)} queries for PubMed")

        try:
            papers = await self._pubmed.fetch_papers_multi(
                queries=expanded_queries,
                max_total=settings.MULTI_QUERY_MAX_PAPERS,
            )
        except Exception as exc:
            logger.error(f"PubMed multi-fetch hard failure: {exc}")
            return _PUBMED_FAILURE

        if not papers:
            logger.warning("No papers returned from any query")
            return _INSUFFICIENT_EVIDENCE

        diagnostics.total_unique_papers = len(papers)
        logger.info(f"Total unique papers after merge: {len(papers)}")

        papers = _domain_filter(papers, core_keywords, intent=intent)
        diagnostics.papers_after_filter = len(papers)

        if len(papers) < _MIN_PAPERS_FOR_ANSWER:
            logger.warning(f"Only {len(papers)} papers after filter; below minimum {_MIN_PAPERS_FOR_ANSWER}")
            return _LOW_QUALITY_RETRIEVAL

        retrieval_score = score_retrieval(raw_query, papers, core_keywords)
        logger.info(f"Retrieval score: {retrieval_score:.0%} ({len(papers)} papers)")

        papers_with_scores = retrieve_with_scores(
            query=interpreted_query,
            papers=papers,
            embedding_service=self._embedder,
            top_k=settings.FAISS_TOP_K,
        )
        if not papers_with_scores:
            return _INSUFFICIENT_EVIDENCE

        retrieved = [paper for paper, _ in papers_with_scores]
        diagnostics.final_ranked_papers = len(retrieved)
        logger.info(f"Final reranked papers: {len(retrieved)}")

        claims = extract_claims(
            retrieved,
            query=interpreted_query,
            llm_generate=self._llm.generate,
        )
        ranked = rank_evidence(claims, retrieved)
        contradiction = detect_contradictions(ranked, retrieval_score=retrieval_score)

        diagnostics.support_count = contradiction.support_count
        diagnostics.oppose_count = contradiction.oppose_count
        diagnostics.neutral_count = contradiction.neutral_count
        direct_count = contradiction.support_count + contradiction.oppose_count
        diagnostics.direct_evidence_ratio = round(direct_count / len(ranked), 4) if ranked else 0.0
        diagnostics.notes.append(
            f"Retrieved {diagnostics.total_unique_papers} unique papers and surfaced the top {diagnostics.final_ranked_papers}."
        )
        if contradiction.neutral_count > direct_count:
            diagnostics.notes.append("Most top-ranked papers were indirect or neutral for the exact user query.")

        decision = run_decision(ranked, contradiction, retrieval_score=retrieval_score)
        prompt = build_prompt(interpreted_query, claims, ranked, decision)
        raw_llm_output = self._llm.generate(prompt)
        llm_name = _get_active_llm_name(self._llm)
        logger.info(f"LLM backend used: {llm_name}")

        response = format_response(
            query=raw_query,
            raw_llm_output=raw_llm_output,
            claims=claims,
            papers_with_scores=papers_with_scores,
            decision=decision,
            contradiction=contradiction,
            diagnostics=diagnostics,
        )
        response.interpreted_query = interpreted_query
        response.llm_backend_used = llm_name

        response = evaluate_and_log(interpreted_query, retrieved, response, core_keywords)
        self._cache.set(raw_query, response)

        logger.info(
            f"Pipeline complete | verdict={decision.verdict.value} | confidence={decision.confidence.value} "
            f"| retrieval={retrieval_score:.0%} | papers_used={len(retrieved)}/{len(papers)}"
        )
        return response
