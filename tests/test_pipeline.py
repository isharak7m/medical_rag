"""
Pipeline smoke tests — all services mocked, no network calls.
Tests work for ANY biomedical query, not just creatine.

Run with: pytest tests/test_pipeline.py -v
"""

from __future__ import annotations
import json
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from db.schemas import Claim, Confidence, ConfidenceDetail, ContradictionReport, ContradictionSummary, Paper, QueryResponse, RankedEvidence, RichQueryResponse, SourceItem, Stance, Verdict
from core.decision_engine import run as run_decision
from core.pipeline import Pipeline
from modules.claim_extractor import extract_claims
from modules.query_expander import generate_multiple_queries
from services.query_intent import build_query_intent


# ─────────────────────────────────────────────────────────────
# Sample papers — two different topics to test generality
# ─────────────────────────────────────────────────────────────

CREATINE_PAPERS: List[Paper] = [
    Paper(
        pmid="12345678",
        title="Creatine supplementation improves strength in athletes",
        abstract=(
            "This randomised controlled trial investigated creatine monohydrate. "
            "Sixty participants were randomised to creatine or placebo. Significant "
            "improvements in 1-RM bench press were observed (p<0.01). The intervention "
            "was well tolerated with no adverse effects reported. We conclude creatine "
            "significantly improves maximal strength in trained athletes."
        ),
        sample_size=60,
    ),
    Paper(
        pmid="87654321",
        title="No significant effect of creatine on endurance performance",
        abstract=(
            "We examined whether creatine supplementation improved VO2max in cyclists. "
            "Thirty subjects completed a double-blind crossover. No significant difference "
            "was found between creatine and placebo for any endurance measure. "
            "We conclude creatine does not benefit endurance performance."
        ),
        sample_size=30,
    ),
    Paper(
        pmid="11223344",
        title="Creatine meta-analysis: high-intensity exercise benefits",
        abstract=(
            "A meta-analysis of 22 trials confirmed that creatine significantly increases "
            "phosphocreatine resynthesis and improves performance in sprint protocols. "
            "Benefits were most pronounced in activities lasting 10-30 seconds. "
            "We suggest creatine is effective for high-intensity exercise."
        ),
        sample_size=None,
    ),
    Paper(
        pmid="55667788",
        title="Creatine safety profile: systematic review",
        abstract=(
            "A systematic review of 45 studies found creatine supplementation to be "
            "safe at recommended doses (3-5g/day). No serious adverse events were reported. "
            "We conclude creatine is safe for healthy adults in the short and long term."
        ),
        sample_size=1200,
    ),
]

METFORMIN_PAPERS: List[Paper] = [
    Paper(
        pmid="99001122",
        title="Metformin reduces HbA1c in type 2 diabetes patients",
        abstract=(
            "This large RCT of 840 participants demonstrated metformin significantly "
            "reduces HbA1c by 1.2% compared to placebo (p<0.001). Gastrointestinal "
            "side effects were reported in 25% of patients but were mostly mild. "
            "We conclude metformin is effective first-line therapy for type 2 diabetes."
        ),
        sample_size=840,
    ),
    Paper(
        pmid="33445566",
        title="Metformin cardiovascular benefits in diabetes",
        abstract=(
            "Analysis of 12,000 diabetic patients showed metformin was associated with "
            "significantly reduced cardiovascular mortality (HR 0.78, p<0.001). "
            "The drug also demonstrated weight-neutral or modest weight-loss effects. "
            "Findings support metformin as beneficial for cardiovascular outcomes."
        ),
        sample_size=12000,
    ),
    Paper(
        pmid="77889900",
        title="Metformin and kidney function: safety considerations",
        abstract=(
            "Metformin use is contraindicated in patients with severe renal impairment "
            "due to increased lactic acidosis risk. Our review of 2,500 patients found "
            "no significant increase in lactic acidosis at eGFR >30 mL/min. "
            "We suggest regular kidney function monitoring for patients on metformin."
        ),
        sample_size=2500,
    ),
]


# ─────────────────────────────────────────────────────────────
# Mock LLM that returns valid JSON
# ─────────────────────────────────────────────────────────────

def _mock_llm_json(prompt: str) -> str:
    return json.dumps({
        "final_answer": "Based on the evidence, the intervention shows significant benefits.",
        "key_claims": [
            "Multiple RCTs demonstrate significant positive effects.",
            "Effect is consistent across different study populations.",
            "Safety profile is acceptable with minor adverse events.",
        ],
        "summary": (
            "The retrieved evidence consistently supports the primary hypothesis. "
            "Multiple randomised controlled trials demonstrate significant improvements. "
            "Some variability exists across studies, which may reflect differences "
            "in study populations and dosing protocols. Overall confidence is moderate to high."
        ),
    })


# ─────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────

def _make_pipeline(papers: List[Paper] = None) -> Pipeline:
    if papers is None:
        papers = CREATINE_PAPERS

    pubmed = MagicMock()
    # fetch_papers_multi is what the pipeline actually calls
    pubmed.fetch_papers_multi = AsyncMock(return_value=papers)
    pubmed.fetch_papers = AsyncMock(return_value=papers)

    embedding = MagicMock()
    embedding.dim = 384
    embedding.embed.side_effect = lambda text: np.random.rand(384).astype("float32")
    embedding.embed_batch.side_effect = lambda texts: np.random.rand(
        len(texts), 384
    ).astype("float32")

    llm = MagicMock()
    llm.generate.side_effect = _mock_llm_json

    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = None

    return Pipeline(
        pubmed_service=pubmed,
        embedding_service=embedding,
        llm=llm,
        cache_service=cache,
    )


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_returns_rich_response():
    pipeline = _make_pipeline()
    result = await pipeline.run_rich("creatine supplementation muscle strength")

    assert isinstance(result, RichQueryResponse)
    assert result.confidence.label in (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)
    assert result.final_answer != ""
    assert result.summary != ""
    assert len(result.evidence_cards) > 0
    assert result.contradiction is not None


@pytest.mark.asyncio
async def test_pipeline_returns_flat_response():
    pipeline = _make_pipeline()
    result = await pipeline.run("creatine supplementation muscle strength")

    assert isinstance(result, QueryResponse)
    assert result.confidence in (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)
    assert isinstance(result.supporting_studies, int)
    assert isinstance(result.opposing_studies, int)
    assert len(result.sources) > 0


@pytest.mark.asyncio
async def test_pipeline_works_for_different_topics():
    """Pipeline must work for any topic, not just creatine."""
    pipeline = _make_pipeline(papers=METFORMIN_PAPERS)
    result = await pipeline.run_rich("metformin type 2 diabetes efficacy")

    assert isinstance(result, RichQueryResponse)
    assert result.final_answer != ""
    assert len(result.evidence_cards) > 0


@pytest.mark.asyncio
async def test_pipeline_cache_hit_skips_pubmed():
    pipeline = _make_pipeline()
    cached_response = RichQueryResponse(
        query="test",
        final_answer="Cached answer.",
        summary="Cached summary.",
        confidence=ConfidenceDetail(label=Confidence.HIGH, score=85, explanation=""),
        verdict=Verdict.STRONG_SUPPORT,
        evidence_strength="Strong",
        evidence_cards=[],
        claim_links=[],
        contradiction=ContradictionSummary(
            has_conflict=False, support_count=3, oppose_count=0,
            neutral_count=0, explanation="", scope_note=""
        ),
    )
    pipeline._cache.get.return_value = cached_response

    result = await pipeline.run_rich("creatine supplementation")

    pipeline._pubmed.fetch_papers_multi.assert_not_called()
    assert result.final_answer == "Cached answer."


@pytest.mark.asyncio
async def test_pipeline_no_papers_returns_insufficient():
    pipeline = _make_pipeline(papers=[])
    result = await pipeline.run("obscure biomarker xyz unknown")

    assert isinstance(result, QueryResponse)
    assert result.confidence == Confidence.LOW
    assert result.supporting_studies == 0


@pytest.mark.asyncio
async def test_pipeline_pubmed_failure_returns_fallback():
    pipeline = _make_pipeline()
    pipeline._pubmed.fetch_papers_multi = AsyncMock(
        side_effect=Exception("Connection timeout")
    )

    result = await pipeline.run("creatine supplementation")

    assert isinstance(result, QueryResponse)
    assert result.confidence == Confidence.LOW


@pytest.mark.asyncio
async def test_pipeline_stores_rich_result_in_cache():
    pipeline = _make_pipeline()
    await pipeline.run_rich("creatine supplementation muscle strength")

    pipeline._cache.set.assert_called_once()
    stored = pipeline._cache.set.call_args[0][1]
    assert isinstance(stored, RichQueryResponse)


@pytest.mark.asyncio
async def test_pipeline_evidence_cards_have_stances():
    pipeline = _make_pipeline()
    result = await pipeline.run_rich("creatine strength training")

    assert len(result.evidence_cards) > 0
    for card in result.evidence_cards:
        assert card.stance in ("support", "oppose", "neutral")
        assert card.pmid != ""
        assert card.title != ""


@pytest.mark.asyncio
async def test_pipeline_malformed_llm_output_handled():
    """Pipeline must not crash if LLM returns non-JSON."""
    pipeline = _make_pipeline()
    pipeline._llm.generate.side_effect = lambda p: "This is plain text, not JSON."

    result = await pipeline.run("creatine supplementation")

    # Should still return a valid response
    assert isinstance(result, QueryResponse)
    assert result.summary != ""


def test_decision_engine_ignores_neutral_as_negative_weight():
    ranked = [
        RankedEvidence(
            claim=Claim(paper_pmid="1", paper_title="A", claim_text="A", stance=Stance.SUPPORT),
            score=3.0,
        ),
        RankedEvidence(
            claim=Claim(paper_pmid="2", paper_title="B", claim_text="B", stance=Stance.NEUTRAL),
            score=2.0,
        ),
        RankedEvidence(
            claim=Claim(paper_pmid="3", paper_title="C", claim_text="C", stance=Stance.NEUTRAL),
            score=2.0,
        ),
    ]
    contradiction = ContradictionReport(
        support_count=1,
        oppose_count=0,
        neutral_count=2,
        has_conflict=False,
        scope_note="",
    )

    result = run_decision(ranked, contradiction, retrieval_score=1.0)

    assert result.verdict == Verdict.MODERATE_SUPPORT
    assert result.confidence in (Confidence.MEDIUM, Confidence.LOW)
    assert result.support_ratio == 1.0


def test_query_expander_keeps_intervention_anchor():
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return str([
                "creatine monohydrate resistance training strength",
                "muscle protein synthesis adults",
                "creatine power output randomized trial",
            ])

    queries = generate_multiple_queries(
        "creatine supplementation muscle strength adults humans",
        llm=FakeLLM(),
        n=4,
    )

    assert queries[0] == "creatine supplementation muscle strength adults humans"
    assert all("creatine" in query.lower() for query in queries)
    assert not any("protein synthesis" in query.lower() for query in queries[1:])


def test_claim_extractor_marks_direct_positive_strength_paper_as_support():
    paper = Paper(
        pmid="1",
        title="Creatine supplementation enhances maximal strength",
        abstract=(
            "In trained adults, creatine supplementation significantly improved 1RM bench press "
            "and repetitions to failure compared with placebo. The study concluded that creatine "
            "enhances strength performance during resistance training."
        ),
        sample_size=40,
    )

    claims = extract_claims([paper], query="Does creatine improve muscle strength in adults?")

    assert claims[0].stance == Stance.SUPPORT


def test_claim_extractor_marks_harmful_steroid_paper_as_oppose_for_health_query():
    paper = Paper(
        pmid="2",
        title="Anabolic steroid use and health consequences in young men",
        abstract=(
            "Non-medical anabolic steroid use was associated with cardiovascular risk, "
            "fertility consequences, and other adverse health effects. The review highlights "
            "the harms and need for preventive intervention."
        ),
        sample_size=120,
    )

    claims = extract_claims([paper], query="is steroids healthy")

    assert claims[0].stance == Stance.OPPOSE


def test_query_intent_flips_harm_query_direction_generally():
    intent = build_query_intent("is vaping dangerous")

    assert intent.polarity in {"safety", "overall_value"}
    assert intent.hypothesis_direction == "negative"
    assert "vaping" in intent.subject_aliases or "vape" in intent.subject_aliases


def test_query_intent_treats_good_bad_questions_as_safety_value_questions():
    positive = build_query_intent("is vaping good")
    negative = build_query_intent("is alcohol bad")

    assert positive.polarity == "overall_value"
    assert positive.hypothesis_direction == "positive"
    assert negative.polarity == "overall_value"
    assert negative.hypothesis_direction == "negative"


def test_claim_extractor_maps_harmful_evidence_to_support_for_negative_query():
    paper = Paper(
        pmid="3",
        title="Alcohol use and long-term health risks",
        abstract=(
            "Alcohol consumption was associated with increased mortality and liver disease risk. "
            "The review highlights substantial adverse health consequences."
        ),
        sample_size=200,
    )

    claims = extract_claims([paper], query="is alcohol dangerous")

    assert claims[0].stance == Stance.SUPPORT
