"""
Cross-domain diagnostic tests for the evidence pipeline.
Tests 6 conceptually different biomedical queries across domains.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from db.schemas import Paper
from core.pipeline import Pipeline


def _make_llm():
    llm = MagicMock()
    def _side_effect(prompt):
        return json.dumps({
            "final_answer": "Based on the evidence, the intervention shows significant benefits.",
            "key_claims": [
                "Multiple studies demonstrate significant positive effects.",
                "Effect is consistent across different study populations.",
            ],
            "summary": (
                "The retrieved evidence consistently supports the primary hypothesis. "
                "Multiple randomised controlled trials demonstrate significant improvements."
            ),
        })
    llm.generate.side_effect = _side_effect
    return llm


def _make_pipeline(papers):
    pubmed = MagicMock()
    pubmed.fetch_papers_multi = AsyncMock(return_value=papers)
    embedding = MagicMock()
    embedding.dim = 384
    embedding.embed.side_effect = lambda text: np.random.rand(384).astype("float32")
    embedding.embed_batch.side_effect = lambda texts: np.random.rand(len(texts), 384).astype("float32")
    cache = MagicMock()
    cache.get.return_value = None
    return Pipeline(
        pubmed_service=pubmed,
        embedding_service=embedding,
        llm=_make_llm(),
        cache_service=cache,
    )


CARDIOVASCULAR_PAPERS = [
    Paper(pmid="1", title="Statin therapy reduces LDL cholesterol in hyperlipidemia",
          abstract="This RCT of 2000 patients demonstrated that atorvastatin 80mg significantly reduces LDL cholesterol by 50% compared to placebo. Cardiovascular events were reduced by 35%.",
          sample_size=2000),
    Paper(pmid="2", title="Statin side effects: myopathy risk in elderly patients",
          abstract="A cohort study of 15000 elderly patients found that statin use was associated with increased risk of myopathy (OR 2.3, 95% CI 1.8-3.0). Risk was dose-dependent.",
          sample_size=15000),
    Paper(pmid="3", title="Omega-3 fatty acids and cardiovascular prevention",
          abstract="Meta-analysis of 20 trials found omega-3 supplementation did not significantly reduce major cardiovascular events (RR 0.97, 95% CI 0.93-1.01).",
          sample_size=None),
    Paper(pmid="4", title="Statin cardiovascular mortality benefit: systematic review",
          abstract="A systematic review of 30 RCTs confirmed that statins significantly reduce all-cause mortality (RR 0.87, 95% CI 0.82-0.92) and cardiovascular mortality.",
          sample_size=50000),
]

MENTAL_HEALTH_PAPERS = [
    Paper(pmid="5", title="SSRI efficacy in major depressive disorder",
          abstract="A network meta-analysis of 522 trials found that all 21 antidepressants were more effective than placebo. Escitalopram and sertraline had the best efficacy-tolerability profile.",
          sample_size=None),
    Paper(pmid="6", title="SSRI sexual side effects: prevalence and management",
          abstract="Systematic review found sexual dysfunction in 40-65% of patients on SSRIs. Prevalence was higher with paroxetine and sertraline.",
          sample_size=None),
    Paper(pmid="7", title="Exercise therapy for depression: meta-analysis",
          abstract="Meta-analysis of 39 RCTs found exercise had a moderate-to-large effect size (SMD -0.68) for depression. Effects were maintained at 6-month follow-up.",
          sample_size=None),
    Paper(pmid="8", title="Mindfulness-based therapy for anxiety disorders",
          abstract="RCT of 200 patients showed mindfulness-based stress reduction significantly reduced anxiety scores (Hedges g = 0.72) compared to waitlist control.",
          sample_size=200),
]

NUTRITION_PAPERS = [
    Paper(pmid="9", title="Intermittent fasting and metabolic health",
          abstract="RCT of 120 obese adults found 16:8 intermittent fasting resulted in 3.2% weight loss at 12 weeks. No significant difference vs continuous caloric restriction.",
          sample_size=120),
    Paper(pmid="10", title="Ketogenic diet and cognitive function in elderly",
          abstract="Crossover trial of 40 elderly adults found a 12-week ketogenic diet improved memory scores by 15% (p<0.01) compared to standard diet.",
          sample_size=40),
    Paper(pmid="11", title="High protein intake and kidney function",
          abstract="Prospective cohort of 5000 adults found high protein intake (>2g/kg/day) was not associated with decline in kidney function over 10 years in healthy adults.",
          sample_size=5000),
    Paper(pmid="12", title="Mediterranean diet and cardiovascular outcomes",
          abstract="PREDIMED trial of 7447 participants found Mediterranean diet supplemented with olive oil reduced major cardiovascular events by 30% (HR 0.70, 95% CI 0.54-0.92).",
          sample_size=7447),
]


async def test_cardiovascular():
    pipeline = _make_pipeline(CARDIOVASCULAR_PAPERS)
    result = await pipeline.run_rich("Do statins reduce heart disease risk?")
    print(f"\n1. CARDIOVASCULAR: Do statins reduce heart disease risk?")
    if hasattr(result, 'verdict'):
        print(f"   Verdict: {result.verdict.value}")
        print(f"   Confidence: {result.confidence.label.value}")
        print(f"   Evidence cards: {len(result.evidence_cards)}")
        print(f"   Direct ratio: {result.diagnostics.direct_evidence_ratio}")
    else:
        print(f"   Response type: {type(result).__name__}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Supporting: {result.supporting_studies}, Opposing: {result.opposing_studies}")
    return result


async def test_mental_health():
    pipeline = _make_pipeline(MENTAL_HEALTH_PAPERS)
    result = await pipeline.run_rich("Are SSRIs effective for depression?")
    print(f"\n2. MENTAL HEALTH: Are SSRIs effective for depression?")
    if hasattr(result, 'verdict'):
        print(f"   Verdict: {result.verdict.value}")
        print(f"   Confidence: {result.confidence.label.value}")
        print(f"   Evidence cards: {len(result.evidence_cards)}")
        print(f"   Direct ratio: {result.diagnostics.direct_evidence_ratio}")
    else:
        print(f"   Response type: {type(result).__name__}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Supporting: {result.supporting_studies}, Opposing: {result.opposing_studies}")
    return result


async def test_nutrition():
    pipeline = _make_pipeline(NUTRITION_PAPERS)
    result = await pipeline.run_rich("Is intermittent fasting safe and effective for weight loss?")
    print(f"\n3. NUTRITION: Is intermittent fasting safe and effective for weight loss?")
    if hasattr(result, 'verdict'):
        print(f"   Verdict: {result.verdict.value}")
        print(f"   Confidence: {result.confidence.label.value}")
        print(f"   Evidence cards: {len(result.evidence_cards)}")
        print(f"   Direct ratio: {result.diagnostics.direct_evidence_ratio}")
    else:
        print(f"   Response type: {type(result).__name__}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Supporting: {result.supporting_studies}, Opposing: {result.opposing_studies}")
    return result


async def main():
    await test_cardiovascular()
    await test_mental_health()
    await test_nutrition()
    print("\n--- All diagnostic tests completed ---")


if __name__ == "__main__":
    asyncio.run(main())
