"""Test the KG build endpoint by simulating the route handler."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from db.schemas import Paper, QueryRequest
from core.pipeline import Pipeline
from agents.orchestrator import Orchestrator, TaskType
from agents.knowledge_agent import KnowledgeAgent
from services.knowledge_graph import KnowledgeGraphService

papers = [
    Paper(pmid='1', title='Creatine supplementation improves strength', abstract='Creatine supplementation significantly improves muscle strength in athletes. RCT with 60 participants showed 1-RM bench press improvement.', sample_size=60),
    Paper(pmid='2', title='Creatine safety profile', abstract='Systematic review of 45 studies found creatine supplementation safe at recommended doses. No serious adverse events.', sample_size=1200),
]

pubmed = MagicMock()
pubmed.fetch_papers_multi = AsyncMock(return_value=papers)
embedding = MagicMock()
embedding.dim = 384
embedding.embed.side_effect = lambda text: np.random.rand(384).astype("float32")
embedding.embed_batch.side_effect = lambda texts: np.random.rand(len(texts), 384).astype("float32")
llm = MagicMock()
llm.generate.return_value = json.dumps({"final_answer": "test", "key_claims": [], "summary": "test"})
cache = MagicMock()
cache.get.return_value = None

pipeline = Pipeline(pubmed_service=pubmed, embedding_service=embedding, llm=llm, cache_service=cache)
orchestrator = Orchestrator(llm_generate=llm.generate, pipeline=pipeline)
orchestrator.register_agent(KnowledgeAgent(llm_generate=llm.generate))

async def test_route():
    query = "creatine supplementation muscle strength"

    # Step 1: run pipeline
    print("Step 1: Running pipeline...")
    result = await pipeline.run_rich(query)
    print(f"  Result type: {type(result).__name__}")
    print(f"  Has evidence_cards: {hasattr(result, 'evidence_cards')}")

    # Step 2: extract papers
    papers_list = []
    if hasattr(result, "evidence_cards"):
        for card in result.evidence_cards:
            papers_list.append({"pmid": card.pmid, "title": card.title, "abstract": card.claim_text})
    print(f"  Papers extracted: {len(papers_list)}")

    # Step 3: build KG
    print("Step 3: Building KG...")
    kg_result = await orchestrator.route(
        TaskType.KNOWLEDGE_GRAPH_BUILD,
        {"query": query, "papers": papers_list},
    )
    print(f"  KG result keys: {kg_result.keys()}")

    # Step 4: build response
    print("Step 4: Building response...")
    from db.schemas import KnowledgeGraphResponse
    response = KnowledgeGraphResponse(
        nodes=kg_result.get("graph", {}).get("nodes", []),
        edges=kg_result.get("graph", {}).get("edges", []),
        entity_count=kg_result.get("entity_count", 0),
        relation_count=kg_result.get("relation_count", 0),
        entity_types=kg_result.get("entity_types", {}),
        top_entities=kg_result.get("top_entities", []),
    )
    print(f"  Response: entities={response.entity_count}, relations={response.relation_count}")
    print("SUCCESS")

asyncio.run(test_route())
