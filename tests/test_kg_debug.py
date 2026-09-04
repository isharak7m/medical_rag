import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from db.schemas import Paper
from core.pipeline import Pipeline
from agents.orchestrator import Orchestrator, TaskType
from agents.knowledge_agent import KnowledgeAgent

papers = [
    Paper(pmid='1', title='Creatine supplementation improves strength', abstract='Creatine supplementation significantly improves muscle strength in athletes.', sample_size=60),
    Paper(pmid='2', title='Creatine safety profile', abstract='Creatine is safe for healthy adults.', sample_size=1200),
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

async def test():
    papers_dicts = [{"pmid": p.pmid, "title": p.title, "abstract": p.abstract} for p in papers]
    result = await orchestrator.route(TaskType.KNOWLEDGE_GRAPH_BUILD, {"query": "creatine strength", "papers": papers_dicts})
    print("Result keys:", result.keys())
    print("Entity count:", result.get("entity_count"))
    print("Error:", result.get("error"))

asyncio.run(test())
