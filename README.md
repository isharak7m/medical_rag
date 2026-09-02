# MyoCortex 🧠

Production-grade biomedical RAG research engine — PubMed + FAISS + LLM.
Handles **any** biomedical query: supplements, drugs, diseases, interventions, nutrition.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env         # Add your GROQ_API_KEY
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Dashboard: `streamlit run dashboard.py`

## Example Queries

```bash
# Supplement query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "creatine supplementation muscle strength"}'

# Drug query
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"query": "metformin type 2 diabetes efficacy"}'

# Disease query
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"query": "omega-3 cardiovascular disease"}'

# Casual/slang query (auto-normalized)
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"query": "is protien good for gains"}'

# Rich structured response (for dashboard)
curl -X POST http://localhost:8000/api/v1/query/rich \
  -d '{"query": "ashwagandha anxiety"}'
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/query` | Flat response (backward-compatible) |
| POST | `/api/v1/query/rich` | Full structured response with evidence cards |
| GET | `/api/v1/health` | Health check |

## LLM Backends

Configure in `.env`:

| Backend | Config | Notes |
|---------|--------|-------|
| `groq` | `GROQ_API_KEY` | Fast, free daily limit — **recommended** |
| `hf_api` | `HF_API_TOKEN` | HuggingFace Inference API, free |
| `mock` | (none) | Placeholder — for testing only |

Groq → HuggingFace → Mock fallback chain is automatic.

## Docker

```bash
cp .env.example .env   # fill in keys
docker-compose up --build
```

## Tests

```bash
pytest tests/test_pipeline.py -v
```

## Architecture

```
myocortex/
├── app/            FastAPI app + config
├── api/            Routes (thin layer only — no business logic)
├── core/           Pipeline orchestrator, decision engine, prompt builder,
│                   response formatter, evaluation
├── modules/        Pure logic — query expander, retriever, claim extractor,
│                   evidence ranker, contradiction detector
├── services/       External integrations — PubMed, embeddings, LLM,
│                   cache service, query normalizer
├── db/             Storage abstractions — FAISS, cache store, schemas
└── utils/          Logger, text cleaning
```

## Pipeline Flow

```
Raw query
  → Normalize (slang fix + LLM rewrite to PubMed terms)
  → Query expansion (LLM generates 4 diverse alternatives)
  → Concurrent PubMed fetch (all queries in parallel)
  → Deduplication by PMID
  → Soft domain filter (removes clearly off-topic papers)
  → FAISS semantic retrieval + hybrid rerank (semantic + BM25 + length)
  → Claim extraction (broader patterns, conclusion-sentence preference)
  → Evidence ranking (sample-size weighted scoring)
  → Contradiction detection (with honest scope notes)
  → Decision engine (confidence gated by retrieval quality)
  → LLM synthesis (structured JSON output)
  → Response formatting (evidence cards, claim links, confidence detail)
  → Evaluation logging (faithfulness, coverage, retrieval scores)
  → Cache store
  → Return RichQueryResponse or QueryResponse
```
