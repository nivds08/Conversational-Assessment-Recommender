# SHL Conversational Assessment Recommender

A conversational API that helps hiring managers find relevant SHL assessments through multi-turn dialogue. The agent clarifies vague requests, recommends catalog-grounded shortlists, supports refinement and comparison, and refuses off-topic queries.

**Live API:** https://conversational-assessment-recommender-production-4cb3.up.railway.app

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Intent classification | Groq (`llama-3.1-8b-instant`) |
| Retrieval | sentence-transformers (`all-MiniLM-L6-v2`) + FAISS |
| Catalog | 369 filtered SHL individual tests (`data/catalog.json`) |
| Deploy | Docker on Railway |

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env            # add GROQ_API_KEY
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"mid-level Java developer, under 30 minutes\"}]}"
```

Interactive docs:https://conversational-assessment-recommender-production-4cb3.up.railway.app/docs

## API contract

**POST `/chat`**

Request:

```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a mid-level Java developer"}
  ]
}
```

Response:

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "Knowledge & Skills"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying or refusing
- `end_of_conversation` is `true` when a non-empty shortlist is returned

## Architecture

```
POST /chat → classify_turn (Groq) → intent handler → FAISS retrieve → validate vs catalog → JSON response
```

**Intents:** `CLARIFY_NEEDED`, `RECOMMEND`, `REFINE`, `COMPARE`, `REFUSE`

Key modules:

- `src/intent_classifier.py` — turn classification + constraint extraction
- `src/agent_logic.py` — per-intent handlers, retrieval query building, URL validation
- `src/retrieval.py` — local embeddings + persisted FAISS index
- `src/catalog_prep.py` — filters pre-packaged job solutions from raw catalog

## Project layout

```
main.py                 FastAPI app
src/                    Agent, classifier, retrieval
data/                   catalog.json, FAISS index (prebuilt)
scripts/                Catalog prep, retrieval tests, trace replay
traces/                 Evaluation conversation transcripts (C1–C10)
Dockerfile              Baked model + index for fast Railway cold starts
```

## Scripts

```bash
python scripts/prepare_catalog.py      # regenerate catalog.json from raw feed
python scripts/test_retrieve.py        # sanity-check retrieval
python scripts/test_classify.py        # intent classifier smoke tests
python scripts/test_traces.py          # replay traces vs live /chat (Recall@10)
```

Trace replay requires the server running: `uvicorn main:app --port 8000`

## Deployment (Railway)

1. Connect GitHub repo to Railway
2. Set `GROQ_API_KEY` in service variables
3. Clear any custom start command (Dockerfile `CMD` handles `$PORT`)
4. Generate public domain (port **8080**)

The Docker image bakes the embedding model and FAISS index at build time so startup stays fast.

## Environment

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for intent classification |

See `.env.example`.
