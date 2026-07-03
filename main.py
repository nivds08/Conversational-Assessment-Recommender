from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from src.agent_logic import SAFE_FALLBACK_RESPONSE, generate_agent_response
from src.retrieval import warmup_retrieval


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class RecommendationOut(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationOut]
    end_of_conversation: bool


def _clarify_fallback() -> ChatResponse:
    return ChatResponse(
        reply="Please describe your hiring need (role, level, and skills) so I can recommend SHL assessments.",
        recommendations=[],
        end_of_conversation=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    info = warmup_retrieval()
    print(
        "[startup] Retrieval ready:",
        f"catalog_entries={info['catalog_entries']},",
        f"index_size={info['index_size']},",
        f"embedding_model={info['embedding_model']}",
    )
    yield


app = FastAPI(lifespan=lifespan)

# Assignment simplification: allow all origins for local/manual/harness access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request) -> ChatResponse:
    try:
        body = await request.json()
    except Exception:
        return _clarify_fallback()

    try:
        payload = ChatRequest.model_validate(body)
    except ValidationError:
        return _clarify_fallback()

    if not payload.messages or len(payload.messages) > 100:
        return _clarify_fallback()

    # Sanity-check roles/content to avoid malformed message arrays causing crashes.
    for m in payload.messages:
        if m.role not in ("user", "assistant"):
            return _clarify_fallback()
        if not isinstance(m.content, str) or not m.content.strip():
            return _clarify_fallback()

    try:
        result = generate_agent_response([m.model_dump() for m in payload.messages])
        return ChatResponse(
            reply=result.reply,
            recommendations=[
                RecommendationOut(name=r.name, url=r.url, test_type=r.test_type)
                for r in result.recommendations
            ],
            end_of_conversation=result.end_of_conversation,
        )
    except Exception as exc:
        print(f"[warn] /chat fallback due to unhandled error: {exc}")
        return ChatResponse(
            reply=SAFE_FALLBACK_RESPONSE.reply,
            recommendations=[],
            end_of_conversation=False,
        )
