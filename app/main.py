"""
FastAPI AI-agent starter: a webhook receiver + a RAG-style /ask endpoint.

This is the shape I actually use for client automation backends: n8n (or any
caller) hits a webhook, the service does retrieval + composes a grounded
answer, and the response goes straight back into the automation graph.

Run it:
    uvicorn app.main:app --reload

Try it:
    curl -X POST localhost:8000/knowledge-base/index -H "Content-Type: application/json" \
        -d '{"docs": {"shipping": "Orders ship within 2 business days.", "returns": "Returns accepted within 30 days with receipt."}}'

    curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
        -d '{"question": "how long do I have to return something?"}'
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import RetrievedDoc, get_default_retriever

app = FastAPI(title="AI Agent Starter", version="0.1.0")

# In-memory for this template; swap for Redis/Postgres in a real deployment.
_retriever = get_default_retriever()


class IndexRequest(BaseModel):
    docs: dict[str, str]


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedDoc]


class WebhookPayload(BaseModel):
    event: str
    data: dict


@app.post("/knowledge-base/index")
def index_knowledge_base(req: IndexRequest) -> dict:
    """Load/replace the retrievable knowledge base (docs keyed by id)."""
    if not req.docs:
        raise HTTPException(status_code=400, detail="docs must not be empty")
    _retriever.index(req.docs)
    return {"indexed": len(req.docs)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    Retrieval-augmented answer: find the most relevant indexed docs, then
    compose an answer. The LLM call itself is intentionally stubbed (see
    `compose_answer`) so this repo runs with zero API keys — swap that one
    function for a real chat-completion call in a private deployment.
    """
    hits = _retriever.retrieve(req.question, top_k=req.top_k)
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant documents indexed yet.")
    answer = compose_answer(req.question, hits)
    return AskResponse(question=req.question, answer=answer, sources=hits)


def compose_answer(question: str, hits: list[RetrievedDoc]) -> str:
    """
    Stub for the generation step. In production this is a chat-completion
    call with `hits` injected as context and a system prompt constraining
    the model to only answer from the provided sources. Kept as a plain
    function (not a class) on purpose — it's the one seam a reviewer needs
    to swap to go from template to production.
    """
    best = hits[0]
    return f"Based on '{best.doc_id}': {best.text}"


@app.post("/webhook")
def receive_webhook(payload: WebhookPayload) -> dict:
    """
    Generic inbound webhook receiver — the same shape n8n/Zapier/Make expect
    on the other end of an HTTP Request node. Dispatch by `event` to whatever
    handler a real deployment needs.
    """
    handlers = {
        "lead.created": _handle_lead_created,
        "order.updated": _handle_order_updated,
    }
    handler = handlers.get(payload.event)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"Unknown event: {payload.event}")
    return handler(payload.data)


def _handle_lead_created(data: dict) -> dict:
    return {"status": "accepted", "event": "lead.created", "echo": data}


def _handle_order_updated(data: dict) -> dict:
    return {"status": "accepted", "event": "order.updated", "echo": data}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
