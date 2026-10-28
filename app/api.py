from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = RAGPipeline()
    if pipe.store.count() == 0:  # first boot: index the bundled docs
        pipe.ingest()
    state["pipe"] = pipe
    yield
    state.clear()


app = FastAPI(title="Enterprise Document RAG Assistant", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    id: int
    source: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    sources: List[Source]
    latency_ms: dict


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": state["pipe"].store.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return state["pipe"].ask(req.question)


@app.post("/ingest")
def ingest():
    try:
        return state["pipe"].ingest()
    except Exception as e:  # surface ingestion problems clearly
        raise HTTPException(status_code=500, detail=str(e))
