# api.py

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from apscheduler.schedulers.background import BackgroundScheduler

from rag.pipeline import RagPipeline
from ingest import run_ingest

API_KEY = os.environ.get("API_KEY", "dev-local-key")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://127.0.0.1:8501,http://localhost:8501").split(",")
REINDEX_INTERVAL_HOURS = int(os.environ.get("REINDEX_INTERVAL_HOURS", "168"))  # weekly by default

limiter = Limiter(key_func=get_remote_address)
scheduler = BackgroundScheduler()

print("Loading RAG pipeline (retriever + local LLM)...")
pipeline = RagPipeline()
print("Ready.")


def scheduled_reindex():
    run_ingest()
    pipeline.retriever.reload()
    print("Scheduled reindex complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(scheduled_reindex, "interval", hours=REINDEX_INTERVAL_HOURS)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(
    title="KB Assistant API",
    description="RAG-based Employee Self-Service Assistant over GitLab's public company handbook",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class Turn(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str
    history: list[Turn] = []


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model": "Qwen2.5-3B-Instruct (GGUF, CPU)",
        "chunks_indexed": len(pipeline.retriever.chunks),
    }


@app.post("/ask")
@limiter.limit("10/minute")
def ask_question(request: Request, req: QuestionRequest, _auth=Depends(require_api_key)):
    history = [t.model_dump() for t in req.history]
    return pipeline.ask(req.question, history=history)


@app.post("/admin/reindex")
def trigger_reindex(_auth=Depends(require_api_key)):
    scheduled_reindex()
    return {"status": "reindexed", "chunks_indexed": len(pipeline.retriever.chunks)}
