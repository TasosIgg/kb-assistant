# api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import RagPipeline

app = FastAPI(
    title="KB Assistant API",
    description="RAG-based Employee Self-Service Assistant over GitLab's public company handbook",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading RAG pipeline (retriever + local LLM)...")
pipeline = RagPipeline()
print("Ready.")


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
def ask_question(req: QuestionRequest):
    history = [t.model_dump() for t in req.history]
    return pipeline.ask(req.question, history=history)
