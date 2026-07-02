---
title: KB Assistant
emoji: 🗂️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# KB Assistant — Employee Self-Service RAG Chatbot

A retrieval-augmented generation (RAG) assistant that answers employee questions
(onboarding, benefits, time off, learning & development, remote-work policy) grounded
in a real company knowledge base. The knowledge base is a curated subset of
[GitLab's public company handbook](https://handbook.gitlab.com/handbook/) — GitLab
runs its actual company on this handbook, so it's a legitimate stand-in for an
internal enterprise KB, and every answer is independently verifiable against the
live source.

## Architecture

```
                     ┌─────────────┐
  question ─────────▶│  Retriever  │  sentence-transformers (MiniLM) + FAISS
                     └──────┬──────┘  over 1.7k chunks of handbook markdown
                            │ top-k relevant chunks
                            ▼
                     ┌─────────────┐
                     │  Generator  │  Qwen2.5-3B-Instruct, GGUF q4, CPU inference
                     └──────┬──────┘  (llama.cpp) — grounded, cites sources,
                            │          declines when context is irrelevant
                            ▼
                       answer + sources
```

- `ingest.py` — pulls markdown pages from GitLab's public handbook source repo
  (via the GitLab API, avoiding the site's bot-protection layer), chunks by heading,
  embeds with `all-MiniLM-L6-v2`, and builds a FAISS index (`data/kb_index.faiss`,
  `data/kb_chunks.pkl`).
- `rag/retriever.py` — embeds the query and does nearest-neighbor search over the
  index; low-relevance results are filtered by distance threshold before ever
  reaching the LLM (see "Why this design" below).
- `rag/generator.py` — wraps a locally-run, quantized instruct model via
  `llama-cpp-python`. No external API calls, no per-query cost.
- `rag/pipeline.py` — orchestrates retrieve → filter → generate, with a short
  conversation history window for multi-turn follow-ups.
- `api.py` — FastAPI service exposing `POST /ask` and `GET /health`.
- `app.py` — Streamlit chat UI with source citations and example questions.

## Running locally

```bash
pip install -r requirements.txt
python scripts/download_model.py   # one-time, ~2GB
python ingest.py                   # one-time, builds data/kb_index.faiss + kb_chunks.pkl

uvicorn api:app --reload           # terminal 1
streamlit run app.py               # terminal 2
```

## Running with Docker

```bash
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501

## Evaluation

`python scripts/evaluate_retrieval.py` runs the retriever against a hand-written set
of 22 in-domain questions (each tied to a known-correct handbook page) and 5
out-of-domain questions (should be rejected, not answered). Current results:

| Metric | Value |
|---|---|
| Hit-rate@1 | 77% (17/22) |
| Hit-rate@3 | 82% (18/22) |
| Hit-rate@5 | 91% (20/22) |
| MRR | 0.818 |
| Correct rejection rate (out-of-domain) | 80% (4/5) |

Two honest findings from this run: (1) hit-rate@1 lags hit-rate@5 by 14 points,
meaning the correct page is usually retrieved but not always ranked first — a
cross-encoder reranker over the top-5 candidates would likely close most of that
gap; (2) the one rejection failure was "What is GitLab's current stock price?" —
topically close enough to the benefits/compensation content to slip under the
distance threshold even though it isn't actually answerable, showing that a purely
distance-based relevance filter has a real, identifiable failure mode. Both are
concrete, evidence-based next steps rather than guesses.

## Why this design

- **Grounded, not hallucinated.** The system prompt restricts the model to the
  retrieved context, and low-relevance retrievals are filtered out before
  generation — the assistant explicitly says "I don't have that information in
  the knowledge base" for off-topic questions rather than guessing. Verified in
  testing: asking about GitLab's parental-leave policy returns a grounded,
  cited answer; asking an unrelated question (e.g. "what's the capital of
  France?") is declined in under 100ms without ever invoking the LLM.
- **Small quantized model, CPU-only.** This runs on a laptop with no GPU
  (~7.6GB RAM). A 3B-parameter model in q4 quantization (~2GB) was chosen over
  a larger 7B+ model to fit that budget. The tradeoff is latency: real
  questions currently take ~20–40 seconds end-to-end on CPU. In a production
  deployment this would be traded for a GPU-backed model server or a hosted
  LLM API for sub-second responses — this project deliberately optimizes for
  "runs anywhere, zero API cost" instead.
- **Self-contained container.** The Docker image bakes in the model and index
  so the whole stack (retrieval + generation) ships as one artifact, which is
  what makes a single-container deploy target (e.g. Hugging Face Spaces)
  possible without a separate model-serving dependency.

## Live demo

**https://huggingface.co/spaces/TasosIggl/kb-assistant**
