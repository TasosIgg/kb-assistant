# ingest.py
#
# Pulls a curated subset of GitLab's public company handbook (real HR/IT/onboarding
# policy content, served as markdown from GitLab's own handbook source repo),
# chunks it by heading, embeds each chunk, and builds a FAISS index + chunk store
# for the RAG pipeline in rag/.

import re
import time
import pickle
import urllib.parse

import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = "gitlab-com/content-sites/handbook"
API_ROOT = f"https://gitlab.com/api/v4/projects/{urllib.parse.quote_plus(PROJECT)}/repository"
REF = "main"
HANDBOOK_BASE_URL = "https://handbook.gitlab.com"

# Curated directories: real HR / IT / onboarding / benefits content, kept bounded
# so the demo index stays small and fast on a CPU-only machine.
SEED_DIRS = [
    "content/handbook/people-group/general-onboarding",
    "content/handbook/people-group/offboarding",
    "content/handbook/people-group/time-off-and-absence",
    "content/handbook/people-group/learning-and-development",
    "content/handbook/total-rewards",
    "content/handbook/total-rewards/benefits",
    "content/handbook/company/culture/all-remote",
]

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_MAX_CHARS = 1800   # ~300-500 tokens
CHUNK_OVERLAP_CHARS = 200

INDEX_PATH = "data/kb_index.faiss"
CHUNKS_PATH = "data/kb_chunks.pkl"


def list_markdown_files(dir_path: str) -> list[str]:
    """Recursively list .md file paths under a handbook source directory."""
    resp = requests.get(
        f"{API_ROOT}/tree",
        params={"path": dir_path, "per_page": 100, "ref": REF},
        timeout=20,
    )
    resp.raise_for_status()
    entries = resp.json()

    files = []
    for entry in entries:
        if entry["type"] == "tree":
            files.extend(list_markdown_files(entry["path"]))
        elif entry["name"].endswith(".md"):
            files.append(entry["path"])
    return files


def fetch_markdown(file_path: str) -> str:
    encoded = urllib.parse.quote_plus(file_path)
    resp = requests.get(f"{API_ROOT}/files/{encoded}/raw", params={"ref": REF}, timeout=20)
    resp.raise_for_status()
    return resp.text


def strip_frontmatter_and_title(md: str) -> tuple[str, str]:
    """Return (title, body) with the YAML frontmatter removed."""
    title = ""
    body = md
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            frontmatter = md[3:end]
            body = md[end + 4:]
            m = re.search(r'^title:\s*"?([^"\n]+)"?', frontmatter, re.MULTILINE)
            if m:
                title = m.group(1).strip()
    return title, body


def chunk_by_heading(body: str, title: str) -> list[str]:
    """Split on markdown headings, then hard-wrap any oversized section."""
    sections = re.split(r"\n(?=#{1,4} )", body)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_MAX_CHARS:
            chunks.append(section)
        else:
            start = 0
            while start < len(section):
                end = start + CHUNK_MAX_CHARS
                chunks.append(section[start:end])
                start = end - CHUNK_OVERLAP_CHARS
    # Prefix every chunk with the page title for retrieval/generation context.
    return [f"{title}\n\n{c}" if title else c for c in chunks if len(c.strip()) > 40]


def file_path_to_url(file_path: str) -> str:
    # content/handbook/people-group/foo/_index.md -> /handbook/people-group/foo/
    rel = file_path.replace("content/handbook", "/handbook").replace("_index.md", "").replace(".md", "/")
    return HANDBOOK_BASE_URL + rel


def main():
    print("Discovering handbook pages...")
    md_files = []
    for seed in SEED_DIRS:
        try:
            md_files.extend(list_markdown_files(seed))
        except requests.HTTPError as e:
            print(f"  skip {seed}: {e}")
    md_files = sorted(set(md_files))
    print(f"Found {len(md_files)} markdown pages.")

    chunks = []  # list of {"text", "title", "url"}
    for i, path in enumerate(md_files):
        try:
            raw = fetch_markdown(path)
        except requests.HTTPError as e:
            print(f"  skip {path}: {e}")
            continue

        title, body = strip_frontmatter_and_title(raw)
        url = file_path_to_url(path)
        for text in chunk_by_heading(body, title):
            chunks.append({"text": text, "title": title or path, "url": url})

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(md_files)} pages, {len(chunks)} chunks so far")
        time.sleep(0.05)  # be polite to the API

    print(f"Total chunks: {len(chunks)}")

    print(f"Embedding chunks with {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index to {INDEX_PATH} and chunk store to {CHUNKS_PATH}.")


if __name__ == "__main__":
    main()
