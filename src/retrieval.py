"""Embedding-based catalog retrieval with Gemini + FAISS."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from embedding_text import build_embedding_text

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
INDEX_PATH = ROOT / "data" / "catalog.index"
IDS_PATH = ROOT / "data" / "catalog_ids.json"
META_PATH = ROOT / "data" / "catalog_index_meta.json"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 20
# Free tier: 100 embed requests/min (each text in a batch counts). ~20 texts → wait ≥12s.
INTER_BATCH_SLEEP_SEC = 15.0
MAX_RETRIES = 8
INITIAL_BACKOFF_SEC = 5.0

_index: faiss.Index | None = None
_catalog_by_id: dict[str, dict] | None = None
_id_order: list[str] | None = None


def _catalog_hash(path: Path = CATALOG_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _get_client() -> genai.Client:
    """Load .env then delegate auth entirely to the google-genai SDK."""
    load_dotenv(ROOT / ".env")
    return genai.Client()


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    """Use API-suggested retry delay when present, else exponential backoff."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0
    return INITIAL_BACKOFF_SEC * (2**attempt)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "rate limit", "quota", "resource exhausted", "too many requests")
    )


def embed_texts(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    client: genai.Client | None = None,
) -> np.ndarray:
    """Embed a list of texts with retry/backoff. Never silently drops entries."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    client = client or _get_client()
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        batch_vectors: list[list[float]] | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                batch_vectors = [emb.values for emb in response.embeddings]
                if len(batch_vectors) != len(batch):
                    raise RuntimeError(
                        f"Embedding batch size mismatch: sent {len(batch)}, "
                        f"got {len(batch_vectors)}"
                    )
                break
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < MAX_RETRIES - 1:
                    wait = _retry_wait_seconds(exc, attempt)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Failed to embed batch starting at index {start} "
                    f"after {attempt + 1} attempt(s): {exc}"
                ) from exc

        assert batch_vectors is not None
        all_vectors.extend(batch_vectors)

        # Free-tier quota: each embedded text counts toward 100/min limit
        if start + EMBED_BATCH_SIZE < len(texts):
            time.sleep(INTER_BATCH_SLEEP_SEC)

    return np.array(all_vectors, dtype=np.float32)


def _write_meta(catalog_hash: str, dimension: int) -> None:
    meta = {
        "catalog_hash": catalog_hash,
        "embedding_model": EMBEDDING_MODEL,
        "dimension": dimension,
        "index_type": "IndexFlatIP (cosine via L2-normalized inner product)",
        "entry_count": len(json.loads(IDS_PATH.read_text(encoding="utf-8"))),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def build_index(*, force: bool = False) -> None:
    """
    Build and persist FAISS index from catalog.json.

    Skips rebuild if index exists and catalog hash is unchanged, unless force=True.
    """
    catalog_hash = _catalog_hash()
    if (
        not force
        and INDEX_PATH.exists()
        and IDS_PATH.exists()
        and META_PATH.exists()
    ):
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if meta.get("catalog_hash") == catalog_hash:
            return

    catalog = load_catalog()
    texts = [build_embedding_text(entry) for entry in catalog]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    entity_ids = [entry["entity_id"] for entry in catalog]
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    IDS_PATH.write_text(json.dumps(entity_ids, indent=2) + "\n", encoding="utf-8")
    _write_meta(catalog_hash, dim)


def _load_index_resources() -> tuple[faiss.Index, list[str], dict[str, dict]]:
    global _index, _id_order, _catalog_by_id

    if _index is None or _id_order is None or _catalog_by_id is None:
        if not INDEX_PATH.exists():
            build_index()
        _index = faiss.read_index(str(INDEX_PATH))
        _id_order = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        catalog = load_catalog()
        _catalog_by_id = {entry["entity_id"]: entry for entry in catalog}

    return _index, _id_order, _catalog_by_id


def retrieve(query: str, k: int = 10) -> list[dict]:
    """
    Embed query, search FAISS, return top-k full catalog entry dicts with scores.

    Each result dict is the catalog entry plus a '_score' field (cosine similarity).
    """
    index, id_order, catalog_by_id = _load_index_resources()

    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
    faiss.normalize_L2(query_vec)

    k = min(k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue
        entity_id = id_order[idx]
        entry = dict(catalog_by_id[entity_id])
        entry["_score"] = float(score)
        results.append(entry)

    return results


def example_embedding_texts(n: int = 3) -> list[dict[str, str]]:
    """Return name + constructed embedding text for the first n catalog entries."""
    catalog = load_catalog()
    samples = catalog[:n]
    return [
        {"name": entry["name"], "embedding_text": build_embedding_text(entry)}
        for entry in samples
    ]
