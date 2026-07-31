"""Level-3 semantic chunks and three small retrieval experiments."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from functools import lru_cache
from pathlib import Path

from qdrant_client import models

from ..config import get_settings
from ..llm.base import LLMError
from ..llm.factory import get_client
from ..vectorstore.qdrant_store import VectorStore, get_store
from .chunking import Chunk
from .embeddings import get_embedder
from .ingest import extract_pages
from .retrieve import Context


_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000004")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@lru_cache
def get_semantic_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(settings.qdrant_url, f"{settings.qdrant_collection}_semantic")


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(clean) if part.strip()]


def _cosine(left: list[float], right: list[float]) -> float:
    # E5 embeddings are already normalised, so cosine similarity is a dot product.
    return sum(a * b for a, b in zip(left, right))


def semantic_chunk_pages(
    pages: list[str],
    similarity_threshold: float = 0.78,
    min_chars: int = 250,
    max_chars: int = 1000,
) -> list[Chunk]:
    """Split at low adjacent-sentence similarity while keeping useful chunk sizes."""
    embedder = get_embedder()
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_no, page_text in enumerate(pages, start=1):
        sentences = _sentences(page_text)
        if not sentences:
            continue
        vectors = embedder.embed(sentences, is_query=False)
        current = [sentences[0]]

        for position in range(1, len(sentences)):
            next_sentence = sentences[position]
            current_text = " ".join(current)
            similarity = _cosine(vectors[position - 1], vectors[position])
            would_be_too_long = len(current_text) + 1 + len(next_sentence) > max_chars
            semantic_break = len(current_text) >= min_chars and similarity < similarity_threshold

            if would_be_too_long or semantic_break:
                chunks.append(Chunk(text=current_text, page=page_no, index=chunk_index))
                chunk_index += 1
                current = [next_sentence]
            else:
                current.append(next_sentence)

        if current:
            chunks.append(
                Chunk(text=" ".join(current), page=page_no, index=chunk_index)
            )
            chunk_index += 1

    return chunks


def build_semantic_index(filename: str, reset: bool = True) -> dict[str, object]:
    settings = get_settings()
    path = Path(settings.in_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"no such PDF: {path}")

    pages = extract_pages(path)
    return build_semantic_index_from_pages(path.name, pages, reset=reset)


def build_semantic_index_from_pages(
    document: str,
    pages: list[str],
    reset: bool = True,
) -> dict[str, object]:
    """Build the isolated semantic index from pages already extracted by ingest."""
    chunks = semantic_chunk_pages(pages)
    embedder = get_embedder()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 32):
        vectors.extend(
            embedder.embed(
                [chunk.text for chunk in chunks[start : start + 32]],
                is_query=False,
            )
        )

    store = get_semantic_store()
    store.ensure_collection(dim=len(vectors[0]), reset=reset)
    store.upsert(
        [
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{document}:{chunk.index}")),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                    "source": document,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
    )
    return {
        "document": document,
        "pages": len(pages),
        "chunks": len(chunks),
        "collection": store.collection,
    }


def _contexts(hits: list[models.ScoredPoint]) -> list[Context]:
    return [
        Context(
            text=str(hit.payload.get("text", "")),
            page=int(hit.payload.get("page", 0)),
            score=float(hit.score),
        )
        for hit in hits
    ]


def _indexed_source(store: VectorStore) -> str | None:
    if not store.exists() or store.count() == 0:
        return None
    points, _ = store.client.scroll(
        collection_name=store.collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    source = points[0].payload.get("source")
    return str(source) if source else None


def _ensure_semantic_index() -> None:
    """Build Level-3 data lazily for the PDF already loaded by shared ingest."""
    original_source = _indexed_source(get_store())
    if not original_source:
        raise RuntimeError("run POST /ingest before asking a Level-3 question")

    semantic_store = get_semantic_store()
    if _indexed_source(semantic_store) == original_source:
        return

    print(f"[level3] building semantic index for {original_source}", flush=True)
    build_semantic_index(original_source, reset=True)


def _search_semantic(query: str, top_k: int) -> list[Context]:
    _ensure_semantic_index()
    vector = get_embedder().embed([query], is_query=True)[0]
    return _contexts(get_semantic_store().search(vector, top_k))


def _search_original(query: str, top_k: int) -> list[Context]:
    vector = get_embedder().embed([query], is_query=True)[0]
    return _contexts(get_store().search(vector, top_k))


def _missing_query(question: str, contexts: list[Context]) -> str | None:
    evidence = "\n\n".join(
        f"[page {context.page}] {context.text}" for context in contexts
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Decide whether the retrieved evidence is sufficient to answer every "
                "part of the document question. If sufficient, return exactly ENOUGH. "
                "Otherwise return exactly SEARCH: followed by one standalone search "
                "query for the most important missing evidence. Do not answer the "
                "original question."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nRetrieved evidence:\n{evidence}",
        },
    ]
    try:
        decision = get_client().chat(messages).strip()
    except LLMError:
        return None

    if decision.upper().startswith("SEARCH:"):
        query = decision.split(":", 1)[1].strip()
        if query:
            print(f"[level3] second-hop query: {query}", flush=True)
            return query
    print("[level3] first-hop evidence judged sufficient", flush=True)
    return None


def _merge_unique(contexts: list[Context], limit: int) -> list[Context]:
    merged: list[Context] = []
    seen: set[tuple[int, str]] = set()
    for context in contexts:
        key = (context.page, context.text)
        if key in seen:
            continue
        seen.add(key)
        merged.append(context)
        if len(merged) == limit:
            break
    return merged


def retrieve_semantic_single(question: str, top_k: int) -> list[Context]:
    """Experiment 1: one query over semantic chunks."""
    return _search_semantic(question, top_k)


def retrieve_semantic_two_hop(question: str, top_k: int) -> list[Context]:
    """Experiment 2: semantic retrieval with at most one follow-up query."""
    first = _search_semantic(question, top_k)
    missing_query = _missing_query(question, first)
    if not missing_query:
        return first
    second = _search_semantic(missing_query, top_k)
    return _merge_unique(first + second, limit=top_k * 2)


def retrieve_page_to_semantic(question: str, top_k: int) -> list[Context]:
    """Experiment 3: rank pages, then two semantic chunks inside every page."""
    _ensure_semantic_index()
    ranked_pages = _search_original(question, top_k)
    query_vector = get_embedder().embed([question], is_query=True)[0]
    store = get_semantic_store()
    contexts: list[Context] = []

    for page_context in ranked_pages:
        page_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="page",
                    match=models.MatchValue(value=page_context.page),
                )
            ]
        )
        response = store.client.query_points(
            collection_name=store.collection,
            query=query_vector,
            query_filter=page_filter,
            limit=2,
            with_payload=True,
        )
        contexts.extend(_contexts(response.points))

    print(
        f"[level3] page-to-semantic pages: {[item.page for item in ranked_pages]}",
        flush=True,
    )
    return _merge_unique(contexts, limit=top_k * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the semantic Level-3 index")
    parser.add_argument("filename")
    args = parser.parse_args()
    print(json.dumps(build_semantic_index(args.filename), indent=2))


if __name__ == "__main__":
    main()
