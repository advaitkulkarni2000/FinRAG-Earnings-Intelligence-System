"""
ChromaDB Vector Store Interface
--------------------------------
Manages a persistent ChromaDB collection for the ingested chunks.
Handles add, query, and metadata filtering.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, COLLECTION_NAME, DENSE_TOP_K
from src.ingestion import Chunk


def _get_client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_collection(client=None):
    if client is None:
        client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: List[Chunk]) -> None:
    """
    Embed and add chunks to the vector store.
    Skips chunks already present (idempotent).
    """
    from src.embeddings import embed_texts

    collection = _get_collection()
    existing_ids = set(collection.get()["ids"])

    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    if not new_chunks:
        print("All chunks already in vector store — nothing to add.")
        return

    print(f"Embedding {len(new_chunks)} chunks...")
    texts = [c.text for c in new_chunks]
    embeddings = embed_texts(texts, show_progress=True)

    collection.add(
        ids=[c.chunk_id for c in new_chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            "ticker":      c.ticker,
            "doc_type":    c.doc_type,
            "period":      c.period,
            "section":     c.section,
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
        } for c in new_chunks],
    )
    print(f"Added {len(new_chunks)} chunks to vector store.")


def query_dense(
    query_embedding: np.ndarray,
    top_k: int = DENSE_TOP_K,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Dense retrieval: returns top_k most similar chunks.

    Returns list of dicts with keys: chunk_id, text, metadata, score
    """
    collection = _get_collection()

    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for i, (chunk_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        output.append({
            "chunk_id": chunk_id,
            "text": doc,
            "metadata": meta,
            "score": 1 - dist,  # cosine distance → similarity
        })
    return output


def get_all_chunks() -> List[Dict[str, Any]]:
    """Retrieve all chunks (for BM25 index building)."""
    collection = _get_collection()
    results = collection.get(include=["documents", "metadatas"])
    return [
        {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        )
    ]


def count() -> int:
    return _get_collection().count()
