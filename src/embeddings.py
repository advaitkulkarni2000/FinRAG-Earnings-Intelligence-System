"""
Embedding Model Wrapper
-----------------------
Thin wrapper around sentence-transformers with:
- Lazy loading (model loaded once on first use)
- Batch encoding with progress logging
- Normalised embeddings for cosine similarity via dot product
"""
from __future__ import annotations

from typing import List
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import sys
        sys.path.insert(0, __file__.replace("src/embeddings.py", ""))
        from config import EMBEDDING_MODEL
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of texts.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM), L2-normalised.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2 norm → dot product = cosine sim
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (EMBEDDING_DIM,)."""
    return embed_texts([query])[0]
