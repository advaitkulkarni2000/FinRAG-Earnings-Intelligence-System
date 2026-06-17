"""
Hybrid Retriever: Dense + BM25 Reranking
-----------------------------------------
Dense retrieval alone misses exact keyword matches — ticker symbols,
specific financial figures, named executives. BM25 reranking over the
dense candidates combines semantic similarity with lexical overlap.

Pipeline:
1. Dense retrieval: top-DENSE_TOP_K candidates from ChromaDB
2. BM25 reranking: score each candidate against the query
3. Reciprocal Rank Fusion: combine dense and BM25 scores
4. Return top-RERANK_TOP_K results
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DENSE_TOP_K, RERANK_TOP_K


# ── BM25 implementation ───────────────────────────────────────────────────

class BM25:
    """
    Lightweight BM25 scorer.
    k1=1.5, b=0.75 are standard defaults from the original paper.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = []
        self._doc_freqs: List[Counter] = []
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    def fit(self, corpus: List[str]) -> None:
        """Fit BM25 on a corpus of strings."""
        self._corpus = [self._tokenise(doc) for doc in corpus]
        self._n_docs = len(self._corpus)
        self._doc_freqs = [Counter(doc) for doc in self._corpus]

        # Average document length
        total_len = sum(len(doc) for doc in self._corpus)
        self._avg_dl = total_len / self._n_docs if self._n_docs > 0 else 1.0

        # IDF for each term
        term_doc_count: Counter = Counter()
        for doc in self._corpus:
            for term in set(doc):
                term_doc_count[term] += 1

        self._idf = {}
        for term, df in term_doc_count.items():
            self._idf[term] = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1
            )

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score for a single query-document pair."""
        tokens = self._tokenise(query)
        doc = self._corpus[doc_index]
        doc_len = len(doc)
        tf_counts = self._doc_freqs[doc_index]

        score = 0.0
        for term in tokens:
            if term not in self._idf:
                continue
            tf = tf_counts.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self._avg_dl
            )
            score += self._idf[term] * numerator / denominator
        return score

    def rank(self, query: str) -> List[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted descending."""
        scores = [(i, self.score(query, i)) for i in range(self._n_docs)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_ranks: List[tuple[int, float]],
    k: int = 60,
) -> List[int]:
    """
    Combine dense and BM25 rankings via RRF.
    Returns sorted indices into dense_results.
    """
    rrf_scores: Dict[int, float] = {}

    # Dense rank contribution
    for rank, result in enumerate(dense_results):
        rrf_scores[rank] = rrf_scores.get(rank, 0) + 1.0 / (k + rank + 1)

    # BM25 rank contribution (indices are into dense_results)
    for bm25_rank, (doc_idx, _) in enumerate(bm25_ranks):
        if doc_idx < len(dense_results):
            rrf_scores[doc_idx] = (
                rrf_scores.get(doc_idx, 0) + 1.0 / (k + bm25_rank + 1)
            )

    sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    return sorted_indices


# ── Main retriever ────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = RERANK_TOP_K,
    ticker_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval for a query string.

    Args:
        query:         Natural language question
        top_k:         Number of chunks to return after reranking
        ticker_filter: Optional ticker symbol to restrict results

    Returns:
        List of chunk dicts (chunk_id, text, metadata, score, bm25_score)
        sorted by RRF score descending.
    """
    from src.embeddings import embed_query
    from src.vectorstore import query_dense

    # 1. Dense retrieval
    query_emb = embed_query(query)
    where = {"ticker": ticker_filter.upper()} if ticker_filter else None
    dense_results = query_dense(query_emb, top_k=DENSE_TOP_K, where=where)

    if not dense_results:
        return []

    # 2. BM25 reranking over dense candidates
    corpus = [r["text"] for r in dense_results]
    bm25 = BM25()
    bm25.fit(corpus)
    bm25_ranks = bm25.rank(query)

    # 3. RRF fusion
    fused_indices = _reciprocal_rank_fusion(dense_results, bm25_ranks)

    # 4. Return top_k
    results = []
    for idx in fused_indices[:top_k]:
        result = dense_results[idx].copy()
        result["bm25_score"] = bm25.score(query, idx)
        results.append(result)

    return results
