"""
Evaluation Metrics
-------------------
Three metrics, each measuring a different failure mode:

1. Retrieval Precision@k
   Measures whether retrieved chunks are actually relevant.
   Failure mode: retrieving plausible-looking but wrong passages.

2. Answer Faithfulness
   Measures whether the answer is grounded in the context.
   Failure mode: hallucination — answering from parametric knowledge
   rather than the retrieved evidence.
   Computed at claim level: decompose answer into individual claims,
   check each claim against context. This catches partial hallucination
   that answer-level scoring would hide.

3. Answer Relevance
   Measures whether the answer addresses the question asked.
   Failure mode: technically grounded answer that doesn't answer the question.

Design note: faithfulness and relevance use an LLM judge (judges.py).
This is standard practice (RAGAS, TruLens use the same approach).
The limitation — same model family as generator — is documented honestly
in the README.
"""
from __future__ import annotations

import re
from typing import List, Set, Dict, Any


# ── Retrieval Precision@k ─────────────────────────────────────────────────

def precision_at_k(
    retrieved_ids: List[str],
    gold_ids: Set[str],
    k: int,
) -> float:
    """
    Fraction of top-k retrieved chunks that are in the gold set.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs
        gold_ids:      Set of chunk IDs that should have been retrieved
        k:             Cutoff

    Returns:
        Float in [0, 1]
    """
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in gold_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: List[str],
    gold_ids: Set[str],
    k: int,
) -> float:
    """Fraction of gold chunks that appear in top-k retrieved."""
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & gold_ids)
    return hits / len(gold_ids)


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    gold_ids: Set[str],
) -> float:
    """MRR: 1/rank of the first correct result. 0 if none found."""
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


# ── Answer Faithfulness ────────────────────────────────────────────────────

def _extract_claims(answer: str) -> List[str]:
    """
    Decompose an answer into individual factual claims.

    Simple heuristic: split on sentence boundaries, filter to
    sentences that look like factual assertions (contain a verb,
    are not questions, are not "I don't know" style refusals).
    """
    # Split on sentence-ending punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    claims = []
    for sent in raw_sentences:
        sent = sent.strip()
        # Filter out non-claims
        if len(sent) < 20:
            continue
        if sent.endswith("?"):
            continue
        if any(phrase in sent.lower() for phrase in [
            "does not contain", "not enough information",
            "cannot determine", "i don't", "i cannot"
        ]):
            continue
        claims.append(sent)

    return claims if claims else [answer]  # fallback: treat whole answer as one claim


def faithfulness_score(
    answer: str,
    context_chunks: List[Dict[str, Any]],
    judge_fn,  # callable(claim: str, context: str) -> bool
) -> Dict[str, Any]:
    """
    Claim-level faithfulness score.

    Args:
        answer:        Generated answer to evaluate
        context_chunks: Retrieved chunks used to generate the answer
        judge_fn:      Function that returns True if a claim is supported

    Returns:
        Dict with score (float), claims, supported_claims, unsupported_claims
    """
    claims = _extract_claims(answer)
    context = "\n\n".join(c["text"] for c in context_chunks)

    supported = []
    unsupported = []

    for claim in claims:
        is_supported = judge_fn(claim, context)
        if is_supported:
            supported.append(claim)
        else:
            unsupported.append(claim)

    score = len(supported) / len(claims) if claims else 0.0

    return {
        "score": score,
        "n_claims": len(claims),
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
    }


# ── Answer Relevance ───────────────────────────────────────────────────────

def relevance_score(
    query: str,
    answer: str,
    judge_fn,  # callable(query: str, answer: str) -> float (1-5)
) -> Dict[str, Any]:
    """
    Answer relevance: does the answer address the question?

    Args:
        query:    The original question
        answer:   The generated answer
        judge_fn: Function returning float in [1, 5]

    Returns:
        Dict with score (float normalised to [0,1]), raw_score
    """
    raw_score = judge_fn(query, answer)
    normalised = (raw_score - 1) / 4  # map [1,5] → [0,1]

    return {
        "score": normalised,
        "raw_score": raw_score,
    }


# ── Aggregate metrics ─────────────────────────────────────────────────────

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute mean metrics across a benchmark run.

    Expected keys per result: precision, faithfulness, relevance
    """
    if not results:
        return {}

    metrics = ["precision", "faithfulness", "relevance"]
    agg = {}
    for m in metrics:
        values = [r[m] for r in results if m in r]
        if values:
            agg[f"mean_{m}"] = sum(values) / len(values)
            agg[f"min_{m}"] = min(values)
            agg[f"max_{m}"] = max(values)

    agg["n_questions"] = len(results)
    return agg
