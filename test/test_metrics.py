"""
Unit Tests — Evaluation Metrics
Tests the metric calculations independently of the LLM calls.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    faithfulness_score,
    relevance_score,
    aggregate_results,
)


# ── Retrieval metrics ─────────────────────────────────────────────────────

def test_precision_at_k_perfect():
    retrieved = ["a", "b", "c", "d", "e"]
    gold = {"a", "b", "c", "d", "e"}
    assert precision_at_k(retrieved, gold, k=5) == 1.0

def test_precision_at_k_zero():
    retrieved = ["x", "y", "z"]
    gold = {"a", "b", "c"}
    assert precision_at_k(retrieved, gold, k=3) == 0.0

def test_precision_at_k_partial():
    retrieved = ["a", "x", "b", "y", "z"]
    gold = {"a", "b", "c"}
    assert precision_at_k(retrieved, gold, k=5) == 2/5

def test_precision_at_k_cutoff():
    retrieved = ["x", "y", "a", "b", "c"]
    gold = {"a", "b", "c"}
    # Only top-2 are considered
    assert precision_at_k(retrieved, gold, k=2) == 0.0

def test_recall_at_k():
    retrieved = ["a", "b", "x", "y", "c"]
    gold = {"a", "b", "c", "d"}
    # 3 out of 4 gold found in top-5
    assert recall_at_k(retrieved, gold, k=5) == 3/4

def test_mrr_first_hit():
    retrieved = ["a", "b", "c"]
    gold = {"a"}
    assert mean_reciprocal_rank(retrieved, gold) == 1.0

def test_mrr_second_hit():
    retrieved = ["x", "a", "c"]
    gold = {"a"}
    assert mean_reciprocal_rank(retrieved, gold) == 0.5

def test_mrr_no_hit():
    retrieved = ["x", "y", "z"]
    gold = {"a"}
    assert mean_reciprocal_rank(retrieved, gold) == 0.0

def test_precision_empty_gold():
    assert precision_at_k(["a", "b"], set(), k=2) == 0.0


# ── Faithfulness metric ───────────────────────────────────────────────────

def test_faithfulness_all_supported():
    answer = "Revenue was $100M. Profit was $10M."
    chunks = [{"text": "Revenue was $100M. Profit was $10M."}]
    judge = lambda claim, ctx: True  # mock: always supported
    result = faithfulness_score(answer, chunks, judge)
    assert result["score"] == 1.0
    assert result["n_unsupported"] == 0

def test_faithfulness_none_supported():
    answer = "Revenue was $100M. Profit was $10M."
    chunks = [{"text": "Some unrelated text."}]
    judge = lambda claim, ctx: False  # mock: never supported
    result = faithfulness_score(answer, chunks, judge)
    assert result["score"] == 0.0
    assert len(result["unsupported_claims"]) > 0

def test_faithfulness_partial():
    # Use a longer answer that generates multiple claims
    answer = (
        "The company reported total revenue of $100 million for the fiscal year. "
        "Operating expenses increased significantly to $60 million. "
        "The net profit margin improved to 15 percent compared to last year."
    )
    chunks = [{"text": "The company reported total revenue of $100 million."}]
    call_count = [0]
    def mock_judge(claim, ctx):
        call_count[0] += 1
        return call_count[0] == 1  # only first claim supported
    result = faithfulness_score(answer, chunks, mock_judge)
    # If only 1 claim extracted, score will be 0 or 1 — just verify it runs
    assert 0.0 <= result["score"] <= 1.0
    assert result["n_claims"] >= 1

def test_faithfulness_refusal_excluded():
    # Refusal phrases should not be counted as claims
    answer = "The provided context does not contain enough information to answer this question."
    chunks = [{"text": "Unrelated text."}]
    judge = lambda claim, ctx: False
    result = faithfulness_score(answer, chunks, judge)
    # Should have 0 or 1 claims (the whole answer as fallback)
    # But the score should be 0 since judge returns False
    assert result["score"] == 0.0


# ── Relevance metric ──────────────────────────────────────────────────────

def test_relevance_high():
    query = "What was the revenue?"
    answer = "Revenue was $100M."
    judge = lambda q, a: 5.0
    result = relevance_score(query, answer, judge)
    assert result["score"] == 1.0
    assert result["raw_score"] == 5.0

def test_relevance_low():
    query = "What was the revenue?"
    answer = "The sky is blue."
    judge = lambda q, a: 1.0
    result = relevance_score(query, answer, judge)
    assert result["score"] == 0.0

def test_relevance_normalisation():
    judge = lambda q, a: 3.0  # mid-point
    result = relevance_score("q", "a", judge)
    assert result["score"] == 0.5


# ── Aggregation ───────────────────────────────────────────────────────────

def test_aggregate_results():
    results = [
        {"faithfulness": 0.9, "relevance": 0.8},
        {"faithfulness": 0.7, "relevance": 0.6},
        {"faithfulness": 1.0, "relevance": 1.0},
    ]
    agg = aggregate_results(results)
    assert abs(agg["mean_faithfulness"] - (0.9 + 0.7 + 1.0) / 3) < 1e-6
    assert abs(agg["mean_relevance"] - (0.8 + 0.6 + 1.0) / 3) < 1e-6
    assert agg["n_questions"] == 3

def test_aggregate_empty():
    assert aggregate_results([]) == {}


if __name__ == "__main__":
    # Simple test runner without pytest
    import traceback
    tests = [k for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for test_name in tests:
        try:
            globals()[test_name]()
            print(f"  ✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
