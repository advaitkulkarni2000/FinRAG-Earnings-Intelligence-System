"""
Full Evaluation Runner
-----------------------
Runs the complete benchmark: retrieval + generation + evaluation
for all 50 QA pairs, then reports metrics by category.

Usage:
    python eval/run_benchmark.py
    python eval/run_benchmark.py --ticker AAPL
    python eval/run_benchmark.py --difficulty easy
    python eval/run_benchmark.py --n 10  # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.benchmark import BENCHMARK, QAPair, get_benchmark
from eval.metrics import (
    precision_at_k, faithfulness_score, relevance_score, aggregate_results
)
from eval.judges import faithfulness_judge, relevance_judge
from config import PRECISION_AT_K, FAITHFULNESS_THRESHOLD, RELEVANCE_THRESHOLD


def run_single(qa: QAPair, verbose: bool = False) -> dict:
    """Run retrieval + generation + evaluation for one QA pair."""
    from src.retriever import retrieve
    from src.generator import generate_answer

    result = {
        "question":   qa.question,
        "ticker":     qa.ticker,
        "difficulty": qa.difficulty,
        "section":    qa.gold_section,
    }

    # ── Retrieval ─────────────────────────────────────────────────────────
    chunks = retrieve(qa.question, ticker_filter=qa.ticker)
    retrieved_ids = [c["chunk_id"] for c in chunks]
    gold_ids = set(qa.gold_chunk_ids)

    # Precision@k (only meaningful if gold_chunk_ids are annotated)
    if gold_ids:
        result["precision"] = precision_at_k(retrieved_ids, gold_ids, PRECISION_AT_K)
    else:
        result["precision"] = None  # unannotated — skip

    # ── Generation ────────────────────────────────────────────────────────
    gen = generate_answer(qa.question, chunks)
    result["answer"] = gen.answer
    result["expected"] = qa.answer
    result["tokens"] = gen.input_tokens + gen.output_tokens

    # ── Faithfulness ──────────────────────────────────────────────────────
    faith = faithfulness_score(gen.answer, chunks, faithfulness_judge)
    result["faithfulness"] = faith["score"]
    result["n_claims"] = faith["n_claims"]
    result["unsupported_claims"] = faith["unsupported_claims"]

    # ── Relevance ─────────────────────────────────────────────────────────
    rel = relevance_score(qa.question, gen.answer, relevance_judge)
    result["relevance"] = rel["score"]
    result["relevance_raw"] = rel["raw_score"]

    # ── Pass/fail flags ───────────────────────────────────────────────────
    result["faithful_pass"] = result["faithfulness"] >= FAITHFULNESS_THRESHOLD
    result["relevant_pass"] = rel["raw_score"] >= RELEVANCE_THRESHOLD

    if verbose:
        status = (
            "✓" if result["faithful_pass"] and result["relevant_pass"] else "✗"
        )
        print(
            f"  [{status}] {qa.ticker} | {qa.difficulty} | "
            f"faith={result['faithfulness']:.2f} "
            f"rel={result['relevance_raw']:.1f} "
            f"| {qa.question[:60]}..."
        )

    return result


def run_benchmark(
    questions: List[QAPair],
    verbose: bool = True,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run full benchmark and return aggregated results.
    """
    print(f"\n{'='*60}")
    print(f"FinRAG Evaluation Benchmark")
    print(f"Questions: {len(questions)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    results = []
    total_tokens = 0
    start_time = time.time()

    for i, qa in enumerate(questions, 1):
        print(f"[{i:02d}/{len(questions)}] ", end="", flush=True)
        try:
            r = run_single(qa, verbose=verbose)
            results.append(r)
            total_tokens += r.get("tokens", 0)
        except Exception as e:
            print(f"ERROR on question {i}: {e}")
        time.sleep(0.2)  # rate limit buffer

    elapsed = time.time() - start_time

    # ── Aggregated metrics ────────────────────────────────────────────────
    # Filter to questions with all metrics
    complete = [r for r in results if "faithfulness" in r and "relevance" in r]
    agg = aggregate_results(complete)
    agg["total_tokens"] = total_tokens
    agg["elapsed_seconds"] = round(elapsed, 1)

    # ── By difficulty ─────────────────────────────────────────────────────
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in complete if r["difficulty"] == diff]
        if subset:
            agg[f"faithfulness_{diff}"] = sum(r["faithfulness"] for r in subset) / len(subset)
            agg[f"relevance_{diff}"] = sum(r["relevance"] for r in subset) / len(subset)
            agg[f"n_{diff}"] = len(subset)

    # ── By ticker ─────────────────────────────────────────────────────────
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        subset = [r for r in complete if r["ticker"] == ticker]
        if subset:
            agg[f"faithfulness_{ticker}"] = sum(r["faithfulness"] for r in subset) / len(subset)
            agg[f"relevance_{ticker}"] = sum(r["relevance"] for r in subset) / len(subset)

    # ── Pass rates ────────────────────────────────────────────────────────
    agg["faithful_pass_rate"] = sum(1 for r in complete if r.get("faithful_pass")) / len(complete)
    agg["relevant_pass_rate"] = sum(1 for r in complete if r.get("relevant_pass")) / len(complete)
    agg["both_pass_rate"] = sum(
        1 for r in complete if r.get("faithful_pass") and r.get("relevant_pass")
    ) / len(complete)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Questions evaluated: {len(complete)}/{len(questions)}")
    print(f"Time elapsed: {elapsed:.0f}s | Tokens used: {total_tokens:,}")
    print()
    print(f"{'Metric':<30} {'Score':>8} {'Threshold':>10} {'Pass':>6}")
    print(f"{'-'*56}")
    print(
        f"{'Mean Faithfulness':<30} "
        f"{agg.get('mean_faithfulness', 0):.3f}    "
        f"{FAITHFULNESS_THRESHOLD:.1f}       "
        f"{'✓' if agg.get('mean_faithfulness', 0) >= FAITHFULNESS_THRESHOLD else '✗'}"
    )
    print(
        f"{'Mean Relevance (0-1)':<30} "
        f"{agg.get('mean_relevance', 0):.3f}    "
        f"{(RELEVANCE_THRESHOLD-1)/4:.2f}       "
        f"{'✓' if agg.get('mean_relevance', 0) >= (RELEVANCE_THRESHOLD-1)/4 else '✗'}"
    )
    print(f"\nPass rates:")
    print(f"  Faithful: {agg.get('faithful_pass_rate', 0):.1%}")
    print(f"  Relevant: {agg.get('relevant_pass_rate', 0):.1%}")
    print(f"  Both:     {agg.get('both_pass_rate', 0):.1%}")

    if "faithfulness_easy" in agg:
        print(f"\nFaithfulness by difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if f"faithfulness_{diff}" in agg:
                print(f"  {diff:<8}: {agg[f'faithfulness_{diff}']:.3f}")

    # ── Failed questions ──────────────────────────────────────────────────
    failed = [
        r for r in complete
        if not r.get("faithful_pass") or not r.get("relevant_pass")
    ]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed[:10]:  # show first 10
            print(
                f"  ✗ [{r['ticker']}|{r['difficulty']}] "
                f"faith={r['faithfulness']:.2f} "
                f"rel={r['relevance_raw']:.1f} | "
                f"{r['question'][:55]}..."
            )

    # ── Save results ──────────────────────────────────────────────────────
    output = {"summary": agg, "results": results}
    if output_path:
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nFull results saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run FinRAG evaluation benchmark")
    parser.add_argument("--ticker", help="Filter by ticker (AAPL/MSFT/NVDA)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--n", type=int, help="Run only first N questions (smoke test)")
    parser.add_argument("--output", default="eval/results_latest.json",
                        help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    questions = get_benchmark(ticker=args.ticker, difficulty=args.difficulty)
    if args.n:
        questions = questions[:args.n]

    if not questions:
        print("No questions match the filter.")
        return

    output_path = Path(args.output)
    run_benchmark(questions, verbose=not args.quiet, output_path=output_path)


if __name__ == "__main__":
    main()
