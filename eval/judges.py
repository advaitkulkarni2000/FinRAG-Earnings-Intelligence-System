"""
LLM Judges
-----------
Programmatic evaluation using Claude as a judge.

These are deterministic (temperature=0) LLM calls that score:
- Faithfulness: is a claim supported by the context?
- Relevance: does the answer address the question?

Important limitation documented honestly:
We use the same model family (Claude) as both generator and judge.
An ideal setup would use a different model family as judge to avoid
systematic bias. This is noted in the README as a known limitation.

Each judge is a pure function: same inputs → same outputs.
"""
from __future__ import annotations

import os
import re
from typing import Optional
from functools import lru_cache

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAITHFULNESS_JUDGE_PROMPT, RELEVANCE_JUDGE_PROMPT


def _call_claude(prompt: str, max_tokens: int = 10) -> str:
    """Make a minimal Claude API call for judging."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheapest model for judging
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def faithfulness_judge(claim: str, context: str) -> bool:
    """
    Returns True if the claim is directly supported by the context.

    Uses a binary supported/not_supported classification.
    Haiku is used here (not Sonnet) to keep evaluation cost low.
    """
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        claim=claim,
        context=context[:3000],  # truncate very long contexts
    )
    response = _call_claude(prompt, max_tokens=10)
    return "supported" in response.lower() and "not_supported" not in response.lower()


def relevance_judge(query: str, answer: str) -> float:
    """
    Returns a float in [1, 5] for answer relevance to the query.
    """
    prompt = RELEVANCE_JUDGE_PROMPT.format(
        question=query,
        answer=answer[:1000],  # truncate very long answers
    )
    response = _call_claude(prompt, max_tokens=5)

    # Extract number from response
    match = re.search(r'[1-5]', response)
    if match:
        return float(match.group())

    # Fallback: try to parse directly
    try:
        val = float(response.strip())
        return max(1.0, min(5.0, val))
    except ValueError:
        return 3.0  # neutral fallback, logged as warning


# ── Batch judging with rate limiting ─────────────────────────────────────

def batch_faithfulness(
    claims_and_contexts: list[tuple[str, str]],
    delay_seconds: float = 0.1,
) -> list[bool]:
    """
    Judge faithfulness for a batch of (claim, context) pairs.
    Adds a small delay between calls to respect rate limits.
    """
    import time
    results = []
    for claim, context in claims_and_contexts:
        results.append(faithfulness_judge(claim, context))
        time.sleep(delay_seconds)
    return results


def batch_relevance(
    queries_and_answers: list[tuple[str, str]],
    delay_seconds: float = 0.1,
) -> list[float]:
    """Judge relevance for a batch of (query, answer) pairs."""
    import time
    results = []
    for query, answer in queries_and_answers:
        results.append(relevance_judge(query, answer))
        time.sleep(delay_seconds)
    return results
