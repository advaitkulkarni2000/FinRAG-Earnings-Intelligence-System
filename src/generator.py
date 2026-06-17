"""
Grounded Answer Generator
--------------------------
Calls the Anthropic API with retrieved context and the user query.
The system prompt enforces grounding: the model must answer only from
context or explicitly say the context is insufficient.

This is the honesty constraint — the same instinct behind walk-forward
validation in quantitative research. We don't want flattering answers
that hallucinate from parametric knowledge; we want answers that are
strictly traceable to the retrieved evidence.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any
from dataclasses import dataclass

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTHROPIC_MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT


@dataclass
class GenerationResult:
    query:          str
    answer:         str
    retrieved_chunks: List[Dict[str, Any]]
    model:          str
    input_tokens:   int
    output_tokens:  int


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = (
            f"[Passage {i}] "
            f"{meta.get('ticker', '?')} | "
            f"{meta.get('doc_type', '?')} | "
            f"{meta.get('period', '?')} | "
            f"Section: {meta.get('section', '?')}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> GenerationResult:
    """
    Generate a grounded answer using the retrieved context.

    The system prompt instructs the model to answer ONLY from context.
    This is enforced at the prompt level; faithfulness evaluation
    then verifies it empirically.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    context = _format_context(retrieved_chunks)

    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer based strictly on the context above:"
    )

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    return GenerationResult(
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        model=ANTHROPIC_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
