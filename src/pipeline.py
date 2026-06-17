"""
End-to-End RAG Pipeline
------------------------
Ties ingestion → vectorstore → retrieval → generation together.
Also serves as the main CLI entry point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, SAMPLE_DIR


def ingest(data_dir: Path = SAMPLE_DIR) -> None:
    """Ingest all documents in data_dir into the vector store."""
    from src.ingestion import ingest_directory
    from src.vectorstore import add_chunks, count

    print(f"Ingesting documents from: {data_dir}")
    chunks = ingest_directory(data_dir)
    print(f"Found {len(chunks)} chunks across all documents.")
    add_chunks(chunks)
    print(f"Vector store now contains {count()} chunks.")


def ask(
    query: str,
    ticker: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline: retrieve + generate.

    Returns dict with query, answer, sources.
    """
    from src.retriever import retrieve
    from src.generator import generate_answer

    # Retrieve
    chunks = retrieve(query, ticker_filter=ticker)
    if not chunks:
        return {
            "query": query,
            "answer": "No relevant documents found in the vector store.",
            "sources": [],
        }

    # Generate
    result = generate_answer(query, chunks)

    output = {
        "query": result.query,
        "answer": result.answer,
        "sources": [
            {
                "passage": i + 1,
                "ticker": c["metadata"].get("ticker"),
                "doc_type": c["metadata"].get("doc_type"),
                "period": c["metadata"].get("period"),
                "section": c["metadata"].get("section"),
                "score": round(c["score"], 4),
                "preview": c["text"][:150] + "...",
            }
            for i, c in enumerate(chunks)
        ],
        "tokens": {
            "input": result.input_tokens,
            "output": result.output_tokens,
        },
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources ({len(chunks)} passages):")
        for s in output["sources"]:
            print(
                f"  [{s['passage']}] {s['ticker']} | "
                f"{s['doc_type']} | {s['period']} | "
                f"{s['section']} (score={s['score']})"
            )
        print(f"\nTokens: {result.input_tokens} in / {result.output_tokens} out")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="FinRAG — Financial Document Intelligence"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "--data-dir", default=str(SAMPLE_DIR),
        help="Directory containing ticker subdirectories"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument(
        "--ticker", help="Filter by ticker symbol (e.g. AAPL)"
    )
    query_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show vector store stats")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(Path(args.data_dir))

    elif args.command == "query":
        result = ask(args.question, ticker=args.ticker, verbose=not args.json)
        if args.json:
            print(json.dumps(result, indent=2))

    elif args.command == "stats":
        from src.vectorstore import count, _get_collection
        col = _get_collection()
        n = count()
        print(f"Vector store: {n} chunks")
        if n > 0:
            # Sample some metadata
            sample = col.get(limit=min(n, 100), include=["metadatas"])
            tickers = set(m["ticker"] for m in sample["metadatas"])
            doc_types = set(m["doc_type"] for m in sample["metadatas"])
            print(f"Tickers: {sorted(tickers)}")
            print(f"Doc types: {sorted(doc_types)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
