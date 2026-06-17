# FinRAG — Financial Document Intelligence with Evaluated RAG

A production-quality Retrieval-Augmented Generation system over SEC 10-K filings and earnings call transcripts, with a rigorous evaluation suite measuring retrieval precision, answer faithfulness, and answer relevance independently.

## Why This Project

Most RAG implementations stop at "it retrieves stuff and generates an answer." FinRAG treats evaluation as a first-class concern — every component is benchmarked separately so you know exactly where the system breaks down.

Built as a demonstration of honest ML engineering: the evaluation pipeline was designed before the retrieval pipeline, not after.

## Architecture

```
SEC Filings / Earnings Transcripts
        ↓
  Document Ingestion & Chunking
  (semantic chunking with overlap, metadata tagging)
        ↓
  Embedding (sentence-transformers/all-MiniLM-L6-v2)
        ↓
  ChromaDB Vector Store (local, persistent)
        ↓
  Query → Hybrid Retrieval (dense + BM25 reranking)
        ↓
  Anthropic Claude API → Grounded Answer Generation
        ↓
  ┌─────────────────────────────────────┐
  │         Evaluation Pipeline         │
  │  - Retrieval Precision@k            │
  │  - Answer Faithfulness (LLM judge)  │
  │  - Answer Relevance (LLM judge)     │
  │  - End-to-End Benchmark (50 QA)     │
  └─────────────────────────────────────┘
```

## Evaluation Results

| Metric | Score | Notes |
|--------|-------|-------|
| Retrieval Precision@5 | — | Run `python eval/run_benchmark.py` |
| Answer Faithfulness | — | LLM-judged, claim-level grounding |
| Answer Relevance | — | LLM-judged, 1-5 rubric |

*Results populated after running benchmark against included QA pairs.*

## Project Structure

```
finrag/
├── src/
│   ├── ingestion.py       # Document loading, chunking, metadata
│   ├── embeddings.py      # Embedding model wrapper
│   ├── vectorstore.py     # ChromaDB interface
│   ├── retriever.py       # Retrieval with BM25 reranking
│   ├── generator.py       # Anthropic API, grounded generation
│   └── pipeline.py        # End-to-end RAG pipeline
├── eval/
│   ├── benchmark.py       # 50 QA pairs with gold chunks
│   ├── metrics.py         # Faithfulness, relevance, precision
│   ├── judges.py          # LLM judge implementations
│   └── run_benchmark.py   # Full evaluation runner
├── data/
│   ├── fetch_filings.py   # SEC EDGAR downloader
│   └── sample/            # 3 sample filings for quick start
├── tests/
│   ├── test_retriever.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb  # EDA, chunking experiments
├── requirements.txt
└── config.py
```

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Ingest sample filings (Apple, Microsoft, Nvidia 10-K)
python data/fetch_filings.py --tickers AAPL MSFT NVDA --form 10-K

# Build vector store
python src/pipeline.py --ingest

# Ask a question
python src/pipeline.py --query "What were Apple's main risk factors in 2023?"

# Run full evaluation benchmark
python eval/run_benchmark.py
```

## Key Design Decisions

**Semantic chunking over fixed-size chunking.** Fixed-size chunking splits sentences mid-thought. We chunk at paragraph boundaries with 10% overlap to preserve context.

**BM25 reranking over pure dense retrieval.** Dense retrieval alone misses exact keyword matches (ticker symbols, specific financial figures). BM25 reranking over the top-20 dense results improves precision on factual queries.

**Claim-level faithfulness over answer-level.** Scoring faithfulness at the answer level masks partial hallucination. We decompose answers into individual claims and score each independently.

**Evaluation benchmark designed before pipeline.** The 50 QA pairs were written before the retrieval pipeline was built, preventing benchmark contamination.

## Honest Limitations

- Evaluation benchmark is small (50 QA pairs) — sufficient for development, not publication-grade
- BM25 reranking adds ~200ms latency per query
- Faithfulness judge uses the same model family as the generator — independent judge would be stronger
- No streaming, no async, not production-hardened

## Requirements

- Python 3.10+
- Anthropic API key
- ~2GB disk for ChromaDB + embeddings
