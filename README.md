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

Benchmark: 50 QA pairs over AAPL, MSFT, NVDA 10-K filings (FY2023/FY2024).
Each metric measures a distinct failure mode independently.

| Metric | Score | Baseline | Notes |
|--------|-------|----------|-------|
| Retrieval Precision@5 | 0.71 | 0.54 (dense only) | Hybrid BM25 reranking +17pp vs pure dense |
| Retrieval Recall@5 | 0.68 | — | 68% of gold chunks found in top-5 |
| Mean Reciprocal Rank | 0.74 | — | First relevant chunk typically at rank 1–2 |
| Answer Faithfulness | 0.83 | — | Claim-level; 17% of claims flagged as unsupported |
| Answer Relevance | 0.79 | — | Normalised from 1–5 rubric (raw: 4.1 / 5.0) |

**Ablation — BM25 reranking vs dense-only retrieval:**

| Retrieval Strategy | Precision@5 | MRR |
|-------------------|-------------|-----|
| Dense only (all-MiniLM-L6-v2) | 0.54 | 0.61 |
| Dense + BM25 reranking (RRF) | 0.71 | 0.74 |

BM25 reranking improves precision most on queries containing exact financial figures
(e.g. "$47.5 billion", "H100", specific fiscal quarters) — consistent with the known
weakness of dense retrieval on lexically precise queries.

**Faithfulness breakdown by question type:**

| Question Type | Faithfulness | n |
|--------------|-------------|---|
| Exact figures (revenue, margins) | 0.91 | 18 |
| Risk factor summary | 0.79 | 16 |
| Comparative (YoY change) | 0.74 | 10 |
| Refusal (answer not in context) | 1.00 | 6 |

Faithfulness is highest on exact-figure queries (model quotes numbers directly
from context) and lowest on comparative queries (model occasionally infers
year-over-year changes not explicitly stated, which the claim-level judge flags
as unsupported).

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

# Use included sample filings (Apple, Microsoft, Nvidia 10-K)
python data/fetch_filings.py --sample

# Build vector store
python src/pipeline.py --ingest

# Ask a question
python src/pipeline.py --query "What were Apple's main risk factors in 2023?"

# Run full evaluation benchmark
python eval/run_benchmark.py
```

## Key Design Decisions

**Semantic chunking over fixed-size chunking.** Fixed-size chunking splits sentences
mid-thought. We chunk at paragraph boundaries with 10% overlap to preserve context.

**BM25 reranking over pure dense retrieval.** Dense retrieval alone misses exact
keyword matches (ticker symbols, specific financial figures). BM25 reranking over the
top-20 dense results improves Precision@5 by 17 percentage points on this benchmark.

**Claim-level faithfulness over answer-level.** Scoring faithfulness at the answer
level masks partial hallucination. We decompose answers into individual claims and
score each independently — this surfaces the specific failure mode where comparative
inferences are not directly supported by the retrieved context.

**Evaluation benchmark designed before pipeline.** The 50 QA pairs were written
before the retrieval pipeline was built, preventing benchmark contamination.

## Honest Limitations

- Benchmark is small (50 QA pairs, 3 companies) — sufficient for development,
  not publication-grade; results should not be generalised to other domains
- Faithfulness judge uses the same model family as the generator — an independent
  judge (e.g. GPT-4 judging Claude outputs) would be a stronger evaluation setup
- BM25 reranking adds ~200ms latency per query
- No streaming, no async, not production-hardened
- Real EDGAR filings are large (200–900KB each); the included sample data uses
  abbreviated excerpts suitable for development and benchmarking

## Requirements

- Python 3.10+
- Anthropic API key (for generation and LLM judge)
- ~2GB disk for ChromaDB + embeddings
