"""
FinRAG Configuration
Central config — change here, propagates everywhere.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
SAMPLE_DIR      = DATA_DIR / "sample"
CHROMA_DIR      = ROOT_DIR / ".chromadb"
EVAL_DIR        = ROOT_DIR / "eval"

# ── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim
EMBEDDING_DIM   = 384

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE      = 512    # tokens (approximate via word count)
CHUNK_OVERLAP   = 64     # overlap between consecutive chunks
MIN_CHUNK_SIZE  = 100    # discard chunks shorter than this

# ── Retrieval ──────────────────────────────────────────────────────────────
DENSE_TOP_K     = 20     # dense retrieval candidates
RERANK_TOP_K    = 5      # final chunks after BM25 reranking
COLLECTION_NAME = "finrag_filings"

# ── Generation ────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_TOKENS      = 1024
TEMPERATURE     = 0.0    # deterministic for evaluation

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions 
using ONLY the provided context passages. 

Rules:
1. If the answer is in the context, answer directly and cite which passage.
2. If the answer is NOT in the context, say exactly: "The provided context 
   does not contain enough information to answer this question."
3. Never use prior knowledge about the company beyond what is in the context.
4. Be precise with numbers — quote figures exactly as they appear in context.
"""

# ── Evaluation ────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD  = 0.8   # minimum acceptable faithfulness score
RELEVANCE_THRESHOLD     = 3.5   # minimum acceptable relevance score (1-5)
PRECISION_AT_K          = 5     # k for Precision@k metric

FAITHFULNESS_JUDGE_PROMPT = """You are evaluating whether a claim is 
supported by the provided context.

Claim: {claim}

Context:
{context}

Is this claim directly supported by the context above?
Answer with ONLY: "supported" or "not_supported"
"""

RELEVANCE_JUDGE_PROMPT = """You are evaluating whether an answer addresses 
the question asked.

Question: {question}
Answer: {answer}

Score the answer's relevance to the question on a scale of 1-5:
1 = Completely irrelevant or refuses to answer
2 = Tangentially related but misses the point
3 = Partially addresses the question
4 = Mostly addresses the question with minor gaps
5 = Directly and completely addresses the question

Answer with ONLY a number from 1 to 5.
"""

# ── SEC EDGAR ────────────────────────────────────────────────────────────
EDGAR_BASE_URL  = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
EDGAR_HEADERS   = {"User-Agent": "FinRAG Research Project finrag@example.com"}
