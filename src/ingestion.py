"""
Document Ingestion & Semantic Chunking
--------------------------------------
Loads SEC filings and earnings transcripts, chunks them at paragraph
boundaries (not fixed token windows), and attaches metadata for
filtering and citation.

Design decision: paragraph-boundary chunking over fixed-size chunking.
Fixed windows split sentences mid-thought, which destroys the coherence
of the chunk and degrades retrieval quality for dense embeddings.
"""
from __future__ import annotations

import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    chunk_id:   str
    text:       str
    ticker:     str
    doc_type:   str          # "10-K", "10-Q", "earnings_transcript"
    period:     str          # e.g. "FY2023", "Q3-2023"
    section:    str          # e.g. "Risk Factors", "MD&A"
    source_file: str
    chunk_index: int         # position within document
    char_start: int
    char_end:   int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@dataclass
class Document:
    """A raw loaded document before chunking."""
    ticker:     str
    doc_type:   str
    period:     str
    source_file: str
    text:       str
    sections:   dict = field(default_factory=dict)  # section_name -> text


# ── Section detection ─────────────────────────────────────────────────────

# Common 10-K section headers (SEC Item numbering)
SECTION_PATTERNS = [
    (r"Item\s+1A[\.\s]+Risk Factors",           "Risk Factors"),
    (r"Item\s+1B[\.\s]+Unresolved Staff",        "Unresolved Staff Comments"),
    (r"Item\s+1[\.\s]+Business",                 "Business"),
    (r"Item\s+2[\.\s]+Properties",               "Properties"),
    (r"Item\s+7A[\.\s]+Quantitative",            "Market Risk"),
    (r"Item\s+7[\.\s]+Management",               "MD&A"),
    (r"Item\s+8[\.\s]+Financial Statements",     "Financial Statements"),
    (r"Item\s+9A[\.\s]+Controls",                "Controls and Procedures"),
    (r"RESULTS OF OPERATIONS",                   "Results of Operations"),
    (r"LIQUIDITY AND CAPITAL",                   "Liquidity"),
    (r"CRITICAL ACCOUNTING",                     "Critical Accounting"),
]


def detect_section(text_before: str) -> str:
    """Return the most recently seen section heading."""
    for pattern, name in reversed(SECTION_PATTERNS):
        if re.search(pattern, text_before, re.IGNORECASE):
            return name
    return "General"


# ── Chunking ──────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _chunk_id(ticker: str, source_file: str, index: int) -> str:
    raw = f"{ticker}:{source_file}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def chunk_document(doc: Document) -> List[Chunk]:
    """
    Semantic chunking at paragraph boundaries.

    Algorithm:
    1. Split text into paragraphs (double newline boundaries).
    2. Accumulate paragraphs into a chunk until we exceed CHUNK_SIZE words.
    3. When we overflow, emit the current chunk and start a new one,
       carrying forward the last CHUNK_OVERLAP words as context.
    4. Discard chunks shorter than MIN_CHUNK_SIZE words (boilerplate).
    """
    text = doc.text

    # Split into paragraphs, filter empty
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text)]
    paragraphs = [p for p in paragraphs if len(p.split()) > 5]

    chunks: List[Chunk] = []
    current_paras: List[str] = []
    current_words = 0
    chunk_index = 0
    char_pos = 0

    for para in paragraphs:
        para_words = _word_count(para)

        # If adding this paragraph exceeds limit and we have content, emit
        if current_words + para_words > CHUNK_SIZE and current_words > 0:
            chunk_text = "\n\n".join(current_paras)

            if _word_count(chunk_text) >= MIN_CHUNK_SIZE:
                # Detect section from the text up to this point
                text_so_far = text[:char_pos]
                section = detect_section(text_so_far)

                chunks.append(Chunk(
                    chunk_id=_chunk_id(doc.ticker, doc.source_file, chunk_index),
                    text=chunk_text,
                    ticker=doc.ticker,
                    doc_type=doc.doc_type,
                    period=doc.period,
                    section=section,
                    source_file=doc.source_file,
                    chunk_index=chunk_index,
                    char_start=char_pos - len(chunk_text),
                    char_end=char_pos,
                ))
                chunk_index += 1

            # Carry forward overlap: last N words
            overlap_text = " ".join(chunk_text.split()[-CHUNK_OVERLAP:])
            current_paras = [overlap_text]
            current_words = _word_count(overlap_text)

        current_paras.append(para)
        current_words += para_words
        char_pos += len(para) + 2  # +2 for "\n\n"

    # Emit final chunk
    if current_paras:
        chunk_text = "\n\n".join(current_paras)
        if _word_count(chunk_text) >= MIN_CHUNK_SIZE:
            section = detect_section(text[:char_pos])
            chunks.append(Chunk(
                chunk_id=_chunk_id(doc.ticker, doc.source_file, chunk_index),
                text=chunk_text,
                ticker=doc.ticker,
                doc_type=doc.doc_type,
                period=doc.period,
                section=section,
                source_file=doc.source_file,
                chunk_index=chunk_index,
                char_start=max(0, char_pos - len(chunk_text)),
                char_end=char_pos,
            ))

    return chunks


# ── File loaders ──────────────────────────────────────────────────────────

def load_text_file(path: Path, ticker: str, doc_type: str, period: str) -> Document:
    """Load a plain text or pre-cleaned filing."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Basic cleanup: collapse excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    return Document(
        ticker=ticker,
        doc_type=doc_type,
        period=period,
        source_file=str(path),
        text=text,
    )


def load_json_transcript(path: Path, ticker: str, period: str) -> Document:
    """
    Load an earnings call transcript in JSON format.
    Expected format: {"ticker": str, "period": str, "text": str}
    or {"ticker": str, "period": str, "segments": [{"speaker": str, "text": str}]}
    """
    data = json.loads(path.read_text())
    if "text" in data:
        text = data["text"]
    elif "segments" in data:
        text = "\n\n".join(
            f"{seg.get('speaker', 'Speaker')}: {seg['text']}"
            for seg in data["segments"]
        )
    else:
        raise ValueError(f"Unrecognised transcript format in {path}")

    return Document(
        ticker=ticker,
        doc_type="earnings_transcript",
        period=period,
        source_file=str(path),
        text=text,
    )


def ingest_directory(data_dir: Path) -> List[Chunk]:
    """
    Walk a data directory and ingest all supported files.

    Expected directory structure:
        data/
          AAPL/
            10-K_FY2023.txt
            earnings_Q4-2023.json
          MSFT/
            10-K_FY2023.txt
    """
    all_chunks: List[Chunk] = []

    for ticker_dir in sorted(data_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()

        for fpath in sorted(ticker_dir.iterdir()):
            if fpath.suffix not in (".txt", ".json"):
                continue

            # Parse doc_type and period from filename
            # Convention: {doc_type}_{period}.{ext}
            stem = fpath.stem  # e.g. "10-K_FY2023"
            parts = stem.split("_", 1)
            doc_type = parts[0] if parts else "unknown"
            period   = parts[1] if len(parts) > 1 else "unknown"

            print(f"  Ingesting {ticker}/{fpath.name} ...", end=" ")
            try:
                if fpath.suffix == ".json":
                    doc = load_json_transcript(fpath, ticker, period)
                else:
                    doc = load_text_file(fpath, ticker, doc_type, period)

                chunks = chunk_document(doc)
                all_chunks.extend(chunks)
                print(f"{len(chunks)} chunks")

            except Exception as e:
                print(f"ERROR: {e}")

    return all_chunks


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import SAMPLE_DIR

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_DIR
    print(f"Ingesting from: {data_dir}")
    chunks = ingest_directory(data_dir)
    print(f"\nTotal chunks: {len(chunks)}")

    if chunks:
        print(f"\nSample chunk:")
        c = chunks[0]
        print(f"  ticker={c.ticker} doc_type={c.doc_type} section={c.section}")
        print(f"  words={_word_count(c.text)}")
        print(f"  preview: {c.text[:200]}...")
