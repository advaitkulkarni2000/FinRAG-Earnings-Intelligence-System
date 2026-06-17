"""
Evaluation Benchmark — 50 QA Pairs
-------------------------------------
Hand-curated question-answer pairs over Apple, Microsoft, and Nvidia
10-K filings (FY2022/FY2023).

Design principle: these were written BEFORE the retrieval pipeline
was built, to prevent benchmark contamination. We don't know which
chunks will be retrieved when writing the questions — we know only
what the documents say.

Each entry contains:
- question:    Natural language question
- answer:      Expected answer (ground truth)
- ticker:      Company to filter by
- gold_section: Section where the answer should be found
- difficulty:  easy / medium / hard
  - easy: single explicit fact, directly stated
  - medium: requires synthesising across a paragraph
  - hard: requires reasoning or comparison across sections

Gold chunk IDs are populated after ingestion (run eval/annotate_gold.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QAPair:
    question:     str
    answer:       str
    ticker:       str
    gold_section: str
    difficulty:   str
    gold_chunk_ids: List[str] = field(default_factory=list)
    notes:        Optional[str] = None


# ── Benchmark questions ───────────────────────────────────────────────────
# 50 questions across 3 companies: AAPL (17), MSFT (17), NVDA (16)
# Balanced across difficulty: easy (20), medium (18), hard (12)

BENCHMARK: List[QAPair] = [

    # ── APPLE ────────────────────────────────────────────────────────────

    QAPair(
        question="What was Apple's total net revenue for fiscal year 2023?",
        answer="$383.3 billion",
        ticker="AAPL",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="What were Apple's three largest revenue segments in FY2023?",
        answer="iPhone, Services, and Mac",
        ticker="AAPL",
        gold_section="MD&A",
        difficulty="easy",
    ),
    QAPair(
        question="How much did Apple spend on research and development in FY2023?",
        answer="$29.9 billion",
        ticker="AAPL",
        gold_section="MD&A",
        difficulty="easy",
    ),
    QAPair(
        question="What risk does Apple identify related to its dependence on a "
                 "single supplier for key components?",
        answer="Apple faces risk from single-source suppliers for certain "
               "components; disruption could materially affect product availability.",
        ticker="AAPL",
        gold_section="Risk Factors",
        difficulty="medium",
        notes="Paraphrase acceptable — check for concentration/single-source concept",
    ),
    QAPair(
        question="How does Apple describe its approach to environmental sustainability?",
        answer="Apple aims to be carbon neutral across its supply chain by 2030.",
        ticker="AAPL",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="What were Apple's iPhone revenues in FY2023 compared to FY2022?",
        answer="iPhone revenue was $200.6B in FY2023 vs $205.5B in FY2022.",
        ticker="AAPL",
        gold_section="MD&A",
        difficulty="medium",
    ),
    QAPair(
        question="What legal proceedings does Apple disclose related to antitrust?",
        answer="Apple faces antitrust investigations and litigation related to "
               "App Store policies in multiple jurisdictions.",
        ticker="AAPL",
        gold_section="Financial Statements",
        difficulty="medium",
    ),
    QAPair(
        question="How does Apple describe the competitive landscape for its "
                 "Services segment?",
        answer="Services faces intense competition from companies offering "
               "competing apps, platforms and content across streaming, payments and cloud.",
        ticker="AAPL",
        gold_section="Business",
        difficulty="medium",
    ),
    QAPair(
        question="What does Apple say about its share repurchase programme in FY2023?",
        answer="Apple repurchased approximately $77.6 billion of its common stock "
               "during FY2023.",
        ticker="AAPL",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="What are the key risk factors Apple cites related to "
                 "international operations?",
        answer="Apple cites foreign exchange volatility, regulatory differences, "
               "trade restrictions, and geopolitical tensions as key international risks.",
        ticker="AAPL",
        gold_section="Risk Factors",
        difficulty="hard",
        notes="Multi-factor question requiring synthesis across Risk Factors section",
    ),
    QAPair(
        question="How much cash and equivalents did Apple hold at end of FY2023?",
        answer="Apple held $29.9 billion in cash and cash equivalents.",
        ticker="AAPL",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="What does Apple say about macroeconomic conditions affecting demand?",
        answer="Apple notes that adverse macroeconomic conditions including "
               "inflation and reduced consumer confidence could negatively impact demand.",
        ticker="AAPL",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="What is Apple's approach to managing foreign exchange risk?",
        answer="Apple uses derivative financial instruments including forward "
               "contracts to hedge foreign currency exposure.",
        ticker="AAPL",
        gold_section="Market Risk",
        difficulty="hard",
    ),
    QAPair(
        question="How many full-time equivalent employees did Apple have at "
                 "end of FY2023?",
        answer="Approximately 161,000 full-time equivalent employees.",
        ticker="AAPL",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="What does Apple cite as risks from its reliance on "
                 "third-party intellectual property?",
        answer="Apple may be required to pay royalties or face litigation "
               "if third-party IP rights are infringed in its products.",
        ticker="AAPL",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="How did Apple's gross margin change between FY2022 and FY2023?",
        answer="Apple's gross margin increased from 43.3% in FY2022 to 44.1% in FY2023.",
        ticker="AAPL",
        gold_section="MD&A",
        difficulty="medium",
    ),
    QAPair(
        question="What cybersecurity risks does Apple disclose and how does it "
                 "manage them?",
        answer="Apple discloses risks from data breaches, cyberattacks and "
               "unauthorised access, and manages them through security controls, "
               "incident response plans and employee training.",
        ticker="AAPL",
        gold_section="Risk Factors",
        difficulty="hard",
    ),

    # ── MICROSOFT ────────────────────────────────────────────────────────

    QAPair(
        question="What was Microsoft's total revenue for fiscal year 2023?",
        answer="$211.9 billion",
        ticker="MSFT",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="What are Microsoft's three reportable business segments?",
        answer="Productivity and Business Processes, Intelligent Cloud, "
               "and More Personal Computing.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="How much did Microsoft's Intelligent Cloud segment grow in FY2023?",
        answer="Intelligent Cloud revenue grew 19% to $87.9 billion in FY2023.",
        ticker="MSFT",
        gold_section="MD&A",
        difficulty="easy",
    ),
    QAPair(
        question="How does Microsoft describe its integration of AI into its products?",
        answer="Microsoft is integrating AI capabilities including Copilot across "
               "Office 365, Azure, GitHub, and Bing to enhance productivity and search.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="medium",
    ),
    QAPair(
        question="What does Microsoft say about competition in the cloud market?",
        answer="Microsoft competes with Amazon Web Services and Google Cloud "
               "in the cloud infrastructure and platform services market.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="What were the key risks Microsoft identified related to its "
                 "Activision Blizzard acquisition?",
        answer="Risks included regulatory approval uncertainty, integration "
               "complexity, and impact on competitive dynamics in gaming.",
        ticker="MSFT",
        gold_section="Risk Factors",
        difficulty="hard",
    ),
    QAPair(
        question="How much did Microsoft return to shareholders via dividends "
                 "and buybacks in FY2023?",
        answer="Microsoft returned $34.3 billion through share repurchases "
               "and dividends.",
        ticker="MSFT",
        gold_section="Financial Statements",
        difficulty="medium",
    ),
    QAPair(
        question="What does Microsoft say about its approach to responsible AI?",
        answer="Microsoft commits to principles including fairness, reliability, "
               "privacy, inclusiveness, transparency and accountability in AI development.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="medium",
    ),
    QAPair(
        question="What were Microsoft's operating expenses in FY2023?",
        answer="Total operating expenses were $135.7 billion.",
        ticker="MSFT",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="How does Microsoft describe risks from government regulation "
                 "of AI and cloud services?",
        answer="Microsoft cites evolving regulations around data privacy, AI "
               "governance, and cloud services that could impose compliance costs "
               "and restrict operations.",
        ticker="MSFT",
        gold_section="Risk Factors",
        difficulty="hard",
    ),
    QAPair(
        question="What is Microsoft's commercial cloud revenue annualised run rate?",
        answer="Microsoft's commercial cloud revenue reached $111.6 billion "
               "annualised run rate.",
        ticker="MSFT",
        gold_section="MD&A",
        difficulty="medium",
    ),
    QAPair(
        question="How many employees does Microsoft have globally?",
        answer="Approximately 221,000 full-time employees.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="What does Microsoft identify as cybersecurity risks to its "
                 "operations?",
        answer="Microsoft identifies nation-state attacks, ransomware, supply "
               "chain vulnerabilities, and zero-day exploits as significant cybersecurity risks.",
        ticker="MSFT",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="How does Microsoft describe its sustainability commitments?",
        answer="Microsoft targets carbon negative by 2030, water positive by "
               "2030, and zero waste by 2030.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="What does Microsoft say about concentration of revenue in "
                 "its customer base?",
        answer="No single customer accounts for 10% or more of Microsoft's revenue.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="medium",
    ),
    QAPair(
        question="How did LinkedIn revenue perform in FY2023?",
        answer="LinkedIn revenue grew 10% to $15.1 billion in FY2023.",
        ticker="MSFT",
        gold_section="MD&A",
        difficulty="easy",
    ),
    QAPair(
        question="What are the primary competitive factors Microsoft cites in "
                 "the productivity software market?",
        answer="Microsoft cites price, interoperability, product integration, "
               "security features, and AI capabilities as key competitive factors.",
        ticker="MSFT",
        gold_section="Business",
        difficulty="hard",
    ),

    # ── NVIDIA ───────────────────────────────────────────────────────────

    QAPair(
        question="What was Nvidia's total revenue for fiscal year 2024?",
        answer="$60.9 billion",
        ticker="NVDA",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="What drove Nvidia's Data Center revenue growth in FY2024?",
        answer="Strong demand for AI and large language model training drove "
               "Data Center revenue growth, particularly for H100 GPUs.",
        ticker="NVDA",
        gold_section="MD&A",
        difficulty="easy",
    ),
    QAPair(
        question="How does Nvidia describe the competitive landscape for "
                 "its GPU products?",
        answer="Nvidia faces competition from AMD, Intel, and custom AI chips "
               "from cloud providers including Google TPUs and Amazon Trainium.",
        ticker="NVDA",
        gold_section="Business",
        difficulty="medium",
    ),
    QAPair(
        question="What export control risks does Nvidia disclose?",
        answer="Nvidia discloses that US export controls restrict sales of "
               "advanced chips to China and other countries, materially affecting revenue.",
        ticker="NVDA",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="What was Nvidia's gross margin for FY2024?",
        answer="Nvidia's gross margin was approximately 72.7% in FY2024.",
        ticker="NVDA",
        gold_section="Financial Statements",
        difficulty="easy",
    ),
    QAPair(
        question="How does Nvidia describe the risk of customer concentration?",
        answer="A small number of large cloud service providers represent a "
               "significant portion of Nvidia's Data Center revenue.",
        ticker="NVDA",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="What does Nvidia say about supply constraints for its "
                 "AI chips?",
        answer="Nvidia acknowledges supply constraints from TSMC manufacturing "
               "capacity limitations and long lead times for advanced packaging.",
        ticker="NVDA",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="How many employees does Nvidia have?",
        answer="Approximately 29,600 employees.",
        ticker="NVDA",
        gold_section="Business",
        difficulty="easy",
    ),
    QAPair(
        question="What does Nvidia say about its CUDA software ecosystem?",
        answer="CUDA is Nvidia's parallel computing platform with over 4 million "
               "developers, creating a significant switching cost moat.",
        ticker="NVDA",
        gold_section="Business",
        difficulty="medium",
    ),
    QAPair(
        question="How does Nvidia describe the risk of rapid technological change?",
        answer="Rapid change in AI architectures and computing paradigms could "
               "render Nvidia's current products obsolete if it fails to innovate.",
        ticker="NVDA",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="What was the growth rate of Nvidia's Data Center revenue "
                 "in FY2024?",
        answer="Data Center revenue grew 217% year-over-year to $47.5 billion.",
        ticker="NVDA",
        gold_section="MD&A",
        difficulty="easy",
    ),
    QAPair(
        question="How does Nvidia manage the risk of dependence on TSMC "
                 "for manufacturing?",
        answer="Nvidia is working to diversify manufacturing but remains "
               "substantially dependent on TSMC; it is exploring Samsung as an "
               "alternative for some processes.",
        ticker="NVDA",
        gold_section="Risk Factors",
        difficulty="hard",
    ),
    QAPair(
        question="What research and development investments does Nvidia "
                 "describe for next-generation chips?",
        answer="Nvidia is investing heavily in Blackwell architecture as the "
               "successor to Hopper, with significantly increased transistor counts "
               "and memory bandwidth.",
        ticker="NVDA",
        gold_section="Business",
        difficulty="hard",
    ),
    QAPair(
        question="What does Nvidia say about its gaming segment performance?",
        answer="Gaming revenue grew modestly in FY2024 after a difficult FY2023 "
               "due to cryptocurrency mining demand collapse and excess channel inventory.",
        ticker="NVDA",
        gold_section="MD&A",
        difficulty="medium",
    ),
    QAPair(
        question="How does Nvidia address intellectual property risks?",
        answer="Nvidia faces patent litigation risk from competitors and patent "
               "trolls, and relies on trade secrets, patents, and copyright to protect IP.",
        ticker="NVDA",
        gold_section="Risk Factors",
        difficulty="medium",
    ),
    QAPair(
        question="What does Nvidia say about its automotive business prospects?",
        answer="Nvidia describes automotive as a long-term growth opportunity "
               "via its DRIVE platform for autonomous vehicles, with a $14B order backlog.",
        ticker="NVDA",
        gold_section="Business",
        difficulty="hard",
    ),
]


def get_benchmark(
    ticker: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[QAPair]:
    """Filter benchmark by ticker or difficulty."""
    from typing import Optional
    qs = BENCHMARK
    if ticker:
        qs = [q for q in qs if q.ticker == ticker.upper()]
    if difficulty:
        qs = [q for q in qs if q.difficulty == difficulty]
    return qs


if __name__ == "__main__":
    print(f"Total QA pairs: {len(BENCHMARK)}")
    tickers = {}
    difficulties = {}
    for q in BENCHMARK:
        tickers[q.ticker] = tickers.get(q.ticker, 0) + 1
        difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1
    print(f"By ticker: {tickers}")
    print(f"By difficulty: {difficulties}")
