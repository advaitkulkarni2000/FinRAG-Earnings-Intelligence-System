"""
SEC EDGAR Filing Downloader
----------------------------
Downloads 10-K and 10-Q filings from SEC EDGAR for a given ticker.
Saves cleaned text to data/{TICKER}/{form}_{period}.txt

Usage:
    python data/fetch_filings.py --tickers AAPL MSFT NVDA --form 10-K
    python data/fetch_filings.py --tickers AAPL --form 10-Q --year 2023
"""
from __future__ import annotations

import re
import time
import argparse
import requests
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, EDGAR_HEADERS


EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form}&dateRange=custom&startdt={start}&enddt={end}"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FILING = "https://www.sec.gov/Archives/edgar/{path}"


def get_cik(ticker: str) -> str:
    """Look up CIK for a ticker from SEC EDGAR."""
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "company": ticker,
        "CIK": ticker,
        "type": "10-K",
        "dateb": "",
        "owner": "include",
        "count": "10",
        "search_text": "",
        "action": "getcompany",
        "output": "atom",
    }
    resp = requests.get(url, params=params, headers=EDGAR_HEADERS, timeout=10)
    resp.raise_for_status()

    match = re.search(r'CIK=(\d+)', resp.url)
    if match:
        return match.group(1).lstrip("0")

    # Try company-tickers.json
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(tickers_url, headers=EDGAR_HEADERS, timeout=10)
    data = resp.json()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry["cik_str"])

    raise ValueError(f"Could not find CIK for ticker {ticker}")


def clean_filing_text(raw: str) -> str:
    """
    Basic cleaning of raw SEC filing text.
    Removes SGML/HTML tags, normalises whitespace.
    """
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', raw)
    # Remove SGML headers
    text = re.sub(r'<\?xml[^>]+\?>', '', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&#160;', ' ')
    # Collapse excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    # Remove very short lines (page numbers, headers)
    lines = [l for l in text.split('\n') if len(l.strip()) > 3]
    return '\n'.join(lines)


def _get_filing_document_url(cik: str, accession_no: str) -> str | None:
    """
    Given a CIK and accession number, fetch the filing index and return
    the URL of the primary text document (.htm or .txt).

    EDGAR accession numbers look like: 0000320193-23-000106
    The filing index lives at:
    https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}
    &type=10-K&dateb=&owner=include&count=10

    More reliably, the index JSON is at:
    https://data.sec.gov/submissions/CIK{cik:010d}.json
    """
    # Normalise accession number to the no-dash form used in URLs
    acc_nodash = accession_no.replace("-", "")
    cik_padded = cik.zfill(10)

    # Filing index page
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{acc_nodash}/{acc_nodash}-index.htm"
    )
    resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    # Find the primary document — look for .htm or .txt link
    matches = re.findall(
        r'href="(/Archives/edgar/data/[^"]+\.(?:htm|txt))"',
        resp.text,
        re.IGNORECASE,
    )
    if not matches:
        return None

    # Prefer the file that isn't the index itself
    for m in matches:
        if "index" not in m.lower():
            return f"https://www.sec.gov{m}"

    return f"https://www.sec.gov{matches[0]}"


def download_filing(
    ticker: str,
    form: str = "10-K",
    year: int = 2023,
    output_dir: Path = None,
) -> Path | None:
    """
    Download a single SEC filing for a ticker and save as cleaned text.

    Strategy:
    1. Look up the ticker's CIK using EDGAR's company-tickers.json
    2. Fetch the submissions JSON which lists all filings with accession numbers
    3. Find the most recent filing of the requested form in the requested year
    4. Fetch the filing index to find the primary document URL
    5. Download, clean, and save the document text

    Returns the path to the saved file, or None if the filing was not found.

    Note: SEC EDGAR rate-limits to 10 requests/second. This function
    sleeps 0.2s between requests to stay compliant.
    """
    if output_dir is None:
        output_dir = DATA_DIR / ticker.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    period = f"FY{year}" if form == "10-K" else str(year)
    output_path = output_dir / f"{form}_{period}.txt"
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    # Step 1: resolve CIK
    print(f"  [{ticker}] Resolving CIK...")
    try:
        cik = get_cik(ticker)
    except ValueError as e:
        print(f"  [{ticker}] ERROR: {e}")
        return None
    time.sleep(0.2)

    # Step 2: fetch submissions JSON — contains full filing history
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    resp = requests.get(submissions_url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    submissions = resp.json()
    time.sleep(0.2)

    # Step 3: find the right filing
    filings = submissions.get("filings", {}).get("recent", {})
    form_types  = filings.get("form", [])
    dates       = filings.get("filingDate", [])
    accessions  = filings.get("accessionNumber", [])

    target_accession = None
    target_date = None
    for i, (ftype, fdate, acc) in enumerate(zip(form_types, dates, accessions)):
        if ftype != form:
            continue
        filing_year = int(fdate[:4])
        if filing_year == year or filing_year == year + 1:
            # 10-Ks for fiscal year X are often filed in year X+1
            target_accession = acc
            target_date = fdate
            break

    if not target_accession:
        print(f"  [{ticker}] No {form} found for year {year} in submissions history.")
        return None

    print(f"  [{ticker}] Found {form} filed {target_date} (accession {target_accession})")
    time.sleep(0.2)

    # Step 4: get primary document URL from filing index
    doc_url = _get_filing_document_url(cik, target_accession)
    if not doc_url:
        print(f"  [{ticker}] Could not resolve primary document URL.")
        return None

    # Step 5: download and clean
    print(f"  [{ticker}] Downloading: {doc_url}")
    resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)

    cleaned = clean_filing_text(resp.text)
    output_path.write_text(cleaned, encoding="utf-8")
    size_kb = output_path.stat().st_size // 1024
    print(f"  [{ticker}] Saved {size_kb}KB → {output_path}")
    return output_path


def create_sample_data(output_dir: Path = None) -> None:
    """
    Create minimal sample data files for testing without hitting EDGAR.
    These are abbreviated excerpts, not real filings.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "sample"

    samples = {
        "AAPL": {
            "10-K_FY2023.txt": """APPLE INC.
ANNUAL REPORT ON FORM 10-K
For the fiscal year ended September 30, 2023

Item 1. Business

Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories and sells a variety of related services. The Company's fiscal year is the 52 or 53-week period that ends on the last Saturday of September.

The Company sells its products and resells third-party products in most of its major markets directly to consumers, small and mid-sized businesses, and education, enterprise and government customers through its retail and online stores and its direct sales force. The Company also employs a variety of indirect distribution channels, such as third-party cellular network carriers, wholesalers, retailers and resellers.

Services
The Company's Services segment includes the Company's advertising, AppleCare, cloud services, digital content and payment services.

Item 1A. Risk Factors

The Company depends on component and product manufacturing and logistical services provided by outsourcing partners, many of which are located outside of the U.S. Substantially all of the Company's manufacturing is performed by outsourcing partners located primarily in China, India, Japan, South Korea, Taiwan and Vietnam. A significant concentration of this manufacturing is performed by a small number of outsourcing partners.

The Company faces substantial competition in the markets it operates in. This competition is intense and includes well-established companies that have extensive experience in developing and marketing consumer electronics, software, and services. These competitors have significant technical, marketing, distribution and other resources, as well as broad customer bases.

Changes in global macroeconomic conditions, including inflation, interest rate changes, currency fluctuations, and reduced consumer confidence, could adversely affect consumer demand for the Company's products and services.

Item 7. Management's Discussion and Analysis

Net revenue for 2023 was $383.3 billion, a decrease of $2.8 billion or 1% compared to 2022. iPhone net revenue was $200.6 billion, a decrease of 2% compared to 2022. Services net revenue was $85.2 billion, an increase of 9% compared to 2022.

Gross margin percentage was 44.1% during 2023, compared to 43.3% during 2022. The increase in gross margin percentage was primarily driven by a different mix toward Services and cost savings, partially offset by the weakness in foreign currencies relative to the U.S. dollar.

Research and development expense was $29.9 billion during 2023.

The Company repurchased approximately $77.6 billion of its common stock and paid dividends of $15.0 billion during 2023.

As of September 30, 2023, the Company had approximately 161,000 full-time equivalent employees.

Cash and cash equivalents were $29.9 billion as of September 30, 2023.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

The Company uses derivative financial instruments, including forward exchange contracts and option contracts, to manage exposure to fluctuations in foreign currency exchange rates.

Item 8. Financial Statements

The Company had 2,166,758 shareholders of record as of October 13, 2023.

Total net revenue: $383,285 million (2023), $394,328 million (2022)

Apple is committed to becoming carbon neutral across its entire supply chain and product life cycle by 2030.

Legal Proceedings: The Company is subject to legal proceedings and claims that have arisen in the ordinary course of business, including antitrust investigations and litigation related to App Store policies in multiple jurisdictions including the European Union and the United States.
""",
        },
        "MSFT": {
            "10-K_FY2023.txt": """MICROSOFT CORPORATION
ANNUAL REPORT ON FORM 10-K
For the fiscal year ended June 30, 2023

Item 1. Business

Microsoft is a technology company whose mission is to empower every person and every organization on the planet to achieve more. We develop, license, and support a wide range of software products, services, and devices that deliver new opportunities, greater convenience, and enhanced value to people's lives.

We have three operating segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.

We are incorporating AI capabilities across our product portfolio. Copilot, our AI companion, is now integrated across Microsoft 365, Azure, GitHub, Bing, and other products to enhance productivity and enable new experiences.

We compete with companies across all our markets. In cloud infrastructure and platform services, we primarily compete with Amazon Web Services and Google Cloud Platform. In productivity software, we compete with Google Workspace, Salesforce, and other providers.

Item 1A. Risk Factors

Evolving regulatory requirements around AI, data privacy, and cloud services could impose significant compliance costs and restrict our ability to operate in certain markets. Regulations such as the EU AI Act and various national AI governance frameworks are creating new compliance obligations.

Cybersecurity risks remain significant. We face threats from nation-state actors, ransomware groups, and supply chain vulnerabilities. Zero-day exploits targeting our products could cause significant harm to our customers and reputation.

The pending acquisition of Activision Blizzard presents integration risks and faces regulatory scrutiny in multiple jurisdictions that could result in conditions being imposed on the transaction or blocking it entirely.

No single customer accounted for 10% or more of revenue in fiscal year 2023.

Item 7. Management's Discussion and Analysis

Revenue was $211,915 million for fiscal year 2023, compared to $198,270 million for fiscal year 2022, an increase of $13,645 million or 7%.

Intelligent Cloud revenue increased $13,979 million or 19% to $87,907 million, driven by Azure and other cloud services growth of 27%.

LinkedIn revenue increased 10% to $15,145 million.

Our commercial cloud revenue annualized run rate reached $111.6 billion.

We returned $34.3 billion to shareholders through share repurchases and dividends.

Total operating expenses were $135,700 million for fiscal year 2023.

As of June 30, 2023, we had approximately 221,000 full-time employees.

Microsoft has committed to being carbon negative by 2030, water positive by 2030, and zero waste by 2030.

Microsoft's responsible AI principles include fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability.
""",
        },
        "NVDA": {
            "10-K_FY2024.txt": """NVIDIA CORPORATION
ANNUAL REPORT ON FORM 10-K
For the fiscal year ended January 28, 2024

Item 1. Business

NVIDIA pioneered accelerated computing to help solve the most challenging computational problems. Since our original focus on PC graphics, we have expanded to several other large and important computationally intensive fields.

Our two operating segments are: Graphics, and Compute & Networking.

CUDA is our parallel computing platform with over 4 million developers. This platform creates significant switching costs and ecosystem advantages. CUDA enables developers to harness the computing power of NVIDIA GPUs for general purpose processing.

We compete with Advanced Micro Devices and Intel in our GPU markets. Large cloud providers including Google (TPUs) and Amazon (Trainium/Inferentia) are developing custom AI chips that could compete with our Data Center products.

We are developing Blackwell, the successor to our Hopper architecture, with significantly increased transistor counts, memory bandwidth, and inference efficiency.

Item 1A. Risk Factors

Export control regulations imposed by the U.S. government restrict our ability to sell advanced computing chips to China and certain other countries. These regulations materially affect our Data Center revenue and could further tighten.

Our manufacturing is substantially concentrated at TSMC in Taiwan. Advanced packaging capacity is limited industry-wide, creating supply constraints. We are exploring Samsung as an alternative for some manufacturing processes.

A small number of large cloud service providers, including Microsoft, Google, Amazon, and Meta, represent a significant and growing portion of our Data Center revenue, creating customer concentration risk.

Rapid changes in AI model architectures could reduce demand for our current products or require significant investment in new architectures.

We face patent litigation from competitors and patent assertion entities. We rely on trade secrets, patents, and copyright to protect our intellectual property.

Item 7. Management's Discussion and Analysis

Revenue for fiscal year 2024 was $60,922 million, compared to $26,974 million for fiscal year 2023, an increase of 126%.

Data Center revenue was $47,532 million, an increase of 217% from fiscal year 2023. Strong demand from cloud service providers for AI training and inference, particularly for our H100 GPUs, drove this growth.

Gaming revenue was $10,447 million, a modest increase from $9,067 million in fiscal year 2023, recovering from the crypto mining demand collapse and channel inventory correction.

Gross margin was 72.7% for fiscal year 2024, compared to 56.9% for fiscal year 2023.

Our automotive business had an order backlog of approximately $14 billion for our DRIVE platform for autonomous vehicles, representing a long-term growth opportunity.

As of January 28, 2024, we had 29,600 employees.
""",
        },
    }

    for ticker, files in samples.items():
        ticker_dir = output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            fpath = ticker_dir / filename
            fpath.write_text(content)
            print(f"  Created sample: {fpath}")

    print(f"\nSample data created in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--form", default="10-K")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--sample", action="store_true",
                        help="Create sample data instead of downloading")
    args = parser.parse_args()

    if args.sample:
        create_sample_data()
    else:
        print("Downloading from SEC EDGAR...")
        print("Note: For quickstart, use --sample flag to use included sample data.")
        for ticker in args.tickers:
            print(f"\n{ticker}:")
            download_filing(ticker, form=args.form, year=args.year)
