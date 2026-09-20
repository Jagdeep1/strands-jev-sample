"""Example 4: extract facts from Amazon's 10-K with a Strands agent, then verify each one with Jev.

Extraction is generation, so the chat model does it: it searches the filing with tools, reads the
pages it needs, and returns a structured `FilingFacts` object where every value carries the page it
came from and a verbatim quote. Verification is a decision, so Jev does it: one request checks all
fields against their cited pages and returns a probability per check. Code accepts, rejects, or flags
each field for review.

    python examples/04_document_extraction.py [--model global.anthropic.claude-sonnet-5]

The agent model runs on Amazon Bedrock; Jev runs on OpenRouter.

The PDF (Amazon's 2025 Annual Report, which contains the FY2025 Form 10-K) is downloaded to data/ on
first run.
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

import httpx
from common import POWERFUL_MODEL, chat_model, enable_logs, jev_client
from pydantic import BaseModel, Field
from pypdf import PdfReader
from strands import Agent, tool

from strands_jev import ExtractedField, JevFieldVerifier

enable_logs()
logging.getLogger("pypdf").setLevel(logging.ERROR)  # the report's fonts trigger a harmless fontTools notice

PDF_URL = "https://s2.q4cdn.com/299287126/files/doc_financials/2026/ar/Amazon-2025-Annual-Report.pdf"
PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "amazon-2025-annual-report.pdf"


# ----------------------------------------------------------------------------- the document


def load_pages() -> dict[int, str]:
    if not PDF_PATH.exists():
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {PDF_URL} ...")
        with httpx.stream("GET", PDF_URL, follow_redirects=True, headers={"User-Agent": "strands-jev-example"}) as r:
            r.raise_for_status()
            PDF_PATH.write_bytes(b"".join(r.iter_bytes()))
    reader = PdfReader(str(PDF_PATH))
    return {i + 1: (page.extract_text() or "") for i, page in enumerate(reader.pages)}


PAGES = load_pages()
print(f"loaded {len(PAGES)} pages from {PDF_PATH.name}")


@tool
def search_filing(query: str, max_results: int = 8) -> list[dict]:
    """Case-insensitive search of the filing. Returns page numbers with a short snippet around each match."""
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits = []
    for page, text in PAGES.items():
        m = pattern.search(text)
        if m:
            start, end = max(0, m.start() - 150), min(len(text), m.end() + 250)
            hits.append({"page": page, "snippet": re.sub(r"\s+", " ", text[start:end])})
            if len(hits) >= max_results:
                break
    return hits or [{"error": f"no page contains {query!r}"}]


@tool
def read_page(page: int) -> str:
    """Return the full text of one page (1-based)."""
    return PAGES.get(page, f"page {page} does not exist; the filing has {len(PAGES)} pages")


# ----------------------------------------------------------------------------- what to extract


class Fact(BaseModel):
    value: str = Field(description="The extracted value, as written in the filing")
    page: int = Field(description="1-based page number the value was read from")
    quote: str = Field(description="Short verbatim quote from that page containing the value")


class FilingFacts(BaseModel):
    registrant_name: Fact = Field(description="Exact name of registrant as specified in its charter")
    fiscal_year_end: Fact = Field(description="Fiscal year end date the 10-K covers")
    state_of_incorporation: Fact = Field(description="State or other jurisdiction of incorporation")
    commission_file_number: Fact = Field(description="SEC Commission File Number")
    exchange: Fact = Field(description="Exchange on which the common stock is registered")
    shares_outstanding: Fact = Field(description="Number of shares of common stock outstanding on the cover page")
    total_net_sales: Fact = Field(description="Total net sales for fiscal 2025 from the consolidated statements of operations, USD millions")
    net_income: Fact = Field(description="Net income for fiscal 2025 from the consolidated statements of operations, USD millions")
    employees: Fact = Field(description="Approximate number of full-time and part-time employees as of fiscal year end")
    auditor: Fact = Field(description="Independent registered public accounting firm")
    ceo: Fact = Field(description="Name of the President and Chief Executive Officer")
    # The three below are deliberately confusable. Each sits next to a look-alike figure in the filing
    # (another segment, another year, a GAAP vs non-GAAP cousin, or a total before a deduction), which is
    # exactly where Jev's `right_item` check earns its keep.
    aws_operating_income: Fact = Field(
        description="Operating income of the AWS segment alone for fiscal 2025 from the segment information note, "
        "USD millions. Not AWS net sales, not consolidated operating income, not a prior year."
    )
    free_cash_flow: Fact = Field(
        description="Free cash flow for fiscal 2025 as defined in the MD&A non-GAAP reconciliation (operating cash "
        "flow less purchases of property and equipment net of proceeds and incentives), USD millions. Not net cash "
        "provided by operating activities, and not one of the other free-cash-flow variants."
    )
    long_term_debt: Fact = Field(
        description="Long-term debt as of December 31, 2025 from the consolidated balance sheet, excluding the current "
        "portion, USD millions. Not the December 31, 2024 column, not total debt including the current portion, "
        "and not lease liabilities."
    )


def to_extracted(facts: FilingFacts) -> list[ExtractedField]:
    return [
        ExtractedField(name=name, description=field.description or name, value=fact.value, page=fact.page, quote=fact.quote)
        for name, field in FilingFacts.model_fields.items()
        for fact in [getattr(facts, name)]
    ]


# ----------------------------------------------------------------------------- run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=POWERFUL_MODEL)
    args = parser.parse_args()

    agent = Agent(
        model=chat_model(args.model, max_tokens=4096),
        tools=[search_filing, read_page],
        system_prompt=(
            "You extract facts from an SEC Form 10-K. Use search_filing to locate each fact and read_page to "
            "confirm it in context. For every field report the page you read it from and a short quote copied "
            "verbatim from that page (keep numbers and punctuation exactly as printed). Financial statement "
            "figures must come from the fiscal 2025 column of the consolidated statements of operations. "
            "Never guess; if a fact truly is not in the filing, say so in the value."
        ),
        callback_handler=None,
    )

    print(f"\nextracting with {args.model} ...")
    result = agent(
        "Extract every field of FilingFacts from this Form 10-K.",
        structured_output_model=FilingFacts,
    )
    facts = result.structured_output
    if not isinstance(facts, FilingFacts):
        print("agent did not return FilingFacts:", str(result)[:500])
        return 1

    print("verifying with Jev ...")
    verifier = JevFieldVerifier(jev_client())
    verdicts = asyncio.run(verifier.verify(to_extracted(facts), PAGES))

    print(f"\n{'field':24} {'page':>4}  {'value':34} {'supported':>9} {'right':>6}  verdict")
    for v in verdicts:
        sup = f"{v.supported:.2f}" if v.supported is not None else "  -  "
        right = f"{v.right_item:.2f}" if v.right_item is not None else "  -  "
        value = str(v.field.value)[:34]
        print(f"{v.field.name:24} {v.field.page:>4}  {value:34} {sup:>9} {right:>6}  {v.outcome:7} {v.reason}")

    counts = {o: sum(1 for v in verdicts if v.outcome == o) for o in ("accept", "review", "reject")}
    print(f"\n{counts}  jev cost: ${verifier.last_cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
