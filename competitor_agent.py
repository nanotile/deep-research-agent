"""
Competitor Intelligence Agent
Auto-detects competitors for a given stock and generates a comparative analysis
with competitive positioning insights.
"""

import os
import asyncio
import time
from typing import AsyncIterator, Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from stock_data_fetchers import fetch_yfinance_data, validate_ticker
from stock_data_models import YFinanceData
from market_context_2026 import GLOBAL_MARKET_CONTEXT_2026
from utils.logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Initialize Claude client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

logger.info("=" * 70)
logger.info("COMPETITOR INTELLIGENCE AGENT")
logger.info("=" * 70)


# =============================================================================
# Competitor Mappings by Industry
# =============================================================================

# Maps companies to their known competitors
COMPETITOR_MAP = {
    # Tech Giants
    "AAPL": ["MSFT", "GOOGL", "SAMSUNG.KS", "DELL", "HPQ"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL", "CRM"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL", "SNAP"],
    "META": ["GOOGL", "SNAP", "PINS", "TWTR", "MSFT"],
    "AMZN": ["WMT", "TGT", "SHOP", "EBAY", "BABA"],

    # Semiconductors
    "NVDA": ["AMD", "INTC", "QCOM", "AVGO", "MU"],
    "AMD": ["NVDA", "INTC", "QCOM", "TXN", "AVGO"],
    "INTC": ["AMD", "NVDA", "QCOM", "TXN", "AVGO"],
    "AVGO": ["QCOM", "TXN", "MRVL", "ADI", "NXPI"],
    "QCOM": ["AVGO", "MRVL", "INTC", "AMD", "TXN"],

    # EVs & Auto
    "TSLA": ["RIVN", "LCID", "F", "GM", "NIO"],
    "RIVN": ["TSLA", "LCID", "F", "GM", "FSR"],
    "F": ["GM", "TSLA", "TM", "HMC", "STLA"],
    "GM": ["F", "TSLA", "TM", "HMC", "STLA"],

    # Financials
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "BAC": ["JPM", "WFC", "C", "USB", "PNC"],
    "GS": ["MS", "JPM", "SCHW", "BLK", "BX"],
    "V": ["MA", "AXP", "PYPL", "SQ", "DFS"],
    "MA": ["V", "AXP", "PYPL", "DFS", "SQ"],

    # Cloud/SaaS
    "CRM": ["MSFT", "ORCL", "SAP", "WDAY", "NOW"],
    "ORCL": ["MSFT", "SAP", "CRM", "IBM", "WDAY"],
    "NOW": ["CRM", "WDAY", "MSFT", "SNOW", "DDOG"],

    # Retail
    "WMT": ["TGT", "COST", "AMZN", "KR", "DG"],
    "TGT": ["WMT", "COST", "AMZN", "KR", "DG"],
    "COST": ["WMT", "TGT", "BJ", "AMZN", "KR"],
    "HD": ["LOW", "WMT", "TGT", "MEN", "TSCO"],
    "LOW": ["HD", "WMT", "TGT", "MEN", "TSCO"],

    # Healthcare/Pharma
    "JNJ": ["PFE", "MRK", "ABBV", "LLY", "BMY"],
    "PFE": ["JNJ", "MRK", "ABBV", "BMY", "GSK"],
    "UNH": ["CVS", "CI", "ELV", "HUM", "CNC"],

    # Energy
    "XOM": ["CVX", "COP", "BP", "SHEL", "TTE"],
    "CVX": ["XOM", "COP", "BP", "SHEL", "TTE"],

    # Streaming/Entertainment
    "NFLX": ["DIS", "WBD", "PARA", "CMCSA", "AMZN"],
    "DIS": ["NFLX", "WBD", "PARA", "CMCSA", "SONY"],
}

# Industry to competitors fallback
INDUSTRY_COMPETITORS = {
    "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU"],
    "Software—Infrastructure": ["MSFT", "ORCL", "CRM", "NOW", "WDAY"],
    "Software—Application": ["CRM", "ADBE", "INTU", "WDAY", "SPLK"],
    "Internet Content & Information": ["GOOGL", "META", "SNAP", "PINS", "TWTR"],
    "Internet Retail": ["AMZN", "EBAY", "ETSY", "W", "CHWY"],
    "Auto Manufacturers": ["TSLA", "F", "GM", "TM", "RIVN"],
    "Banks—Diversified": ["JPM", "BAC", "WFC", "C", "USB"],
    "Drug Manufacturers": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
    "Oil & Gas Integrated": ["XOM", "CVX", "COP", "BP", "SHEL"],
}


@dataclass
class CompetitorProgressUpdate:
    """Progress update for UI display"""
    stage: str
    stage_display: str
    current_step: int
    total_steps: int
    elapsed_time: float
    message: str
    report: Optional[str] = None


# =============================================================================
# Competitor Detection
# =============================================================================

async def find_competitors(
    ticker: str,
    max_competitors: int = 5
) -> List[str]:
    """
    Find competitors for a given ticker.
    Uses predefined mappings first, then falls back to industry matching.
    """
    ticker = ticker.upper()

    # Check direct mapping first
    if ticker in COMPETITOR_MAP:
        return COMPETITOR_MAP[ticker][:max_competitors]

    # Fallback: Get company's industry and find peers
    try:
        data = await fetch_yfinance_data(ticker)
        if data.fetch_success and data.industry:
            industry = data.industry
            if industry in INDUSTRY_COMPETITORS:
                competitors = [t for t in INDUSTRY_COMPETITORS[industry] if t != ticker]
                return competitors[:max_competitors]
    except Exception as e:
        logger.error(f"Error finding competitors: {e}")

    return []


async def fetch_competitor_data(
    primary_ticker: str,
    competitors: List[str]
) -> Dict[str, Any]:
    """
    Fetch data for the primary company and its competitors.
    """
    all_tickers = [primary_ticker] + competitors
    company_data = {}
    errors = []

    for ticker in all_tickers:
        try:
            data = await fetch_yfinance_data(ticker)
            if data.fetch_success:
                company_data[ticker] = data
            else:
                errors.append(f"{ticker}: {data.error_message}")
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    return {
        "primary": primary_ticker,
        "competitors": competitors,
        "data": company_data,
        "errors": errors,
    }


# =============================================================================
# Report Generation
# =============================================================================

def _build_comparison_matrix(data: Dict[str, YFinanceData]) -> str:
    """Build a markdown comparison table."""
    if not data:
        return "*No data available*"

    tickers = list(data.keys())

    # Header
    lines = [
        f"| Metric | {' | '.join(tickers)} |",
        f"|--------|{'|'.join(['------' for _ in tickers])}|",
    ]

    # Price
    row = ["**Price**"]
    for t in tickers:
        if data[t].price:
            row.append(f"${data[t].price.current_price:.2f}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # Market Cap
    row = ["**Market Cap**"]
    for t in tickers:
        if data[t].price and data[t].price.market_cap:
            mcap = data[t].price.market_cap
            if mcap >= 1e12:
                row.append(f"${mcap/1e12:.2f}T")
            elif mcap >= 1e9:
                row.append(f"${mcap/1e9:.2f}B")
            else:
                row.append(f"${mcap/1e6:.2f}M")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # P/E
    row = ["**P/E (TTM)**"]
    for t in tickers:
        if data[t].ratios and data[t].ratios.pe_trailing:
            row.append(f"{data[t].ratios.pe_trailing:.2f}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # Forward P/E
    row = ["**P/E (Fwd)**"]
    for t in tickers:
        if data[t].ratios and data[t].ratios.pe_forward:
            row.append(f"{data[t].ratios.pe_forward:.2f}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # Profit Margin
    row = ["**Profit Margin**"]
    for t in tickers:
        if data[t].ratios and data[t].ratios.profit_margin:
            row.append(f"{data[t].ratios.profit_margin:.1%}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # ROE
    row = ["**ROE**"]
    for t in tickers:
        if data[t].ratios and data[t].ratios.roe:
            row.append(f"{data[t].ratios.roe:.1%}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # Debt/Equity
    row = ["**Debt/Equity**"]
    for t in tickers:
        if data[t].ratios and data[t].ratios.debt_to_equity:
            row.append(f"{data[t].ratios.debt_to_equity:.2f}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # 52-Week Performance (vs low)
    row = ["**52W Range**"]
    for t in tickers:
        if data[t].price:
            row.append(f"${data[t].price.fifty_two_week_low:.0f}-${data[t].price.fifty_two_week_high:.0f}")
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    # Analyst Rating
    row = ["**Analyst Rating**"]
    for t in tickers:
        if data[t].recommendation_key:
            row.append(data[t].recommendation_key.upper())
        else:
            row.append("N/A")
    lines.append(f"| {' | '.join(row)} |")

    return "\n".join(lines)


def _format_competitor_data(comp_data: Dict[str, Any]) -> str:
    """Format competitor data for Claude analysis."""
    lines = []
    primary = comp_data["primary"]
    data = comp_data["data"]

    lines.append(f"# Competitive Analysis: {primary}")
    lines.append(f"\n**Primary Company:** {primary}")
    lines.append(f"**Competitors:** {', '.join(comp_data['competitors'])}")

    # Primary company details
    if primary in data:
        pdata = data[primary]
        lines.append(f"\n## {primary} (Primary)")
        if pdata.long_name:
            lines.append(f"**Name:** {pdata.long_name}")
        if pdata.sector:
            lines.append(f"**Sector/Industry:** {pdata.sector} / {pdata.industry}")

    # Comparison matrix
    lines.append("\n## Comparative Metrics")
    lines.append(_build_comparison_matrix(data))

    # Individual company details
    lines.append("\n## Company Details")
    for ticker, d in data.items():
        lines.append(f"\n### {ticker}")
        if d.long_name:
            lines.append(f"- Name: {d.long_name}")
        if d.sector:
            lines.append(f"- Industry: {d.industry}")
        if d.price and d.price.market_cap:
            mcap = d.price.market_cap
            if mcap >= 1e12:
                lines.append(f"- Market Cap: ${mcap/1e12:.2f}T")
            elif mcap >= 1e9:
                lines.append(f"- Market Cap: ${mcap/1e9:.2f}B")

    return "\n".join(lines)


async def generate_competitor_report(comp_data: Dict[str, Any]) -> str:
    """Generate competitive intelligence report using Claude."""
    formatted_data = _format_competitor_data(comp_data)
    primary = comp_data["primary"]

    system_prompt = f"""You are a competitive intelligence analyst providing strategic insights.
Generate a comprehensive competitive analysis report.

{GLOBAL_MARKET_CONTEXT_2026}

Your report should include:
1. **Executive Summary** - Key competitive insights
2. **Market Position Analysis** - Where the primary company stands vs competitors
3. **Competitive Advantages** - What makes the primary company stronger
4. **Competitive Vulnerabilities** - Where competitors have advantages
5. **Comparative Metrics Analysis** - Deep dive into the numbers
6. **Strategic Recommendations** - How the primary company should position
7. **Competitive Threats** - Key risks from competitors
8. **Winner Picks** - Which companies are best positioned

Use markdown formatting. Be specific and cite data from the comparison."""

    user_prompt = f"""Generate a competitive intelligence report for {primary} against its competitors.

{formatted_data}

Provide strategic insights on competitive positioning, strengths, weaknesses, and recommendations."""

    response = await client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


# =============================================================================
# Main Entry Points
# =============================================================================

async def competitor_analysis(ticker: str, max_competitors: int = 5) -> str:
    """
    Simple entry point for competitor analysis.
    Returns the final report as a string.
    """
    logger.info(f"Starting competitor analysis for: {ticker}")

    # Find competitors
    competitors = await find_competitors(ticker, max_competitors)
    if not competitors:
        return f"Could not find competitors for {ticker}. Please check the ticker symbol."

    # Fetch data
    comp_data = await fetch_competitor_data(ticker, competitors)

    # Generate report
    report = await generate_competitor_report(comp_data)

    logger.info(f"Competitor analysis complete for: {ticker}")
    return report


async def competitor_analysis_with_progress(
    ticker: str,
    max_competitors: int = 5
) -> AsyncIterator[CompetitorProgressUpdate]:
    """
    Async generator that yields progress updates during competitor analysis.
    """
    start_time = time.time()
    ticker = ticker.upper()

    # Stage 1: Finding competitors
    yield CompetitorProgressUpdate(
        stage="finding",
        stage_display="Finding Competitors",
        current_step=1,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Identifying competitors for {ticker}...",
    )

    competitors = await find_competitors(ticker, max_competitors)

    if not competitors:
        yield CompetitorProgressUpdate(
            stage="error",
            stage_display="Error",
            current_step=1,
            total_steps=4,
            elapsed_time=time.time() - start_time,
            message=f"Could not find competitors for {ticker}. Check the ticker symbol.",
        )
        return

    # Stage 2: Fetching primary data
    yield CompetitorProgressUpdate(
        stage="fetching",
        stage_display="Fetching Data",
        current_step=2,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Fetching data for {ticker} and {len(competitors)} competitors...",
    )

    comp_data = await fetch_competitor_data(ticker, competitors)

    # Stage 3: Analyzing - with sub-updates
    yield CompetitorProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Competition",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Building competitive comparison matrix...",
    )

    await asyncio.sleep(0.5)

    yield CompetitorProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Competition",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Sending to Claude AI for competitive analysis... (this may take 30-60 seconds)",
    )

    report = await generate_competitor_report(comp_data)

    yield CompetitorProgressUpdate(
        stage="writing",
        stage_display="Generating Report",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Formatting competitive intelligence report...",
    )

    # Stage 4: Complete
    yield CompetitorProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=4,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Competitive analysis complete!",
        report=report,
    )


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python competitor_agent.py <ticker>")
        print("Example: python competitor_agent.py NVDA")
        sys.exit(1)

    ticker = sys.argv[1]

    async def main():
        async for update in competitor_analysis_with_progress(ticker):
            print(f"[{update.stage}] {update.message}")
            if update.report:
                print("\n" + "="*70)
                print(update.report)

    asyncio.run(main())
