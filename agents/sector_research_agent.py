"""
Sector Research Agent
Generates comprehensive sector-wide analysis reports covering top companies,
industry trends, and regulatory impacts.
"""

import os
import asyncio
import time
from typing import AsyncIterator, Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from services.stock_data_fetchers import fetch_yfinance_data, validate_ticker
from models.stock_data_models import YFinanceData
from services.market_context_2026 import GLOBAL_MARKET_CONTEXT_2026
from utils.logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Initialize Claude client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

logger.info("=" * 70)
logger.info("SECTOR RESEARCH AGENT")
logger.info("=" * 70)


# =============================================================================
# Sector Definitions
# =============================================================================

SECTOR_DEFINITIONS = {
    "Technology": {
        "description": "Software, hardware, IT services, and internet companies",
        "tickers": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "IBM"],
        "etf": "XLK",
        "key_themes": ["AI/ML adoption", "Cloud computing", "Cybersecurity", "Digital transformation"],
    },
    "Semiconductors": {
        "description": "Chip designers, manufacturers, and equipment makers",
        "tickers": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC"],
        "etf": "SMH",
        "key_themes": ["AI chips", "Process node race", "China restrictions", "Supply chain resilience"],
    },
    "Healthcare": {
        "description": "Pharmaceuticals, biotech, medical devices, and healthcare services",
        "tickers": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY"],
        "etf": "XLV",
        "key_themes": ["Drug pricing", "Aging demographics", "Biotech innovation", "Healthcare AI"],
    },
    "Financial Services": {
        "description": "Banks, insurance, asset management, and fintech",
        "tickers": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA"],
        "etf": "XLF",
        "key_themes": ["Interest rates", "Digital banking", "Regulatory environment", "Fintech disruption"],
    },
    "Energy": {
        "description": "Oil & gas, renewable energy, and utilities",
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG", "NEE", "DUK", "SO", "D", "AEP"],
        "etf": "XLE",
        "key_themes": ["Energy transition", "Oil prices", "Renewable growth", "Grid modernization"],
    },
    "Consumer Discretionary": {
        "description": "Retail, automotive, entertainment, and luxury goods",
        "tickers": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
        "etf": "XLY",
        "key_themes": ["E-commerce", "EV adoption", "Consumer spending", "Experience economy"],
    },
    "Industrials": {
        "description": "Aerospace, defense, machinery, and transportation",
        "tickers": ["CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "DE", "UNP", "MMM"],
        "etf": "XLI",
        "key_themes": ["Infrastructure spending", "Supply chain", "Automation", "Defense budgets"],
    },
    "Real Estate": {
        "description": "REITs across residential, commercial, and specialized properties",
        "tickers": ["PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
        "etf": "XLRE",
        "key_themes": ["Interest rate sensitivity", "Remote work impact", "Data center demand", "E-commerce logistics"],
    },
}


@dataclass
class SectorProgressUpdate:
    """Progress update for UI display"""
    stage: str
    stage_display: str
    current_step: int
    total_steps: int
    elapsed_time: float
    message: str
    companies_fetched: int = 0
    total_companies: int = 0
    report: Optional[str] = None
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class TokenAccumulator:
    """Accumulates token usage across multiple API calls."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, response):
        if hasattr(response, 'usage'):
            self.input_tokens += getattr(response.usage, 'input_tokens', 0)
            self.output_tokens += getattr(response.usage, 'output_tokens', 0)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self):
        return (self.input_tokens / 1_000_000) * 3.0 + (self.output_tokens / 1_000_000) * 15.0


# =============================================================================
# Data Fetching
# =============================================================================

async def fetch_sector_data(
    sector: str,
    max_companies: int = 10
) -> Dict[str, Any]:
    """
    Fetch data for all companies in a sector.

    Returns dict with company data, sector stats, and any errors.
    """
    if sector not in SECTOR_DEFINITIONS:
        return {
            "error": f"Unknown sector: {sector}",
            "available_sectors": list(SECTOR_DEFINITIONS.keys())
        }

    sector_info = SECTOR_DEFINITIONS[sector]
    tickers = sector_info["tickers"][:max_companies]

    company_data = {}
    errors = []

    for ticker in tickers:
        try:
            data = await fetch_yfinance_data(ticker)
            if data.fetch_success:
                company_data[ticker] = data
            else:
                errors.append(f"{ticker}: {data.error_message}")
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    # Calculate sector aggregates
    total_market_cap = 0
    avg_pe = []
    avg_margin = []

    for ticker, data in company_data.items():
        if data.price and data.price.market_cap:
            total_market_cap += data.price.market_cap
        if data.ratios:
            if data.ratios.pe_trailing:
                avg_pe.append(data.ratios.pe_trailing)
            if data.ratios.profit_margin:
                avg_margin.append(data.ratios.profit_margin)

    return {
        "sector": sector,
        "description": sector_info["description"],
        "etf": sector_info["etf"],
        "key_themes": sector_info["key_themes"],
        "companies": company_data,
        "errors": errors,
        "aggregates": {
            "total_market_cap": total_market_cap,
            "avg_pe": sum(avg_pe) / len(avg_pe) if avg_pe else None,
            "avg_profit_margin": sum(avg_margin) / len(avg_margin) if avg_margin else None,
            "companies_analyzed": len(company_data),
        }
    }


async def fetch_sector_data_with_progress(
    sector: str,
    max_companies: int = 10,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Fetch sector data with progress updates.
    """
    if sector not in SECTOR_DEFINITIONS:
        return {
            "error": f"Unknown sector: {sector}",
            "available_sectors": list(SECTOR_DEFINITIONS.keys())
        }

    sector_info = SECTOR_DEFINITIONS[sector]
    tickers = sector_info["tickers"][:max_companies]

    company_data = {}
    errors = []

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, len(tickers), ticker)

        try:
            data = await fetch_yfinance_data(ticker)
            if data.fetch_success:
                company_data[ticker] = data
            else:
                errors.append(f"{ticker}: {data.error_message}")
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    # Calculate aggregates
    total_market_cap = 0
    avg_pe = []
    avg_margin = []

    for ticker, data in company_data.items():
        if data.price and data.price.market_cap:
            total_market_cap += data.price.market_cap
        if data.ratios:
            if data.ratios.pe_trailing:
                avg_pe.append(data.ratios.pe_trailing)
            if data.ratios.profit_margin:
                avg_margin.append(data.ratios.profit_margin)

    return {
        "sector": sector,
        "description": sector_info["description"],
        "etf": sector_info["etf"],
        "key_themes": sector_info["key_themes"],
        "companies": company_data,
        "errors": errors,
        "aggregates": {
            "total_market_cap": total_market_cap,
            "avg_pe": sum(avg_pe) / len(avg_pe) if avg_pe else None,
            "avg_profit_margin": sum(avg_margin) / len(avg_margin) if avg_margin else None,
            "companies_analyzed": len(company_data),
        }
    }


# =============================================================================
# Report Generation
# =============================================================================

def _format_sector_data_for_analysis(sector_data: Dict[str, Any]) -> str:
    """Format sector data for Claude analysis."""
    lines = []

    lines.append(f"# {sector_data['sector']} Sector Analysis Data")
    lines.append(f"\n**Description:** {sector_data['description']}")
    lines.append(f"**Sector ETF:** {sector_data['etf']}")
    lines.append(f"**Key Themes:** {', '.join(sector_data['key_themes'])}")

    # Aggregates
    agg = sector_data['aggregates']
    lines.append(f"\n## Sector Aggregates")
    lines.append(f"- Companies Analyzed: {agg['companies_analyzed']}")

    if agg['total_market_cap']:
        mcap = agg['total_market_cap']
        if mcap >= 1e12:
            lines.append(f"- Combined Market Cap: ${mcap/1e12:.2f}T")
        else:
            lines.append(f"- Combined Market Cap: ${mcap/1e9:.2f}B")

    if agg['avg_pe']:
        lines.append(f"- Average P/E Ratio: {agg['avg_pe']:.2f}")
    if agg['avg_profit_margin']:
        lines.append(f"- Average Profit Margin: {agg['avg_profit_margin']:.1%}")

    # Individual companies
    lines.append(f"\n## Company Data")

    for ticker, data in sector_data['companies'].items():
        lines.append(f"\n### {ticker}")

        if data.long_name:
            lines.append(f"**Company:** {data.long_name}")
        if data.sector:
            lines.append(f"**Sector/Industry:** {data.sector} / {data.industry}")

        if data.price:
            lines.append(f"**Price:** ${data.price.current_price:.2f}")
            if data.price.market_cap:
                mcap = data.price.market_cap
                if mcap >= 1e12:
                    lines.append(f"**Market Cap:** ${mcap/1e12:.2f}T")
                elif mcap >= 1e9:
                    lines.append(f"**Market Cap:** ${mcap/1e9:.2f}B")
                else:
                    lines.append(f"**Market Cap:** ${mcap/1e6:.2f}M")
            lines.append(f"**52-Week Range:** ${data.price.fifty_two_week_low:.2f} - ${data.price.fifty_two_week_high:.2f}")

        if data.ratios:
            ratios = []
            if data.ratios.pe_trailing:
                ratios.append(f"P/E: {data.ratios.pe_trailing:.2f}")
            if data.ratios.profit_margin:
                ratios.append(f"Margin: {data.ratios.profit_margin:.1%}")
            if data.ratios.roe:
                ratios.append(f"ROE: {data.ratios.roe:.1%}")
            if data.ratios.debt_to_equity:
                ratios.append(f"D/E: {data.ratios.debt_to_equity:.2f}")
            if ratios:
                lines.append(f"**Ratios:** {' | '.join(ratios)}")

        if data.recommendation_key:
            lines.append(f"**Analyst Rating:** {data.recommendation_key.upper()}")

    if sector_data['errors']:
        lines.append(f"\n## Data Fetch Errors")
        for error in sector_data['errors']:
            lines.append(f"- {error}")

    return "\n".join(lines)


async def generate_sector_report(sector_data: Dict[str, Any], tokens: TokenAccumulator = None, model: str = None) -> str:
    """
    Use Claude to generate a comprehensive sector analysis report.
    """
    formatted_data = _format_sector_data_for_analysis(sector_data)

    system_prompt = f"""You are a senior equity research analyst specializing in sector analysis.
Generate a comprehensive sector research report based on the provided data.

{GLOBAL_MARKET_CONTEXT_2026}

Your report should include:
1. **Executive Summary** - Key takeaways for the sector
2. **Sector Overview** - Current state, size, and structure
3. **Top Companies Analysis** - Leaders, their positions, and key metrics
4. **Competitive Landscape** - How companies compare, market share dynamics
5. **Key Trends & Themes** - What's driving the sector
6. **Risks & Challenges** - Headwinds facing the sector
7. **Investment Outlook** - Overall sector rating and top picks
8. **Valuation Summary Table** - Key metrics for all companies

Use markdown formatting. Be data-driven and cite specific metrics from the data provided.
Focus on actionable insights for investors."""

    user_prompt = f"""Generate a comprehensive sector research report for the {sector_data['sector']} sector.

Here is the data:

{formatted_data}

Please provide a thorough analysis covering all major aspects of this sector."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=8000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    # Track tokens
    if tokens:
        tokens.add(response)

    return response.content[0].text


# =============================================================================
# Main Entry Points
# =============================================================================

async def sector_research(sector: str) -> str:
    """
    Simple entry point for sector research.
    Returns the final report as a string.
    """
    logger.info(f"Starting sector research for: {sector}")

    # Fetch data
    sector_data = await fetch_sector_data(sector)

    if "error" in sector_data:
        return f"Error: {sector_data['error']}\n\nAvailable sectors: {', '.join(sector_data.get('available_sectors', []))}"

    # Generate report
    report = await generate_sector_report(sector_data)

    logger.info(f"Sector research complete for: {sector}")
    return report


async def sector_research_with_progress(sector: str, model: str = None) -> AsyncIterator[SectorProgressUpdate]:
    """
    Async generator that yields progress updates during sector research.
    """
    start_time = time.time()

    # Validate sector
    if sector not in SECTOR_DEFINITIONS:
        yield SectorProgressUpdate(
            stage="error",
            stage_display="Error",
            current_step=0,
            total_steps=1,
            elapsed_time=0,
            message=f"Unknown sector: {sector}. Available: {', '.join(SECTOR_DEFINITIONS.keys())}",
        )
        return

    sector_info = SECTOR_DEFINITIONS[sector]
    tickers = sector_info["tickers"]

    # Stage 1: Fetching data
    yield SectorProgressUpdate(
        stage="fetching",
        stage_display="Fetching Company Data",
        current_step=1,
        total_steps=3,
        elapsed_time=time.time() - start_time,
        message=f"Fetching data for {len(tickers)} companies in {sector}...",
        companies_fetched=0,
        total_companies=len(tickers),
    )

    company_data = {}
    errors = []

    for i, ticker in enumerate(tickers):
        yield SectorProgressUpdate(
            stage="fetching",
            stage_display="Fetching Company Data",
            current_step=1,
            total_steps=3,
            elapsed_time=time.time() - start_time,
            message=f"Fetching {ticker} ({i+1}/{len(tickers)})...",
            companies_fetched=i,
            total_companies=len(tickers),
        )

        try:
            data = await fetch_yfinance_data(ticker)
            if data.fetch_success:
                company_data[ticker] = data
            else:
                errors.append(f"{ticker}: {data.error_message}")
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    # Calculate aggregates
    total_market_cap = 0
    avg_pe = []
    avg_margin = []

    for ticker, data in company_data.items():
        if data.price and data.price.market_cap:
            total_market_cap += data.price.market_cap
        if data.ratios:
            if data.ratios.pe_trailing:
                avg_pe.append(data.ratios.pe_trailing)
            if data.ratios.profit_margin:
                avg_margin.append(data.ratios.profit_margin)

    sector_data = {
        "sector": sector,
        "description": sector_info["description"],
        "etf": sector_info["etf"],
        "key_themes": sector_info["key_themes"],
        "companies": company_data,
        "errors": errors,
        "aggregates": {
            "total_market_cap": total_market_cap,
            "avg_pe": sum(avg_pe) / len(avg_pe) if avg_pe else None,
            "avg_profit_margin": sum(avg_margin) / len(avg_margin) if avg_margin else None,
            "companies_analyzed": len(company_data),
        }
    }

    # Stage 2: Analyzing - with multiple sub-updates
    yield SectorProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Sector",
        current_step=2,
        total_steps=3,
        elapsed_time=time.time() - start_time,
        message=f"Preparing analysis for {len(company_data)} companies...",
        companies_fetched=len(company_data),
        total_companies=len(tickers),
    )

    # Brief pause to show the message
    await asyncio.sleep(0.5)

    yield SectorProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Sector",
        current_step=2,
        total_steps=3,
        elapsed_time=time.time() - start_time,
        message=f"Sending data to Claude AI for analysis... (this may take 30-60 seconds)",
        companies_fetched=len(company_data),
        total_companies=len(tickers),
    )

    # Generate report with token tracking
    tokens = TokenAccumulator()
    report = await generate_sector_report(sector_data, tokens, model=model)

    yield SectorProgressUpdate(
        stage="writing",
        stage_display="Generating Report",
        current_step=2,
        total_steps=3,
        elapsed_time=time.time() - start_time,
        message=f"Formatting final report...",
        companies_fetched=len(company_data),
        total_companies=len(tickers),
    )

    # Stage 3: Complete
    yield SectorProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=3,
        total_steps=3,
        elapsed_time=time.time() - start_time,
        message=f"Sector analysis complete!",
        companies_fetched=len(company_data),
        total_companies=len(tickers),
        report=report,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
    )


def get_available_sectors() -> List[str]:
    """Return list of available sectors."""
    return list(SECTOR_DEFINITIONS.keys())


def get_sector_info(sector: str) -> Optional[Dict]:
    """Get info about a specific sector."""
    return SECTOR_DEFINITIONS.get(sector)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sector_research_agent.py <sector>")
        print(f"Available sectors: {', '.join(get_available_sectors())}")
        sys.exit(1)

    sector = sys.argv[1]

    async def main():
        async for update in sector_research_with_progress(sector):
            print(f"[{update.stage}] {update.message}")
            if update.report:
                print("\n" + "="*70)
                print(update.report)

    asyncio.run(main())
