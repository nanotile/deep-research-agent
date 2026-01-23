"""
Portfolio Analyzer Agent
Accepts a portfolio of holdings (CSV or manual input) and generates
risk analysis, sector exposure, and rebalancing suggestions.
"""

import os
import asyncio
import time
import csv
import io
from typing import AsyncIterator, Optional, Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from stock_data_fetchers import fetch_yfinance_data, validate_ticker
from stock_data_models import YFinanceData
from market_context_2026 import GLOBAL_MARKET_CONTEXT_2026
from utils.logging_config import get_logger
from utils.validators import sanitize_ticker

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Initialize Claude client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

logger.info("=" * 70)
logger.info("PORTFOLIO ANALYZER AGENT")
logger.info("=" * 70)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PortfolioHolding:
    """A single holding in the portfolio."""
    ticker: str
    shares: float
    purchase_price: Optional[float] = None  # Cost basis per share
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    gain_loss: Optional[float] = None
    gain_loss_pct: Optional[float] = None
    weight: Optional[float] = None  # Portfolio weight
    sector: Optional[str] = None
    industry: Optional[str] = None
    company_name: Optional[str] = None


@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis results."""
    holdings: List[PortfolioHolding]
    total_value: float
    total_cost: Optional[float]
    total_gain_loss: Optional[float]
    total_gain_loss_pct: Optional[float]
    sector_exposure: Dict[str, float]  # sector -> weight
    top_holdings: List[Tuple[str, float]]  # top by weight
    concentration_warnings: List[str]
    diversification_score: float  # 0-100
    fetch_errors: List[str]


@dataclass
class PortfolioProgressUpdate:
    """Progress update for UI display."""
    stage: str
    stage_display: str
    current_step: int
    total_steps: int
    elapsed_time: float
    message: str
    holdings_processed: int = 0
    total_holdings: int = 0
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
# Portfolio Parsing
# =============================================================================

def parse_portfolio_csv(csv_content: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse CSV content into portfolio holdings.

    Expected format:
    ticker,shares[,cost_basis]

    Returns:
        Tuple of (holdings_list, errors_list)
    """
    holdings = []
    errors = []

    try:
        # Try to detect if there's a header
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        if not rows:
            return [], ["Empty CSV file"]

        # Check if first row is a header
        first_row = rows[0]
        start_idx = 0
        if first_row[0].lower() in ['ticker', 'symbol', 'stock']:
            start_idx = 1

        for i, row in enumerate(rows[start_idx:], start=start_idx + 1):
            if not row or not row[0].strip():
                continue

            ticker = row[0].strip().upper()
            sanitized, is_valid, error = sanitize_ticker(ticker)

            if not is_valid:
                errors.append(f"Row {i}: Invalid ticker '{ticker}' - {error}")
                continue

            try:
                shares = float(row[1]) if len(row) > 1 else 0
                if shares <= 0:
                    errors.append(f"Row {i}: Invalid shares for {ticker}")
                    continue
            except ValueError:
                errors.append(f"Row {i}: Invalid shares value for {ticker}")
                continue

            cost_basis = None
            if len(row) > 2 and row[2].strip():
                try:
                    cost_basis = float(row[2])
                except ValueError:
                    pass  # Cost basis is optional

            holdings.append({
                "ticker": sanitized,
                "shares": shares,
                "cost_basis": cost_basis,
            })

    except Exception as e:
        errors.append(f"CSV parsing error: {str(e)}")

    return holdings, errors


def parse_portfolio_text(text: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse simple text format into holdings.

    Formats supported:
    - AAPL 100
    - AAPL: 100
    - AAPL, 100
    - AAPL 100 150.00 (with cost basis)
    """
    holdings = []
    errors = []

    for line_num, line in enumerate(text.strip().split('\n'), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Try different separators
        parts = None
        for sep in [',', ':', '\t', ' ']:
            if sep in line:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                break

        if not parts or len(parts) < 2:
            parts = line.split()

        if len(parts) < 2:
            errors.append(f"Line {line_num}: Could not parse '{line}'")
            continue

        ticker = parts[0].upper()
        sanitized, is_valid, error = sanitize_ticker(ticker)

        if not is_valid:
            errors.append(f"Line {line_num}: Invalid ticker '{ticker}'")
            continue

        try:
            shares = float(parts[1])
            if shares <= 0:
                errors.append(f"Line {line_num}: Shares must be positive for {ticker}")
                continue
        except ValueError:
            errors.append(f"Line {line_num}: Invalid shares value for {ticker}")
            continue

        cost_basis = None
        if len(parts) > 2:
            try:
                cost_basis = float(parts[2])
            except ValueError:
                pass

        holdings.append({
            "ticker": sanitized,
            "shares": shares,
            "cost_basis": cost_basis,
        })

    return holdings, errors


# =============================================================================
# Portfolio Analysis
# =============================================================================

async def analyze_portfolio(
    holdings: List[Dict],
) -> PortfolioAnalysis:
    """
    Analyze a portfolio by fetching current data for all holdings.
    """
    analyzed_holdings = []
    fetch_errors = []
    sector_values = {}
    total_value = 0
    total_cost = 0
    has_cost_basis = False

    for holding in holdings:
        ticker = holding["ticker"]
        shares = holding["shares"]
        cost_basis = holding.get("cost_basis")

        try:
            data = await fetch_yfinance_data(ticker)

            if not data.fetch_success or not data.price:
                fetch_errors.append(f"{ticker}: Failed to fetch data")
                continue

            current_price = data.price.current_price
            current_value = current_price * shares
            total_value += current_value

            # Calculate gain/loss if cost basis provided
            gain_loss = None
            gain_loss_pct = None
            if cost_basis:
                has_cost_basis = True
                cost_value = cost_basis * shares
                total_cost += cost_value
                gain_loss = current_value - cost_value
                gain_loss_pct = (gain_loss / cost_value) * 100 if cost_value > 0 else 0

            # Get sector
            sector = data.sector or "Unknown"
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += current_value

            analyzed_holdings.append(PortfolioHolding(
                ticker=ticker,
                shares=shares,
                purchase_price=cost_basis,
                current_price=current_price,
                current_value=current_value,
                gain_loss=gain_loss,
                gain_loss_pct=gain_loss_pct,
                sector=sector,
                industry=data.industry,
                company_name=data.long_name,
            ))

        except Exception as e:
            fetch_errors.append(f"{ticker}: {str(e)}")

    # Calculate weights
    for holding in analyzed_holdings:
        if total_value > 0:
            holding.weight = (holding.current_value / total_value) * 100

    # Sort holdings by value
    analyzed_holdings.sort(key=lambda h: h.current_value or 0, reverse=True)

    # Calculate sector exposure as percentages
    sector_exposure = {}
    for sector, value in sector_values.items():
        if total_value > 0:
            sector_exposure[sector] = (value / total_value) * 100

    # Sort sectors by exposure
    sector_exposure = dict(sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True))

    # Top holdings
    top_holdings = [(h.ticker, h.weight) for h in analyzed_holdings[:5]]

    # Concentration warnings
    concentration_warnings = []

    # Check for single stock concentration (>25%)
    for holding in analyzed_holdings:
        if holding.weight and holding.weight > 25:
            concentration_warnings.append(
                f"High concentration: {holding.ticker} is {holding.weight:.1f}% of portfolio"
            )

    # Check for sector concentration (>40%)
    for sector, weight in sector_exposure.items():
        if weight > 40:
            concentration_warnings.append(
                f"Sector concentration: {sector} is {weight:.1f}% of portfolio"
            )

    # Check for low diversification (less than 5 holdings)
    if len(analyzed_holdings) < 5:
        concentration_warnings.append(
            f"Low diversification: Only {len(analyzed_holdings)} holdings"
        )

    # Calculate diversification score (simplified)
    # Based on: number of holdings, sector spread, concentration
    div_score = 50  # Start at 50

    # Bonus for number of holdings (up to +20)
    div_score += min(len(analyzed_holdings) * 2, 20)

    # Bonus for sector spread (up to +20)
    div_score += min(len(sector_exposure) * 4, 20)

    # Penalty for concentration
    max_holding_weight = max((h.weight or 0) for h in analyzed_holdings) if analyzed_holdings else 0
    if max_holding_weight > 30:
        div_score -= (max_holding_weight - 30)

    max_sector_weight = max(sector_exposure.values()) if sector_exposure else 0
    if max_sector_weight > 40:
        div_score -= (max_sector_weight - 40) / 2

    div_score = max(0, min(100, div_score))

    # Calculate totals
    total_gain_loss = None
    total_gain_loss_pct = None
    if has_cost_basis and total_cost > 0:
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost) * 100

    return PortfolioAnalysis(
        holdings=analyzed_holdings,
        total_value=total_value,
        total_cost=total_cost if has_cost_basis else None,
        total_gain_loss=total_gain_loss,
        total_gain_loss_pct=total_gain_loss_pct,
        sector_exposure=sector_exposure,
        top_holdings=top_holdings,
        concentration_warnings=concentration_warnings,
        diversification_score=div_score,
        fetch_errors=fetch_errors,
    )


# =============================================================================
# Report Generation
# =============================================================================

def _format_portfolio_data(analysis: PortfolioAnalysis) -> str:
    """Format portfolio analysis for Claude."""
    lines = []

    lines.append("# Portfolio Analysis Data")

    # Summary
    lines.append("\n## Portfolio Summary")
    lines.append(f"- **Total Value:** ${analysis.total_value:,.2f}")
    if analysis.total_cost:
        lines.append(f"- **Total Cost Basis:** ${analysis.total_cost:,.2f}")
    if analysis.total_gain_loss is not None:
        sign = "+" if analysis.total_gain_loss >= 0 else ""
        lines.append(f"- **Total Gain/Loss:** {sign}${analysis.total_gain_loss:,.2f} ({sign}{analysis.total_gain_loss_pct:.1f}%)")
    lines.append(f"- **Number of Holdings:** {len(analysis.holdings)}")
    lines.append(f"- **Diversification Score:** {analysis.diversification_score:.0f}/100")

    # Sector Exposure
    lines.append("\n## Sector Exposure")
    for sector, weight in analysis.sector_exposure.items():
        lines.append(f"- {sector}: {weight:.1f}%")

    # Holdings Table
    lines.append("\n## Holdings")
    lines.append("| Ticker | Shares | Price | Value | Weight | Sector |")
    lines.append("|--------|--------|-------|-------|--------|--------|")

    for h in analysis.holdings:
        lines.append(
            f"| {h.ticker} | {h.shares:.2f} | ${h.current_price:.2f} | "
            f"${h.current_value:,.2f} | {h.weight:.1f}% | {h.sector} |"
        )

    # Performance (if cost basis available)
    if analysis.total_cost:
        lines.append("\n## Performance by Holding")
        lines.append("| Ticker | Cost | Current | Gain/Loss | % |")
        lines.append("|--------|------|---------|-----------|---|")
        for h in analysis.holdings:
            if h.gain_loss is not None:
                sign = "+" if h.gain_loss >= 0 else ""
                lines.append(
                    f"| {h.ticker} | ${h.purchase_price:.2f} | ${h.current_price:.2f} | "
                    f"{sign}${h.gain_loss:,.2f} | {sign}{h.gain_loss_pct:.1f}% |"
                )

    # Warnings
    if analysis.concentration_warnings:
        lines.append("\n## Risk Warnings")
        for warning in analysis.concentration_warnings:
            lines.append(f"- ⚠️ {warning}")

    # Errors
    if analysis.fetch_errors:
        lines.append("\n## Data Fetch Errors")
        for error in analysis.fetch_errors:
            lines.append(f"- {error}")

    return "\n".join(lines)


async def generate_portfolio_report(analysis: PortfolioAnalysis, tokens: TokenAccumulator = None) -> str:
    """Generate portfolio analysis report using Claude."""
    formatted_data = _format_portfolio_data(analysis)

    system_prompt = f"""You are a portfolio manager and financial advisor providing analysis.
Generate a comprehensive portfolio analysis report.

{GLOBAL_MARKET_CONTEXT_2026}

Your report should include:
1. **Executive Summary** - Key portfolio metrics and overall assessment
2. **Portfolio Composition** - Analysis of holdings and weights
3. **Sector Analysis** - Exposure analysis with commentary on diversification
4. **Risk Assessment** - Concentration risks, volatility, and concerns
5. **Performance Analysis** - If cost basis provided, analyze winners/losers
6. **Rebalancing Suggestions** - Specific recommendations to improve portfolio
7. **Strategic Recommendations** - What to add, reduce, or eliminate
8. **Action Items** - Prioritized list of suggested changes

Use markdown formatting. Be specific with numbers and percentages.
Provide actionable advice, not generic statements."""

    user_prompt = f"""Analyze this portfolio and provide recommendations:

{formatted_data}

Give specific, actionable advice on improving diversification, reducing risk, and optimizing returns."""

    response = await client.messages.create(
        model=MODEL,
        max_tokens=6000,
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

async def portfolio_analysis_from_text(text: str) -> str:
    """
    Analyze portfolio from text input.
    """
    holdings, parse_errors = parse_portfolio_text(text)

    if not holdings:
        return f"Could not parse any holdings.\n\nErrors:\n" + "\n".join(parse_errors)

    analysis = await analyze_portfolio(holdings)
    report = await generate_portfolio_report(analysis)

    return report


async def portfolio_analysis_from_csv(csv_content: str) -> str:
    """
    Analyze portfolio from CSV content.
    """
    holdings, parse_errors = parse_portfolio_csv(csv_content)

    if not holdings:
        return f"Could not parse any holdings from CSV.\n\nErrors:\n" + "\n".join(parse_errors)

    analysis = await analyze_portfolio(holdings)
    report = await generate_portfolio_report(analysis)

    return report


async def portfolio_analysis_with_progress(
    text_or_csv: str,
    is_csv: bool = False
) -> AsyncIterator[PortfolioProgressUpdate]:
    """
    Async generator that yields progress updates during portfolio analysis.
    """
    start_time = time.time()

    # Stage 1: Parsing
    yield PortfolioProgressUpdate(
        stage="parsing",
        stage_display="Parsing Portfolio",
        current_step=1,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message="Parsing portfolio data...",
    )

    if is_csv:
        holdings, parse_errors = parse_portfolio_csv(text_or_csv)
    else:
        holdings, parse_errors = parse_portfolio_text(text_or_csv)

    if not holdings:
        yield PortfolioProgressUpdate(
            stage="error",
            stage_display="Error",
            current_step=1,
            total_steps=4,
            elapsed_time=time.time() - start_time,
            message=f"Could not parse holdings. Errors: {'; '.join(parse_errors)}",
        )
        return

    # Stage 2: Fetching data
    yield PortfolioProgressUpdate(
        stage="fetching",
        stage_display="Fetching Market Data",
        current_step=2,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Fetching data for {len(holdings)} holdings...",
        holdings_processed=0,
        total_holdings=len(holdings),
    )

    analysis = await analyze_portfolio(holdings)

    # Stage 3: Analyzing - with sub-updates
    yield PortfolioProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Portfolio",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message="Calculating sector exposure and risk metrics...",
        holdings_processed=len(analysis.holdings),
        total_holdings=len(holdings),
    )

    await asyncio.sleep(0.5)

    yield PortfolioProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Portfolio",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Portfolio value: ${analysis.total_value:,.2f} | Diversification score: {analysis.diversification_score:.0f}/100",
        holdings_processed=len(analysis.holdings),
        total_holdings=len(holdings),
    )

    await asyncio.sleep(0.5)

    yield PortfolioProgressUpdate(
        stage="analyzing",
        stage_display="Analyzing Portfolio",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message="Sending to Claude AI for recommendations... (this may take 30-60 seconds)",
        holdings_processed=len(analysis.holdings),
        total_holdings=len(holdings),
    )

    # Generate report with token tracking
    tokens = TokenAccumulator()
    report = await generate_portfolio_report(analysis, tokens)

    yield PortfolioProgressUpdate(
        stage="writing",
        stage_display="Generating Report",
        current_step=3,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message="Formatting portfolio analysis report...",
        holdings_processed=len(analysis.holdings),
        total_holdings=len(holdings),
    )

    # Stage 4: Complete
    yield PortfolioProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=4,
        total_steps=4,
        elapsed_time=time.time() - start_time,
        message=f"Portfolio analysis complete! Analyzed {len(analysis.holdings)} holdings.",
        holdings_processed=len(analysis.holdings),
        total_holdings=len(holdings),
        report=report,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
    )


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    # Example portfolio for testing
    example_portfolio = """
AAPL 50
MSFT 30
GOOGL 20
NVDA 15
AMZN 25
JPM 40
JNJ 35
"""

    print("Portfolio Analyzer - Using example portfolio:")
    print(example_portfolio)
    print("\n" + "=" * 70)

    async def main():
        async for update in portfolio_analysis_with_progress(example_portfolio):
            print(f"[{update.stage}] {update.message}")
            if update.report:
                print("\n" + "=" * 70)
                print(update.report)

    asyncio.run(main())
