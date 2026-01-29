"""
Commodity/Futures Research Agent
Generates comprehensive commodity market analysis reports with macro context,
supply/demand assessment, and price outlook using multiple data sources.
"""

import os
import asyncio
import time
from typing import AsyncIterator, Optional
from datetime import datetime
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from models.commodity_data_models import (
    CommodityDataBundle,
    CommodityAnalysis,
    CommodityThesis,
    CommodityProgressUpdate,
    OutlookType,
    TrendStrength,
)
from services.commodity_data_fetchers import (
    validate_commodity_symbol,
    fetch_all_commodity_data,
    COMMODITY_SYMBOLS,
)
from services.commodity_context_2026 import get_commodity_macro_context
from utils.logging_config import get_logger

# Load environment variables
load_dotenv()

# Initialize logging
logger = get_logger(__name__)

# Initialize Claude client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")


# =============================================================================
# Token Tracking
# =============================================================================

class TokenAccumulator:
    """Accumulates token usage across multiple API calls."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, response):
        if hasattr(response, "usage"):
            self.input_tokens += getattr(response.usage, "input_tokens", 0)
            self.output_tokens += getattr(response.usage, "output_tokens", 0)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self):
        return (self.input_tokens / 1_000_000) * 3.0 + (self.output_tokens / 1_000_000) * 15.0

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0


logger.info("=" * 70)
logger.info("COMMODITY RESEARCH AGENT")
logger.info("=" * 70)


# =============================================================================
# Tool Definition for Structured Analysis
# =============================================================================

COMMODITY_ANALYSIS_TOOL = {
    "name": "commodity_analysis",
    "description": "Provide structured commodity market analysis with outlook",
    "input_schema": {
        "type": "object",
        "properties": {
            "outlook": {
                "type": "string",
                "enum": ["bullish", "neutral", "bearish"],
                "description": "Market outlook for this commodity",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence level from 0 to 1",
            },
            "trend_strength": {
                "type": "string",
                "enum": ["strong", "moderate", "weak"],
                "description": "Strength of the current price trend",
            },
            "bull_case": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 bullish arguments",
                "minItems": 3,
                "maxItems": 5,
            },
            "bear_case": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 bearish arguments",
                "minItems": 3,
                "maxItems": 5,
            },
            "key_catalysts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential positive catalysts",
            },
            "key_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Primary risk factors",
            },
            "price_target_high": {
                "type": "number",
                "description": "High-end price target (6-12 month)",
            },
            "price_target_low": {
                "type": "number",
                "description": "Low-end price target (6-12 month)",
            },
            "supply_demand_balance": {
                "type": "string",
                "description": "Assessment of current supply/demand dynamics (e.g., 'deficit', 'surplus', 'balanced')",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence executive summary of the commodity outlook",
            },
        },
        "required": [
            "outlook",
            "confidence",
            "trend_strength",
            "bull_case",
            "bear_case",
            "key_risks",
            "summary",
        ],
    },
}


# =============================================================================
# Analysis Agent
# =============================================================================

async def analyze_commodity_data(
    symbol: str,
    display_name: str,
    data: CommodityDataBundle,
    tokens: TokenAccumulator,
    model: str = None,
) -> CommodityAnalysis:
    """Use Claude to analyze commodity data and generate market outlook."""
    print(f"\n  Analyzing {display_name} with Claude...")

    context = _format_data_for_analysis(data)
    macro_context = get_commodity_macro_context(data.category)

    prompt = f"""You are a senior commodities analyst. Analyze the following market data for {display_name} and provide a comprehensive outlook.

{context}

{macro_context}

Based on this data, provide your analysis using the commodity_analysis tool. Be specific and data-driven.
Consider:
- Current price action and momentum (1d, 1w, 1m, YTD changes)
- Macro environment (USD, interest rates, inflation)
- Supply/demand fundamentals specific to this commodity
- Geopolitical and weather risks
- Seasonal patterns
- Recent news sentiment

Provide a balanced, objective analysis with clear bull and bear cases."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=2048,
        system="You are a professional commodities research analyst. Provide data-driven, balanced analysis. Today's date is January 2026.",
        messages=[{"role": "user", "content": prompt}],
        tools=[COMMODITY_ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "commodity_analysis"},
    )

    tokens.add(response)

    tool_use = next(block for block in response.content if block.type == "tool_use")
    result = tool_use.input

    analysis = CommodityAnalysis(
        symbol=symbol,
        display_name=display_name,
        outlook=OutlookType(result["outlook"]),
        confidence=result["confidence"],
        trend_strength=TrendStrength(result["trend_strength"]),
        thesis=CommodityThesis(
            bull_case=result["bull_case"],
            bear_case=result["bear_case"],
            key_catalysts=result.get("key_catalysts", []),
            key_risks=result["key_risks"],
        ),
        price_target_high=result.get("price_target_high"),
        price_target_low=result.get("price_target_low"),
        supply_demand_balance=result.get("supply_demand_balance"),
        summary=result["summary"],
    )

    print(f"  Analysis complete: {analysis.outlook.value.upper()} ({analysis.confidence:.0%} confidence)")
    return analysis


def _format_data_for_analysis(data: CommodityDataBundle) -> str:
    """Format CommodityDataBundle into text for Claude analysis."""
    sections = []

    # Price data
    if data.yfinance and data.yfinance.price:
        p = data.yfinance.price
        price_lines = [f"## Current Price Data - {data.display_name}"]
        price_lines.append(f"- Spot Price: ${p.spot_price:.2f}")
        if p.day_high and p.day_low:
            price_lines.append(f"- Day Range: ${p.day_low:.2f} - ${p.day_high:.2f}")
        if p.fifty_two_week_high and p.fifty_two_week_low:
            price_lines.append(f"- 52-Week Range: ${p.fifty_two_week_low:.2f} - ${p.fifty_two_week_high:.2f}")
        if p.volume:
            price_lines.append(f"- Volume: {p.volume:,}")

        changes = []
        if p.pct_change_1d is not None:
            changes.append(f"1D: {p.pct_change_1d:+.2f}%")
        if p.pct_change_1w is not None:
            changes.append(f"1W: {p.pct_change_1w:+.2f}%")
        if p.pct_change_1m is not None:
            changes.append(f"1M: {p.pct_change_1m:+.2f}%")
        if p.pct_change_ytd is not None:
            changes.append(f"YTD: {p.pct_change_ytd:+.2f}%")
        if changes:
            price_lines.append(f"- Performance: {' | '.join(changes)}")

        sections.append("\n".join(price_lines))

    # Alpha Vantage spot
    if data.alpha_vantage and data.alpha_vantage.fetch_success and data.alpha_vantage.spot_price:
        sections.append(
            f"## Alpha Vantage Spot Price\n"
            f"- Price: ${data.alpha_vantage.spot_price:.2f}"
            f" (as of {data.alpha_vantage.last_refreshed or 'N/A'})"
        )

    # FRED macro
    if data.fred_macro and data.fred_macro.fetch_success:
        m = data.fred_macro
        macro_lines = ["## Macro Environment (FRED)"]
        if m.dxy_index:
            macro_lines.append(f"- US Dollar Index (DXY): {m.dxy_index:.2f}")
        if m.treasury_10y:
            macro_lines.append(f"- 10-Year Treasury Yield: {m.treasury_10y:.2f}%")
        if m.treasury_2y:
            macro_lines.append(f"- 2-Year Treasury Yield: {m.treasury_2y:.2f}%")
        if m.fed_funds_rate:
            macro_lines.append(f"- Fed Funds Rate: {m.fed_funds_rate:.2f}%")
        if m.inflation_expectations_5y:
            macro_lines.append(f"- 5Y Inflation Expectations: {m.inflation_expectations_5y:.2f}%")
        if m.cpi_yoy:
            macro_lines.append(f"- CPI YoY: {m.cpi_yoy:.2f}%")
        sections.append("\n".join(macro_lines))

    # News
    if data.tavily_news and data.tavily_news.articles:
        news_lines = [f"## Recent News ({len(data.tavily_news.articles)} articles)"]
        for article in data.tavily_news.articles[:8]:
            news_lines.append(f"- **{article.title}**\n  {article.content[:200]}...")
        sections.append("\n".join(news_lines))

    return "\n\n".join(sections)


# =============================================================================
# Report Writer
# =============================================================================

async def write_commodity_report(
    symbol: str,
    display_name: str,
    data: CommodityDataBundle,
    analysis: CommodityAnalysis,
) -> str:
    """Generate comprehensive markdown report."""
    print(f"\n  Generating report for {display_name}...")

    info = COMMODITY_SYMBOLS.get(symbol, {})
    unit = info.get("unit", "USD")
    contract_size = info.get("contract_size", "N/A")

    # Outlook formatting
    outlook_emoji = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}
    trend_emoji = {"strong": "💪", "moderate": "➡️", "weak": "〰️"}

    sections = []

    # Header
    sections.append(f"""# {display_name} Commodity Research Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## Executive Summary

{analysis.summary}

| | |
|---|---|
| **Outlook** | {outlook_emoji.get(analysis.outlook.value, '')} **{analysis.outlook.value.upper()}** |
| **Confidence** | {analysis.confidence:.0%} |
| **Trend Strength** | {trend_emoji.get(analysis.trend_strength.value, '')} {analysis.trend_strength.value.title()} |
{f"| **Price Target Range** | ${analysis.price_target_low:.2f} - ${analysis.price_target_high:.2f} |" if analysis.price_target_low and analysis.price_target_high else ""}
{f"| **Supply/Demand** | {analysis.supply_demand_balance} |" if analysis.supply_demand_balance else ""}
| **Unit** | {unit} |
| **Contract Size** | {contract_size} |
""")

    # Market Outlook
    sections.append(f"""---

## Market Outlook

### Bull Case
{chr(10).join(f"- {point}" for point in analysis.thesis.bull_case)}

### Bear Case
{chr(10).join(f"- {point}" for point in analysis.thesis.bear_case)}

### Key Catalysts
{chr(10).join(f"- {c}" for c in analysis.thesis.key_catalysts) if analysis.thesis.key_catalysts else "- No specific catalysts identified"}

### Key Risks
{chr(10).join(f"- {r}" for r in analysis.thesis.key_risks)}
""")

    # Price Action
    if data.yfinance and data.yfinance.price:
        p = data.yfinance.price
        perf_rows = []
        if p.pct_change_1d is not None:
            perf_rows.append(f"| 1 Day | {p.pct_change_1d:+.2f}% |")
        if p.pct_change_1w is not None:
            perf_rows.append(f"| 1 Week | {p.pct_change_1w:+.2f}% |")
        if p.pct_change_1m is not None:
            perf_rows.append(f"| 1 Month | {p.pct_change_1m:+.2f}% |")
        if p.pct_change_ytd is not None:
            perf_rows.append(f"| YTD | {p.pct_change_ytd:+.2f}% |")

        sections.append(f"""---

## Price Action

| Metric | Value |
|--------|-------|
| Spot Price | ${p.spot_price:.2f} |
{f"| Day Range | ${p.day_low:.2f} - ${p.day_high:.2f} |" if p.day_high and p.day_low else ""}
{f"| 52-Week Range | ${p.fifty_two_week_low:.2f} - ${p.fifty_two_week_high:.2f} |" if p.fifty_two_week_high and p.fifty_two_week_low else ""}
{f"| Volume | {p.volume:,} |" if p.volume else ""}

### Performance
| Period | Change |
|--------|--------|
{chr(10).join(perf_rows)}
""")

    # Macro Environment
    if data.fred_macro and data.fred_macro.fetch_success:
        m = data.fred_macro
        macro_rows = []
        if m.dxy_index:
            macro_rows.append(f"| US Dollar Index (DXY) | {m.dxy_index:.2f} |")
        if m.treasury_10y:
            macro_rows.append(f"| 10-Year Treasury Yield | {m.treasury_10y:.2f}% |")
        if m.treasury_2y:
            macro_rows.append(f"| 2-Year Treasury Yield | {m.treasury_2y:.2f}% |")
        if m.fed_funds_rate:
            macro_rows.append(f"| Fed Funds Rate | {m.fed_funds_rate:.2f}% |")
        if m.inflation_expectations_5y:
            macro_rows.append(f"| 5Y Inflation Expectations | {m.inflation_expectations_5y:.2f}% |")
        if m.cpi_yoy:
            macro_rows.append(f"| CPI (YoY) | {m.cpi_yoy:.2f}% |")

        if macro_rows:
            sections.append(f"""---

## Macro Environment

| Indicator | Value |
|-----------|-------|
{chr(10).join(macro_rows)}
""")

    # News
    news_items = []
    if data.tavily_news and data.tavily_news.articles:
        for article in data.tavily_news.articles[:8]:
            news_items.append(
                f"1. **{article.title}**\n"
                f"   > {article.content[:200]}...\n"
                f"   [Read more]({article.url})"
            )

    if news_items:
        sections.append(f"""---

## Latest News

{chr(10).join(news_items)}
""")

    # Sources
    all_sources = data.get_all_source_urls()
    if all_sources:
        seen = set()
        source_links = []
        for src in all_sources:
            if src.url in seen:
                continue
            seen.add(src.url)
            source_links.append(f"- [{src.title}]({src.url})")

        sections.append(f"""---

## Sources

{chr(10).join(source_links[:15])}
""")

    # Disclaimer
    sections.append("""
---

*Disclaimer: This report is generated by AI for informational purposes only. It is not financial advice. Commodity futures trading involves substantial risk of loss and is not suitable for all investors. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
""")

    report = "\n".join(sections)
    print("  Report generated")
    return report


# =============================================================================
# Main Orchestrators
# =============================================================================

async def commodity_research(symbol_input: str, model: str = None) -> str:
    """
    Simple async entry point for commodity research.
    Pipeline: validate -> fetch_all -> analyze -> write_report
    """
    print(f"\n  Researching commodity: {symbol_input}")
    print("=" * 70)

    tokens = TokenAccumulator()

    # Validate
    validation = validate_commodity_symbol(symbol_input)
    if not validation["is_valid"]:
        return f"# Error\n\n{validation['error']}"

    symbol = validation["symbol"]
    display_name = validation["display_name"]
    category = validation["category"]
    yf_symbol = validation["yfinance_symbol"]
    av_function = validation["alpha_vantage_function"]

    print(f"  Valid: {display_name} ({yf_symbol})")

    # Fetch
    data = await fetch_all_commodity_data(symbol, display_name, category, yf_symbol, av_function)

    if not data.get_successful_sources():
        return f"# Error\n\nFailed to fetch data from any source.\n\nErrors:\n" + "\n".join(data.fetch_errors)

    # Analyze
    analysis = await analyze_commodity_data(symbol, display_name, data, tokens, model=model)

    # Report
    report = await write_commodity_report(symbol, display_name, data, analysis)

    return report


async def commodity_research_with_progress(
    symbol_input: str, model: str = None
) -> AsyncIterator[CommodityProgressUpdate]:
    """
    Async generator yielding CommodityProgressUpdate for UI.
    Stages: validating -> fetching -> analyzing -> writing -> complete
    """
    start_time = time.time()
    tokens = TokenAccumulator()

    # Stage 1: Validate
    yield CommodityProgressUpdate(
        stage="validating",
        stage_display="Validating Commodity",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message=f"Validating: {symbol_input}...",
    )

    validation = validate_commodity_symbol(symbol_input)

    if not validation["is_valid"]:
        yield CommodityProgressUpdate(
            stage="complete",
            stage_display="Error",
            current_step=1,
            total_steps=1,
            elapsed_time=time.time() - start_time,
            message=validation["error"],
            report=f"# Error\n\n{validation['error']}",
        )
        return

    symbol = validation["symbol"]
    display_name = validation["display_name"]
    category = validation["category"]
    yf_symbol = validation["yfinance_symbol"]
    av_function = validation["alpha_vantage_function"]

    yield CommodityProgressUpdate(
        stage="validating",
        stage_display="Validating Commodity",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message=f"Valid: {display_name} ({yf_symbol})",
    )

    # Stage 2: Fetch data
    stage_start = time.time()
    sources = ["yfinance", "alpha_vantage", "fred_macro", "tavily_news"]
    source_status = {s: "pending" for s in sources}

    yield CommodityProgressUpdate(
        stage="fetching",
        stage_display="Fetching Data",
        current_step=0,
        total_steps=len(sources),
        elapsed_time=0,
        message="Starting data collection...",
        source_status=source_status.copy(),
    )

    data = await fetch_all_commodity_data(symbol, display_name, category, yf_symbol, av_function)

    # Update source status
    if data.yfinance and data.yfinance.fetch_success:
        source_status["yfinance"] = "success"
    else:
        source_status["yfinance"] = "failed"
    if data.alpha_vantage and data.alpha_vantage.fetch_success:
        source_status["alpha_vantage"] = "success"
    else:
        source_status["alpha_vantage"] = "failed"
    if data.fred_macro and data.fred_macro.fetch_success:
        source_status["fred_macro"] = "success"
    else:
        source_status["fred_macro"] = "failed"
    if data.tavily_news and data.tavily_news.fetch_success:
        source_status["tavily_news"] = "success"
    else:
        source_status["tavily_news"] = "failed"

    successful = sum(1 for s in source_status.values() if s == "success")

    yield CommodityProgressUpdate(
        stage="fetching",
        stage_display="Fetching Data",
        current_step=len(sources),
        total_steps=len(sources),
        elapsed_time=time.time() - stage_start,
        message=f"Collected data from {successful}/{len(sources)} sources",
        source_status=source_status.copy(),
    )

    if not data.get_successful_sources():
        yield CommodityProgressUpdate(
            stage="complete",
            stage_display="Error",
            current_step=1,
            total_steps=1,
            elapsed_time=time.time() - start_time,
            message="Failed to fetch data from any source",
            report=f"# Error\n\nFailed to fetch data from any source.\n\nErrors:\n" + "\n".join(data.fetch_errors),
        )
        return

    # Stage 3: Analyze
    stage_start = time.time()
    yield CommodityProgressUpdate(
        stage="analyzing",
        stage_display="AI Analysis",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Analyzing data with Claude...",
    )

    analysis = await analyze_commodity_data(symbol, display_name, data, tokens, model=model)

    yield CommodityProgressUpdate(
        stage="analyzing",
        stage_display="AI Analysis",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message=f"Analysis complete: {analysis.outlook.value.upper()} ({analysis.confidence:.0%})",
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
    )

    # Stage 4: Write report
    stage_start = time.time()
    yield CommodityProgressUpdate(
        stage="writing",
        stage_display="Writing Report",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Generating comprehensive report...",
    )

    report = await write_commodity_report(symbol, display_name, data, analysis)

    yield CommodityProgressUpdate(
        stage="writing",
        stage_display="Writing Report",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message="Report generated",
    )

    # Complete
    yield CommodityProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message="Research complete!",
        report=report,
        analysis=analysis,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for testing."""
    symbol = "gold"
    report = await commodity_research(symbol)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(report)
    print("\n" + "=" * 70)
    print("Research complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
