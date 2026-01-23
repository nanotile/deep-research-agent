"""
Stock Research Agent
A multi-agent system for comprehensive stock analysis using multiple financial data sources.
Generates investment thesis reports with source citations.
"""

import os
import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from stock_data_models import (
    StockDataBundle, StockAnalysis, StockProgressUpdate,
    RecommendationType, ValuationAssessment, InvestmentThesis,
    SourceURL, MacroRiskAssessment, RiskLevel, GeopoliticalImpact2026,
)
from stock_data_fetchers import (
    validate_ticker,
    fetch_yfinance_data,
    fetch_finnhub_data,
    fetch_sec_edgar_filings,
    fetch_alpha_vantage_data,
    fetch_tavily_news,
    fetch_macro_sentiment,
)
from market_context_2026 import (
    is_tech_semiconductor_sector,
    get_china_exposure_level,
    estimate_surcharge_eps_impact,
    get_process_node_status,
    get_software_moat,
    get_2026_analysis_prompt_injection,
    CORE_2026_VARIABLES,
    GLOBAL_MARKET_CONTEXT_2026,
)
from utils.logging_config import get_logger

# Load environment variables
load_dotenv()

# Initialize logging
logger = get_logger(__name__)

# Initialize Claude client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration - read from .env with defaults
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

logger.info("=" * 70)
logger.info("STOCK RESEARCH AGENT")
logger.info("=" * 70)


# =============================================================================
# Tool Definition for Structured Analysis
# =============================================================================

STOCK_ANALYSIS_TOOL = {
    "name": "stock_analysis",
    "description": "Provide structured stock analysis with investment recommendation",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendation": {
                "type": "string",
                "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                "description": "Investment recommendation"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence level from 0 to 1"
            },
            "valuation_assessment": {
                "type": "string",
                "enum": ["undervalued", "fairly_valued", "overvalued"],
                "description": "Current valuation assessment"
            },
            "bull_case": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 bullish arguments",
                "minItems": 3,
                "maxItems": 5
            },
            "bear_case": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 bearish arguments",
                "minItems": 3,
                "maxItems": 5
            },
            "key_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key risk factors"
            },
            "key_catalysts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential positive catalysts"
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence executive summary"
            },
            "price_target": {
                "type": "number",
                "description": "Price target if determinable"
            },
            # 2026 Strategic Outlook fields (for tech/semiconductor)
            "china_exposure": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "China revenue/supply chain exposure level (for tech/semiconductor)"
            },
            "surcharge_eps_impact": {
                "type": "number",
                "description": "Estimated EPS impact percentage from 2026 BIS surcharge"
            },
            "agentic_ai_positioning": {
                "type": "string",
                "description": "Company's positioning for Agentic AI revenue opportunities"
            },
            "process_node_status": {
                "type": "string",
                "description": "Process node technology status (2nm, 3nm, etc.)"
            },
            "software_moat": {
                "type": "string",
                "description": "Software/ecosystem competitive moat"
            },
            "supply_chain_dependency": {
                "type": "string",
                "description": "Key supply chain dependencies (TSMC, Samsung, etc.)"
            },
            "technology_sovereignty": {
                "type": "string",
                "description": "CHIPS Act benefits and domestic production status"
            }
        },
        "required": ["recommendation", "confidence", "valuation_assessment",
                     "bull_case", "bear_case", "key_risks", "summary"]
    }
}


# =============================================================================
# Data Fetching
# =============================================================================

async def fetch_all_stock_data(ticker: str, company_name: str) -> StockDataBundle:
    """
    Fetch data from all sources in parallel.
    Handles failures gracefully - partial data is acceptable.
    """
    print(f"\n🔍 Fetching data for {ticker} ({company_name})...")

    # Define all fetch tasks (first batch - need alpha_vantage for sector)
    tasks = {
        'yfinance': fetch_yfinance_data(ticker),
        'finnhub': fetch_finnhub_data(ticker),
        'sec_filings': fetch_sec_edgar_filings(ticker),
        'alpha_vantage': fetch_alpha_vantage_data(ticker),
        'tavily_news': fetch_tavily_news(ticker, company_name),
    }

    # Execute all in parallel
    results = await asyncio.gather(
        *tasks.values(),
        return_exceptions=True
    )

    # Process results
    bundle = StockDataBundle(ticker=ticker, company_name=company_name)
    errors = []

    for (source_name, _), result in zip(tasks.items(), results):
        if isinstance(result, Exception):
            errors.append(f"{source_name}: {str(result)}")
            print(f"  ❌ {source_name}: Error - {str(result)[:50]}")
        elif hasattr(result, 'fetch_success') and not result.fetch_success:
            errors.append(f"{source_name}: {result.error_message}")
            print(f"  ⚠️ {source_name}: {result.error_message}")
        else:
            setattr(bundle, source_name, result)
            print(f"  ✅ {source_name}: Success")

    # Fetch macro sentiment (get sector from already-fetched data - no duplicate calls)
    sector, _ = bundle.get_sector_industry()

    try:
        macro_result = await fetch_macro_sentiment(ticker, sector=sector)
        if macro_result.fetch_success:
            bundle.macro_sentiment = macro_result
            print(f"  ✅ macro_sentiment: Success (VIX: {macro_result.vix_current})")
        else:
            errors.append(f"macro_sentiment: {macro_result.error_message}")
            print(f"  ⚠️ macro_sentiment: {macro_result.error_message}")
    except Exception as e:
        errors.append(f"macro_sentiment: {str(e)}")
        print(f"  ❌ macro_sentiment: Error - {str(e)[:50]}")

    bundle.fetch_errors = errors
    print(f"\n📊 Data collected from {len(bundle.get_successful_sources())} sources")

    return bundle


# =============================================================================
# Analysis Agent
# =============================================================================

async def analyze_stock_data(ticker: str, company_name: str, data: StockDataBundle) -> StockAnalysis:
    """
    Use Claude to analyze the aggregated data and generate investment thesis.
    Returns structured StockAnalysis via tool calling.
    For tech/semiconductor stocks, includes 2026 Strategic Outlook analysis.
    """
    print("\n🧠 Analyzing data with Claude...")

    # Format data for Claude
    analysis_context = _format_data_for_analysis(data)

    # Determine sector for 2026 context injection (use already-fetched data - no duplicate calls)
    sector, industry = data.get_sector_industry()
    if sector:
        print(f"  📂 Sector: {sector} / {industry}")

    # Check if this is a tech/semiconductor stock
    is_tech_semi = is_tech_semiconductor_sector(sector, industry)

    # Build the analysis prompt
    base_prompt = f"""You are a senior equity analyst. Analyze the following data for {company_name} ({ticker}) and provide a comprehensive investment analysis.

{analysis_context}

Based on this data, provide your analysis using the stock_analysis tool. Be specific and data-driven in your bull/bear cases.
Consider:
- Financial health and trends
- Valuation metrics relative to growth
- Market sentiment and analyst opinions
- Recent news and SEC filings
- Technical indicators if available
- Risk factors and potential catalysts
- MACRO/POLITICAL RISK: Pay special attention to current VIX levels, trade policy news (tariffs, trade wars), Federal Reserve policy, and government economic announcements. Factor in the stock's sector sensitivity to political risk.

Provide a balanced, objective analysis. Include any relevant macro/political risks in your key_risks assessment."""

    # Inject 2026 context for tech/semiconductor stocks
    if is_tech_semi:
        context_2026 = get_2026_analysis_prompt_injection(ticker, sector)
        prompt = base_prompt + context_2026
        print("  📊 2026 Strategic Outlook mode (tech/semiconductor detected)")
    else:
        prompt = base_prompt

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system="You are a professional equity research analyst. Provide data-driven, balanced analysis. Today's date is January 2026.",
        messages=[{"role": "user", "content": prompt}],
        tools=[STOCK_ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "stock_analysis"}
    )

    # Extract tool use result
    tool_use = next(block for block in response.content if block.type == "tool_use")
    analysis_data = tool_use.input

    # Build GeopoliticalImpact2026 for tech/semiconductor stocks
    geopolitical_2026 = None
    if is_tech_semi:
        # Use Claude's assessment if provided, otherwise use our estimates
        china_exposure = analysis_data.get('china_exposure') or get_china_exposure_level(ticker, data)
        surcharge_impact = analysis_data.get('surcharge_eps_impact') or estimate_surcharge_eps_impact(ticker, china_exposure, data)

        geopolitical_2026 = GeopoliticalImpact2026(
            china_exposure=china_exposure,
            surcharge_eps_impact_percent=surcharge_impact,
            technology_sovereignty_status=analysis_data.get('technology_sovereignty'),
            process_node_status=analysis_data.get('process_node_status') or get_process_node_status(ticker),
            software_moat=analysis_data.get('software_moat') or get_software_moat(ticker),
            agentic_ai_positioning=analysis_data.get('agentic_ai_positioning'),
            supply_chain_dependency=analysis_data.get('supply_chain_dependency'),
            export_control_risk=f"BIS 25% surcharge impact: {surcharge_impact}% estimated EPS headwind"
        )

    # Build StockAnalysis model
    analysis = StockAnalysis(
        ticker=ticker,
        company_name=company_name,
        recommendation=RecommendationType(analysis_data['recommendation']),
        confidence=analysis_data['confidence'],
        valuation_assessment=ValuationAssessment(analysis_data['valuation_assessment']),
        thesis=InvestmentThesis(
            bull_case=analysis_data['bull_case'],
            bear_case=analysis_data['bear_case'],
            key_risks=analysis_data['key_risks'],
            key_catalysts=analysis_data.get('key_catalysts', []),
        ),
        geopolitical_2026=geopolitical_2026,
        summary=analysis_data['summary'],
        price_target=analysis_data.get('price_target'),
    )

    print(f"✅ Analysis complete: {analysis.recommendation.value.upper()}")
    if is_tech_semi:
        print(f"  🌏 China Exposure: {geopolitical_2026.china_exposure.upper()}")
        print(f"  📉 Surcharge Impact: {geopolitical_2026.surcharge_eps_impact_percent}% EPS")
    return analysis


def _format_data_for_analysis(data: StockDataBundle) -> str:
    """Format StockDataBundle into readable text for Claude analysis."""
    sections = []

    # Price and basic info from yfinance
    if data.yfinance and data.yfinance.price:
        p = data.yfinance.price
        sections.append(f"""## Current Price Data
- Current Price: ${p.current_price:.2f}
- Day Range: ${p.day_low:.2f} - ${p.day_high:.2f}
- 52-Week Range: ${p.fifty_two_week_low:.2f} - ${p.fifty_two_week_high:.2f}
- Volume: {p.volume:,}
- Market Cap: ${p.market_cap:,}""" if p.market_cap else f"""## Current Price Data
- Current Price: ${p.current_price:.2f}
- Day Range: ${p.day_low:.2f} - ${p.day_high:.2f}
- 52-Week Range: ${p.fifty_two_week_low:.2f} - ${p.fifty_two_week_high:.2f}
- Volume: {p.volume:,}""")

    # Ratios from yfinance
    if data.yfinance and data.yfinance.ratios:
        r = data.yfinance.ratios
        ratios_text = ["## Valuation Ratios"]
        if r.pe_trailing: ratios_text.append(f"- P/E (TTM): {r.pe_trailing:.2f}")
        if r.pe_forward: ratios_text.append(f"- P/E (Forward): {r.pe_forward:.2f}")
        if r.price_to_book: ratios_text.append(f"- P/B: {r.price_to_book:.2f}")
        if r.price_to_sales: ratios_text.append(f"- P/S: {r.price_to_sales:.2f}")
        if r.ev_to_ebitda: ratios_text.append(f"- EV/EBITDA: {r.ev_to_ebitda:.2f}")
        if r.debt_to_equity: ratios_text.append(f"- Debt/Equity: {r.debt_to_equity:.2f}")
        if r.profit_margin: ratios_text.append(f"- Profit Margin: {r.profit_margin:.1%}")
        if r.roe: ratios_text.append(f"- ROE: {r.roe:.1%}")
        if len(ratios_text) > 1:
            sections.append("\n".join(ratios_text))

    # Alpha Vantage overview
    if data.alpha_vantage and data.alpha_vantage.overview:
        o = data.alpha_vantage.overview
        sections.append(f"""## Company Overview
- Sector: {o.sector}
- Industry: {o.industry}
- Revenue TTM: ${o.revenue_ttm:,}""" if o.revenue_ttm else f"""## Company Overview
- Sector: {o.sector}
- Industry: {o.industry}""")

    # Analyst recommendations from Finnhub
    if data.finnhub and data.finnhub.analyst_recommendations:
        rec = data.finnhub.analyst_recommendations[0]
        sections.append(f"""## Analyst Recommendations ({rec.period})
- Strong Buy: {rec.strong_buy}
- Buy: {rec.buy}
- Hold: {rec.hold}
- Sell: {rec.sell}
- Strong Sell: {rec.strong_sell}""")

        if data.finnhub.price_target_mean:
            sections.append(f"""## Price Targets
- Mean: ${data.finnhub.price_target_mean:.2f}
- High: ${data.finnhub.price_target_high:.2f}
- Low: ${data.finnhub.price_target_low:.2f}""")

    # Sentiment from Finnhub
    if data.finnhub and data.finnhub.overall_sentiment is not None:
        sentiment_label = "Bullish" if data.finnhub.overall_sentiment > 0.1 else "Bearish" if data.finnhub.overall_sentiment < -0.1 else "Neutral"
        sections.append(f"## News Sentiment: {sentiment_label} ({data.finnhub.overall_sentiment:.2f})")

    # Recent news headlines
    if data.finnhub and data.finnhub.news:
        news_text = ["## Recent News Headlines"]
        for article in data.finnhub.news[:5]:
            news_text.append(f"- {article.headline} ({article.source})")
        sections.append("\n".join(news_text))

    # SEC Filings
    if data.sec_filings and data.sec_filings.fetch_success:
        sec_text = ["## SEC Filings"]
        if data.sec_filings.latest_10k_date:
            sec_text.append(f"- Latest 10-K: {data.sec_filings.latest_10k_date.strftime('%Y-%m-%d')}")
        if data.sec_filings.latest_10q_date:
            sec_text.append(f"- Latest 10-Q: {data.sec_filings.latest_10q_date.strftime('%Y-%m-%d')}")
        if data.sec_filings.recent_8k_filings:
            sec_text.append(f"- Recent 8-K filings: {len(data.sec_filings.recent_8k_filings)}")
        if len(sec_text) > 1:
            sections.append("\n".join(sec_text))

    # Insider transactions
    if data.finnhub and data.finnhub.insider_transactions:
        insider_text = ["## Recent Insider Transactions"]
        for txn in data.finnhub.insider_transactions[:5]:
            action = "bought" if txn.transaction_type == "P" else "sold"
            insider_text.append(f"- {txn.name} {action} {abs(txn.share_change):,} shares on {txn.transaction_date.strftime('%Y-%m-%d')}")
        sections.append("\n".join(insider_text))

    # Macro/Political Risk Data
    if data.macro_sentiment:
        macro = data.macro_sentiment
        macro_text = ["## Macro & Political Risk Environment"]

        # VIX data
        if macro.vix_current:
            vix_level = "Low" if macro.vix_current < 15 else "Elevated" if macro.vix_current < 20 else "High" if macro.vix_current < 30 else "Extreme"
            macro_text.append(f"- VIX (Volatility Index): {macro.vix_current:.2f} ({vix_level})")
            if macro.vix_change_percent:
                direction = "up" if macro.vix_change_percent > 0 else "down"
                macro_text.append(f"- VIX Change: {direction} {abs(macro.vix_change_percent):.1f}% from previous close")

        # Sector sensitivity
        if macro.sector and macro.sector_political_sensitivity:
            macro_text.append(f"- Sector: {macro.sector}")
            macro_text.append(f"- Political/Trade Policy Sensitivity: {macro.sector_political_sensitivity.upper()}")

        # Political news summary
        if macro.political_news:
            macro_text.append(f"\n### Recent Political/Macro News ({len(macro.political_news)} articles)")
            # Group by category
            by_category = {}
            for item in macro.political_news:
                by_category.setdefault(item.category, []).append(item)

            for category, items in by_category.items():
                macro_text.append(f"\n**{category.replace('_', ' ').title()}:**")
                for item in items[:3]:
                    macro_text.append(f"- {item.title}")

        sections.append("\n".join(macro_text))

    return "\n\n".join(sections)


# =============================================================================
# Report Writer Agent
# =============================================================================

async def write_stock_report(
    ticker: str,
    company_name: str,
    data: StockDataBundle,
    analysis: StockAnalysis
) -> str:
    """
    Generate comprehensive markdown report with all data and source URLs.
    For tech/semiconductor stocks, uses the 2026 Strategic Outlook template.
    """
    print("\n📝 Generating comprehensive report...")

    # Determine sector for template selection (use already-fetched data - no duplicate calls)
    sector, industry = data.get_sector_industry()

    # Check if this is a tech/semiconductor stock
    is_tech_semi = is_tech_semiconductor_sector(sector, industry)

    # Build report sections
    report_sections = []

    # Use 2026 Strategic Outlook template for tech/semiconductor
    if is_tech_semi and analysis.geopolitical_2026:
        print("  📊 Using 2026 Strategic Outlook template")
        geo = analysis.geopolitical_2026

        # China exposure emoji
        china_emoji = "🔴" if geo.china_exposure == "high" else "🟡" if geo.china_exposure == "medium" else "🟢"

        # Build agentic insight
        agentic_insight = geo.agentic_ai_positioning or "Assessment pending - evaluate enterprise AI agent deployment trajectory"

        # Build geopolitical insight
        geopolitical_insight = f"China exposure {geo.china_exposure.upper()} with {geo.surcharge_eps_impact_percent}% estimated EPS impact from BIS surcharges"

        # Process node info for competitive table
        process_node = geo.process_node_status or "N/A"
        software_moat = geo.software_moat or "N/A"
        supply_chain = geo.supply_chain_dependency or "N/A"

        # Peer comparisons (simplified)
        peer_process_node = "TSMC 2nm (leader)" if ticker.upper() != "TSM" else "Samsung 2nm"
        peer_software_moat = "CUDA (NVIDIA)" if ticker.upper() != "NVDA" else "ROCm (AMD)"
        peer_supply_chain = "TSMC primary" if "TSMC" not in str(supply_chain) else "Samsung/Intel alternative"

        # Technology sovereignty status
        tech_sovereignty = geo.technology_sovereignty_status or "Evaluate CHIPS Act benefits and domestic capacity"

        # Header with 2026 Strategic Outlook
        report_sections.append(f"""# {company_name} ({ticker}) Research Report
## 2026 Strategic Outlook

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## 1. Executive Summary

| | |
|---|---|
| **Recommendation** | **{analysis.recommendation.value.replace('_', ' ').upper()}** |
| **Confidence** | {analysis.confidence:.0%} |
| **Valuation** | {analysis.valuation_assessment.value.replace('_', ' ').title()} |
{f"| **Price Target** | ${analysis.price_target:.2f} |" if analysis.price_target else ""}

### Key 2026 Insights
* **Agentic AI Positioning:** {agentic_insight}
* **Geopolitical Risk vs. Opportunity:** {geopolitical_insight}

{analysis.summary}

---

## 2. Technical & Competitive Moat

| Feature | {ticker} Status | Top Peer Comparison |
| :--- | :--- | :--- |
| **Process Node** | {process_node} | {peer_process_node} |
| **Software Moat** | {software_moat} | {peer_software_moat} |
| **Supply Chain** | {supply_chain} | {peer_supply_chain} |

---

## 3. 2026 Geopolitical Impact Table

| Factor | Assessment |
|--------|------------|
| **China Exposure** | {china_emoji} {geo.china_exposure.upper()} |
| **Surcharge Impact** | Estimated {geo.surcharge_eps_impact_percent}% EPS impact from 2026 BIS fee structures |
| **Technology Sovereignty** | {tech_sovereignty} |

### Export Control Analysis
{geo.export_control_risk or "BIS export restrictions apply to high-performance compute exports to China. Monitor quarterly revenue disclosures for China segment exposure."}

---

## 4. Investment Thesis (Bull vs. Bear)

### Bull Case
*Focus on Agentic AI revenue and technology leadership:*
{chr(10).join(f"- {point}" for point in analysis.thesis.bull_case)}

### Bear Case
*Focus on margin compression and regulatory headwinds:*
{chr(10).join(f"- {point}" for point in analysis.thesis.bear_case)}

### Key Catalysts
{chr(10).join(f"- {catalyst}" for catalyst in analysis.thesis.key_catalysts) if analysis.thesis.key_catalysts else "- Agentic AI enterprise adoption acceleration"}

### Key Risks
{chr(10).join(f"- {risk}" for risk in analysis.thesis.key_risks)}
""")
    else:
        # Standard report template for non-tech stocks
        report_sections.append(f"""# Stock Research Report: {ticker}
## {company_name}
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## Executive Summary

{analysis.summary}

| | |
|---|---|
| **Recommendation** | **{analysis.recommendation.value.replace('_', ' ').upper()}** |
| **Confidence** | {analysis.confidence:.0%} |
| **Valuation** | {analysis.valuation_assessment.value.replace('_', ' ').title()} |
{f"| **Price Target** | ${analysis.price_target:.2f} |" if analysis.price_target else ""}

---

## Investment Thesis

### Bull Case
{chr(10).join(f"- {point}" for point in analysis.thesis.bull_case)}

### Bear Case
{chr(10).join(f"- {point}" for point in analysis.thesis.bear_case)}

### Key Catalysts
{chr(10).join(f"- {catalyst}" for catalyst in analysis.thesis.key_catalysts) if analysis.thesis.key_catalysts else "- No specific catalysts identified"}

### Key Risks
{chr(10).join(f"- {risk}" for risk in analysis.thesis.key_risks)}
""")

    # Financial Snapshot
    if data.yfinance and data.yfinance.price:
        p = data.yfinance.price
        report_sections.append(f"""---

## Financial Snapshot

### Current Price Data
| Metric | Value |
|--------|-------|
| Current Price | ${p.current_price:.2f} |
| Day Range | ${p.day_low:.2f} - ${p.day_high:.2f} |
| 52-Week Range | ${p.fifty_two_week_low:.2f} - ${p.fifty_two_week_high:.2f} |
| Volume | {p.volume:,} |
{f"| Market Cap | ${p.market_cap:,} |" if p.market_cap else ""}
""")

    # Valuation Ratios
    if data.yfinance and data.yfinance.ratios:
        r = data.yfinance.ratios
        ratio_rows = []
        if r.pe_trailing: ratio_rows.append(f"| P/E (TTM) | {r.pe_trailing:.2f} |")
        if r.pe_forward: ratio_rows.append(f"| P/E (Forward) | {r.pe_forward:.2f} |")
        if r.price_to_book: ratio_rows.append(f"| Price/Book | {r.price_to_book:.2f} |")
        if r.price_to_sales: ratio_rows.append(f"| Price/Sales | {r.price_to_sales:.2f} |")
        if r.ev_to_ebitda: ratio_rows.append(f"| EV/EBITDA | {r.ev_to_ebitda:.2f} |")
        if r.debt_to_equity: ratio_rows.append(f"| Debt/Equity | {r.debt_to_equity:.2f} |")

        if ratio_rows:
            report_sections.append(f"""### Valuation Ratios
| Metric | Value |
|--------|-------|
{chr(10).join(ratio_rows)}
""")

    # Financial Health
    if data.yfinance and data.yfinance.ratios:
        r = data.yfinance.ratios
        health_rows = []
        if r.profit_margin: health_rows.append(f"| Profit Margin | {r.profit_margin:.1%} |")
        if r.operating_margin: health_rows.append(f"| Operating Margin | {r.operating_margin:.1%} |")
        if r.roe: health_rows.append(f"| ROE | {r.roe:.1%} |")
        if r.roa: health_rows.append(f"| ROA | {r.roa:.1%} |")
        if r.current_ratio: health_rows.append(f"| Current Ratio | {r.current_ratio:.2f} |")

        if health_rows:
            report_sections.append(f"""### Financial Health
| Metric | Value |
|--------|-------|
{chr(10).join(health_rows)}
""")

    # Macro/Political Risk Section
    if data.macro_sentiment:
        macro = data.macro_sentiment
        macro_section = ["---\n\n## Macro & Political Risk"]

        # VIX indicator
        if macro.vix_current:
            vix_level = "Low" if macro.vix_current < 15 else "Elevated" if macro.vix_current < 20 else "High" if macro.vix_current < 30 else "Extreme"
            vix_emoji = "🟢" if macro.vix_current < 15 else "🟡" if macro.vix_current < 20 else "🟠" if macro.vix_current < 30 else "🔴"
            change_str = ""
            if macro.vix_change_percent:
                change_dir = "↑" if macro.vix_change_percent > 0 else "↓"
                change_str = f" ({change_dir} {abs(macro.vix_change_percent):.1f}%)"

            macro_section.append(f"""
### Market Volatility (VIX)
| Indicator | Value |
|-----------|-------|
| VIX Level | {vix_emoji} {macro.vix_current:.2f} ({vix_level}){change_str} |
""")

        # Sector sensitivity
        if macro.sector and macro.sector_political_sensitivity:
            sensitivity_emoji = "🔴" if macro.sector_political_sensitivity == "high" else "🟡" if macro.sector_political_sensitivity == "medium" else "🟢"
            macro_section.append(f"""### Sector Political Sensitivity
| Sector | Sensitivity |
|--------|-------------|
| {macro.sector} | {sensitivity_emoji} {macro.sector_political_sensitivity.upper()} |

*High sensitivity sectors are more exposed to tariffs, trade policy changes, and government regulation.*
""")

        # Political news highlights
        if macro.political_news:
            macro_section.append("### Recent Political & Trade Policy News")
            # Group by category
            by_category = {}
            for item in macro.political_news:
                by_category.setdefault(item.category, []).append(item)

            for category in ['government', 'tariffs', 'trade_policy', 'fed', 'geopolitical', 'regulation']:
                if category in by_category:
                    macro_section.append(f"\n**{category.replace('_', ' ').title()}:**")
                    for item in by_category[category][:2]:
                        macro_section.append(f"- [{item.title}]({item.url})")

        report_sections.append("\n".join(macro_section))

    # Analyst Opinions
    if data.finnhub and data.finnhub.analyst_recommendations:
        rec = data.finnhub.analyst_recommendations[0]
        total = rec.strong_buy + rec.buy + rec.hold + rec.sell + rec.strong_sell
        report_sections.append(f"""---

## Analyst Opinions

### Recommendation Distribution ({rec.period})
| Rating | Count | % |
|--------|-------|---|
| Strong Buy | {rec.strong_buy} | {rec.strong_buy/total*100:.0f}% |
| Buy | {rec.buy} | {rec.buy/total*100:.0f}% |
| Hold | {rec.hold} | {rec.hold/total*100:.0f}% |
| Sell | {rec.sell} | {rec.sell/total*100:.0f}% |
| Strong Sell | {rec.strong_sell} | {rec.strong_sell/total*100:.0f}% |
""")

        if data.finnhub.price_target_mean:
            report_sections.append(f"""### Price Targets
| | Price |
|---|------|
| Mean | ${data.finnhub.price_target_mean:.2f} |
| High | ${data.finnhub.price_target_high:.2f} |
| Low | ${data.finnhub.price_target_low:.2f} |
""")

    # News Section with Links
    news_items = []
    if data.finnhub and data.finnhub.news:
        for article in data.finnhub.news[:5]:
            news_items.append(f"1. **{article.headline}** - {article.source}, {article.datetime.strftime('%Y-%m-%d')}\n   > {article.summary[:200]}...\n   [Read more]({article.url})")

    if data.tavily_news and data.tavily_news.articles:
        for article in data.tavily_news.articles[:5]:
            news_items.append(f"1. **{article.title}**\n   > {article.content[:200]}...\n   [Read more]({article.url})")

    if news_items:
        report_sections.append(f"""---

## Latest News

{chr(10).join(news_items[:8])}
""")

    # SEC Filings with Links
    if data.sec_filings and data.sec_filings.fetch_success:
        sec_items = []
        if data.sec_filings.latest_10k_url:
            sec_items.append(f"- **10-K Annual Report** ({data.sec_filings.latest_10k_date.strftime('%Y-%m-%d')}): [View on SEC.gov]({data.sec_filings.latest_10k_url})")
        if data.sec_filings.latest_10q_url:
            sec_items.append(f"- **10-Q Quarterly Report** ({data.sec_filings.latest_10q_date.strftime('%Y-%m-%d')}): [View on SEC.gov]({data.sec_filings.latest_10q_url})")

        if data.sec_filings.recent_8k_filings:
            sec_items.append("\n### Recent 8-K Filings (Material Events)")
            for filing in data.sec_filings.recent_8k_filings[:3]:
                sec_items.append(f"- {filing.filing_date.strftime('%Y-%m-%d')}: [View on SEC.gov]({filing.url})")

        if sec_items:
            report_sections.append(f"""---

## SEC Filings

{chr(10).join(sec_items)}
""")

    # Insider Activity
    if data.finnhub and data.finnhub.insider_transactions:
        insider_rows = []
        for txn in data.finnhub.insider_transactions[:5]:
            action = "Purchase" if txn.transaction_type == "P" else "Sale"
            insider_rows.append(f"| {txn.name} | {action} | {abs(txn.share_change):,} | {txn.transaction_date.strftime('%Y-%m-%d')} |")

        report_sections.append(f"""---

## Insider Activity

| Name | Transaction | Shares | Date |
|------|-------------|--------|------|
{chr(10).join(insider_rows)}
""")

    # Institutional Ownership
    if data.yfinance and data.yfinance.institutional_holders:
        holder_rows = []
        for h in data.yfinance.institutional_holders[:5]:
            pct = f"{h.percent_held:.1%}" if h.percent_held else "N/A"
            holder_rows.append(f"| {h.holder} | {h.shares:,} | {pct} |")

        report_sections.append(f"""---

## Top Institutional Holders

| Institution | Shares | % Held |
|-------------|--------|--------|
{chr(10).join(holder_rows)}
""")

    # Sources Section
    all_sources = data.get_all_source_urls()
    if all_sources:
        # Deduplicate and categorize
        seen = set()
        news_sources = []
        filing_sources = []
        data_sources = []

        for src in all_sources:
            if src.url in seen:
                continue
            seen.add(src.url)

            link = f"- [{src.title}]({src.url})"
            if src.source_type == "news":
                news_sources.append(link)
            elif src.source_type == "sec_filing":
                filing_sources.append(link)
            else:
                data_sources.append(link)

        sources_text = ["---\n\n## Sources"]
        if data_sources:
            sources_text.append("\n### Data Sources\n" + "\n".join(data_sources[:5]))
        if news_sources:
            sources_text.append("\n### News Articles\n" + "\n".join(news_sources[:10]))
        if filing_sources:
            sources_text.append("\n### SEC Filings\n" + "\n".join(filing_sources[:5]))

        report_sections.append("\n".join(sources_text))

    # Disclaimer
    report_sections.append("""
---

*Disclaimer: This report is generated by AI for informational purposes only. It is not financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
""")

    report = "\n".join(report_sections)
    print("✅ Report generated")
    return report


# =============================================================================
# Main Orchestrator
# =============================================================================

async def stock_research(ticker: str) -> str:
    """
    Main orchestrator function for stock research.
    Pipeline: validate -> fetch_all -> analyze -> write_report
    """
    ticker = ticker.upper().strip()
    print(f"\n🎯 Researching: {ticker}")
    print("=" * 70)

    # Stage 1: Validate ticker
    validation = await validate_ticker(ticker)
    if not validation['is_valid']:
        return f"# Error\n\nInvalid ticker: {validation['error']}"

    company_name = validation['company_name']
    print(f"✅ Valid ticker: {company_name} ({validation['exchange']})")

    # Stage 2: Fetch all data
    data = await fetch_all_stock_data(ticker, company_name)

    if not data.get_successful_sources():
        return f"# Error\n\nFailed to fetch data from any source.\n\nErrors:\n" + "\n".join(data.fetch_errors)

    # Stage 3: Analyze
    analysis = await analyze_stock_data(ticker, company_name, data)

    # Stage 4: Write report
    report = await write_stock_report(ticker, company_name, data, analysis)

    return report


async def stock_research_with_progress(ticker: str) -> AsyncIterator[StockProgressUpdate]:
    """
    Main orchestrator with progress updates for UI.
    Yields StockProgressUpdate objects as research progresses.
    """
    ticker = ticker.upper().strip()
    start_time = time.time()

    # Stage 1: Validate
    yield StockProgressUpdate(
        stage="validating",
        stage_display="Validating Ticker",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message=f"Validating ticker: {ticker}..."
    )

    validation = await validate_ticker(ticker)

    if not validation['is_valid']:
        yield StockProgressUpdate(
            stage="complete",
            stage_display="Error",
            current_step=1,
            total_steps=1,
            elapsed_time=time.time() - start_time,
            message=f"Invalid ticker: {validation['error']}",
            report=f"# Error\n\nInvalid ticker: {validation['error']}"
        )
        return

    company_name = validation['company_name']

    yield StockProgressUpdate(
        stage="validating",
        stage_display="Validating Ticker",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message=f"Valid: {company_name}"
    )

    # Stage 2: Fetch data (with source-by-source updates)
    stage_start = time.time()
    sources = ['yfinance', 'finnhub', 'sec_filings', 'alpha_vantage', 'tavily_news', 'macro_sentiment']
    source_status = {s: "pending" for s in sources}

    yield StockProgressUpdate(
        stage="fetching",
        stage_display="Fetching Data",
        current_step=0,
        total_steps=len(sources),
        elapsed_time=0,
        message="Starting data collection...",
        source_status=source_status.copy()
    )

    data = await fetch_all_stock_data(ticker, company_name)

    # Update source status based on results
    if data.yfinance and data.yfinance.fetch_success: source_status['yfinance'] = "success"
    else: source_status['yfinance'] = "failed"
    if data.finnhub and data.finnhub.fetch_success: source_status['finnhub'] = "success"
    else: source_status['finnhub'] = "failed"
    if data.sec_filings and data.sec_filings.fetch_success: source_status['sec_filings'] = "success"
    else: source_status['sec_filings'] = "failed"
    if data.alpha_vantage and data.alpha_vantage.fetch_success: source_status['alpha_vantage'] = "success"
    else: source_status['alpha_vantage'] = "failed"
    if data.tavily_news and data.tavily_news.fetch_success: source_status['tavily_news'] = "success"
    else: source_status['tavily_news'] = "failed"
    if data.macro_sentiment and data.macro_sentiment.fetch_success: source_status['macro_sentiment'] = "success"
    else: source_status['macro_sentiment'] = "failed"

    successful = sum(1 for s in source_status.values() if s == "success")

    yield StockProgressUpdate(
        stage="fetching",
        stage_display="Fetching Data",
        current_step=len(sources),
        total_steps=len(sources),
        elapsed_time=time.time() - stage_start,
        message=f"Collected data from {successful}/{len(sources)} sources",
        source_status=source_status.copy()
    )

    if not data.get_successful_sources():
        yield StockProgressUpdate(
            stage="complete",
            stage_display="Error",
            current_step=1,
            total_steps=1,
            elapsed_time=time.time() - start_time,
            message="Failed to fetch data from any source",
            report=f"# Error\n\nFailed to fetch data from any source.\n\nErrors:\n" + "\n".join(data.fetch_errors)
        )
        return

    # Stage 3: Analyze
    stage_start = time.time()
    yield StockProgressUpdate(
        stage="analyzing",
        stage_display="AI Analysis",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Analyzing data with Claude..."
    )

    analysis = await analyze_stock_data(ticker, company_name, data)

    yield StockProgressUpdate(
        stage="analyzing",
        stage_display="AI Analysis",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message=f"Analysis complete: {analysis.recommendation.value.replace('_', ' ').upper()}"
    )

    # Stage 4: Write report
    stage_start = time.time()
    yield StockProgressUpdate(
        stage="writing",
        stage_display="Writing Report",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Generating comprehensive report..."
    )

    report = await write_stock_report(ticker, company_name, data, analysis)

    yield StockProgressUpdate(
        stage="writing",
        stage_display="Writing Report",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message="Report generated"
    )

    # Complete
    yield StockProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message="Research complete!",
        report=report,
        analysis=analysis
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for testing."""
    ticker = "AAPL"  # Change this to test different tickers

    report = await stock_research(ticker)

    print("\n" + "=" * 70)
    print("📊 FINAL REPORT")
    print("=" * 70)
    print(report)
    print("\n" + "=" * 70)
    print("✅ Research complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
