"""
2026 Global Market Context Module
Provides 2026-specific market intelligence for Stock Research Agent including:
- Trade policy context (BIS 25% China Surcharge)
- Tech milestones (2nm process node race)
- Macro risks (Big Tech capex scrutiny)
"""

from typing import List, Dict, Any, Optional


# =============================================================================
# 2026 Global Market Context
# =============================================================================

GLOBAL_MARKET_CONTEXT_2026 = {
    "trade_policy": {
        "china_status": "BIS 25% Federal Surcharge for high-performance compute exports (Jan 2026)",
        "impact_sectors": ["Semiconductors", "AI Infrastructure", "High-End Networking"],
        "key_regulations": [
            "Bureau of Industry and Security (BIS) export controls",
            "Advanced computing chip restrictions to China",
            "AI model training hardware limitations"
        ]
    },
    "tech_milestones": {
        "process_node": "Entry into 2nm volume production (TSMC/Samsung)",
        "ai_paradigm": "Shift from 'Chatbots' to 'Agentic AI Workflows' with measurable ROI benchmarks",
        "key_players": {
            "foundry_leaders": ["TSMC", "Samsung", "Intel"],
            "ai_infrastructure": ["NVIDIA", "AMD", "Intel", "Broadcom"],
            "hyperscalers": ["Microsoft", "Google", "Amazon", "Meta"]
        }
    },
    "macro_risks": {
        "capex_scrutiny": "Big Tech $500B annual spend monitoring; transition from 'infrastructure build' to 'revenue realization'",
        "interest_rates": "Fed policy normalization impact on growth stocks",
        "geopolitical": "US-China tech decoupling acceleration"
    }
}


# =============================================================================
# Core 2026 Analysis Variables
# =============================================================================

CORE_2026_VARIABLES = {
    "china_surcharge": {
        "name": "25% China Surcharge",
        "rate": 0.25,
        "description": "BIS 25% Federal Surcharge on high-performance compute exports to China",
        "effective_date": "January 2026",
        "affected_sectors": ["Semiconductors", "Technology", "AI Infrastructure", "High-End Networking"],
        "affected_products": [
            "High-performance GPUs (H100, H200, B100 class)",
            "AI accelerators",
            "Advanced networking chips",
            "High-bandwidth memory (HBM)"
        ],
        "key_companies": ["NVDA", "AMD", "AVGO", "MRVL", "INTC", "MU", "QCOM"]
    },
    "tech_war_2nm": {
        "name": "2nm Tech War",
        "description": "Race for 2nm process node leadership between TSMC and Samsung",
        "leaders": ["TSMC", "Samsung"],
        "challengers": ["Intel"],
        "strategic_importance": "Next-gen AI/mobile chip manufacturing capability",
        "customer_dependencies": {
            "TSMC": ["Apple", "NVIDIA", "AMD", "Qualcomm", "MediaTek"],
            "Samsung": ["Qualcomm", "Google"],
            "Intel": ["Internal"]
        }
    },
    "agentic_ai": {
        "name": "Agentic AI Revolution",
        "description": "Enterprise shift from chatbots to autonomous AI agents with measurable ROI",
        "key_metrics": ["AI revenue contribution", "Enterprise AI adoption rate", "Agent deployment count"],
        "beneficiaries": ["MSFT", "GOOGL", "AMZN", "META", "CRM", "NOW", "PLTR"]
    }
}


# =============================================================================
# Tech/Semiconductor Sector Definition
# =============================================================================

TECH_SEMICONDUCTOR_SECTORS = [
    "Semiconductors",
    "Technology",
    "AI Infrastructure",
    "High-End Networking",
    "Data Center Equipment",
    "Consumer Electronics",
    "Software - Infrastructure",
    "Information Technology Services",
    "Electronic Components",
    "Communication Equipment"
]

TECH_SEMICONDUCTOR_INDUSTRIES = [
    "Semiconductors",
    "Semiconductor Equipment & Materials",
    "Software - Application",
    "Software - Infrastructure",
    "Information Technology Services",
    "Computer Hardware",
    "Electronic Components",
    "Consumer Electronics",
    "Communication Equipment",
    "Scientific & Technical Instruments"
]

# Tickers known to have high China exposure
HIGH_CHINA_EXPOSURE_TICKERS = [
    "NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "MRVL", "LRCX", "AMAT", "KLAC",
    "ASML", "TSM", "TXN", "ADI", "NXPI", "ON", "MCHP"
]

MEDIUM_CHINA_EXPOSURE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "CRM", "NOW", "ADBE", "ORCL", "IBM"
]


# =============================================================================
# Helper Functions
# =============================================================================

def is_tech_semiconductor_sector(sector: Optional[str], industry: Optional[str] = None) -> bool:
    """
    Check if a sector/industry qualifies as tech/semiconductor for 2026 analysis.

    Args:
        sector: Company sector (e.g., "Technology")
        industry: Company industry (e.g., "Semiconductors")

    Returns:
        True if the company should receive 2026 Strategic Outlook treatment
    """
    if not sector and not industry:
        return False

    # Check sector
    if sector:
        sector_lower = sector.lower()
        for tech_sector in TECH_SEMICONDUCTOR_SECTORS:
            if tech_sector.lower() in sector_lower or sector_lower in tech_sector.lower():
                return True

    # Check industry
    if industry:
        industry_lower = industry.lower()
        for tech_industry in TECH_SEMICONDUCTOR_INDUSTRIES:
            if tech_industry.lower() in industry_lower or industry_lower in tech_industry.lower():
                return True

    return False


def get_china_exposure_level(ticker: str, data: Optional[Dict[str, Any]] = None) -> str:
    """
    Determine China exposure level for a ticker.

    Args:
        ticker: Stock ticker symbol
        data: Optional StockDataBundle dict with additional context

    Returns:
        "high", "medium", or "low"
    """
    ticker_upper = ticker.upper()

    # Check known high exposure tickers
    if ticker_upper in HIGH_CHINA_EXPOSURE_TICKERS:
        return "high"

    # Check known medium exposure tickers
    if ticker_upper in MEDIUM_CHINA_EXPOSURE_TICKERS:
        return "medium"

    # If we have data, check sector
    if data:
        sector = None
        if hasattr(data, 'alpha_vantage') and data.alpha_vantage and data.alpha_vantage.overview:
            sector = data.alpha_vantage.overview.sector

        if sector and is_tech_semiconductor_sector(sector):
            return "medium"  # Default for tech companies

    return "low"


def estimate_surcharge_eps_impact(ticker: str, china_exposure: str, data: Optional[Dict[str, Any]] = None) -> float:
    """
    Estimate the EPS impact percentage from the 2026 BIS surcharge.

    This is a rough estimate based on:
    - China exposure level
    - Sector characteristics
    - Known revenue exposure patterns

    Args:
        ticker: Stock ticker symbol
        china_exposure: "high", "medium", or "low"
        data: Optional StockDataBundle for additional context

    Returns:
        Estimated EPS impact as a percentage (e.g., 5.0 for 5%)
    """
    ticker_upper = ticker.upper()

    # Base impact by exposure level
    base_impacts = {
        "high": 8.0,    # 8% base for high exposure
        "medium": 3.0,  # 3% base for medium exposure
        "low": 0.5      # 0.5% base for low exposure
    }

    impact = base_impacts.get(china_exposure, 0.5)

    # Adjust for specific high-impact tickers
    if ticker_upper in ["NVDA", "AMD", "AVGO", "MRVL"]:
        impact *= 1.5  # GPU/AI chip companies have higher exposure
    elif ticker_upper in ["LRCX", "AMAT", "KLAC"]:
        impact *= 1.3  # Semiconductor equipment
    elif ticker_upper in ["MU", "QCOM"]:
        impact *= 1.2  # Memory and mobile chips

    return round(impact, 1)


def get_process_node_status(ticker: str) -> Optional[str]:
    """
    Get the process node status for semiconductor companies.

    Returns:
        Process node information or None if not applicable
    """
    ticker_upper = ticker.upper()

    process_node_map = {
        "TSM": "2nm volume production leader (N2), 3nm mature",
        "INTC": "Intel 18A (1.8nm equivalent) ramping, competing with TSMC",
        "NVDA": "TSMC 3nm/2nm for next-gen GPUs (Blackwell successor)",
        "AMD": "TSMC 3nm current, transitioning to 2nm",
        "AAPL": "TSMC 2nm for A-series and M-series chips",
        "QCOM": "TSMC/Samsung 3nm, evaluating 2nm transition",
        "AVGO": "Mixed nodes, custom silicon partnerships"
    }

    return process_node_map.get(ticker_upper)


def get_software_moat(ticker: str) -> Optional[str]:
    """
    Get the software/ecosystem moat status for tech companies.

    Returns:
        Software moat description or None if not applicable
    """
    ticker_upper = ticker.upper()

    moat_map = {
        "NVDA": "CUDA ecosystem dominance, cuDNN, TensorRT, Triton",
        "AMD": "ROCm growing, HIP compatibility layer",
        "MSFT": "Azure AI, Copilot, GitHub, enterprise integration",
        "GOOGL": "TensorFlow, JAX, TPU ecosystem, Gemini",
        "AMZN": "AWS Bedrock, SageMaker, custom Trainium/Inferentia",
        "META": "PyTorch leadership, open-source AI models (Llama)",
        "CRM": "Einstein AI, Data Cloud, industry-specific AI",
        "NOW": "Now Assist, workflow automation AI",
        "PLTR": "AIP, Foundry, Gotham - government and enterprise AI"
    }

    return moat_map.get(ticker_upper)


# =============================================================================
# 2026 Search Query Filters
# =============================================================================

def get_2026_search_queries(ticker: str, company_name: str, sector: Optional[str] = None) -> List[tuple]:
    """
    Get 2026-specific search queries for tech/semiconductor stocks.

    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        sector: Company sector

    Returns:
        List of (query, category) tuples
    """
    queries = []

    # Always add for tech/semiconductor
    if is_tech_semiconductor_sector(sector):
        queries.extend([
            (f"{ticker} China revenue exposure 2026 BIS restrictions", "china_surcharge"),
            (f"{ticker} export controls China high-performance compute", "china_surcharge"),
            (f"{ticker} 2nm chip production supply chain 2026", "tech_war"),
            (f"{ticker} TSMC Samsung supply dependency", "tech_war"),
            (f"{ticker} agentic AI enterprise revenue 2026", "agentic_ai"),
            (f"{company_name} AI infrastructure spending ROI", "agentic_ai"),
        ])

    # Add semiconductor-specific queries
    if sector and "semiconductor" in sector.lower():
        queries.extend([
            (f"{ticker} BIS export restriction impact EPS", "china_surcharge"),
            (f"{ticker} process node roadmap 2nm", "tech_war"),
            (f"{ticker} CHIPS Act funding domestic production", "tech_sovereignty"),
        ])

    return queries


def filter_searches_for_2026(queries: List[str], sector: Optional[str] = None) -> List[str]:
    """
    Filter and prioritize search queries for 2026 context.

    Args:
        queries: Original search queries
        sector: Company sector for prioritization

    Returns:
        Filtered and prioritized list of queries
    """
    # Add 2026 context to queries for tech sector
    if is_tech_semiconductor_sector(sector):
        enhanced_queries = []
        for query in queries:
            # Add year context if not present
            if "2026" not in query and "2025" not in query:
                enhanced_queries.append(f"{query} 2026")
            else:
                enhanced_queries.append(query)
        return enhanced_queries

    return queries


# =============================================================================
# 2026 Strategic Outlook Report Template
# =============================================================================

def get_2026_report_template() -> str:
    """
    Get the 2026 Strategic Outlook report template for tech/semiconductor stocks.
    """
    return '''# {company_name} ({ticker}) Research Report
## 2026 Strategic Outlook

*Generated: {timestamp}*

---

## 1. Executive Summary

| | |
|---|---|
| **Recommendation** | **{recommendation}** |
| **Confidence** | {confidence} |
| **Valuation** | {valuation} |
{price_target_row}

### Key 2026 Insights
* **Agentic AI Positioning:** {agentic_insight}
* **Geopolitical Risk vs. Opportunity:** {geopolitical_insight}

{executive_summary}

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
| **China Exposure** | {china_exposure_emoji} {china_exposure} |
| **Surcharge Impact** | Estimated {surcharge_impact}% EPS impact from 2026 BIS fee structures |
| **Technology Sovereignty** | {tech_sovereignty} |

### Export Control Analysis
{export_control_analysis}

---

## 4. Investment Thesis (Bull vs. Bear)

### Bull Case
{bull_case}

### Bear Case
{bear_case}

### Key Catalysts
{key_catalysts}

### Key Risks
{key_risks}

---

{financial_snapshot}

{analyst_opinions}

{macro_risk_section}

{news_section}

{sec_filings_section}

{insider_activity}

{institutional_holders}

{sources_section}

---

*Disclaimer: This report is generated by AI for informational purposes only. It is not financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
'''


def get_standard_report_template() -> str:
    """
    Get the standard report template for non-tech stocks.
    """
    return '''# Stock Research Report: {ticker}
## {company_name}
*Generated: {timestamp}*

---

## Executive Summary

{executive_summary}

| | |
|---|---|
| **Recommendation** | **{recommendation}** |
| **Confidence** | {confidence} |
| **Valuation** | {valuation} |
{price_target_row}

---

## Investment Thesis

### Bull Case
{bull_case}

### Bear Case
{bear_case}

### Key Catalysts
{key_catalysts}

### Key Risks
{key_risks}

---

{financial_snapshot}

{analyst_opinions}

{macro_risk_section}

{news_section}

{sec_filings_section}

{insider_activity}

{institutional_holders}

{sources_section}

---

*Disclaimer: This report is generated by AI for informational purposes only. It is not financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
'''


# =============================================================================
# 2026 Analysis Prompt Enhancements
# =============================================================================

def get_2026_analysis_prompt_injection(ticker: str, sector: Optional[str] = None) -> str:
    """
    Get the 2026-specific prompt injection for analysis.

    Args:
        ticker: Stock ticker symbol
        sector: Company sector

    Returns:
        Prompt text to inject into analysis
    """
    if not is_tech_semiconductor_sector(sector):
        return ""

    china_exposure = get_china_exposure_level(ticker)
    surcharge_impact = estimate_surcharge_eps_impact(ticker, china_exposure)
    process_node = get_process_node_status(ticker) or "N/A"
    software_moat = get_software_moat(ticker) or "N/A"

    return f"""

## 2026 STRATEGIC CONTEXT (CRITICAL FOR ANALYSIS)

You are analyzing this stock in January 2026. Apply the following 2026-specific context:

### Core 2026 Market Variables:
1. **25% China Surcharge**: The BIS has implemented a 25% Federal Surcharge on high-performance compute exports to China (effective January 2026). This affects GPUs, AI accelerators, and advanced networking chips.

2. **2nm Tech War**: TSMC and Samsung are in volume production of 2nm process nodes. This is a critical competitive differentiator for AI and mobile chip manufacturers.

3. **Agentic AI Shift**: Enterprise AI is transitioning from chatbots to autonomous AI agents with measurable ROI. Companies are expected to demonstrate concrete AI revenue contribution.

### Company-Specific 2026 Assessment for {ticker}:
- **China Exposure Level**: {china_exposure.upper()}
- **Estimated Surcharge EPS Impact**: {surcharge_impact}%
- **Process Node Status**: {process_node}
- **Software Moat**: {software_moat}

### Required Analysis Points:
In your analysis, you MUST address:
1. How the 25% China surcharge affects this company's revenue and margins
2. The company's position in the 2nm process node race (if applicable)
3. Agentic AI revenue potential and enterprise adoption trajectory
4. Technology sovereignty status (CHIPS Act benefits, domestic production)

### Bull/Bear Case Requirements:
- **Bull case** should specifically address Agentic AI revenue upside and 2nm technology leadership
- **Bear case** should specifically address margin compression from China restrictions and capex normalization pressure
"""


# =============================================================================
# 2026 Deep Research Context
# =============================================================================

DEEP_RESEARCH_2026_CONTEXT = """
## 2026 Market Context

You are conducting research in January 2026. Be aware of the following market conditions:

1. **AI Market Maturation**: The AI industry has shifted from infrastructure buildout to revenue realization. Investors are scrutinizing AI ROI more carefully.

2. **Agentic AI Era**: Enterprise AI has evolved from chatbots to autonomous AI agents capable of complex multi-step tasks. Companies demonstrating measurable AI agent deployments are rewarded.

3. **US-China Tech Decoupling**: Export controls have tightened significantly. The BIS 25% surcharge on high-performance compute exports to China is affecting semiconductor and AI hardware companies.

4. **2nm Process Node**: TSMC and Samsung have entered 2nm volume production, creating new competitive dynamics in chip manufacturing.

5. **Big Tech Capex Scrutiny**: After years of massive AI infrastructure spending ($500B+ annually by hyperscalers), investors are demanding clear paths to AI revenue monetization.

When researching tech, AI, or semiconductor topics, consider these 2026-specific factors in your analysis.
"""
