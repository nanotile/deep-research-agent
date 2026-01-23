"""
Pydantic models for stock research data.
All models track source URLs for citation in final report.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# Base Classes
# =============================================================================

class SourceURL(BaseModel):
    """Individual source with URL and metadata"""
    url: str
    title: str
    source_type: str  # "news", "sec_filing", "api_data", "analysis"
    accessed_at: datetime = Field(default_factory=datetime.now)


class DataWithSources(BaseModel):
    """Base class for all data models that track sources"""
    source_urls: List[SourceURL] = Field(default_factory=list)
    fetch_timestamp: datetime = Field(default_factory=datetime.now)
    fetch_success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# Finnhub Models
# =============================================================================

class NewsArticle(BaseModel):
    """News article from Finnhub"""
    headline: str
    summary: str
    source: str
    url: str
    datetime: datetime
    sentiment_score: Optional[float] = None  # -1 to 1


class AnalystRecommendation(BaseModel):
    """Analyst recommendation data"""
    period: str  # "2024-01"
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0


class InsiderTransaction(BaseModel):
    """Insider trading transaction"""
    name: str
    share_change: int
    transaction_type: str  # "P" (purchase), "S" (sale)
    transaction_date: datetime
    filing_date: datetime
    sec_url: Optional[str] = None


class FinnhubData(DataWithSources):
    """All data from Finnhub API"""
    news: List[NewsArticle] = Field(default_factory=list)
    overall_sentiment: Optional[float] = None  # Aggregated sentiment
    analyst_recommendations: List[AnalystRecommendation] = Field(default_factory=list)
    insider_transactions: List[InsiderTransaction] = Field(default_factory=list)
    price_target_high: Optional[float] = None
    price_target_low: Optional[float] = None
    price_target_mean: Optional[float] = None


# =============================================================================
# Alpha Vantage Models
# =============================================================================

class CompanyOverview(BaseModel):
    """Company overview from Alpha Vantage"""
    name: str
    symbol: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    book_value: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    revenue_ttm: Optional[int] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    fifty_day_moving_avg: Optional[float] = None
    two_hundred_day_moving_avg: Optional[float] = None
    shares_outstanding: Optional[int] = None
    dividend_date: Optional[str] = None
    ex_dividend_date: Optional[str] = None


class FinancialStatement(BaseModel):
    """Financial statement line items"""
    fiscal_date: str
    period: str  # "annual" or "quarterly"
    total_revenue: Optional[int] = None
    gross_profit: Optional[int] = None
    operating_income: Optional[int] = None
    net_income: Optional[int] = None
    ebitda: Optional[int] = None
    # Balance sheet
    total_assets: Optional[int] = None
    total_liabilities: Optional[int] = None
    total_equity: Optional[int] = None
    cash_and_equivalents: Optional[int] = None
    total_debt: Optional[int] = None
    # Cash flow
    operating_cash_flow: Optional[int] = None
    capital_expenditure: Optional[int] = None
    free_cash_flow: Optional[int] = None


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None


class AlphaVantageData(DataWithSources):
    """All data from Alpha Vantage API"""
    overview: Optional[CompanyOverview] = None
    income_statements: List[FinancialStatement] = Field(default_factory=list)
    balance_sheets: List[FinancialStatement] = Field(default_factory=list)
    cash_flows: List[FinancialStatement] = Field(default_factory=list)
    technical: Optional[TechnicalIndicators] = None


# =============================================================================
# SEC EDGAR Models
# =============================================================================

class SECFiling(BaseModel):
    """SEC filing metadata"""
    form_type: str  # "10-K", "10-Q", "8-K", "4"
    filing_date: datetime
    accession_number: str
    url: str  # Direct SEC URL
    description: Optional[str] = None
    primary_document: Optional[str] = None


class SECFilingsData(DataWithSources):
    """All SEC filings data"""
    cik: str  # Central Index Key
    company_name: str
    filings: List[SECFiling] = Field(default_factory=list)
    latest_10k_url: Optional[str] = None
    latest_10k_date: Optional[datetime] = None
    latest_10q_url: Optional[str] = None
    latest_10q_date: Optional[datetime] = None
    recent_8k_filings: List[SECFiling] = Field(default_factory=list)
    insider_form4_filings: List[SECFiling] = Field(default_factory=list)


# =============================================================================
# Yahoo Finance (yfinance) Models
# =============================================================================

class PriceData(BaseModel):
    """Current price data"""
    current_price: float
    day_high: float
    day_low: float
    open_price: float
    previous_close: float
    volume: int
    avg_volume: Optional[int] = None
    fifty_two_week_high: float
    fifty_two_week_low: float
    market_cap: Optional[int] = None


class KeyRatios(BaseModel):
    """Key financial ratios"""
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[int] = None
    ev_to_revenue: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None


class InstitutionalHolder(BaseModel):
    """Institutional holder data"""
    holder: str
    shares: int
    date_reported: Optional[datetime] = None
    percent_held: Optional[float] = None
    value: Optional[int] = None


class YFinanceData(DataWithSources):
    """All data from yfinance"""
    price: Optional[PriceData] = None
    ratios: Optional[KeyRatios] = None
    institutional_holders: List[InstitutionalHolder] = Field(default_factory=list)
    major_holders_breakdown: Optional[Dict[str, float]] = None
    recommendation_key: Optional[str] = None  # "buy", "hold", "sell"
    recommendation_mean: Optional[float] = None  # 1.0 (strong buy) to 5.0 (strong sell)
    target_mean_price: Optional[float] = None
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None


# =============================================================================
# Tavily News Models
# =============================================================================

class TavilyArticle(BaseModel):
    """News article from Tavily search"""
    title: str
    url: str
    content: str
    score: Optional[float] = None  # Relevance score
    published_date: Optional[str] = None


class TavilyNewsData(DataWithSources):
    """News data from Tavily search"""
    articles: List[TavilyArticle] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


# =============================================================================
# Macro Sentiment Data
# =============================================================================

class PoliticalNewsItem(BaseModel):
    """Political/macro news item from Tavily search"""
    title: str
    url: str
    content: str
    category: str  # "tariffs", "fed", "trade_policy", "regulation", "geopolitical"
    relevance_score: Optional[float] = None


class MacroSentimentData(DataWithSources):
    """Macro-level sentiment data from VIX and political news"""
    # VIX data
    vix_current: Optional[float] = None
    vix_previous_close: Optional[float] = None
    vix_change_percent: Optional[float] = None

    # Political/macro news
    political_news: List[PoliticalNewsItem] = Field(default_factory=list)

    # Sector sensitivity mapping (how sensitive this sector is to political risk)
    sector: Optional[str] = None
    sector_political_sensitivity: Optional[str] = None  # "low", "medium", "high"

    # Search queries used
    search_queries: List[str] = Field(default_factory=list)


# =============================================================================
# Aggregated Data Bundle
# =============================================================================

class StockDataBundle(BaseModel):
    """Complete data bundle from all sources"""
    ticker: str
    company_name: str
    fetch_timestamp: datetime = Field(default_factory=datetime.now)

    # Data from each source (None if fetch failed)
    finnhub: Optional[FinnhubData] = None
    alpha_vantage: Optional[AlphaVantageData] = None
    sec_filings: Optional[SECFilingsData] = None
    yfinance: Optional[YFinanceData] = None
    tavily_news: Optional[TavilyNewsData] = None
    macro_sentiment: Optional[MacroSentimentData] = None

    # Track which sources succeeded/failed
    fetch_errors: List[str] = Field(default_factory=list)

    def get_all_source_urls(self) -> List[SourceURL]:
        """Aggregate all source URLs from all data sources"""
        urls = []
        for source in [self.finnhub, self.alpha_vantage,
                       self.sec_filings, self.yfinance, self.tavily_news,
                       self.macro_sentiment]:
            if source:
                urls.extend(source.source_urls)
        return urls

    def get_successful_sources(self) -> List[str]:
        """Return list of successfully fetched data sources"""
        sources = []
        if self.finnhub and self.finnhub.fetch_success:
            sources.append("Finnhub")
        if self.alpha_vantage and self.alpha_vantage.fetch_success:
            sources.append("Alpha Vantage")
        if self.sec_filings and self.sec_filings.fetch_success:
            sources.append("SEC EDGAR")
        if self.yfinance and self.yfinance.fetch_success:
            sources.append("Yahoo Finance")
        if self.tavily_news and self.tavily_news.fetch_success:
            sources.append("Tavily")
        if self.macro_sentiment and self.macro_sentiment.fetch_success:
            sources.append("Macro Sentiment")
        return sources


# =============================================================================
# Analysis Output Models
# =============================================================================

class RecommendationType(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class ValuationAssessment(str, Enum):
    UNDERVALUED = "undervalued"
    FAIRLY_VALUED = "fairly_valued"
    OVERVALUED = "overvalued"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ELEVATED = "elevated"


class MacroRiskAssessment(BaseModel):
    """Macro-level risk assessment for market and political factors"""
    political_risk_level: RiskLevel = Field(description="Overall political risk level")
    vix_level: Optional[float] = Field(default=None, description="Current VIX volatility index level")
    key_political_factors: List[str] = Field(default_factory=list, description="Key political factors affecting the stock")
    sector_exposure: Dict[str, float] = Field(default_factory=dict, description="Sector exposure breakdown as percentages")


class GeopoliticalImpact2026(BaseModel):
    """2026-specific geopolitical and technology assessment for tech/semiconductor stocks"""
    china_exposure: str = Field(description="China exposure level: 'high', 'medium', or 'low'")
    surcharge_eps_impact_percent: Optional[float] = Field(
        default=None,
        description="Estimated EPS impact percentage from 2026 BIS surcharge"
    )
    technology_sovereignty_status: Optional[str] = Field(
        default=None,
        description="CHIPS Act benefits, domestic production status"
    )
    process_node_status: Optional[str] = Field(
        default=None,
        description="Current process node capability (2nm, 3nm, etc.)"
    )
    software_moat: Optional[str] = Field(
        default=None,
        description="Software ecosystem moat (CUDA, ROCm, cloud AI, etc.)"
    )
    agentic_ai_positioning: Optional[str] = Field(
        default=None,
        description="Company's position in the Agentic AI shift"
    )
    supply_chain_dependency: Optional[str] = Field(
        default=None,
        description="Key supply chain dependencies (TSMC, Samsung, Intel)"
    )
    export_control_risk: Optional[str] = Field(
        default=None,
        description="Assessment of export control and BIS restriction risk"
    )


class InvestmentThesis(BaseModel):
    """Structured investment thesis"""
    bull_case: List[str] = Field(description="3-5 bullish arguments")
    bear_case: List[str] = Field(description="3-5 bearish arguments")
    key_risks: List[str] = Field(description="Primary risk factors")
    key_catalysts: List[str] = Field(description="Potential positive catalysts")


class StockAnalysis(BaseModel):
    """Structured output from Claude analysis"""
    ticker: str
    company_name: str
    recommendation: RecommendationType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level 0-1")
    valuation_assessment: ValuationAssessment
    thesis: InvestmentThesis
    macro_risk: Optional[MacroRiskAssessment] = Field(default=None, description="Macro-level risk assessment")
    geopolitical_2026: Optional[GeopoliticalImpact2026] = Field(
        default=None,
        description="2026-specific geopolitical and technology assessment for tech/semiconductor stocks"
    )
    summary: str = Field(description="2-3 sentence executive summary")
    price_target: Optional[float] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Progress Tracking
# =============================================================================

class StockProgressUpdate(BaseModel):
    """Progress update for UI display"""
    stage: str  # "validating", "fetching", "analyzing", "writing", "complete"
    stage_display: str  # Human-readable stage name
    current_step: int
    total_steps: int
    elapsed_time: float
    message: str

    # Source-specific status (for fetching stage)
    source_status: Optional[Dict[str, str]] = None  # {"finnhub": "complete", "sec": "in_progress"}

    # Final outputs (only set when complete)
    report: Optional[str] = None
    analysis: Optional[StockAnalysis] = None
