"""
Pydantic models for commodity/futures research data.
All models track source URLs for citation in final report.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

from models.stock_data_models import SourceURL, DataWithSources, TavilyArticle, TavilyNewsData


# =============================================================================
# Enums
# =============================================================================

class OutlookType(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class TrendStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# =============================================================================
# Commodity Price Data
# =============================================================================

class CommodityPriceData(BaseModel):
    """Current commodity price data from yfinance"""
    spot_price: float
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    open_price: Optional[float] = None
    previous_close: Optional[float] = None
    volume: Optional[int] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    pct_change_1d: Optional[float] = None
    pct_change_1w: Optional[float] = None
    pct_change_1m: Optional[float] = None
    pct_change_ytd: Optional[float] = None


class HistoricalOHLCV(BaseModel):
    """Single OHLCV data point"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None


# =============================================================================
# Data Source Models
# =============================================================================

class YFinanceCommodityData(DataWithSources):
    """Commodity data from yfinance"""
    price: Optional[CommodityPriceData] = None
    historical: List[HistoricalOHLCV] = Field(default_factory=list)
    currency: Optional[str] = None
    exchange: Optional[str] = None


class AlphaVantageCommodityData(DataWithSources):
    """Commodity spot price from Alpha Vantage"""
    spot_price: Optional[float] = None
    unit: Optional[str] = None
    last_refreshed: Optional[str] = None


class FREDMacroData(DataWithSources):
    """Macro data from FRED API"""
    dxy_index: Optional[float] = None
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    fed_funds_rate: Optional[float] = None
    inflation_expectations_5y: Optional[float] = None
    cpi_yoy: Optional[float] = None


# =============================================================================
# Aggregated Data Bundle
# =============================================================================

class CommodityDataBundle(BaseModel):
    """Complete data bundle from all sources for a commodity"""
    symbol: str
    display_name: str
    category: str
    fetch_timestamp: datetime = Field(default_factory=datetime.now)

    # Data from each source
    yfinance: Optional[YFinanceCommodityData] = None
    alpha_vantage: Optional[AlphaVantageCommodityData] = None
    fred_macro: Optional[FREDMacroData] = None
    tavily_news: Optional[TavilyNewsData] = None

    # Track errors
    fetch_errors: List[str] = Field(default_factory=list)

    def get_all_source_urls(self) -> List[SourceURL]:
        """Aggregate all source URLs from all data sources"""
        urls = []
        for source in [self.yfinance, self.alpha_vantage, self.fred_macro, self.tavily_news]:
            if source:
                urls.extend(source.source_urls)
        return urls

    def get_successful_sources(self) -> List[str]:
        """Return list of successfully fetched data sources"""
        sources = []
        if self.yfinance and self.yfinance.fetch_success:
            sources.append("Yahoo Finance")
        if self.alpha_vantage and self.alpha_vantage.fetch_success:
            sources.append("Alpha Vantage")
        if self.fred_macro and self.fred_macro.fetch_success:
            sources.append("FRED")
        if self.tavily_news and self.tavily_news.fetch_success:
            sources.append("Tavily News")
        return sources


# =============================================================================
# Analysis Output Models
# =============================================================================

class CommodityThesis(BaseModel):
    """Structured commodity thesis"""
    bull_case: List[str] = Field(description="3-5 bullish arguments")
    bear_case: List[str] = Field(description="3-5 bearish arguments")
    key_catalysts: List[str] = Field(description="Potential positive catalysts")
    key_risks: List[str] = Field(description="Primary risk factors")


class CommodityAnalysis(BaseModel):
    """Structured output from Claude commodity analysis"""
    symbol: str
    display_name: str
    outlook: OutlookType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level 0-1")
    trend_strength: TrendStrength
    thesis: CommodityThesis
    price_target_high: Optional[float] = None
    price_target_low: Optional[float] = None
    supply_demand_balance: Optional[str] = None
    summary: str = Field(description="2-3 sentence executive summary")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Progress Tracking
# =============================================================================

class CommodityProgressUpdate(BaseModel):
    """Progress update for UI display"""
    stage: str  # "validating", "fetching", "analyzing", "writing", "complete"
    stage_display: str
    current_step: int
    total_steps: int
    elapsed_time: float
    message: str

    source_status: Optional[Dict[str, str]] = None

    # Final outputs
    report: Optional[str] = None
    analysis: Optional[CommodityAnalysis] = None

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
