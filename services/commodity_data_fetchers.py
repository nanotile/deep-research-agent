"""
Commodity/futures data fetcher functions.
Each function returns typed Pydantic models with source URLs preserved.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import httpx

from models.stock_data_models import SourceURL, TavilyNewsData, TavilyArticle
from models.commodity_data_models import (
    CommodityPriceData,
    HistoricalOHLCV,
    YFinanceCommodityData,
    AlphaVantageCommodityData,
    FREDMacroData,
    CommodityDataBundle,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Commodity Symbol Registry
# =============================================================================

COMMODITY_SYMBOLS: Dict[str, Dict[str, Any]] = {
    "gold": {
        "yfinance_symbol": "GC=F",
        "alpha_vantage_function": None,
        "display_name": "Gold",
        "category": "precious_metals",
        "unit": "USD/oz",
        "contract_size": "100 troy oz",
    },
    "silver": {
        "yfinance_symbol": "SI=F",
        "alpha_vantage_function": None,
        "display_name": "Silver",
        "category": "precious_metals",
        "unit": "USD/oz",
        "contract_size": "5,000 troy oz",
    },
    "platinum": {
        "yfinance_symbol": "PL=F",
        "alpha_vantage_function": None,
        "display_name": "Platinum",
        "category": "precious_metals",
        "unit": "USD/oz",
        "contract_size": "50 troy oz",
    },
    "crude": {
        "yfinance_symbol": "CL=F",
        "alpha_vantage_function": "WTI",
        "display_name": "Crude Oil WTI",
        "category": "energy",
        "unit": "USD/bbl",
        "contract_size": "1,000 barrels",
    },
    "brent": {
        "yfinance_symbol": "BZ=F",
        "alpha_vantage_function": "BRENT",
        "display_name": "Brent Crude",
        "category": "energy",
        "unit": "USD/bbl",
        "contract_size": "1,000 barrels",
    },
    "natgas": {
        "yfinance_symbol": "NG=F",
        "alpha_vantage_function": "NATURAL_GAS",
        "display_name": "Natural Gas",
        "category": "energy",
        "unit": "USD/MMBtu",
        "contract_size": "10,000 MMBtu",
    },
    "copper": {
        "yfinance_symbol": "HG=F",
        "alpha_vantage_function": "COPPER",
        "display_name": "Copper",
        "category": "industrial_metals",
        "unit": "USD/lb",
        "contract_size": "25,000 lbs",
    },
    "corn": {
        "yfinance_symbol": "ZC=F",
        "alpha_vantage_function": "CORN",
        "display_name": "Corn",
        "category": "agriculture",
        "unit": "USc/bu",
        "contract_size": "5,000 bushels",
    },
    "wheat": {
        "yfinance_symbol": "ZW=F",
        "alpha_vantage_function": "WHEAT",
        "display_name": "Wheat",
        "category": "agriculture",
        "unit": "USc/bu",
        "contract_size": "5,000 bushels",
    },
    "soybeans": {
        "yfinance_symbol": "ZS=F",
        "alpha_vantage_function": None,
        "display_name": "Soybeans",
        "category": "agriculture",
        "unit": "USc/bu",
        "contract_size": "5,000 bushels",
    },
}

# Reverse lookup: yfinance symbol -> commodity name
_YF_REVERSE = {v["yfinance_symbol"]: k for k, v in COMMODITY_SYMBOLS.items()}


def validate_commodity_symbol(user_input: str) -> Dict[str, Any]:
    """
    Validate and resolve a commodity symbol from user input.
    Supports friendly names ("gold"), yfinance symbols ("GC=F"), and fuzzy matching.

    Returns:
        Dict with keys: is_valid, symbol, display_name, category, yfinance_symbol,
        alpha_vantage_function, unit, contract_size, error
    """
    cleaned = user_input.strip().lower()

    # Direct name match
    if cleaned in COMMODITY_SYMBOLS:
        info = COMMODITY_SYMBOLS[cleaned]
        return {
            "is_valid": True,
            "symbol": cleaned,
            **info,
            "error": None,
        }

    # yfinance symbol match (e.g., "GC=F")
    upper = user_input.strip().upper()
    if upper in _YF_REVERSE:
        name = _YF_REVERSE[upper]
        info = COMMODITY_SYMBOLS[name]
        return {
            "is_valid": True,
            "symbol": name,
            **info,
            "error": None,
        }

    # Fuzzy match on display_name
    for name, info in COMMODITY_SYMBOLS.items():
        if cleaned in info["display_name"].lower() or info["display_name"].lower() in cleaned:
            return {
                "is_valid": True,
                "symbol": name,
                **info,
                "error": None,
            }

    # Not found
    available = ", ".join(sorted(COMMODITY_SYMBOLS.keys()))
    return {
        "is_valid": False,
        "symbol": None,
        "error": f"Unknown commodity: '{user_input}'. Available: {available}",
    }


# =============================================================================
# yfinance Fetcher
# =============================================================================

async def fetch_yfinance_commodity(symbol: str, yf_symbol: str) -> YFinanceCommodityData:
    """Fetch commodity data from yfinance."""
    try:
        import yfinance as yf

        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")
            return info, hist

        info, hist = await asyncio.to_thread(_fetch)

        if not info or info.get("regularMarketPrice") is None:
            return YFinanceCommodityData(
                fetch_success=False,
                error_message=f"No data returned for {yf_symbol}",
            )

        current_price = info.get("regularMarketPrice", info.get("previousClose", 0))
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        # Calculate percent changes from history
        pct_1d = None
        pct_1w = None
        pct_1m = None
        pct_ytd = None

        if not hist.empty and current_price:
            closes = hist["Close"]
            if len(closes) >= 2:
                pct_1d = ((current_price - closes.iloc[-2]) / closes.iloc[-2]) * 100
            if len(closes) >= 5:
                pct_1w = ((current_price - closes.iloc[-5]) / closes.iloc[-5]) * 100
            if len(closes) >= 22:
                pct_1m = ((current_price - closes.iloc[-22]) / closes.iloc[-22]) * 100
            # YTD: find first trading day of year
            try:
                current_year = datetime.now().year
                ytd_data = closes[closes.index.year == current_year]
                if len(ytd_data) > 0:
                    pct_ytd = ((current_price - ytd_data.iloc[0]) / ytd_data.iloc[0]) * 100
            except Exception:
                pass

        price_data = CommodityPriceData(
            spot_price=current_price,
            day_high=info.get("dayHigh"),
            day_low=info.get("dayLow"),
            open_price=info.get("open"),
            previous_close=prev_close,
            volume=info.get("volume"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            pct_change_1d=round(pct_1d, 2) if pct_1d is not None else None,
            pct_change_1w=round(pct_1w, 2) if pct_1w is not None else None,
            pct_change_1m=round(pct_1m, 2) if pct_1m is not None else None,
            pct_change_ytd=round(pct_ytd, 2) if pct_ytd is not None else None,
        )

        # Build historical OHLCV (last 30 days for report)
        historical = []
        if not hist.empty:
            recent = hist.tail(30)
            for date, row in recent.iterrows():
                historical.append(HistoricalOHLCV(
                    date=date.strftime("%Y-%m-%d"),
                    open=round(row["Open"], 2),
                    high=round(row["High"], 2),
                    low=round(row["Low"], 2),
                    close=round(row["Close"], 2),
                    volume=int(row["Volume"]) if row.get("Volume") else None,
                ))

        return YFinanceCommodityData(
            price=price_data,
            historical=historical,
            currency=info.get("currency", "USD"),
            exchange=info.get("exchange"),
            source_urls=[
                SourceURL(
                    url=f"https://finance.yahoo.com/quote/{yf_symbol}",
                    title=f"Yahoo Finance - {yf_symbol}",
                    source_type="api_data",
                )
            ],
        )

    except ImportError:
        return YFinanceCommodityData(
            fetch_success=False,
            error_message="yfinance not installed",
        )
    except Exception as e:
        logger.error(f"yfinance commodity error for {yf_symbol}: {e}")
        return YFinanceCommodityData(
            fetch_success=False,
            error_message=str(e),
        )


# =============================================================================
# Alpha Vantage Commodity Fetcher
# =============================================================================

async def fetch_alpha_vantage_commodity(name: str, av_function: Optional[str]) -> AlphaVantageCommodityData:
    """Fetch commodity spot price from Alpha Vantage."""
    if not av_function:
        return AlphaVantageCommodityData(
            fetch_success=False,
            error_message=f"No Alpha Vantage mapping for {name}",
        )

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key == "your_alpha_vantage_api_key_here":
        return AlphaVantageCommodityData(
            fetch_success=False,
            error_message="ALPHA_VANTAGE_API_KEY not set",
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": av_function,
                    "interval": "monthly",
                    "apikey": api_key,
                },
                timeout=30.0,
            )
            data = response.json()

        # Alpha Vantage commodity endpoints return data in varying formats
        # Try to extract the most recent price
        spot_price = None
        last_refreshed = None

        # Try "data" key (commodity endpoints)
        if "data" in data and isinstance(data["data"], list) and data["data"]:
            latest = data["data"][0]
            val = latest.get("value")
            if val and val != ".":
                spot_price = float(val)
            last_refreshed = latest.get("date")

        if spot_price is None:
            return AlphaVantageCommodityData(
                fetch_success=False,
                error_message=f"No price data from Alpha Vantage for {av_function}",
            )

        return AlphaVantageCommodityData(
            spot_price=spot_price,
            unit=data.get("unit"),
            last_refreshed=last_refreshed,
            source_urls=[
                SourceURL(
                    url=f"https://www.alphavantage.co/query?function={av_function}",
                    title=f"Alpha Vantage - {av_function}",
                    source_type="api_data",
                )
            ],
        )

    except Exception as e:
        logger.error(f"Alpha Vantage commodity error for {av_function}: {e}")
        return AlphaVantageCommodityData(
            fetch_success=False,
            error_message=str(e),
        )


# =============================================================================
# FRED Macro Data Fetcher
# =============================================================================

async def fetch_fred_macro_data() -> FREDMacroData:
    """Fetch macro indicators from FRED API (DXY, yields, Fed funds, inflation)."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_fred_api_key_here":
        return FREDMacroData(
            fetch_success=False,
            error_message="FRED_API_KEY not set",
        )

    try:
        from fredapi import Fred

        def _fetch():
            fred = Fred(api_key=api_key)
            results = {}
            series_map = {
                "dxy_index": "DTWEXBGS",        # Trade-weighted USD index (broad)
                "treasury_10y": "DGS10",          # 10-Year Treasury yield
                "treasury_2y": "DGS2",            # 2-Year Treasury yield
                "fed_funds_rate": "FEDFUNDS",     # Effective Federal Funds Rate
                "inflation_expectations_5y": "T5YIE",  # 5-Year Breakeven Inflation
                "cpi_yoy": "CPIAUCSL",            # CPI (we'll compute YoY)
            }

            for field, series_id in series_map.items():
                try:
                    data = fred.get_series(series_id, observation_start="2025-01-01")
                    if data is not None and len(data) > 0:
                        val = data.dropna().iloc[-1]
                        if field == "cpi_yoy":
                            # CPI is an index; compute YoY % change
                            full = fred.get_series(series_id, observation_start="2024-01-01")
                            if full is not None and len(full) > 12:
                                clean = full.dropna()
                                results[field] = round(
                                    ((clean.iloc[-1] - clean.iloc[-13]) / clean.iloc[-13]) * 100, 2
                                )
                        else:
                            results[field] = round(float(val), 2)
                except Exception as ex:
                    logger.warning(f"FRED series {series_id} failed: {ex}")
                    continue

            return results

        results = await asyncio.to_thread(_fetch)

        return FREDMacroData(
            dxy_index=results.get("dxy_index"),
            treasury_10y=results.get("treasury_10y"),
            treasury_2y=results.get("treasury_2y"),
            fed_funds_rate=results.get("fed_funds_rate"),
            inflation_expectations_5y=results.get("inflation_expectations_5y"),
            cpi_yoy=results.get("cpi_yoy"),
            source_urls=[
                SourceURL(
                    url="https://fred.stlouisfed.org/",
                    title="Federal Reserve Economic Data (FRED)",
                    source_type="api_data",
                )
            ],
        )

    except ImportError:
        return FREDMacroData(
            fetch_success=False,
            error_message="fredapi not installed (pip install fredapi)",
        )
    except Exception as e:
        logger.error(f"FRED macro data error: {e}")
        return FREDMacroData(
            fetch_success=False,
            error_message=str(e),
        )


# =============================================================================
# Tavily News Fetcher (commodity-specific)
# =============================================================================

async def fetch_tavily_commodity_news(
    symbol: str, display_name: str, category: str
) -> TavilyNewsData:
    """Fetch commodity-specific news via Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        return TavilyNewsData(
            fetch_success=False,
            error_message="TAVILY_API_KEY not set",
        )

    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=api_key)
    except ImportError:
        return TavilyNewsData(
            fetch_success=False,
            error_message="tavily-python not installed",
        )

    # Category-specific search queries
    category_queries = {
        "precious_metals": [
            f"{display_name} price forecast 2026",
            f"{display_name} central bank demand supply",
        ],
        "energy": [
            f"{display_name} price outlook OPEC",
            f"{display_name} supply demand inventory",
        ],
        "industrial_metals": [
            f"{display_name} industrial demand China",
            f"{display_name} supply mining production",
        ],
        "agriculture": [
            f"{display_name} crop production harvest",
            f"{display_name} price weather impact",
        ],
    }

    queries = [f"{display_name} commodity market news"]
    queries.extend(category_queries.get(category, []))

    source_urls = []
    all_articles = []

    try:
        for query in queries:
            response = await client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=False,
            )

            for result in response.get("results", []):
                article = TavilyArticle(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", "")[:500],
                    score=result.get("score"),
                    published_date=result.get("published_date"),
                )
                all_articles.append(article)
                source_urls.append(
                    SourceURL(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        source_type="news",
                    )
                )

        return TavilyNewsData(
            articles=all_articles,
            search_queries=queries,
            source_urls=source_urls,
        )

    except Exception as e:
        logger.error(f"Tavily commodity news error: {e}")
        return TavilyNewsData(
            fetch_success=False,
            error_message=str(e),
        )


# =============================================================================
# Parallel Fetch All
# =============================================================================

async def fetch_all_commodity_data(
    symbol: str,
    display_name: str,
    category: str,
    yf_symbol: str,
    av_function: Optional[str],
) -> CommodityDataBundle:
    """
    Fetch data from all commodity sources in parallel.
    Handles failures gracefully - partial data is acceptable.
    """
    print(f"\n  Fetching data for {display_name} ({yf_symbol})...")

    tasks = {
        "yfinance": fetch_yfinance_commodity(symbol, yf_symbol),
        "alpha_vantage": fetch_alpha_vantage_commodity(symbol, av_function),
        "fred_macro": fetch_fred_macro_data(),
        "tavily_news": fetch_tavily_commodity_news(symbol, display_name, category),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    bundle = CommodityDataBundle(
        symbol=symbol,
        display_name=display_name,
        category=category,
    )
    errors = []

    for (source_name, _), result in zip(tasks.items(), results):
        if isinstance(result, Exception):
            errors.append(f"{source_name}: {str(result)}")
            print(f"    {source_name}: Error - {str(result)[:50]}")
        elif hasattr(result, "fetch_success") and not result.fetch_success:
            errors.append(f"{source_name}: {result.error_message}")
            print(f"    {source_name}: {result.error_message}")
        else:
            setattr(bundle, source_name, result)
            print(f"    {source_name}: Success")

    bundle.fetch_errors = errors
    print(f"\n  Data collected from {len(bundle.get_successful_sources())} sources")

    return bundle
