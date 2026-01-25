"""
Stock data fetcher functions for multiple financial APIs.
Each function returns typed Pydantic models with source URLs preserved.
Includes retry logic with exponential backoff for reliability.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import httpx

from utils.retry_handler import async_retry_with_backoff
from utils.logging_config import get_logger, log_api_call

logger = get_logger(__name__)


# =============================================================================
# HTTP Client with Retry Logic
# =============================================================================

async def _http_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> httpx.Response:
    """
    Make HTTP GET request with exponential backoff retry.
    Retries on network errors, timeouts, and 5xx errors.
    """
    import random

    last_exception = None
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            # Retry on 5xx errors (server-side issues)
            if response.status_code >= 500:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) * (1 + random.uniform(0, 0.25))
                    logger.warning(
                        f"HTTP {response.status_code} from {url}, retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

            # Retry on 429 (rate limit)
            if response.status_code == 429:
                if attempt < max_retries:
                    # Use Retry-After header if present
                    retry_after = response.headers.get('Retry-After', str(base_delay * (2 ** attempt)))
                    delay = float(retry_after) * (1 + random.uniform(0, 0.25))
                    logger.warning(
                        f"Rate limited by {url}, retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

            return response

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) * (1 + random.uniform(0, 0.25))
                logger.warning(
                    f"Network error from {url}: {e}, retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise httpx.RequestError(f"Failed after {max_retries + 1} attempts")


from models.stock_data_models import (
    SourceURL,
    FinnhubData, NewsArticle, AnalystRecommendation, InsiderTransaction,
    AlphaVantageData, CompanyOverview, FinancialStatement, TechnicalIndicators,
    SECFilingsData, SECFiling,
    YFinanceData, PriceData, KeyRatios, InstitutionalHolder,
    TavilyNewsData, TavilyArticle,
    MacroSentimentData, PoliticalNewsItem,
)
from services.market_context_2026 import (
    is_tech_semiconductor_sector,
    get_2026_search_queries,
    filter_searches_for_2026,
    GLOBAL_MARKET_CONTEXT_2026,
)


# =============================================================================
# yfinance Fetcher (No API key required)
# =============================================================================

async def fetch_yfinance_data(ticker: str) -> YFinanceData:
    """
    Fetch stock data from Yahoo Finance via yfinance library.
    yfinance is synchronous, so we wrap it in asyncio.to_thread().
    """
    import yfinance as yf

    def _sync_fetch() -> Dict[str, Any]:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get institutional holders
        try:
            inst_holders = stock.institutional_holders
            if inst_holders is not None:
                holders_list = inst_holders.to_dict('records')
            else:
                holders_list = []
        except Exception:
            holders_list = []

        return {
            'info': info,
            'holders': holders_list,
            # Cache sector/industry to avoid duplicate yfinance calls
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'long_name': info.get('longName') or info.get('shortName'),
            'exchange': info.get('exchange'),
        }

    try:
        data = await asyncio.to_thread(_sync_fetch)
        info = data['info']

        # Build PriceData
        price = None
        if info.get('currentPrice') or info.get('regularMarketPrice'):
            price = PriceData(
                current_price=info.get('currentPrice') or info.get('regularMarketPrice', 0),
                day_high=info.get('dayHigh') or info.get('regularMarketDayHigh', 0),
                day_low=info.get('dayLow') or info.get('regularMarketDayLow', 0),
                open_price=info.get('open') or info.get('regularMarketOpen', 0),
                previous_close=info.get('previousClose') or info.get('regularMarketPreviousClose', 0),
                volume=info.get('volume') or info.get('regularMarketVolume', 0),
                avg_volume=info.get('averageVolume'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh', 0),
                fifty_two_week_low=info.get('fiftyTwoWeekLow', 0),
                market_cap=info.get('marketCap'),
            )

        # Build KeyRatios
        ratios = KeyRatios(
            pe_trailing=info.get('trailingPE'),
            pe_forward=info.get('forwardPE'),
            price_to_book=info.get('priceToBook'),
            price_to_sales=info.get('priceToSalesTrailing12Months'),
            enterprise_value=info.get('enterpriseValue'),
            ev_to_revenue=info.get('enterpriseToRevenue'),
            ev_to_ebitda=info.get('enterpriseToEbitda'),
            debt_to_equity=info.get('debtToEquity'),
            current_ratio=info.get('currentRatio'),
            quick_ratio=info.get('quickRatio'),
            roe=info.get('returnOnEquity'),
            roa=info.get('returnOnAssets'),
            gross_margin=info.get('grossMargins'),
            operating_margin=info.get('operatingMargins'),
            profit_margin=info.get('profitMargins'),
        )

        # Build institutional holders list
        holders = []
        for h in data['holders']:
            try:
                holders.append(InstitutionalHolder(
                    holder=h.get('Holder', 'Unknown'),
                    shares=int(h.get('Shares', 0)),
                    date_reported=h.get('Date Reported'),
                    percent_held=h.get('% Out'),
                    value=h.get('Value'),
                ))
            except Exception:
                continue

        return YFinanceData(
            price=price,
            ratios=ratios,
            institutional_holders=holders[:10],  # Top 10
            recommendation_key=info.get('recommendationKey'),
            recommendation_mean=info.get('recommendationMean'),
            target_mean_price=info.get('targetMeanPrice'),
            target_high_price=info.get('targetHighPrice'),
            target_low_price=info.get('targetLowPrice'),
            # Sector/industry cached to avoid duplicate yfinance calls
            sector=data.get('sector'),
            industry=data.get('industry'),
            long_name=data.get('long_name'),
            exchange=data.get('exchange'),
            source_urls=[SourceURL(
                url=f"https://finance.yahoo.com/quote/{ticker}",
                title=f"Yahoo Finance - {ticker}",
                source_type="api_data"
            )],
            fetch_success=True,
        )

    except Exception as e:
        return YFinanceData(
            fetch_success=False,
            error_message=str(e)
        )


# =============================================================================
# Finnhub Fetcher
# =============================================================================

async def fetch_finnhub_data(ticker: str) -> FinnhubData:
    """
    Fetch news, sentiment, analyst recommendations, and insider trades from Finnhub.
    Requires FINNHUB_API_KEY environment variable.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return FinnhubData(
            fetch_success=False,
            error_message="FINNHUB_API_KEY not set"
        )

    base_url = "https://finnhub.io/api/v1"
    headers = {"X-Finnhub-Token": api_key}

    source_urls = []
    news_articles = []
    recommendations = []
    insider_txns = []
    sentiment = None
    price_target_high = None
    price_target_low = None
    price_target_mean = None

    async with httpx.AsyncClient() as client:
        try:
            # Fetch company news (last 30 days)
            today = datetime.now()
            from_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
            to_date = today.strftime("%Y-%m-%d")

            news_resp = await _http_get_with_retry(
                client,
                f"{base_url}/company-news",
                params={"symbol": ticker, "from": from_date, "to": to_date},
                headers=headers,
                timeout=30.0
            )
            if news_resp.status_code == 200:
                news_data = news_resp.json()
                for article in news_data[:10]:  # Top 10 articles
                    try:
                        news_articles.append(NewsArticle(
                            headline=article.get('headline', ''),
                            summary=article.get('summary', ''),
                            source=article.get('source', ''),
                            url=article.get('url', ''),
                            datetime=datetime.fromtimestamp(article.get('datetime', 0)),
                        ))
                        if article.get('url'):
                            source_urls.append(SourceURL(
                                url=article['url'],
                                title=article.get('headline', 'News Article'),
                                source_type="news"
                            ))
                    except Exception:
                        continue

            # Fetch analyst recommendations
            rec_resp = await _http_get_with_retry(
                client,
                f"{base_url}/stock/recommendation",
                params={"symbol": ticker},
                headers=headers,
                timeout=30.0
            )
            if rec_resp.status_code == 200:
                rec_data = rec_resp.json()
                for rec in rec_data[:4]:  # Last 4 periods
                    try:
                        recommendations.append(AnalystRecommendation(
                            period=rec.get('period', ''),
                            strong_buy=rec.get('strongBuy', 0),
                            buy=rec.get('buy', 0),
                            hold=rec.get('hold', 0),
                            sell=rec.get('sell', 0),
                            strong_sell=rec.get('strongSell', 0),
                        ))
                    except Exception:
                        continue

            # Fetch price target
            target_resp = await _http_get_with_retry(
                client,
                f"{base_url}/stock/price-target",
                params={"symbol": ticker},
                headers=headers,
                timeout=30.0
            )
            if target_resp.status_code == 200:
                target_data = target_resp.json()
                price_target_high = target_data.get('targetHigh')
                price_target_low = target_data.get('targetLow')
                price_target_mean = target_data.get('targetMean')

            # Fetch insider transactions
            insider_resp = await _http_get_with_retry(
                client,
                f"{base_url}/stock/insider-transactions",
                params={"symbol": ticker},
                headers=headers,
                timeout=30.0
            )
            if insider_resp.status_code == 200:
                insider_data = insider_resp.json().get('data', [])
                for txn in insider_data[:10]:  # Last 10 transactions
                    try:
                        insider_txns.append(InsiderTransaction(
                            name=txn.get('name', 'Unknown'),
                            share_change=txn.get('change', 0),
                            transaction_type=txn.get('transactionCode', ''),
                            transaction_date=datetime.strptime(
                                txn.get('transactionDate', '2000-01-01'), "%Y-%m-%d"
                            ),
                            filing_date=datetime.strptime(
                                txn.get('filingDate', '2000-01-01'), "%Y-%m-%d"
                            ),
                        ))
                    except Exception:
                        continue

            # Fetch news sentiment
            sentiment_resp = await _http_get_with_retry(
                client,
                f"{base_url}/news-sentiment",
                params={"symbol": ticker},
                headers=headers,
                timeout=30.0
            )
            if sentiment_resp.status_code == 200:
                sentiment_data = sentiment_resp.json()
                if sentiment_data.get('sentiment'):
                    sentiment = sentiment_data['sentiment'].get('bullishPercent', 0.5) - 0.5

            source_urls.append(SourceURL(
                url=f"https://finnhub.io/",
                title="Finnhub Financial Data",
                source_type="api_data"
            ))

            return FinnhubData(
                news=news_articles,
                overall_sentiment=sentiment,
                analyst_recommendations=recommendations,
                insider_transactions=insider_txns,
                price_target_high=price_target_high,
                price_target_low=price_target_low,
                price_target_mean=price_target_mean,
                source_urls=source_urls,
                fetch_success=True,
            )

        except Exception as e:
            return FinnhubData(
                fetch_success=False,
                error_message=str(e)
            )


# =============================================================================
# SEC EDGAR Fetcher (No API key required)
# =============================================================================

async def fetch_sec_edgar_filings(ticker: str) -> SECFilingsData:
    """
    Fetch SEC filings from EDGAR.
    No API key required, but must set User-Agent header per SEC requirements.
    """
    user_agent = os.getenv("SEC_EDGAR_USER_AGENT", "StockResearchAgent research@example.com")
    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    source_urls = []
    filings = []
    latest_10k_url = None
    latest_10k_date = None
    latest_10q_url = None
    latest_10q_date = None
    recent_8k = []
    form4_filings = []

    async with httpx.AsyncClient() as client:
        try:
            # First, get CIK from ticker
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            tickers_resp = await _http_get_with_retry(
                client, tickers_url, headers=headers, timeout=30.0
            )

            if tickers_resp.status_code != 200:
                return SECFilingsData(
                    cik="",
                    company_name="",
                    fetch_success=False,
                    error_message="Failed to fetch SEC ticker mapping"
                )

            tickers_data = tickers_resp.json()
            cik = None
            company_name = ""

            for entry in tickers_data.values():
                if entry['ticker'].upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    company_name = entry.get('title', '')
                    break

            if not cik:
                return SECFilingsData(
                    cik="",
                    company_name="",
                    fetch_success=False,
                    error_message=f"CIK not found for ticker: {ticker}"
                )

            # Fetch recent filings
            filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            filings_resp = await _http_get_with_retry(
                client, filings_url, headers=headers, timeout=30.0
            )

            if filings_resp.status_code != 200:
                return SECFilingsData(
                    cik=cik,
                    company_name=company_name,
                    fetch_success=False,
                    error_message="Failed to fetch SEC filings"
                )

            filings_data = filings_resp.json()
            recent_filings = filings_data.get('filings', {}).get('recent', {})

            # Process filings
            form_types = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_docs = recent_filings.get('primaryDocument', [])

            for i in range(min(len(form_types), 50)):  # Check last 50 filings
                form_type = form_types[i]
                filing_date = filing_dates[i]
                accession = accession_numbers[i].replace('-', '')
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""

                # Build SEC URL
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"

                filing = SECFiling(
                    form_type=form_type,
                    filing_date=datetime.strptime(filing_date, "%Y-%m-%d"),
                    accession_number=accession_numbers[i],
                    url=filing_url,
                    primary_document=primary_doc,
                )
                filings.append(filing)

                # Track specific filing types
                if form_type == "10-K" and not latest_10k_url:
                    latest_10k_url = filing_url
                    latest_10k_date = filing.filing_date
                    source_urls.append(SourceURL(
                        url=filing_url,
                        title=f"10-K Annual Report ({filing_date})",
                        source_type="sec_filing"
                    ))
                elif form_type == "10-Q" and not latest_10q_url:
                    latest_10q_url = filing_url
                    latest_10q_date = filing.filing_date
                    source_urls.append(SourceURL(
                        url=filing_url,
                        title=f"10-Q Quarterly Report ({filing_date})",
                        source_type="sec_filing"
                    ))
                elif form_type == "8-K" and len(recent_8k) < 5:
                    recent_8k.append(filing)
                    source_urls.append(SourceURL(
                        url=filing_url,
                        title=f"8-K Material Event ({filing_date})",
                        source_type="sec_filing"
                    ))
                elif form_type == "4" and len(form4_filings) < 5:
                    form4_filings.append(filing)
                    source_urls.append(SourceURL(
                        url=filing_url,
                        title=f"Form 4 Insider Transaction ({filing_date})",
                        source_type="sec_filing"
                    ))

            return SECFilingsData(
                cik=cik,
                company_name=company_name,
                filings=filings[:20],  # Keep most recent 20
                latest_10k_url=latest_10k_url,
                latest_10k_date=latest_10k_date,
                latest_10q_url=latest_10q_url,
                latest_10q_date=latest_10q_date,
                recent_8k_filings=recent_8k,
                insider_form4_filings=form4_filings,
                source_urls=source_urls,
                fetch_success=True,
            )

        except Exception as e:
            return SECFilingsData(
                cik="",
                company_name="",
                fetch_success=False,
                error_message=str(e)
            )


# =============================================================================
# Alpha Vantage Fetcher
# =============================================================================

async def fetch_alpha_vantage_data(ticker: str) -> AlphaVantageData:
    """
    Fetch company overview and financial statements from Alpha Vantage.
    Requires ALPHA_VANTAGE_API_KEY environment variable.
    Note: Free tier is limited to 25 requests/day.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return AlphaVantageData(
            fetch_success=False,
            error_message="ALPHA_VANTAGE_API_KEY not set"
        )

    base_url = "https://www.alphavantage.co/query"
    source_urls = [SourceURL(
        url="https://www.alphavantage.co/",
        title="Alpha Vantage Financial Data",
        source_type="api_data"
    )]

    overview = None
    income_statements = []
    balance_sheets = []
    cash_flows = []
    technical = None

    async with httpx.AsyncClient() as client:
        try:
            # Fetch company overview
            overview_resp = await _http_get_with_retry(
                client,
                base_url,
                params={"function": "OVERVIEW", "symbol": ticker, "apikey": api_key},
                timeout=30.0
            )
            if overview_resp.status_code == 200:
                data = overview_resp.json()
                if data and 'Symbol' in data:
                    overview = CompanyOverview(
                        name=data.get('Name', ''),
                        symbol=data.get('Symbol', ''),
                        exchange=data.get('Exchange', ''),
                        sector=data.get('Sector'),
                        industry=data.get('Industry'),
                        market_cap=_safe_int(data.get('MarketCapitalization')),
                        pe_ratio=_safe_float(data.get('PERatio')),
                        peg_ratio=_safe_float(data.get('PEGRatio')),
                        book_value=_safe_float(data.get('BookValue')),
                        dividend_yield=_safe_float(data.get('DividendYield')),
                        eps=_safe_float(data.get('EPS')),
                        revenue_ttm=_safe_int(data.get('RevenueTTM')),
                        profit_margin=_safe_float(data.get('ProfitMargin')),
                        operating_margin=_safe_float(data.get('OperatingMarginTTM')),
                        return_on_equity=_safe_float(data.get('ReturnOnEquityTTM')),
                        return_on_assets=_safe_float(data.get('ReturnOnAssetsTTM')),
                        fifty_two_week_high=_safe_float(data.get('52WeekHigh')),
                        fifty_two_week_low=_safe_float(data.get('52WeekLow')),
                        fifty_day_moving_avg=_safe_float(data.get('50DayMovingAverage')),
                        two_hundred_day_moving_avg=_safe_float(data.get('200DayMovingAverage')),
                        shares_outstanding=_safe_int(data.get('SharesOutstanding')),
                        dividend_date=data.get('DividendDate'),
                        ex_dividend_date=data.get('ExDividendDate'),
                    )

            # Fetch income statement
            income_resp = await _http_get_with_retry(
                client,
                base_url,
                params={"function": "INCOME_STATEMENT", "symbol": ticker, "apikey": api_key},
                timeout=30.0
            )
            if income_resp.status_code == 200:
                data = income_resp.json()
                for report in data.get('annualReports', [])[:3]:
                    income_statements.append(FinancialStatement(
                        fiscal_date=report.get('fiscalDateEnding', ''),
                        period='annual',
                        total_revenue=_safe_int(report.get('totalRevenue')),
                        gross_profit=_safe_int(report.get('grossProfit')),
                        operating_income=_safe_int(report.get('operatingIncome')),
                        net_income=_safe_int(report.get('netIncome')),
                        ebitda=_safe_int(report.get('ebitda')),
                    ))

            # Fetch balance sheet
            balance_resp = await _http_get_with_retry(
                client,
                base_url,
                params={"function": "BALANCE_SHEET", "symbol": ticker, "apikey": api_key},
                timeout=30.0
            )
            if balance_resp.status_code == 200:
                data = balance_resp.json()
                for report in data.get('annualReports', [])[:3]:
                    balance_sheets.append(FinancialStatement(
                        fiscal_date=report.get('fiscalDateEnding', ''),
                        period='annual',
                        total_assets=_safe_int(report.get('totalAssets')),
                        total_liabilities=_safe_int(report.get('totalLiabilities')),
                        total_equity=_safe_int(report.get('totalShareholderEquity')),
                        cash_and_equivalents=_safe_int(report.get('cashAndCashEquivalentsAtCarryingValue')),
                        total_debt=_safe_int(report.get('shortLongTermDebtTotal')),
                    ))

            # Fetch cash flow
            cash_resp = await _http_get_with_retry(
                client,
                base_url,
                params={"function": "CASH_FLOW", "symbol": ticker, "apikey": api_key},
                timeout=30.0
            )
            if cash_resp.status_code == 200:
                data = cash_resp.json()
                for report in data.get('annualReports', [])[:3]:
                    cf = FinancialStatement(
                        fiscal_date=report.get('fiscalDateEnding', ''),
                        period='annual',
                        operating_cash_flow=_safe_int(report.get('operatingCashflow')),
                        capital_expenditure=_safe_int(report.get('capitalExpenditures')),
                    )
                    # Calculate free cash flow
                    if cf.operating_cash_flow and cf.capital_expenditure:
                        cf.free_cash_flow = cf.operating_cash_flow - abs(cf.capital_expenditure)
                    cash_flows.append(cf)

            return AlphaVantageData(
                overview=overview,
                income_statements=income_statements,
                balance_sheets=balance_sheets,
                cash_flows=cash_flows,
                technical=technical,
                source_urls=source_urls,
                fetch_success=True,
            )

        except Exception as e:
            return AlphaVantageData(
                fetch_success=False,
                error_message=str(e)
            )


# =============================================================================
# Tavily News Fetcher
# =============================================================================

async def fetch_tavily_news(ticker: str, company_name: str) -> TavilyNewsData:
    """
    Fetch news articles using Tavily search API.
    Requires TAVILY_API_KEY environment variable.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        return TavilyNewsData(
            fetch_success=False,
            error_message="TAVILY_API_KEY not set"
        )

    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=api_key)
    except ImportError:
        return TavilyNewsData(
            fetch_success=False,
            error_message="tavily-python not installed"
        )

    source_urls = []
    all_articles = []
    search_queries = [
        f"{ticker} stock news",
        f"{company_name} financial news",
        f"{ticker} earnings analysis",
    ]

    try:
        for query in search_queries:
            response = await client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=False,
            )

            for result in response.get("results", []):
                article = TavilyArticle(
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    content=result.get('content', ''),
                    score=result.get('score'),
                )
                all_articles.append(article)

                if result.get('url'):
                    source_urls.append(SourceURL(
                        url=result['url'],
                        title=result.get('title', 'News Article'),
                        source_type="news"
                    ))

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        return TavilyNewsData(
            articles=unique_articles[:15],  # Top 15 unique articles
            search_queries=search_queries,
            source_urls=source_urls[:15],
            fetch_success=True,
        )

    except Exception as e:
        return TavilyNewsData(
            fetch_success=False,
            error_message=str(e)
        )


# =============================================================================
# Macro Sentiment Fetcher
# =============================================================================

# Sector sensitivity to political/trade policy risk
# Updated for 2026 context with AI and semiconductor focus
SECTOR_POLITICAL_SENSITIVITY = {
    # High sensitivity - directly impacted by tariffs, trade policy, regulation
    # 2026 Update: AI Infrastructure and Data Center added due to BIS export controls
    "Technology": "high",
    "Semiconductors": "high",
    "AI Infrastructure": "high",
    "Data Center Equipment": "high",
    "High-End Networking": "high",
    "Consumer Cyclical": "high",
    "Industrials": "high",
    "Basic Materials": "high",
    "Energy": "high",
    "Automotive": "high",
    "Electronic Components": "high",
    "Communication Equipment": "high",
    # Medium sensitivity - some exposure to macro factors
    "Financial Services": "medium",
    "Communication Services": "medium",
    "Consumer Defensive": "medium",
    "Real Estate": "medium",
    "Software - Infrastructure": "medium",
    "Software - Application": "medium",
    # Low sensitivity - more domestic/defensive
    "Healthcare": "low",
    "Utilities": "low",
}


async def fetch_macro_sentiment(ticker: str, sector: Optional[str] = None) -> MacroSentimentData:
    """
    Fetch macro-level sentiment data including VIX and political/trade news.
    - VIX via yfinance (^VIX)
    - Political news via Tavily (tariffs, Fed, trade policy, Trump)
    - Sector sensitivity mapping
    """
    import yfinance as yf

    source_urls = []
    political_news = []
    search_queries = []

    vix_current = None
    vix_previous_close = None
    vix_change_percent = None

    # -------------------------------------------------------------------------
    # Fetch VIX data via yfinance
    # -------------------------------------------------------------------------
    def _fetch_vix() -> Dict[str, Any]:
        vix = yf.Ticker("^VIX")
        info = vix.info
        return {
            'current': info.get('regularMarketPrice'),
            'previous_close': info.get('regularMarketPreviousClose'),
        }

    try:
        vix_data = await asyncio.to_thread(_fetch_vix)
        vix_current = vix_data.get('current')
        vix_previous_close = vix_data.get('previous_close')
        if vix_current and vix_previous_close:
            vix_change_percent = ((vix_current - vix_previous_close) / vix_previous_close) * 100

        source_urls.append(SourceURL(
            url="https://finance.yahoo.com/quote/%5EVIX",
            title="CBOE Volatility Index (VIX)",
            source_type="api_data"
        ))
    except Exception:
        pass  # VIX fetch failed, continue with news

    # -------------------------------------------------------------------------
    # Fetch political/macro news via Tavily
    # -------------------------------------------------------------------------
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key and api_key != "your_tavily_api_key_here":
        try:
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=api_key)

            # Define search queries for political/macro news
            macro_queries = [
                ("tariffs stock market impact", "tariffs"),
                ("Trump trade policy stocks", "trade_policy"),
                ("Federal Reserve interest rates", "fed"),
                ("China trade war stocks", "geopolitical"),
                (f"{ticker} tariff exposure", "tariffs"),
                ("White House economic policy announcement", "government"),
                ("Treasury Secretary statement markets", "government"),
                ("executive order economy stocks", "regulation"),
            ]

            # 2026 FILTER: Add tech/semiconductor specific queries
            if is_tech_semiconductor_sector(sector):
                macro_queries.extend([
                    (f"{ticker} China revenue BIS surcharge 2026", "china_surcharge"),
                    (f"{ticker} export controls high-performance compute", "china_surcharge"),
                    (f"{ticker} 2nm process node competitive position", "tech_war"),
                    (f"{ticker} TSMC Samsung supply dependency", "tech_war"),
                    (f"{ticker} agentic AI enterprise revenue 2026", "agentic_ai"),
                    ("BIS export restrictions semiconductor 2026", "china_surcharge"),
                    ("2nm chip production TSMC Samsung 2026", "tech_war"),
                    ("Big Tech AI capex spending ROI 2026", "agentic_ai"),
                ])

            for query, category in macro_queries:
                search_queries.append(query)
                try:
                    response = await client.search(
                        query=query,
                        search_depth="basic",
                        max_results=3,
                        include_answer=False,
                    )

                    for result in response.get("results", []):
                        news_item = PoliticalNewsItem(
                            title=result.get('title', ''),
                            url=result.get('url', ''),
                            content=result.get('content', '')[:500],  # Truncate content
                            category=category,
                            relevance_score=result.get('score'),
                        )
                        political_news.append(news_item)

                        if result.get('url'):
                            source_urls.append(SourceURL(
                                url=result['url'],
                                title=result.get('title', 'Political News'),
                                source_type="news"
                            ))
                except Exception:
                    continue  # Skip failed queries

            # Deduplicate news by URL
            seen_urls = set()
            unique_news = []
            for item in political_news:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    unique_news.append(item)
            political_news = unique_news[:20]  # Keep top 20

        except ImportError:
            pass  # Tavily not installed

    # -------------------------------------------------------------------------
    # Map sector sensitivity
    # -------------------------------------------------------------------------
    sector_sensitivity = None
    if sector:
        # Try exact match first, then partial match
        sector_sensitivity = SECTOR_POLITICAL_SENSITIVITY.get(sector)
        if not sector_sensitivity:
            sector_lower = sector.lower()
            for key, value in SECTOR_POLITICAL_SENSITIVITY.items():
                if key.lower() in sector_lower or sector_lower in key.lower():
                    sector_sensitivity = value
                    break
        if not sector_sensitivity:
            sector_sensitivity = "medium"  # Default

    return MacroSentimentData(
        vix_current=vix_current,
        vix_previous_close=vix_previous_close,
        vix_change_percent=vix_change_percent,
        political_news=political_news,
        sector=sector,
        sector_political_sensitivity=sector_sensitivity,
        search_queries=search_queries,
        source_urls=source_urls[:20],  # Limit source URLs
        fetch_success=True,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_float(value) -> Optional[float]:
    """Safely convert to float, returning None on failure."""
    if value is None or value == 'None' or value == '-':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value) -> Optional[int]:
    """Safely convert to int, returning None on failure."""
    if value is None or value == 'None' or value == '-':
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


# =============================================================================
# Ticker Validation
# =============================================================================

async def validate_ticker(ticker: str) -> Dict[str, Any]:
    """
    Validate ticker exists and get basic info.
    Returns: {is_valid: bool, company_name: str, exchange: str, error: str}
    """
    import yfinance as yf

    def _sync_validate():
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if we got valid data
        if not info or info.get('regularMarketPrice') is None:
            return None
        return {
            'company_name': info.get('longName') or info.get('shortName') or ticker,
            'exchange': info.get('exchange', 'Unknown'),
            'currency': info.get('currency', 'USD'),
            'quote_type': info.get('quoteType', 'EQUITY'),
        }

    try:
        result = await asyncio.to_thread(_sync_validate)
        if result:
            return {
                'is_valid': True,
                'company_name': result['company_name'],
                'exchange': result['exchange'],
                'currency': result['currency'],
                'error': None
            }
        else:
            return {
                'is_valid': False,
                'company_name': '',
                'exchange': '',
                'error': f"Ticker '{ticker}' not found or has no market data"
            }
    except Exception as e:
        return {
            'is_valid': False,
            'company_name': '',
            'exchange': '',
            'error': str(e)
        }
