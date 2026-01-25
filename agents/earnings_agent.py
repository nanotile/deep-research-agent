"""
Earnings Calendar Agent - Track upcoming earnings and analyze historical performance.

Features:
- Upcoming earnings calendar for watchlist/sector
- Historical beat/miss rates
- Pre-earnings sentiment analysis
- Earnings surprise history
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, AsyncIterator, List, Dict, Any
from anthropic import Anthropic
import yfinance as yf

# Import shared utilities
from services.stock_data_fetchers import fetch_finnhub_data, fetch_tavily_news
from utils.logging_config import get_logger
from utils.validators import sanitize_ticker
from utils.retry_handler import retry_with_backoff

logger = get_logger(__name__)

# Configuration
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")


@dataclass
class EarningsData:
    """Container for earnings-related data."""
    ticker: str
    company_name: str = ""
    next_earnings_date: Optional[datetime] = None
    days_until_earnings: Optional[int] = None

    # Historical earnings data
    earnings_history: List[Dict[str, Any]] = field(default_factory=list)
    beat_count: int = 0
    miss_count: int = 0
    meet_count: int = 0
    beat_rate: float = 0.0
    avg_surprise_percent: float = 0.0

    # Recent performance
    last_quarter_eps_actual: Optional[float] = None
    last_quarter_eps_estimate: Optional[float] = None
    last_quarter_surprise: Optional[float] = None

    # Analyst estimates
    current_quarter_estimate: Optional[float] = None
    next_quarter_estimate: Optional[float] = None
    current_year_estimate: Optional[float] = None

    # Price reaction history
    avg_post_earnings_move: Optional[float] = None

    # Sentiment
    news_sentiment: Optional[str] = None
    recent_news: List[Dict[str, str]] = field(default_factory=list)

    # Error tracking
    error: Optional[str] = None


@dataclass
class EarningsProgressUpdate:
    """Progress update for earnings analysis."""
    stage: str
    stage_display: str
    message: str
    tickers_processed: int = 0
    total_tickers: int = 0
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


def get_earnings_data(ticker: str) -> EarningsData:
    """Fetch comprehensive earnings data for a ticker."""
    data = EarningsData(ticker=ticker.upper())

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Company name
        data.company_name = info.get('longName', info.get('shortName', ticker))

        # Next earnings date
        try:
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    earnings_dates = calendar.loc['Earnings Date']
                    if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
                        next_date = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
                    else:
                        next_date = earnings_dates

                    if next_date is not None:
                        if hasattr(next_date, 'to_pydatetime'):
                            data.next_earnings_date = next_date.to_pydatetime()
                        elif isinstance(next_date, datetime):
                            data.next_earnings_date = next_date

                        if data.next_earnings_date:
                            data.days_until_earnings = (data.next_earnings_date.date() - datetime.now().date()).days
        except Exception as e:
            logger.warning(f"Could not get earnings calendar for {ticker}: {e}")

        # Earnings history
        try:
            earnings = stock.earnings_history
            if earnings is not None and not earnings.empty:
                for idx, row in earnings.iterrows():
                    record = {
                        'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        'eps_actual': row.get('epsActual'),
                        'eps_estimate': row.get('epsEstimate'),
                        'surprise': row.get('surprise'),
                        'surprise_percent': row.get('surprisePercent')
                    }
                    data.earnings_history.append(record)

                    # Count beats/misses
                    surprise = row.get('surprise', 0) or 0
                    if surprise > 0:
                        data.beat_count += 1
                    elif surprise < 0:
                        data.miss_count += 1
                    else:
                        data.meet_count += 1

                total = data.beat_count + data.miss_count + data.meet_count
                if total > 0:
                    data.beat_rate = data.beat_count / total

                # Average surprise
                surprises = [r.get('surprise_percent', 0) for r in data.earnings_history if r.get('surprise_percent')]
                if surprises:
                    data.avg_surprise_percent = sum(surprises) / len(surprises)

                # Last quarter data
                if data.earnings_history:
                    last = data.earnings_history[0]
                    data.last_quarter_eps_actual = last.get('eps_actual')
                    data.last_quarter_eps_estimate = last.get('eps_estimate')
                    data.last_quarter_surprise = last.get('surprise_percent')
        except Exception as e:
            logger.warning(f"Could not get earnings history for {ticker}: {e}")

        # Analyst estimates
        try:
            data.current_quarter_estimate = info.get('epsCurrentQuarter')
            data.next_quarter_estimate = info.get('epsNextQuarter')
            data.current_year_estimate = info.get('epsForward')
        except Exception as e:
            logger.warning(f"Could not get analyst estimates for {ticker}: {e}")

        # Calculate average post-earnings price move
        try:
            hist = stock.history(period="2y")
            if not hist.empty and data.earnings_history:
                moves = []
                for record in data.earnings_history[:8]:  # Last 8 quarters
                    try:
                        date_str = record['date']
                        # Find price move around earnings date
                        # This is simplified - in production would need more sophisticated logic
                        earnings_date = datetime.strptime(date_str, '%Y-%m-%d')
                        # Look for 1-day move after earnings
                        mask = hist.index.date >= earnings_date.date()
                        if mask.any():
                            post_earnings = hist[mask].head(2)
                            if len(post_earnings) >= 2:
                                move = ((post_earnings['Close'].iloc[1] - post_earnings['Close'].iloc[0])
                                       / post_earnings['Close'].iloc[0] * 100)
                                moves.append(abs(move))
                    except:
                        pass

                if moves:
                    data.avg_post_earnings_move = sum(moves) / len(moves)
        except Exception as e:
            logger.warning(f"Could not calculate post-earnings moves for {ticker}: {e}")

    except Exception as e:
        logger.error(f"Error fetching earnings data for {ticker}: {e}")
        data.error = str(e)

    return data


async def fetch_earnings_sentiment(ticker: str) -> tuple[str, List[Dict[str, str]]]:
    """Fetch pre-earnings sentiment from news."""
    try:
        news = await fetch_tavily_news(ticker, days_back=7)

        if not news:
            return "neutral", []

        # Simple sentiment scoring based on keywords
        positive_words = ['beat', 'exceed', 'strong', 'growth', 'upgrade', 'bullish', 'outperform']
        negative_words = ['miss', 'weak', 'decline', 'downgrade', 'bearish', 'underperform', 'warning']

        positive_count = 0
        negative_count = 0

        recent_news = []
        for article in news[:5]:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            text = title + ' ' + content

            for word in positive_words:
                if word in text:
                    positive_count += 1
            for word in negative_words:
                if word in text:
                    negative_count += 1

            recent_news.append({
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'date': article.get('published_date', '')
            })

        if positive_count > negative_count + 2:
            sentiment = "positive"
        elif negative_count > positive_count + 2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, recent_news

    except Exception as e:
        logger.error(f"Error fetching sentiment for {ticker}: {e}")
        return "unknown", []


async def earnings_calendar_with_progress(
    tickers: List[str],
    include_sentiment: bool = True
) -> AsyncIterator[EarningsProgressUpdate]:
    """
    Generate earnings calendar report with progress updates.

    Args:
        tickers: List of ticker symbols to analyze
        include_sentiment: Whether to fetch news sentiment (slower but more informative)
    """

    # Validate and clean tickers
    valid_tickers = []
    for t in tickers:
        clean, is_valid, _ = sanitize_ticker(t)
        if is_valid:
            valid_tickers.append(clean)

    if not valid_tickers:
        yield EarningsProgressUpdate(
            stage="error",
            stage_display="Error",
            message="No valid ticker symbols provided."
        )
        return

    total = len(valid_tickers)

    yield EarningsProgressUpdate(
        stage="init",
        stage_display="Initializing",
        message=f"Analyzing earnings data for {total} ticker(s)...",
        total_tickers=total
    )

    # Fetch earnings data for each ticker
    earnings_data: List[EarningsData] = []

    for i, ticker in enumerate(valid_tickers):
        yield EarningsProgressUpdate(
            stage="fetching",
            stage_display="Fetching Data",
            message=f"Getting earnings data for {ticker}...",
            tickers_processed=i,
            total_tickers=total
        )

        data = get_earnings_data(ticker)

        # Fetch sentiment if requested
        if include_sentiment and not data.error:
            yield EarningsProgressUpdate(
                stage="sentiment",
                stage_display="Analyzing Sentiment",
                message=f"Analyzing pre-earnings sentiment for {ticker}...",
                tickers_processed=i,
                total_tickers=total
            )

            sentiment, news = await fetch_earnings_sentiment(ticker)
            data.news_sentiment = sentiment
            data.recent_news = news

        earnings_data.append(data)

    yield EarningsProgressUpdate(
        stage="analyzing",
        stage_display="Generating Report",
        message="Sending to Claude AI for analysis... (this may take 30-60 seconds)",
        tickers_processed=total,
        total_tickers=total
    )

    # Generate analysis report using Claude with token tracking
    tokens = TokenAccumulator()
    report = await generate_earnings_report(earnings_data, tokens)

    yield EarningsProgressUpdate(
        stage="complete",
        stage_display="Complete",
        message="Earnings analysis complete!",
        tickers_processed=total,
        total_tickers=total,
        report=report,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
    )


async def generate_earnings_report(earnings_data: List[EarningsData], tokens: TokenAccumulator = None) -> str:
    """Generate comprehensive earnings report using Claude."""

    # Build data summary for Claude
    data_summary = []

    for data in earnings_data:
        # Format optional numeric values safely
        avg_move = f"{data.avg_post_earnings_move:.2f}%" if data.avg_post_earnings_move else "N/A"
        eps_actual = f"${data.last_quarter_eps_actual:.2f}" if data.last_quarter_eps_actual else "N/A"
        eps_estimate = f"${data.last_quarter_eps_estimate:.2f}" if data.last_quarter_eps_estimate else "N/A"
        surprise = f"{data.last_quarter_surprise:.2f}%" if data.last_quarter_surprise else "N/A"
        curr_q_est = f"${data.current_quarter_estimate:.2f}" if data.current_quarter_estimate else "N/A"
        curr_y_est = f"${data.current_year_estimate:.2f}" if data.current_year_estimate else "N/A"
        avg_surprise = f"{data.avg_surprise_percent:.2f}" if data.avg_surprise_percent else "0.00"

        summary = f"""
## {data.company_name} ({data.ticker})

### Upcoming Earnings
- Next Earnings Date: {data.next_earnings_date.strftime('%Y-%m-%d') if data.next_earnings_date else 'Not scheduled'}
- Days Until Earnings: {data.days_until_earnings if data.days_until_earnings is not None else 'N/A'}

### Historical Performance
- Beat Rate: {data.beat_rate:.1%} ({data.beat_count} beats, {data.miss_count} misses, {data.meet_count} meets)
- Average Surprise: {avg_surprise}% vs estimates
- Average Post-Earnings Move: {avg_move}

### Last Quarter
- EPS Actual: {eps_actual}
- EPS Estimate: {eps_estimate}
- Surprise: {surprise}

### Analyst Estimates
- Current Quarter EPS Estimate: {curr_q_est}
- Current Year EPS Estimate: {curr_y_est}

### Pre-Earnings Sentiment: {data.news_sentiment or 'Not analyzed'}
"""

        if data.recent_news:
            summary += "\n### Recent News:\n"
            for news in data.recent_news[:3]:
                summary += f"- {news['title']}\n"

        if data.error:
            summary = f"## {data.ticker}\nError fetching data: {data.error}\n"

        data_summary.append(summary)

    # Sort by days until earnings
    sorted_data = sorted(
        [d for d in earnings_data if d.days_until_earnings is not None],
        key=lambda x: x.days_until_earnings
    )

    upcoming_calendar = ""
    if sorted_data:
        upcoming_calendar = "\n### Upcoming Earnings Calendar:\n"
        for d in sorted_data:
            if d.days_until_earnings is not None and d.days_until_earnings >= 0:
                date_str = d.next_earnings_date.strftime('%b %d, %Y') if d.next_earnings_date else 'TBD'
                upcoming_calendar += f"- **{d.ticker}** ({d.company_name}): {date_str} ({d.days_until_earnings} days)\n"

    prompt = f"""Analyze the following earnings data and provide a comprehensive earnings report.

{upcoming_calendar}

# Detailed Earnings Data:
{''.join(data_summary)}

Please provide:

1. **Earnings Calendar Summary** - Upcoming earnings dates sorted chronologically, highlighting any reporting this week

2. **Beat/Miss Analysis** - Which companies have the strongest track records? Any concerning patterns?

3. **Pre-Earnings Outlook** - Based on historical performance and sentiment, what might we expect?

4. **Trading Considerations** - Historical volatility around earnings, potential opportunities/risks

5. **Key Dates to Watch** - Most important upcoming earnings and why

Format the report in clean markdown with clear sections. Focus on actionable insights."""

    client = Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track tokens
    if tokens:
        tokens.add(response)

    report = response.content[0].text

    # Add header
    header = f"""# Earnings Calendar Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Tickers Analyzed: {', '.join([d.ticker for d in earnings_data])}*

---

"""

    return header + report


# Convenience function for single ticker
async def analyze_single_ticker_earnings(ticker: str) -> str:
    """Analyze earnings for a single ticker and return the report."""
    report = None
    async for update in earnings_calendar_with_progress([ticker]):
        if update.report:
            report = update.report
    return report or "Error generating report"


# Main entry point for CLI testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        tickers = [t.strip() for t in sys.argv[1].split(',')]
    else:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    async def main():
        async for update in earnings_calendar_with_progress(tickers):
            print(f"[{update.stage_display}] {update.message}")
            if update.report:
                print("\n" + "="*60 + "\n")
                print(update.report)

    asyncio.run(main())
