"""
Unified Research Agent Hub - Gradio Web Interface
Combines Deep Research Agent and Stock Research Agent in a single tabbed interface.
Includes input validation, rate limiting, and optional authentication.
"""

import os
import gradio as gr
import asyncio
import time
import uuid
from threading import Thread
from queue import Queue
from dotenv import load_dotenv

# Import Deep Research components
from agents.deep_research_agent import (
    deep_research_with_progress,
    ProgressUpdate,
    tavily_client
)

# Import Stock Research components
from agents.stock_research_agent import stock_research_with_progress
from models.stock_data_models import StockProgressUpdate

# Import new Phase 3 agents
from agents.sector_research_agent import (
    sector_research_with_progress,
    get_available_sectors,
    get_sector_info,
    SectorProgressUpdate,
)
from agents.competitor_agent import (
    competitor_analysis_with_progress,
    CompetitorProgressUpdate,
)
from agents.portfolio_agent import (
    portfolio_analysis_with_progress,
    PortfolioProgressUpdate,
)

# Import Phase 4 agents
from agents.earnings_agent import (
    earnings_calendar_with_progress,
    EarningsProgressUpdate,
)
from agents.alert_system import (
    create_price_alert,
    create_earnings_alert,
    get_alert_summary,
    cancel_alert,
    check_all_alerts,
    send_test_email,
    AlertType,
)

# Import AI Research Agent
from agents.ai_research_agent import (
    ai_research_with_progress,
    AIResearchProgressUpdate,
)
from models.ai_research_models import AIResearchProgressUpdate as AIProgressModel

# Import Commodity Research Agent
from agents.commodity_research_agent import (
    commodity_research_with_progress,
)
from models.commodity_data_models import CommodityProgressUpdate

# Import Google Drive service
from services.google_drive_service import save_report_to_drive, is_drive_configured

# Import utilities
from utils.validators import sanitize_ticker, sanitize_query
from utils.rate_limiter import RateLimiter
from utils.logging_config import get_logger, setup_logging
from utils.pdf_export import markdown_to_pdf, generate_report_filename
from utils.report_history import add_to_history, get_history_choices, get_report_content, report_history
from utils.cache import db_cache

# Load environment variables
load_dotenv()

# Watchlist storage (in-memory, per session)
_watchlists: dict = {}

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Rate limiting configuration
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Initialize rate limiters
research_limiter = RateLimiter(
    max_requests=RATE_LIMIT_MAX_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS
)
stock_limiter = RateLimiter(
    max_requests=RATE_LIMIT_MAX_REQUESTS + 5,  # Stock gets slightly higher limit
    window_seconds=RATE_LIMIT_WINDOW_SECONDS
)

# Session tracking for rate limiting (simple approach)
_session_counter = 0
def get_session_id(request: gr.Request = None) -> str:
    """Generate or retrieve session ID."""
    global _session_counter
    if request and hasattr(request, 'session_hash') and request.session_hash:
        return request.session_hash
    _session_counter += 1
    return f"anon_{_session_counter}"


# ============================================================
# Deep Research Helper Functions
# ============================================================

def format_research_progress(update: ProgressUpdate, total_elapsed: float) -> str:
    """Format progress update for Deep Research display"""
    stage_icons = {
        "planning": "🤔",
        "searching": "🔍",
        "extracting": "🧠",
        "recursing": "🔄",
        "writing": "📝",
        "emailing": "📧",
        "complete": "✅"
    }

    icon = stage_icons.get(update.stage, "⏳")

    # Progress bar for search stage
    if update.stage == "searching" and update.total_steps > 1:
        filled = "█" * update.current_step
        empty = "░" * (update.total_steps - update.current_step)
        progress_bar = f"[{filled}{empty}] {update.current_step}/{update.total_steps}"
    else:
        progress_bar = ""

    stage_time = f"{update.elapsed_time:.1f}s" if update.elapsed_time > 0 else ""

    lines = [
        f"### {icon} {update.stage_display}",
        f"**Status:** {update.message}",
    ]

    if progress_bar:
        lines.append(f"**Progress:** {progress_bar}")
    if stage_time:
        lines.append(f"**Stage time:** {stage_time}")

    # Add depth tracking if depth > 1
    if update.max_depth > 1:
        lines.append(f"\n**Depth:** {update.current_depth}/{update.max_depth}")
        if update.total_searches > 0:
            lines.append(f"**Total Searches:** {update.total_searches}")
        if update.learnings_count > 0:
            lines.append(f"**Learnings:** {update.learnings_count}")
        if update.gaps_identified:
            lines.append(f"\n**Gaps Found ({len(update.gaps_identified)}):**")
            for gap in update.gaps_identified:
                display = gap[:80] + "..." if len(gap) > 80 else gap
                lines.append(f"  • {display}")

    lines.append(f"\n---\n**Total elapsed:** {total_elapsed:.1f}s")

    # Add token usage if available
    if update.total_tokens > 0:
        lines.append(f"\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})")
        lines.append(f"   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}")

    return "\n".join(lines)


def run_research_with_progress(query: str, depth: int = 1, max_searches: int = 6, request: gr.Request = None):
    """Generator function that yields progress updates and final report for Deep Research."""
    # Input validation
    if not query or query.strip() == "":
        yield "### ⚠️ Input Required\n\nPlease enter a research query!", "*Your research report will appear here...*"
        return

    # Sanitize the query
    sanitized_query, is_valid, error_msg = sanitize_query(query)
    if not is_valid:
        yield f"### ⚠️ Invalid Input\n\n{error_msg}", "*Your research report will appear here...*"
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = research_limiter.is_allowed(session_id)
    if not is_allowed:
        remaining = research_limiter.get_remaining(session_id)
        yield f"### ⚠️ Rate Limit Exceeded\n\n{rate_error}\n\nRemaining requests: {remaining}", "*Please wait before making another request...*"
        return

    # Convert slider values to integers
    depth = int(depth)
    max_searches = int(max_searches)

    logger.info(f"Starting deep research for query: {sanitized_query[:50]}... (depth={depth}, max_searches={max_searches})")

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in deep_research_with_progress(query=sanitized_query, depth=depth, max_searches=max_searches):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async_research)
    thread.start()

    start_time = time.time()
    report = "*Research in progress...*"

    yield "### ⏳ Starting...\n\nInitializing research agents...", report

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=0.5)

            if isinstance(update, Exception):
                total_time = time.time() - start_time
                error_status = f"### ❌ Error\n\nAn error occurred: {update}\n\n**Time:** {total_time:.1f}s"
                yield error_status, f"❌ **Error occurred:**\n\n```\n{update}\n```\n\nPlease check your API keys in the .env file."
                break

            total_elapsed = time.time() - start_time

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s"
                # Add token usage to completion status
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            else:
                status = format_research_progress(update, total_elapsed)
                yield status, report

        except:
            pass

    thread.join()


# ============================================================
# PDF Export Helper Functions
# ============================================================

def export_report_to_pdf(report_content: str, title: str = "Research Report"):
    """Export report to PDF and return file path for download."""
    if not report_content or report_content.startswith("*"):
        return None

    try:
        filename = generate_report_filename(title, "report")
        pdf_path = markdown_to_pdf(report_content, title=title)
        if pdf_path:
            logger.info(f"PDF exported: {pdf_path}")
            return pdf_path
    except Exception as e:
        logger.error(f"PDF export failed: {e}")

    return None


# ============================================================
# Watchlist Helper Functions
# ============================================================

def get_watchlist(session_id: str) -> list:
    """Get watchlist for a session."""
    return _watchlists.get(session_id, [])


def add_to_watchlist(ticker: str, session_id: str) -> tuple:
    """Add ticker to watchlist. Returns (success_message, updated_watchlist_display)."""
    sanitized, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        return f"Invalid ticker: {error}", get_watchlist_display(session_id)

    if session_id not in _watchlists:
        _watchlists[session_id] = []

    if sanitized in _watchlists[session_id]:
        return f"{sanitized} is already in your watchlist", get_watchlist_display(session_id)

    if len(_watchlists[session_id]) >= 20:
        return "Watchlist is full (max 20 tickers)", get_watchlist_display(session_id)

    _watchlists[session_id].append(sanitized)
    logger.info(f"Added {sanitized} to watchlist for session {session_id[:8]}...")
    return f"Added {sanitized} to watchlist", get_watchlist_display(session_id)


def remove_from_watchlist(ticker: str, session_id: str) -> tuple:
    """Remove ticker from watchlist."""
    sanitized = ticker.upper().strip()
    if session_id in _watchlists and sanitized in _watchlists[session_id]:
        _watchlists[session_id].remove(sanitized)
        return f"Removed {sanitized} from watchlist", get_watchlist_display(session_id)
    return f"{sanitized} not found in watchlist", get_watchlist_display(session_id)


def clear_watchlist(session_id: str) -> tuple:
    """Clear entire watchlist."""
    if session_id in _watchlists:
        _watchlists[session_id] = []
    return "Watchlist cleared", ""


def get_watchlist_display(session_id: str) -> str:
    """Get formatted watchlist display."""
    watchlist = get_watchlist(session_id)
    if not watchlist:
        return "*No tickers in watchlist*"
    return ", ".join(watchlist)


# ============================================================
# Report History Helper Functions
# ============================================================

def load_history_report(report_id: str, session_id: str) -> str:
    """Load a report from history."""
    if not report_id:
        return "*Select a report from history...*"
    content = get_report_content(session_id, report_id)
    return content if content else "*Report not found*"


def get_research_history_dropdown(session_id: str):
    """Get dropdown choices for research history."""
    choices = get_history_choices(session_id, "research")
    return gr.update(choices=choices, value=None)


def get_stock_history_dropdown(session_id: str):
    """Get dropdown choices for stock history."""
    choices = get_history_choices(session_id, "stock")
    return gr.update(choices=choices, value=None)


# ============================================================
# Stock Research Helper Functions
# ============================================================

def format_stock_progress(update: StockProgressUpdate, total_elapsed: float) -> str:
    """Format progress update for Stock Research display"""
    stage_icons = {
        "validating": "🔍",
        "fetching": "📡",
        "analyzing": "🧠",
        "identifying_gaps": "🔎",
        "followup_research": "🔬",
        "writing": "📝",
        "complete": "✅",
        "error": "❌"
    }

    icon = stage_icons.get(update.stage, "⏳")

    lines = [
        f"### {icon} {update.stage_display}",
        f"**Status:** {update.message}",
    ]

    if update.source_status:
        source_icons = {"success": "✅", "failed": "❌", "pending": "⏳"}
        source_lines = []
        for source, status in update.source_status.items():
            source_name = source.replace('_', ' ').title()
            source_lines.append(f"  {source_icons.get(status, '⏳')} {source_name}")
        lines.append("\n**Data Sources:**\n" + "\n".join(source_lines))

    if update.gaps_identified:
        lines.append(f"\n**Gaps Identified ({len(update.gaps_identified)}):**")
        for gap in update.gaps_identified:
            category = gap['category'].replace('_', ' ').title()
            desc = gap['description'][:60] + "..." if len(gap['description']) > 60 else gap['description']
            lines.append(f"  • **{category}:** {desc}")

    if update.elapsed_time > 0:
        lines.append(f"\n**Stage time:** {update.elapsed_time:.1f}s")

    lines.append(f"\n---\n**Total elapsed:** {total_elapsed:.1f}s")

    return "\n".join(lines)


def run_stock_research(ticker: str, deep_analysis: bool = False, request: gr.Request = None):
    """Generator function that yields progress updates and final report for Stock Research."""
    # Input validation
    if not ticker or ticker.strip() == "":
        yield (
            "### ⚠️ Input Required\n\nPlease enter a stock ticker symbol!",
            "*Enter a ticker like AAPL, TSLA, or MSFT to generate a research report...*"
        )
        return

    # Sanitize the ticker
    sanitized_ticker, is_valid, error_msg = sanitize_ticker(ticker)
    if not is_valid:
        yield (
            f"### ⚠️ Invalid Ticker\n\n{error_msg}",
            "*Please enter a valid ticker symbol (e.g., AAPL, TSLA, MSFT)...*"
        )
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = stock_limiter.is_allowed(session_id)
    if not is_allowed:
        remaining = stock_limiter.get_remaining(session_id)
        yield (
            f"### ⚠️ Rate Limit Exceeded\n\n{rate_error}\n\nRemaining requests: {remaining}",
            "*Please wait before making another request...*"
        )
        return

    logger.info(f"Starting stock research for ticker: {sanitized_ticker} (deep_analysis={deep_analysis})")

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in stock_research_with_progress(sanitized_ticker, deep_analysis=deep_analysis):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async_research)
    thread.start()

    start_time = time.time()
    report = f"*Researching {sanitized_ticker}...*"

    yield (
        f"### ⏳ Starting Research\n\nInitializing research for **{sanitized_ticker}**...",
        report
    )

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=0.5)

            if isinstance(update, Exception):
                total_time = time.time() - start_time
                error_status = f"### ❌ Error\n\nAn error occurred: {update}\n\n**Time:** {total_time:.1f}s"
                yield (
                    error_status,
                    f"❌ **Error occurred:**\n\n```\n{update}\n```\n\nPlease check your API keys and try again."
                )
                break

            total_elapsed = time.time() - start_time

            if update.stage == "complete":
                status = f"### ✅ Research Complete!\n\n**Total time:** {total_elapsed:.1f}s"
                if update.analysis:
                    rec = update.analysis.recommendation.value.replace('_', ' ').upper()
                    status += f"\n\n**Recommendation:** {rec}"
                # Add token usage
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            else:
                status = format_stock_progress(update, total_elapsed)
                yield status, report

        except:
            pass

    thread.join()


# ============================================================
# Commodity Research Helper Functions
# ============================================================

def format_commodity_progress(update: CommodityProgressUpdate, total_elapsed: float) -> str:
    """Format progress update for Commodity Research display"""
    stage_icons = {
        "validating": "🔍",
        "fetching": "📡",
        "analyzing": "🧠",
        "writing": "📝",
        "complete": "✅",
        "error": "❌"
    }

    icon = stage_icons.get(update.stage, "⏳")

    lines = [
        f"### {icon} {update.stage_display}",
        f"**Status:** {update.message}",
    ]

    if update.source_status:
        source_icons = {"success": "✅", "failed": "❌", "pending": "⏳"}
        source_lines = []
        for source, status in update.source_status.items():
            source_name = source.replace('_', ' ').title()
            source_lines.append(f"  {source_icons.get(status, '⏳')} {source_name}")
        lines.append("\n**Data Sources:**\n" + "\n".join(source_lines))

    if update.elapsed_time > 0:
        lines.append(f"\n**Stage time:** {update.elapsed_time:.1f}s")

    lines.append(f"\n---\n**Total elapsed:** {total_elapsed:.1f}s")

    return "\n".join(lines)


def run_commodity_research(symbol_input: str, request: gr.Request = None):
    """Generator function that yields progress updates and final report for Commodity Research."""
    if not symbol_input or symbol_input.strip() == "":
        yield (
            "### ⚠️ Input Required\n\nPlease enter a commodity name or symbol!",
            "*Enter a commodity like gold, crude, or copper to generate a research report...*"
        )
        return

    cleaned = symbol_input.strip()
    logger.info(f"Starting commodity research for: {cleaned}")

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in commodity_research_with_progress(cleaned):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async_research)
    thread.start()

    start_time = time.time()
    report = f"*Researching {cleaned}...*"

    yield (
        f"### ⏳ Starting Research\n\nInitializing research for **{cleaned}**...",
        report
    )

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=0.5)

            if isinstance(update, Exception):
                total_time = time.time() - start_time
                error_status = f"### ❌ Error\n\nAn error occurred: {update}\n\n**Time:** {total_time:.1f}s"
                yield (
                    error_status,
                    f"❌ **Error occurred:**\n\n```\n{update}\n```\n\nPlease check your API keys and try again."
                )
                break

            total_elapsed = time.time() - start_time

            if update.stage == "complete":
                status = f"### ✅ Research Complete!\n\n**Total time:** {total_elapsed:.1f}s"
                if update.analysis:
                    outlook = update.analysis.outlook.value.upper()
                    status += f"\n\n**Outlook:** {outlook}"
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            else:
                status = format_commodity_progress(update, total_elapsed)
                yield status, report

        except:
            pass

    thread.join()


# ============================================================
# Stock Comparison Helper Functions
# ============================================================

def run_stock_comparison(ticker1: str, ticker2: str, ticker3: str = "", request: gr.Request = None):
    """
    Run comparison analysis for 2-3 stocks.
    Fetches data for each stock and creates a comparison table.
    """
    from stock_data_fetchers import fetch_yfinance_data, validate_ticker

    # Validate and collect tickers
    tickers = []
    for t in [ticker1, ticker2, ticker3]:
        if t and t.strip():
            sanitized, is_valid, error = sanitize_ticker(t)
            if is_valid:
                tickers.append(sanitized)

    if len(tickers) < 2:
        return "### ⚠️ Input Required\n\nPlease enter at least 2 valid ticker symbols.", "*Comparison will appear here...*"

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = stock_limiter.is_allowed(session_id)
    if not is_allowed:
        return f"### ⚠️ Rate Limit\n\n{rate_error}", "*Please wait...*"

    logger.info(f"Starting comparison for: {', '.join(tickers)}")

    # Fetch data for all tickers
    import asyncio

    async def fetch_all():
        results = {}
        for ticker in tickers:
            try:
                validation = await validate_ticker(ticker)
                if validation['is_valid']:
                    data = await fetch_yfinance_data(ticker)
                    results[ticker] = {
                        'name': validation['company_name'],
                        'data': data
                    }
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
        return results

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        stock_data = loop.run_until_complete(fetch_all())
    finally:
        loop.close()

    if len(stock_data) < 2:
        return "### ❌ Error\n\nCouldn't fetch data for enough stocks.", "*Please check the ticker symbols and try again.*"

    # Build comparison table
    comparison_md = f"""# Stock Comparison
## {' vs '.join(stock_data.keys())}

| Metric | {' | '.join(stock_data.keys())} |
|--------|{'|'.join(['------' for _ in stock_data])}|
"""

    # Price data
    prices = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.price:
            prices.append(f"${data.price.current_price:.2f}")
        else:
            prices.append("N/A")
    comparison_md += f"| **Current Price** | {' | '.join(prices)} |\n"

    # Market Cap
    mcaps = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.price and data.price.market_cap:
            mcap = data.price.market_cap
            if mcap >= 1e12:
                mcaps.append(f"${mcap/1e12:.2f}T")
            elif mcap >= 1e9:
                mcaps.append(f"${mcap/1e9:.2f}B")
            else:
                mcaps.append(f"${mcap/1e6:.2f}M")
        else:
            mcaps.append("N/A")
    comparison_md += f"| **Market Cap** | {' | '.join(mcaps)} |\n"

    # P/E Ratio
    pes = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.ratios and data.ratios.pe_trailing:
            pes.append(f"{data.ratios.pe_trailing:.2f}")
        else:
            pes.append("N/A")
    comparison_md += f"| **P/E (TTM)** | {' | '.join(pes)} |\n"

    # Forward P/E
    fpes = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.ratios and data.ratios.pe_forward:
            fpes.append(f"{data.ratios.pe_forward:.2f}")
        else:
            fpes.append("N/A")
    comparison_md += f"| **P/E (Forward)** | {' | '.join(fpes)} |\n"

    # Price/Book
    pbs = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.ratios and data.ratios.price_to_book:
            pbs.append(f"{data.ratios.price_to_book:.2f}")
        else:
            pbs.append("N/A")
    comparison_md += f"| **Price/Book** | {' | '.join(pbs)} |\n"

    # Profit Margin
    margins = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.ratios and data.ratios.profit_margin:
            margins.append(f"{data.ratios.profit_margin:.1%}")
        else:
            margins.append("N/A")
    comparison_md += f"| **Profit Margin** | {' | '.join(margins)} |\n"

    # ROE
    roes = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.ratios and data.ratios.roe:
            roes.append(f"{data.ratios.roe:.1%}")
        else:
            roes.append("N/A")
    comparison_md += f"| **ROE** | {' | '.join(roes)} |\n"

    # Debt/Equity
    des = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.ratios and data.ratios.debt_to_equity:
            des.append(f"{data.ratios.debt_to_equity:.2f}")
        else:
            des.append("N/A")
    comparison_md += f"| **Debt/Equity** | {' | '.join(des)} |\n"

    # 52-Week Range
    ranges = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.price:
            ranges.append(f"${data.price.fifty_two_week_low:.2f} - ${data.price.fifty_two_week_high:.2f}")
        else:
            ranges.append("N/A")
    comparison_md += f"| **52-Week Range** | {' | '.join(ranges)} |\n"

    # Recommendation
    recs = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.recommendation_key:
            recs.append(data.recommendation_key.upper())
        else:
            recs.append("N/A")
    comparison_md += f"| **Analyst Rating** | {' | '.join(recs)} |\n"

    # Sector/Industry
    sectors = []
    for ticker in stock_data:
        data = stock_data[ticker]['data']
        if data.sector:
            sectors.append(data.sector)
        else:
            sectors.append("N/A")
    comparison_md += f"| **Sector** | {' | '.join(sectors)} |\n"

    comparison_md += """
---
*Data from Yahoo Finance. For detailed analysis, use the Stock Research tab for individual tickers.*
"""

    status = f"### ✅ Comparison Complete\n\nCompared {len(stock_data)} stocks"
    return status, comparison_md


# ============================================================
# Sector Research Helper Functions
# ============================================================

def run_sector_research(sector: str, request: gr.Request = None):
    """Generator function for sector research with progress updates."""
    if not sector:
        yield "### ⚠️ Input Required\n\nPlease select a sector.", "*Select a sector to analyze...*"
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = research_limiter.is_allowed(session_id)
    if not is_allowed:
        yield f"### ⚠️ Rate Limit\n\n{rate_error}", "*Please wait...*"
        return

    logger.info(f"Starting sector research for: {sector}")

    progress_queue = Queue()

    def run_async():
        async def wrapper():
            async for update in sector_research_with_progress(sector):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async)
    thread.start()

    start_time = time.time()
    report = f"*Analyzing {sector} sector...*"
    last_message = "Initializing..."
    last_stage = "Starting"

    yield f"### ⏳ Starting\n\nInitializing sector analysis for **{sector}**...", report

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=1.0)

            if isinstance(update, Exception):
                yield f"### ❌ Error\n\n{update}", f"Error: {update}"
                break

            total_elapsed = time.time() - start_time
            last_message = update.message
            last_stage = update.stage_display

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s\n\n**Companies analyzed:** {update.companies_fetched}"
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            elif update.stage == "error":
                yield f"### ❌ Error\n\n{update.message}", f"Error: {update.message}"
                break
            else:
                progress = ""
                if update.total_companies > 0:
                    progress = f"\n\n**Progress:** {update.companies_fetched}/{update.total_companies} companies"
                status = f"### ⏳ {update.stage_display}\n\n{update.message}{progress}\n\n**Elapsed:** {total_elapsed:.1f}s"
                yield status, report

        except:
            # Timeout - update elapsed time to show we're still working
            if thread.is_alive():
                total_elapsed = time.time() - start_time
                status = f"### ⏳ {last_stage}\n\n{last_message}\n\n**Elapsed:** {total_elapsed:.1f}s ⟳"
                yield status, report

    thread.join()


# ============================================================
# Competitor Analysis Helper Functions
# ============================================================

def run_competitor_analysis(ticker: str, request: gr.Request = None):
    """Generator function for competitor analysis with progress updates."""
    if not ticker or not ticker.strip():
        yield "### ⚠️ Input Required\n\nPlease enter a ticker symbol.", "*Enter a ticker to analyze its competitors...*"
        return

    sanitized_ticker, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        yield f"### ⚠️ Invalid Ticker\n\n{error}", "*Please enter a valid ticker...*"
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = stock_limiter.is_allowed(session_id)
    if not is_allowed:
        yield f"### ⚠️ Rate Limit\n\n{rate_error}", "*Please wait...*"
        return

    logger.info(f"Starting competitor analysis for: {sanitized_ticker}")

    progress_queue = Queue()

    def run_async():
        async def wrapper():
            async for update in competitor_analysis_with_progress(sanitized_ticker):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async)
    thread.start()

    start_time = time.time()
    report = f"*Analyzing competitors for {sanitized_ticker}...*"
    last_message = "Finding competitors..."
    last_stage = "Starting"

    yield f"### ⏳ Starting\n\nFinding competitors for **{sanitized_ticker}**...", report

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=1.0)

            if isinstance(update, Exception):
                yield f"### ❌ Error\n\n{update}", f"Error: {update}"
                break

            total_elapsed = time.time() - start_time
            last_message = update.message
            last_stage = update.stage_display

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s"
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            elif update.stage == "error":
                yield f"### ❌ Error\n\n{update.message}", f"Error: {update.message}"
                break
            else:
                status = f"### ⏳ {update.stage_display}\n\n{update.message}\n\n**Elapsed:** {total_elapsed:.1f}s"
                yield status, report

        except:
            # Timeout - update elapsed time to show we're still working
            if thread.is_alive():
                total_elapsed = time.time() - start_time
                status = f"### ⏳ {last_stage}\n\n{last_message}\n\n**Elapsed:** {total_elapsed:.1f}s ⟳"
                yield status, report

    thread.join()


# ============================================================
# Portfolio Analysis Helper Functions
# ============================================================

def run_portfolio_analysis(portfolio_text: str, request: gr.Request = None):
    """Generator function for portfolio analysis with progress updates."""
    if not portfolio_text or not portfolio_text.strip():
        yield "### ⚠️ Input Required\n\nPlease enter your portfolio holdings.", "*Enter your holdings to analyze...*"
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = research_limiter.is_allowed(session_id)
    if not is_allowed:
        yield f"### ⚠️ Rate Limit\n\n{rate_error}", "*Please wait...*"
        return

    logger.info("Starting portfolio analysis")

    progress_queue = Queue()

    def run_async():
        async def wrapper():
            async for update in portfolio_analysis_with_progress(portfolio_text, is_csv=False):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async)
    thread.start()

    start_time = time.time()
    report = "*Analyzing portfolio...*"
    last_message = "Parsing portfolio data..."
    last_stage = "Starting"

    yield "### ⏳ Starting\n\nParsing portfolio data...", report

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=1.0)

            if isinstance(update, Exception):
                yield f"### ❌ Error\n\n{update}", f"Error: {update}"
                break

            total_elapsed = time.time() - start_time
            last_message = update.message
            last_stage = update.stage_display

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s\n\n**Holdings analyzed:** {update.holdings_processed}"
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            elif update.stage == "error":
                yield f"### ❌ Error\n\n{update.message}", f"Error: {update.message}"
                break
            else:
                progress = ""
                if update.total_holdings > 0:
                    progress = f"\n\n**Progress:** {update.holdings_processed}/{update.total_holdings} holdings"
                status = f"### ⏳ {update.stage_display}\n\n{update.message}{progress}\n\n**Elapsed:** {total_elapsed:.1f}s"
                yield status, report

        except:
            # Timeout - update elapsed time to show we're still working
            if thread.is_alive():
                total_elapsed = time.time() - start_time
                status = f"### ⏳ {last_stage}\n\n{last_message}\n\n**Elapsed:** {total_elapsed:.1f}s ⟳"
                yield status, report

    thread.join()


def run_portfolio_csv_analysis(file, request: gr.Request = None):
    """Handle CSV file upload for portfolio analysis."""
    if file is None:
        yield "### ⚠️ No File\n\nPlease upload a CSV file.", "*Upload a CSV file with your holdings...*"
        return

    try:
        # Read CSV content
        with open(file.name, 'r') as f:
            csv_content = f.read()
    except Exception as e:
        yield f"### ❌ Error\n\nCould not read file: {e}", "*Please try again...*"
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = research_limiter.is_allowed(session_id)
    if not is_allowed:
        yield f"### ⚠️ Rate Limit\n\n{rate_error}", "*Please wait...*"
        return

    logger.info("Starting portfolio CSV analysis")

    progress_queue = Queue()

    def run_async():
        async def wrapper():
            async for update in portfolio_analysis_with_progress(csv_content, is_csv=True):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async)
    thread.start()

    start_time = time.time()
    report = "*Analyzing portfolio from CSV...*"
    last_message = "Parsing CSV file..."
    last_stage = "Starting"

    yield "### ⏳ Starting\n\nParsing CSV file...", report

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=1.0)

            if isinstance(update, Exception):
                yield f"### ❌ Error\n\n{update}", f"Error: {update}"
                break

            total_elapsed = time.time() - start_time
            last_message = update.message
            last_stage = update.stage_display

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s\n\n**Holdings analyzed:** {update.holdings_processed}"
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            elif update.stage == "error":
                yield f"### ❌ Error\n\n{update.message}", f"Error: {update.message}"
                break
            else:
                progress = ""
                if update.total_holdings > 0:
                    progress = f"\n\n**Progress:** {update.holdings_processed}/{update.total_holdings} holdings"
                status = f"### ⏳ {update.stage_display}\n\n{update.message}{progress}\n\n**Elapsed:** {total_elapsed:.1f}s"
                yield status, report

        except:
            # Timeout - update elapsed time to show we're still working
            if thread.is_alive():
                total_elapsed = time.time() - start_time
                status = f"### ⏳ {last_stage}\n\n{last_message}\n\n**Elapsed:** {total_elapsed:.1f}s ⟳"
                yield status, report

    thread.join()


# ============================================================
# AI Research Helper Functions
# ============================================================

def format_ai_research_progress(update: AIResearchProgressUpdate, total_elapsed: float) -> str:
    """Format progress update for AI Research display"""
    stage_icons = {
        "planning": "🤔",
        "searching": "🔍",
        "extracting": "🧠",
        "recursing": "🔄",
        "synthesizing": "📝",
        "complete": "✅"
    }

    icon = stage_icons.get(update.stage, "⏳")

    lines = [
        f"### {icon} {update.stage_display}",
        f"**Status:** {update.message}",
    ]

    # Depth progress
    if update.max_depth > 0:
        depth_filled = "●" * (update.current_depth + 1)
        depth_empty = "○" * (update.max_depth - update.current_depth - 1)
        lines.append(f"**Depth:** [{depth_filled}{depth_empty}] {update.current_depth + 1}/{update.max_depth}")

    # Search progress
    if update.total_searches > 0:
        search_pct = min(100, (update.searches_completed / update.total_searches) * 100)
        lines.append(f"**Searches:** {update.searches_completed}/{update.total_searches} ({search_pct:.0f}%)")

    # Learnings count
    if update.learnings_count > 0:
        lines.append(f"**Learnings:** {update.learnings_count}")

    if update.elapsed_time > 0:
        lines.append(f"\n**Elapsed:** {update.elapsed_time:.1f}s")

    lines.append(f"\n---\n**Total elapsed:** {total_elapsed:.1f}s")

    return "\n".join(lines)


def run_ai_research(query: str, depth: int, max_searches: int, request: gr.Request = None):
    """Generator function that yields progress updates and final report for AI Research."""
    # Input validation
    if not query or query.strip() == "":
        yield (
            "### ⚠️ Input Required\n\nPlease enter a research query!",
            "*Your AI research report will appear here...*"
        )
        return

    # Sanitize the query
    sanitized_query, is_valid, error_msg = sanitize_query(query)
    if not is_valid:
        yield (
            f"### ⚠️ Invalid Input\n\n{error_msg}",
            "*Your AI research report will appear here...*"
        )
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = research_limiter.is_allowed(session_id)
    if not is_allowed:
        remaining = research_limiter.get_remaining(session_id)
        yield (
            f"### ⚠️ Rate Limit Exceeded\n\n{rate_error}\n\nRemaining requests: {remaining}",
            "*Please wait before making another request...*"
        )
        return

    logger.info(f"Starting AI research for query: {sanitized_query[:50]}...")

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in ai_research_with_progress(
                query=sanitized_query,
                depth=depth,
                max_searches=max_searches
            ):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async_research)
    thread.start()

    start_time = time.time()
    report = "*AI research in progress...*"
    last_message = "Initializing..."
    last_stage = "Starting"

    yield (
        "### ⏳ Starting...\n\nInitializing AI research agents...",
        report
    )

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=1.0)

            if isinstance(update, Exception):
                total_time = time.time() - start_time
                error_status = f"### ❌ Error\n\nAn error occurred: {update}\n\n**Time:** {total_time:.1f}s"
                yield (
                    error_status,
                    f"❌ **Error occurred:**\n\n```\n{update}\n```\n\nPlease check your API keys in the .env file."
                )
                break

            total_elapsed = time.time() - start_time
            last_message = update.message
            last_stage = update.stage_display

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s"
                status += f"\n\n**Depth:** {update.current_depth + 1}/{update.max_depth}"
                status += f"\n**Searches:** {update.searches_completed}"
                status += f"\n**Learnings:** {update.learnings_count}"
                # Add token usage
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                yield status, update.report
                break
            else:
                status = format_ai_research_progress(update, total_elapsed)
                yield status, report

        except:
            # Timeout - update elapsed time to show we're still working
            if thread.is_alive():
                total_elapsed = time.time() - start_time
                status = f"### ⏳ {last_stage}\n\n{last_message}\n\n**Elapsed:** {total_elapsed:.1f}s ⟳"
                yield status, report

    thread.join()


# ============================================================
# Earnings Calendar Helper Functions
# ============================================================

def run_earnings_analysis(tickers_input: str, include_sentiment: bool, request: gr.Request = None):
    """Generator function for earnings calendar with progress updates."""
    if not tickers_input or not tickers_input.strip():
        yield "### ⚠️ Input Required\n\nPlease enter ticker symbols.", "*Enter tickers to analyze earnings...*"
        return

    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.replace(',', ' ').split() if t.strip()]

    if not tickers:
        yield "### ⚠️ Input Required\n\nNo valid ticker symbols found.", "*Enter tickers to analyze earnings...*"
        return

    # Rate limiting
    session_id = get_session_id(request)
    is_allowed, rate_error = stock_limiter.is_allowed(session_id)
    if not is_allowed:
        yield f"### ⚠️ Rate Limit\n\n{rate_error}", "*Please wait...*"
        return

    logger.info(f"Starting earnings analysis for: {tickers}")

    progress_queue = Queue()

    def run_async():
        async def wrapper():
            async for update in earnings_calendar_with_progress(tickers, include_sentiment):
                progress_queue.put(update)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    thread = Thread(target=run_async)
    thread.start()

    start_time = time.time()
    report = f"*Analyzing earnings for {len(tickers)} ticker(s)...*"
    last_message = "Initializing..."
    last_stage = "Starting"

    yield f"### ⏳ Starting\n\nInitializing earnings analysis...", report

    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=1.0)

            if isinstance(update, Exception):
                yield f"### ❌ Error\n\n{update}", f"Error: {update}"
                break

            total_elapsed = time.time() - start_time
            last_message = update.message
            last_stage = update.stage_display

            if update.stage == "complete":
                status = f"### ✅ Complete!\n\n**Total time:** {total_elapsed:.1f}s\n\n**Tickers analyzed:** {update.tickers_processed}"
                if update.total_tokens > 0:
                    status += f"\n\n📊 **Token Usage:** {update.total_tokens:,} tokens (${update.estimated_cost:.4f})"
                    status += f"\n   - Input: {update.input_tokens:,} | Output: {update.output_tokens:,}"
                # Save to cache
                db_cache.save_report(
                    report_type='earnings',
                    query=', '.join(tickers),
                    report_content=update.report
                )
                yield status, update.report
                break
            elif update.stage == "error":
                yield f"### ❌ Error\n\n{update.message}", f"Error: {update.message}"
                break
            else:
                progress = ""
                if update.total_tickers > 0:
                    progress = f"\n\n**Progress:** {update.tickers_processed}/{update.total_tickers} tickers"
                status = f"### ⏳ {update.stage_display}\n\n{update.message}{progress}\n\n**Elapsed:** {total_elapsed:.1f}s"
                yield status, report

        except:
            # Timeout - update elapsed time to show we're still working
            if thread.is_alive():
                total_elapsed = time.time() - start_time
                status = f"### ⏳ {last_stage}\n\n{last_message}\n\n**Elapsed:** {total_elapsed:.1f}s ⟳"
                yield status, report

    thread.join()


# ============================================================
# Alerts Helper Functions
# ============================================================

def create_new_price_alert(ticker: str, condition: str, target_price: float, email: str, request: gr.Request = None):
    """Create a new price alert."""
    if not ticker or not email or not target_price:
        return "### ⚠️ Missing Fields\n\nPlease fill in all required fields.", get_alerts_display()

    clean_ticker, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        return f"### ⚠️ Invalid Ticker\n\n{error}", get_alerts_display()

    try:
        alert_id = create_price_alert(clean_ticker, condition, float(target_price), email)
        logger.info(f"Created price alert {alert_id} for {clean_ticker}")
        return f"### ✅ Alert Created\n\nAlert #{alert_id}: {clean_ticker} price {condition} ${target_price:.2f}", get_alerts_display()
    except Exception as e:
        logger.error(f"Error creating price alert: {e}")
        return f"### ❌ Error\n\n{e}", get_alerts_display()


def create_new_earnings_alert(ticker: str, days_before: int, email: str, request: gr.Request = None):
    """Create a new earnings reminder alert."""
    if not ticker or not email:
        return "### ⚠️ Missing Fields\n\nPlease fill in all required fields.", get_alerts_display()

    clean_ticker, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        return f"### ⚠️ Invalid Ticker\n\n{error}", get_alerts_display()

    try:
        alert_id = create_earnings_alert(clean_ticker, int(days_before), email)
        logger.info(f"Created earnings alert {alert_id} for {clean_ticker}")
        return f"### ✅ Alert Created\n\nAlert #{alert_id}: Earnings reminder for {clean_ticker} ({days_before} days before)", get_alerts_display()
    except Exception as e:
        logger.error(f"Error creating earnings alert: {e}")
        return f"### ❌ Error\n\n{e}", get_alerts_display()


def get_alerts_display():
    """Get formatted display of all active alerts."""
    try:
        summary = get_alert_summary()
        alerts = db_cache.get_active_alerts()

        if not alerts:
            return """### 📋 Active Alerts

*No active alerts. Create one above!*

---

**Tip:** Price alerts trigger when the stock crosses your target. Earnings alerts remind you before the announcement date.
"""

        display = f"""### 📋 Active Alerts ({summary['total_active']} total)

| ID | Ticker | Type | Condition | Email |
|---|---|---|---|---|
"""
        for alert in alerts:
            alert_type = alert['alert_type'].replace('_', ' ').title()
            display += f"| {alert['id']} | {alert['ticker']} | {alert_type} | {alert['condition']} | {alert['email'][:20]}... |\n"

        display += """
---

*Click an ID above and use the cancel button to remove an alert.*
"""
        return display
    except Exception as e:
        logger.error(f"Error getting alerts display: {e}")
        return f"### Error loading alerts\n\n{e}"


def cancel_alert_by_id(alert_id: str):
    """Cancel an alert by ID."""
    if not alert_id:
        return "### ⚠️ Enter Alert ID\n\nPlease enter the ID of the alert to cancel.", get_alerts_display()

    try:
        alert_id_int = int(alert_id)
        success = cancel_alert(alert_id_int)
        if success:
            return f"### ✅ Alert Cancelled\n\nAlert #{alert_id_int} has been cancelled.", get_alerts_display()
        else:
            return f"### ⚠️ Not Found\n\nAlert #{alert_id_int} not found or already cancelled.", get_alerts_display()
    except ValueError:
        return "### ⚠️ Invalid ID\n\nPlease enter a valid numeric alert ID.", get_alerts_display()
    except Exception as e:
        logger.error(f"Error cancelling alert: {e}")
        return f"### ❌ Error\n\n{e}", get_alerts_display()


def run_alert_check():
    """Manually run alert check."""
    try:
        import asyncio
        triggered = asyncio.run(check_all_alerts())
        if triggered:
            result = f"### ✅ Alert Check Complete\n\n**{len(triggered)} alert(s) triggered:**\n\n"
            for r in triggered:
                result += f"- {r.alert.ticker}: {r.message}\n"
            return result, get_alerts_display()
        else:
            return "### ✅ Alert Check Complete\n\nNo alerts triggered.", get_alerts_display()
    except Exception as e:
        logger.error(f"Error running alert check: {e}")
        return f"### ❌ Error\n\n{e}", get_alerts_display()


def run_test_email(email: str):
    """Send a test email to verify configuration."""
    if not email or not email.strip():
        return "### ⚠️ Enter Email\n\nPlease enter an email address to test."

    success, message = send_test_email(email.strip())
    if success:
        return f"### ✅ Test Email Sent\n\n{message}\n\n**Check your inbox (and spam folder).**"
    else:
        return f"### ❌ Email Failed\n\n{message}"


# ============================================================
# Unified Gradio Interface with Tabs
# ============================================================

search_mode = "Tavily web search" if tavily_client else "LLM knowledge base"

# Store current report content for export (global state for simplicity)
_current_research_report = {"content": "", "query": ""}
_current_stock_report = {"content": "", "ticker": ""}


def update_research_report_state(status, report):
    """Update research report state and return values."""
    if report and not report.startswith("*"):
        _current_research_report["content"] = report
    return status, report


def update_stock_report_state(status, report):
    """Update stock report state and return values."""
    if report and not report.startswith("*"):
        _current_stock_report["content"] = report
    return status, report


with gr.Blocks(title="Research Agent Hub", theme=gr.themes.Soft()) as demo:

    # Hidden state for session ID
    session_state = gr.State(value="")

    # Header
    gr.Markdown("""
    # 🎯 Research Agent Hub
    ### Multi-Agent AI Research System

    Select a tab below to access different research capabilities.
    """)

    with gr.Tabs():
        # ========== Deep Research Tab ==========
        with gr.Tab("🔬 Deep Research"):
            gr.Markdown(f"""
            ### Multi-Agent System for Comprehensive Research Reports

            This AI-powered research assistant uses multiple specialized agents to:
            1. **Plan** relevant search queries
            2. **Research** each topic via {search_mode}
            3. **Synthesize** findings into a professional report
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="🎯 Research Query",
                        placeholder="e.g., How effective is Seeking Alpha investment advice in 2025?",
                        lines=2
                    )

                    with gr.Row():
                        research_depth_slider = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=1,
                            step=1,
                            label="Research Depth",
                            info="1=Quick (default), 2=Standard with follow-ups, 3=Comprehensive"
                        )
                        research_max_searches = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=6,
                            step=3,
                            label="Max Searches",
                            info="Total searches across all depths"
                        )

                    with gr.Row():
                        research_submit_btn = gr.Button("🚀 Start Research", variant="primary", scale=2)
                        research_clear_btn = gr.Button("🗑️ Clear", scale=1)

                with gr.Column(scale=1):
                    research_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter a query and click **Start Research**",
                        label="Status"
                    )

            research_output = gr.Markdown(
                label="Research Report",
                value="*Your research report will appear here...*"
            )

            # Export and Copy buttons
            with gr.Row():
                research_copy_btn = gr.Button("📋 Copy Report", scale=1)
                research_pdf_btn = gr.Button("📄 Download PDF", scale=1)
                research_pdf_file = gr.File(label="PDF Download", visible=True, interactive=False)

            # Report History
            with gr.Accordion("📚 Report History", open=False):
                research_history_dropdown = gr.Dropdown(
                    label="Previous Reports",
                    choices=[],
                    interactive=True
                )
                research_history_load_btn = gr.Button("Load Selected Report")

            # Deep Research button actions
            research_submit_btn.click(
                fn=run_research_with_progress,
                inputs=[query_input, research_depth_slider, research_max_searches],
                outputs=[research_status, research_output]
            )

            research_clear_btn.click(
                lambda: ("", 1, 6, "### ⏳ Ready\n\nEnter a query and click **Start Research**", "*Your research report will appear here...*"),
                outputs=[query_input, research_depth_slider, research_max_searches, research_status, research_output]
            )

            # Copy to clipboard (uses JavaScript)
            research_copy_btn.click(
                fn=lambda report: report,
                inputs=[research_output],
                outputs=[],
                js="(report) => { navigator.clipboard.writeText(report); alert('Report copied to clipboard!'); return []; }"
            )

            # PDF Export
            def export_research_pdf(report, query):
                if not report or report.startswith("*"):
                    return None
                return export_report_to_pdf(report, f"Research: {query[:50]}")

            research_pdf_btn.click(
                fn=export_research_pdf,
                inputs=[research_output, query_input],
                outputs=[research_pdf_file]
            )

            with gr.Accordion("📋 Instructions & Requirements", open=False):
                gr.Markdown("""
                ### How to Use
                1. Enter your research question
                2. Click "Start Research"
                3. Watch the progress indicators
                4. Wait for the report
                5. Use **Copy** or **Download PDF** to save

                ### Requirements
                - **Anthropic API key** in `.env` (required)
                - **Tavily API key** in `.env` (optional - enables real-time web search)

                ### Example Queries
                - "AI trends in 2026"
                - "Sustainable energy investment opportunities"
                - "Remote work productivity tools comparison"
                """)

        # ========== AI Research Tab ==========
        with gr.Tab("🤖 AI Research"):
            gr.Markdown("""
            ### Specialized AI Information Research Agent

            This advanced research agent is optimized for AI/ML topics with:
            - **Recursive Research** - Follows up on gaps and unanswered questions
            - **Domain Prioritization** - Favors authoritative sources (arXiv, AI labs, etc.)
            - **Inline Citations** - References with [N] markers and source tiers
            - **Parallel Execution** - Faster searches with concurrency control

            Ideal for: AI capabilities, model comparisons, enterprise AI, policy/governance, technical deep-dives.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    ai_query_input = gr.Textbox(
                        label="🎯 Research Query",
                        placeholder="e.g., Latest advances in reasoning models and their enterprise applications",
                        lines=2
                    )

                    with gr.Row():
                        ai_depth_slider = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=2,
                            step=1,
                            label="Research Depth",
                            info="1=Quick, 2=Standard, 3=Comprehensive"
                        )
                        ai_max_searches = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=15,
                            step=5,
                            label="Max Searches",
                            info="Higher = more thorough but slower"
                        )

                    with gr.Row():
                        ai_submit_btn = gr.Button("🚀 Start AI Research", variant="primary", scale=2)
                        ai_clear_btn = gr.Button("🗑️ Clear", scale=1)

                    gr.Examples(
                        examples=[
                            "Latest advances in reasoning models and their enterprise applications",
                            "How are companies implementing agentic AI workflows in 2026?",
                            "State of AI governance and regulation across US, EU, and China",
                            "Open-source vs proprietary AI models: current landscape and trends",
                            "Multimodal AI capabilities and limitations in production systems",
                        ],
                        inputs=ai_query_input,
                        label="Example Queries"
                    )

                with gr.Column(scale=1):
                    ai_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter a query and click **Start AI Research**",
                        label="Status"
                    )

            ai_output = gr.Markdown(
                label="Research Report",
                value="*Your AI research report will appear here...*"
            )

            # Export buttons
            with gr.Row():
                # Show different label based on Drive configuration
                drive_btn_label = "📁 Save to Drive" if is_drive_configured() else "📁 Save to Drive (Not configured)"
                ai_drive_btn = gr.Button(drive_btn_label, scale=1)
                ai_pdf_btn = gr.Button("📄 Download PDF", scale=1)
                ai_pdf_file = gr.File(label="PDF Download", visible=True, interactive=False)

            # Drive save status
            ai_drive_status = gr.Markdown(value="", visible=True)

            # AI Research button actions
            ai_submit_btn.click(
                fn=run_ai_research,
                inputs=[ai_query_input, ai_depth_slider, ai_max_searches],
                outputs=[ai_status, ai_output]
            )

            ai_clear_btn.click(
                lambda: ("", 2, 15, "### ⏳ Ready\n\nEnter a query and click **Start AI Research**", "*Your AI research report will appear here...*", ""),
                outputs=[ai_query_input, ai_depth_slider, ai_max_searches, ai_status, ai_output, ai_drive_status]
            )

            # Save to Google Drive
            ai_drive_btn.click(
                fn=save_report_to_drive,
                inputs=[ai_output, ai_query_input],
                outputs=[ai_drive_status]
            )

            # PDF Export
            def export_ai_pdf(report, query):
                if not report or report.startswith("*"):
                    return None
                return export_report_to_pdf(report, f"AI Research: {query[:50]}")

            ai_pdf_btn.click(
                fn=export_ai_pdf,
                inputs=[ai_output, ai_query_input],
                outputs=[ai_pdf_file]
            )

            with gr.Accordion("📋 Source Tiers & Features", open=False):
                gr.Markdown("""
                ### Source Prioritization Tiers

                | Tier | Description | Examples |
                |------|-------------|----------|
                | 🎓 **Tier 1** | Academic & AI Labs | arXiv, OpenAI, DeepMind, Anthropic, MIT, Stanford |
                | 📊 **Tier 2** | Strategic Intelligence | Bloomberg, McKinsey, Gartner, The Information |
                | 📜 **Tier 3** | Policy & Governance | AI Now Institute, Future of Life, NIST, EU AI Office |
                | 💻 **Tier 4** | Practitioner | Hugging Face, Towards Data Science, AI blogs |

                ### Research Depth Levels

                - **Depth 1**: Quick overview with 5-8 initial searches
                - **Depth 2**: Standard research with follow-up queries on gaps
                - **Depth 3**: Comprehensive with multiple rounds of follow-ups

                ### Features

                - **Recursive Research**: Automatically identifies knowledge gaps and generates follow-up queries
                - **Circuit Breakers**: Prevents runaway recursion with configurable limits
                - **Inline Citations**: Reports include [N] markers linking to references
                - **Parallel Execution**: Up to 5 concurrent searches with rate limiting
                """)

        # ========== Stock Research Tab ==========
        with gr.Tab("📈 Stock Research"):
            gr.Markdown("""
            ### AI-Powered Comprehensive Stock Analysis

            Enter a stock ticker to generate a detailed research report including:
            - **Investment Thesis** (Bull/Bear case, recommendation)
            - **Financial Metrics** (Valuation, profitability, health)
            - **Analyst Opinions** (Ratings, price targets, sentiment)
            - **Latest News** (with source links)
            - **SEC Filings** (10-K, 10-Q, 8-K with direct links)
            - **Insider Activity** (Recent transactions)
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    ticker_input = gr.Textbox(
                        label="📌 Stock Ticker",
                        placeholder="e.g., AAPL, TSLA, MSFT, NVDA",
                        max_lines=1,
                        scale=2
                    )

                    with gr.Row():
                        stock_deep_analysis = gr.Checkbox(
                            label="Deep Analysis",
                            value=False,
                            info="Identify gaps and conduct follow-up research"
                        )

                    with gr.Row():
                        stock_submit_btn = gr.Button("🚀 Generate Report", variant="primary", scale=2)
                        stock_clear_btn = gr.Button("🗑️ Clear", scale=1)

                    gr.Examples(
                        examples=["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
                        inputs=ticker_input,
                        label="Popular Tickers"
                    )

                with gr.Column(scale=1):
                    stock_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter a ticker symbol and click **Generate Report**",
                        label="Status"
                    )

            stock_output = gr.Markdown(
                label="Research Report",
                value="*Your comprehensive stock research report will appear here...*"
            )

            # Export and Copy buttons
            with gr.Row():
                stock_copy_btn = gr.Button("📋 Copy Report", scale=1)
                stock_pdf_btn = gr.Button("📄 Download PDF", scale=1)
                stock_pdf_file = gr.File(label="PDF Download", visible=True, interactive=False)

            # Watchlist Section
            with gr.Accordion("⭐ Watchlist", open=False):
                gr.Markdown("*Save tickers for quick access. Add from current ticker or type a new one.*")
                with gr.Row():
                    watchlist_add_input = gr.Textbox(
                        label="Add Ticker",
                        placeholder="AAPL",
                        max_lines=1,
                        scale=2
                    )
                    watchlist_add_btn = gr.Button("➕ Add", scale=1)
                    watchlist_add_current_btn = gr.Button("⭐ Add Current", scale=1)

                watchlist_display = gr.Markdown(value="*No tickers in watchlist*")

                with gr.Row():
                    watchlist_remove_input = gr.Textbox(
                        label="Remove Ticker",
                        placeholder="AAPL",
                        max_lines=1,
                        scale=2
                    )
                    watchlist_remove_btn = gr.Button("➖ Remove", scale=1)
                    watchlist_clear_btn = gr.Button("🗑️ Clear All", scale=1)

                watchlist_status = gr.Markdown(value="")

            # Report History
            with gr.Accordion("📚 Report History", open=False):
                stock_history_dropdown = gr.Dropdown(
                    label="Previous Reports",
                    choices=[],
                    interactive=True
                )
                stock_history_load_btn = gr.Button("Load Selected Report")

            # Stock Research button actions
            stock_submit_btn.click(
                fn=run_stock_research,
                inputs=[ticker_input, stock_deep_analysis],
                outputs=[stock_status, stock_output]
            )

            stock_clear_btn.click(
                lambda: ("", False, "### ⏳ Ready\n\nEnter a ticker symbol and click **Generate Report**", "*Your comprehensive stock research report will appear here...*"),
                outputs=[ticker_input, stock_deep_analysis, stock_status, stock_output]
            )

            ticker_input.submit(
                fn=run_stock_research,
                inputs=[ticker_input, stock_deep_analysis],
                outputs=[stock_status, stock_output]
            )

            # Copy to clipboard
            stock_copy_btn.click(
                fn=lambda report: report,
                inputs=[stock_output],
                outputs=[],
                js="(report) => { navigator.clipboard.writeText(report); alert('Report copied to clipboard!'); return []; }"
            )

            # PDF Export
            def export_stock_pdf(report, ticker):
                if not report or report.startswith("*"):
                    return None
                return export_report_to_pdf(report, f"Stock Analysis: {ticker}")

            stock_pdf_btn.click(
                fn=export_stock_pdf,
                inputs=[stock_output, ticker_input],
                outputs=[stock_pdf_file]
            )

            # Watchlist handlers (simplified - uses default session)
            def handle_watchlist_add(ticker):
                return add_to_watchlist(ticker, "default")

            def handle_watchlist_add_current(current_ticker):
                if current_ticker:
                    return add_to_watchlist(current_ticker, "default")
                return "No ticker entered", get_watchlist_display("default")

            def handle_watchlist_remove(ticker):
                return remove_from_watchlist(ticker, "default")

            def handle_watchlist_clear():
                return clear_watchlist("default")

            watchlist_add_btn.click(
                fn=handle_watchlist_add,
                inputs=[watchlist_add_input],
                outputs=[watchlist_status, watchlist_display]
            )

            watchlist_add_current_btn.click(
                fn=handle_watchlist_add_current,
                inputs=[ticker_input],
                outputs=[watchlist_status, watchlist_display]
            )

            watchlist_remove_btn.click(
                fn=handle_watchlist_remove,
                inputs=[watchlist_remove_input],
                outputs=[watchlist_status, watchlist_display]
            )

            watchlist_clear_btn.click(
                fn=handle_watchlist_clear,
                outputs=[watchlist_status, watchlist_display]
            )

            with gr.Accordion("📋 Data Sources & API Requirements", open=False):
                gr.Markdown("""
                ### Data Sources Used

                | Source | Data Provided | API Key Required |
                |--------|---------------|------------------|
                | **Yahoo Finance** | Price, ratios, institutional holders | No |
                | **Finnhub** | News, sentiment, analyst ratings, insider trades | Yes (free tier: 60/min) |
                | **SEC EDGAR** | 10-K, 10-Q, 8-K filings | No |
                | **Alpha Vantage** | Financials, company overview | Yes (free tier: 25/day) |
                | **Tavily** | Additional news search | Yes |

                ### Setup

                1. Get free API keys:
                   - [Finnhub](https://finnhub.io/register) - Recommended for news & analyst data
                   - [Alpha Vantage](https://www.alphavantage.co/support/#api-key) - For detailed financials
                   - [Tavily](https://tavily.com) - For web search (optional)

                2. Add to your `.env` file:
                ```
                FINNHUB_API_KEY=your_key_here
                ALPHA_VANTAGE_API_KEY=your_key_here
                TAVILY_API_KEY=your_key_here
                ```

                **Note:** The agent works with partial data. If some APIs are unavailable, it will use available sources.
                """)

        # ========== Stock Comparison Tab ==========
        with gr.Tab("⚖️ Compare Stocks"):
            gr.Markdown("""
            ### Side-by-Side Stock Comparison

            Compare 2-3 stocks with key metrics in a table format.
            For detailed analysis, use the **Stock Research** tab for individual reports.
            """)

            with gr.Row():
                compare_ticker1 = gr.Textbox(
                    label="Stock 1",
                    placeholder="e.g., AAPL",
                    max_lines=1
                )
                compare_ticker2 = gr.Textbox(
                    label="Stock 2",
                    placeholder="e.g., MSFT",
                    max_lines=1
                )
                compare_ticker3 = gr.Textbox(
                    label="Stock 3 (optional)",
                    placeholder="e.g., GOOGL",
                    max_lines=1
                )

            compare_btn = gr.Button("⚖️ Compare Stocks", variant="primary")

            compare_status = gr.Markdown(
                value="### ⏳ Ready\n\nEnter 2-3 ticker symbols to compare"
            )

            compare_output = gr.Markdown(
                value="*Comparison table will appear here...*"
            )

            # Quick comparison examples
            gr.Markdown("**Quick Comparisons:**")
            with gr.Row():
                gr.Button("FAANG").click(
                    lambda: ("META", "AAPL", "GOOGL"),
                    outputs=[compare_ticker1, compare_ticker2, compare_ticker3]
                )
                gr.Button("Tech Giants").click(
                    lambda: ("MSFT", "AAPL", "NVDA"),
                    outputs=[compare_ticker1, compare_ticker2, compare_ticker3]
                )
                gr.Button("EVs").click(
                    lambda: ("TSLA", "RIVN", "F"),
                    outputs=[compare_ticker1, compare_ticker2, compare_ticker3]
                )
                gr.Button("Chips").click(
                    lambda: ("NVDA", "AMD", "INTC"),
                    outputs=[compare_ticker1, compare_ticker2, compare_ticker3]
                )

            compare_btn.click(
                fn=run_stock_comparison,
                inputs=[compare_ticker1, compare_ticker2, compare_ticker3],
                outputs=[compare_status, compare_output]
            )

        # ========== Sector Research Tab ==========
        with gr.Tab("🏭 Sector Research"):
            gr.Markdown("""
            ### Sector-Wide Analysis

            Analyze an entire sector: top companies, industry trends, competitive dynamics, and investment outlook.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    sector_dropdown = gr.Dropdown(
                        label="Select Sector",
                        choices=get_available_sectors(),
                        value="Technology",
                        interactive=True
                    )

                    sector_submit_btn = gr.Button("🏭 Analyze Sector", variant="primary")

                    # Sector quick info
                    gr.Markdown("**Available Sectors:** Technology, Semiconductors, Healthcare, Financial Services, Energy, Consumer Discretionary, Industrials, Real Estate")

                with gr.Column(scale=1):
                    sector_status = gr.Markdown(
                        value="### ⏳ Ready\n\nSelect a sector and click **Analyze Sector**",
                        label="Status"
                    )

            sector_output = gr.Markdown(
                value="*Sector analysis report will appear here...*"
            )

            sector_submit_btn.click(
                fn=run_sector_research,
                inputs=[sector_dropdown],
                outputs=[sector_status, sector_output]
            )

        # ========== Competitor Intelligence Tab ==========
        with gr.Tab("🎯 Competitors"):
            gr.Markdown("""
            ### Competitive Intelligence

            Enter a stock ticker to automatically identify competitors and generate a competitive analysis.
            Includes market position, competitive advantages, and strategic recommendations.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    competitor_ticker_input = gr.Textbox(
                        label="Stock Ticker",
                        placeholder="e.g., NVDA, AAPL, TSLA",
                        max_lines=1
                    )

                    competitor_submit_btn = gr.Button("🎯 Analyze Competitors", variant="primary")

                    gr.Examples(
                        examples=["NVDA", "AAPL", "TSLA", "JPM", "AMZN", "NFLX"],
                        inputs=competitor_ticker_input,
                        label="Popular Tickers"
                    )

                with gr.Column(scale=1):
                    competitor_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter a ticker to analyze its competitors",
                        label="Status"
                    )

            competitor_output = gr.Markdown(
                value="*Competitive analysis will appear here...*"
            )

            competitor_submit_btn.click(
                fn=run_competitor_analysis,
                inputs=[competitor_ticker_input],
                outputs=[competitor_status, competitor_output]
            )

            competitor_ticker_input.submit(
                fn=run_competitor_analysis,
                inputs=[competitor_ticker_input],
                outputs=[competitor_status, competitor_output]
            )

        # ========== Portfolio Analyzer Tab ==========
        with gr.Tab("💼 Portfolio"):
            gr.Markdown("""
            ### Portfolio Analyzer

            Enter your portfolio holdings to get:
            - **Sector Exposure Analysis** - Understand your diversification
            - **Risk Assessment** - Concentration warnings and risk metrics
            - **Rebalancing Suggestions** - AI-powered recommendations
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Enter holdings (one per line):** `TICKER SHARES [COST_BASIS]`")

                    portfolio_input = gr.Textbox(
                        label="Portfolio Holdings",
                        placeholder="AAPL 50\nMSFT 30\nGOOGL 20 150.00\nNVDA 15\nAMZN 25",
                        lines=8
                    )

                    with gr.Row():
                        portfolio_submit_btn = gr.Button("💼 Analyze Portfolio", variant="primary", scale=2)
                        portfolio_clear_btn = gr.Button("🗑️ Clear", scale=1)

                    # CSV Upload option
                    with gr.Accordion("📁 Or Upload CSV", open=False):
                        gr.Markdown("CSV format: `ticker,shares,cost_basis` (cost_basis optional)")
                        portfolio_file = gr.File(
                            label="Upload Portfolio CSV",
                            file_types=[".csv"],
                        )
                        portfolio_csv_btn = gr.Button("📁 Analyze CSV")

                with gr.Column(scale=1):
                    portfolio_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter your holdings and click **Analyze Portfolio**",
                        label="Status"
                    )

            portfolio_output = gr.Markdown(
                value="*Portfolio analysis will appear here...*"
            )

            # Example portfolios
            with gr.Accordion("📋 Example Portfolios", open=False):
                gr.Markdown("Click to load an example portfolio:")
                with gr.Row():
                    gr.Button("Tech Heavy").click(
                        lambda: "AAPL 50\nMSFT 40\nGOOGL 30\nNVDA 25\nMETA 20",
                        outputs=[portfolio_input]
                    )
                    gr.Button("Balanced").click(
                        lambda: "AAPL 30\nJPM 25\nJNJ 25\nXOM 20\nPG 20\nVZ 15\nHD 15",
                        outputs=[portfolio_input]
                    )
                    gr.Button("Dividend").click(
                        lambda: "JNJ 40\nPG 35\nKO 30\nPEP 30\nVZ 25\nT 25\nO 20",
                        outputs=[portfolio_input]
                    )

            portfolio_submit_btn.click(
                fn=run_portfolio_analysis,
                inputs=[portfolio_input],
                outputs=[portfolio_status, portfolio_output]
            )

            portfolio_clear_btn.click(
                lambda: ("", "### ⏳ Ready\n\nEnter your holdings and click **Analyze Portfolio**", "*Portfolio analysis will appear here...*"),
                outputs=[portfolio_input, portfolio_status, portfolio_output]
            )

            portfolio_csv_btn.click(
                fn=run_portfolio_csv_analysis,
                inputs=[portfolio_file],
                outputs=[portfolio_status, portfolio_output]
            )

        # ========== Earnings Calendar Tab ==========
        with gr.Tab("📅 Earnings"):
            gr.Markdown("""
            ### Earnings Calendar & Analysis

            Track upcoming earnings dates, analyze historical beat/miss rates, and get pre-earnings sentiment analysis.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    earnings_tickers = gr.Textbox(
                        label="Ticker Symbols",
                        placeholder="AAPL, MSFT, NVDA, GOOGL",
                        info="Enter multiple tickers separated by commas or spaces"
                    )

                    with gr.Row():
                        earnings_sentiment = gr.Checkbox(
                            label="Include Sentiment Analysis",
                            value=True,
                            info="Adds news sentiment (slower but more insightful)"
                        )

                    with gr.Row():
                        earnings_submit_btn = gr.Button("📅 Analyze Earnings", variant="primary", scale=2)
                        earnings_clear_btn = gr.Button("🗑️ Clear", scale=1)

                with gr.Column(scale=1):
                    earnings_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter ticker symbols and click **Analyze Earnings**",
                        label="Status"
                    )

            earnings_output = gr.Markdown(
                value="*Earnings analysis will appear here...*"
            )

            # Quick watchlist buttons
            with gr.Accordion("📋 Quick Watchlists", open=False):
                gr.Markdown("Click to load a watchlist:")
                with gr.Row():
                    gr.Button("Tech Giants").click(
                        lambda: "AAPL, MSFT, GOOGL, AMZN, META, NVDA",
                        outputs=[earnings_tickers]
                    )
                    gr.Button("Financials").click(
                        lambda: "JPM, BAC, WFC, GS, MS, C",
                        outputs=[earnings_tickers]
                    )
                    gr.Button("Semiconductors").click(
                        lambda: "NVDA, AMD, INTC, TSM, AVGO, QCOM",
                        outputs=[earnings_tickers]
                    )

            earnings_submit_btn.click(
                fn=run_earnings_analysis,
                inputs=[earnings_tickers, earnings_sentiment],
                outputs=[earnings_status, earnings_output]
            )

            earnings_clear_btn.click(
                lambda: ("", "### ⏳ Ready\n\nEnter ticker symbols and click **Analyze Earnings**", "*Earnings analysis will appear here...*"),
                outputs=[earnings_tickers, earnings_status, earnings_output]
            )

        # ========== Commodities Tab ==========
        with gr.Tab("🛢️ Commodities"):
            gr.Markdown("""
            ### AI-Powered Commodity & Futures Research

            Enter a commodity name or futures symbol to generate a detailed market analysis:
            - **Market Outlook** (Bullish/Neutral/Bearish with confidence)
            - **Bull & Bear Cases** with catalysts and risks
            - **Price Action** (spot, range, performance)
            - **Macro Environment** (USD, yields, inflation from FRED)
            - **Supply & Demand Assessment**
            - **Latest News** (commodity-specific search)
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    commodity_input = gr.Textbox(
                        label="🛢️ Commodity",
                        placeholder="e.g., gold, crude, copper, natgas, GC=F",
                        max_lines=1,
                        scale=2
                    )

                    with gr.Row():
                        commodity_submit_btn = gr.Button("🚀 Generate Report", variant="primary", scale=2)
                        commodity_clear_btn = gr.Button("🗑️ Clear", scale=1)

                    gr.Examples(
                        examples=["gold", "silver", "crude", "natgas", "copper", "corn", "wheat"],
                        inputs=commodity_input,
                        label="Popular Commodities"
                    )

                with gr.Column(scale=1):
                    commodity_status = gr.Markdown(
                        value="### ⏳ Ready\n\nEnter a commodity and click **Generate Report**",
                        label="Status"
                    )

            commodity_output = gr.Markdown(
                label="Research Report",
                value="*Your commodity research report will appear here...*"
            )

            # Export buttons
            with gr.Row():
                commodity_copy_btn = gr.Button("📋 Copy Report", scale=1)
                commodity_pdf_btn = gr.Button("📄 Download PDF", scale=1)
                commodity_pdf_file = gr.File(label="PDF Download", visible=True, interactive=False)

            # Event handlers
            commodity_submit_btn.click(
                fn=run_commodity_research,
                inputs=[commodity_input],
                outputs=[commodity_status, commodity_output]
            )

            commodity_input.submit(
                fn=run_commodity_research,
                inputs=[commodity_input],
                outputs=[commodity_status, commodity_output]
            )

            commodity_clear_btn.click(
                lambda: ("", "### ⏳ Ready\n\nEnter a commodity and click **Generate Report**", "*Your commodity research report will appear here...*"),
                outputs=[commodity_input, commodity_status, commodity_output]
            )

            # Copy button (clipboard JS)
            commodity_copy_btn.click(
                fn=None,
                inputs=[commodity_output],
                js="(text) => { navigator.clipboard.writeText(text); }"
            )

            # PDF export
            def export_commodity_pdf(report_md):
                if not report_md or report_md.startswith("*Your"):
                    return None
                try:
                    filename = generate_report_filename("commodity", "report")
                    pdf_path = markdown_to_pdf(report_md, filename)
                    return pdf_path
                except Exception as e:
                    logger.error(f"PDF export error: {e}")
                    return None

            commodity_pdf_btn.click(
                fn=export_commodity_pdf,
                inputs=[commodity_output],
                outputs=[commodity_pdf_file]
            )

        # ========== Alerts Tab ==========
        with gr.Tab("🔔 Alerts"):
            gr.Markdown("""
            ### Price & Earnings Alerts

            Set up alerts to be notified when stocks hit your price targets or when earnings dates approach.

            **Important:**
            - Alerts are checked when you click **"Check Alerts Now"** button
            - Email notifications require `RESEND_API_KEY` in your `.env` file
            - With Resend free tier, emails can only be sent to your verified email address
            """)

            with gr.Row():
                # Left column - Create alerts
                with gr.Column(scale=1):
                    gr.Markdown("#### Create New Alert")

                    with gr.Tab("Price Alert"):
                        price_alert_ticker = gr.Textbox(
                            label="Ticker",
                            placeholder="AAPL"
                        )
                        price_alert_condition = gr.Radio(
                            label="Condition",
                            choices=["above", "below"],
                            value="above"
                        )
                        price_alert_target = gr.Number(
                            label="Target Price ($)",
                            value=150.00
                        )
                        price_alert_email = gr.Textbox(
                            label="Email",
                            placeholder="your@email.com"
                        )
                        price_alert_btn = gr.Button("📈 Create Price Alert", variant="primary")

                    with gr.Tab("Earnings Alert"):
                        earnings_alert_ticker = gr.Textbox(
                            label="Ticker",
                            placeholder="AAPL"
                        )
                        earnings_alert_days = gr.Slider(
                            label="Days Before Earnings",
                            minimum=1,
                            maximum=14,
                            value=3,
                            step=1
                        )
                        earnings_alert_email = gr.Textbox(
                            label="Email",
                            placeholder="your@email.com"
                        )
                        earnings_alert_btn = gr.Button("📅 Create Earnings Alert", variant="primary")

                # Right column - Manage alerts
                with gr.Column(scale=1):
                    gr.Markdown("#### Manage Alerts")

                    alert_status = gr.Markdown(
                        value="### Ready\n\nCreate an alert using the form on the left."
                    )

                    alerts_display = gr.Markdown(
                        value=get_alerts_display()
                    )

                    with gr.Row():
                        cancel_alert_id = gr.Textbox(
                            label="Alert ID to Cancel",
                            placeholder="1"
                        )
                        cancel_alert_btn = gr.Button("❌ Cancel Alert")

                    with gr.Row():
                        check_alerts_btn = gr.Button("🔄 Check Alerts Now", variant="secondary")
                        refresh_alerts_btn = gr.Button("🔃 Refresh List")

                    # Test email section
                    with gr.Accordion("📧 Test Email Configuration", open=False):
                        test_email_input = gr.Textbox(
                            label="Email Address",
                            placeholder="your@email.com",
                            info="Send a test email to verify your setup"
                        )
                        test_email_btn = gr.Button("📧 Send Test Email")
                        test_email_status = gr.Markdown(value="")

            # Event handlers
            price_alert_btn.click(
                fn=create_new_price_alert,
                inputs=[price_alert_ticker, price_alert_condition, price_alert_target, price_alert_email],
                outputs=[alert_status, alerts_display]
            )

            earnings_alert_btn.click(
                fn=create_new_earnings_alert,
                inputs=[earnings_alert_ticker, earnings_alert_days, earnings_alert_email],
                outputs=[alert_status, alerts_display]
            )

            cancel_alert_btn.click(
                fn=cancel_alert_by_id,
                inputs=[cancel_alert_id],
                outputs=[alert_status, alerts_display]
            )

            check_alerts_btn.click(
                fn=run_alert_check,
                outputs=[alert_status, alerts_display]
            )

            refresh_alerts_btn.click(
                fn=lambda: ("### ✅ Refreshed", get_alerts_display()),
                outputs=[alert_status, alerts_display]
            )

            test_email_btn.click(
                fn=run_test_email,
                inputs=[test_email_input],
                outputs=[test_email_status]
            )

    # Footer
    gr.Markdown("""
    ---
    **Note:** This hub uses Anthropic's Claude API and various data sources.
    Each research uses approximately 5,000-20,000 tokens (~$0.05-$0.20) plus API costs if enabled.

    *Stock research disclaimer: Reports are for informational purposes only, not financial advice.*
    """)


# Launch the app
def get_external_ip() -> str:
    """Fetch the VM's external IP address."""
    import urllib.request
    services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://checkip.amazonaws.com",
    ]
    for url in services:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.read().decode().strip()
        except Exception:
            continue
    return "unknown"


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("RESEARCH AGENT HUB")
    logger.info("=" * 70)

    # Detect external IP for GCP VMs with non-static IPs
    external_ip = get_external_ip()

    # Check for authentication credentials
    gradio_username = os.getenv("GRADIO_USERNAME", "").strip()
    gradio_password = os.getenv("GRADIO_PASSWORD", "").strip()

    # Set up authentication if credentials are provided
    auth = None
    if gradio_username and gradio_password:
        auth = (gradio_username, gradio_password)
        logger.info("Authentication: ENABLED (username/password required)")
    else:
        logger.info("Authentication: DISABLED (set GRADIO_USERNAME and GRADIO_PASSWORD in .env to enable)")

    logger.info(f"Rate Limiting: {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds")
    logger.info("Starting unified interface on port 7860...")
    logger.info(f"Local:    http://0.0.0.0:7860")
    logger.info(f"External: http://{external_ip}:7860")
    logger.info("=" * 70)

    # SSL for HTTPS (required for clipboard API and clean PDF downloads)
    ssl_certfile = os.path.join(os.path.dirname(__file__), "certs", "cert.pem")
    ssl_keyfile = os.path.join(os.path.dirname(__file__), "certs", "key.pem")
    ssl_kwargs = {}
    if os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
        ssl_kwargs = {
            "ssl_certfile": ssl_certfile,
            "ssl_keyfile": ssl_keyfile,
            "ssl_verify": False,
        }
        logger.info("SSL: ENABLED (HTTPS)")
    else:
        logger.info("SSL: DISABLED (HTTP only — clipboard and PDF download may not work in Chrome)")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        auth=auth,
        allowed_paths=["/tmp"],
        **ssl_kwargs,
    )
