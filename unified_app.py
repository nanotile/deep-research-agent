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
from deep_research_agent import (
    deep_research_with_progress,
    ProgressUpdate,
    tavily_client
)

# Import Stock Research components
from stock_research_agent import stock_research_with_progress
from stock_data_models import StockProgressUpdate

# Import utilities
from utils.validators import sanitize_ticker, sanitize_query
from utils.rate_limiter import RateLimiter
from utils.logging_config import get_logger, setup_logging
from utils.pdf_export import markdown_to_pdf, generate_report_filename
from utils.report_history import add_to_history, get_history_choices, get_report_content, report_history

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

    lines.append(f"\n---\n**Total elapsed:** {total_elapsed:.1f}s")

    return "\n".join(lines)


def run_research_with_progress(query: str, request: gr.Request = None):
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

    logger.info(f"Starting deep research for query: {sanitized_query[:50]}...")

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in deep_research_with_progress(query=sanitized_query):
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


def run_stock_research(ticker: str, request: gr.Request = None):
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

    logger.info(f"Starting stock research for ticker: {sanitized_ticker}")

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in stock_research_with_progress(sanitized_ticker):
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
                yield status, update.report
                break
            else:
                status = format_stock_progress(update, total_elapsed)
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
                research_pdf_download = gr.File(label="PDF Download", visible=False)

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
                inputs=[query_input],
                outputs=[research_status, research_output]
            )

            research_clear_btn.click(
                lambda: ("", "### ⏳ Ready\n\nEnter a query and click **Start Research**", "*Your research report will appear here...*"),
                outputs=[query_input, research_status, research_output]
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
                outputs=[research_pdf_download]
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
                stock_pdf_download = gr.File(label="PDF Download", visible=False)

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
                inputs=[ticker_input],
                outputs=[stock_status, stock_output]
            )

            stock_clear_btn.click(
                lambda: ("", "### ⏳ Ready\n\nEnter a ticker symbol and click **Generate Report**", "*Your comprehensive stock research report will appear here...*"),
                outputs=[ticker_input, stock_status, stock_output]
            )

            ticker_input.submit(
                fn=run_stock_research,
                inputs=[ticker_input],
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
                outputs=[stock_pdf_download]
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

    # Footer
    gr.Markdown("""
    ---
    **Note:** This hub uses Anthropic's Claude API and various data sources.
    Each research uses approximately 5,000-20,000 tokens (~$0.05-$0.20) plus API costs if enabled.

    *Stock research disclaimer: Reports are for informational purposes only, not financial advice.*
    """)


# Launch the app
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("RESEARCH AGENT HUB")
    logger.info("=" * 70)

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
    logger.info("Access at: http://0.0.0.0:7860")
    logger.info("=" * 70)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        auth=auth,
    )
