"""
Unified Research Agent Hub - Gradio Web Interface
Combines Deep Research Agent and Stock Research Agent in a single tabbed interface
"""

import gradio as gr
import asyncio
import time
from threading import Thread
from queue import Queue

# Import Deep Research components
from deep_research_agent import (
    deep_research_with_progress,
    ProgressUpdate,
    tavily_client
)

# Import Stock Research components
from stock_research_agent import stock_research_with_progress
from stock_data_models import StockProgressUpdate


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


def run_research_with_progress(query: str):
    """Generator function that yields progress updates and final report for Deep Research."""
    if not query or query.strip() == "":
        yield "### ⚠️ Input Required\n\nPlease enter a research query!", "*Your research report will appear here...*"
        return

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in deep_research_with_progress(query=query.strip()):
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


def run_stock_research(ticker: str):
    """Generator function that yields progress updates and final report for Stock Research."""
    if not ticker or ticker.strip() == "":
        yield (
            "### ⚠️ Input Required\n\nPlease enter a stock ticker symbol!",
            "*Enter a ticker like AAPL, TSLA, or MSFT to generate a research report...*"
        )
        return

    ticker = ticker.upper().strip()

    progress_queue = Queue()

    def run_async_research():
        async def async_wrapper():
            async for update in stock_research_with_progress(ticker):
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
    report = f"*Researching {ticker}...*"

    yield (
        f"### ⏳ Starting Research\n\nInitializing research for **{ticker}**...",
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
# Unified Gradio Interface with Tabs
# ============================================================

search_mode = "Tavily web search" if tavily_client else "LLM knowledge base"

with gr.Blocks(title="Research Agent Hub", theme=gr.themes.Soft()) as demo:

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

            with gr.Accordion("📋 Instructions & Requirements", open=False):
                gr.Markdown("""
                ### How to Use
                1. Enter your research question
                2. Click "Start Research"
                3. Watch the progress indicators
                4. Wait for the report

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

    # Footer
    gr.Markdown("""
    ---
    **Note:** This hub uses Anthropic's Claude API and various data sources.
    Each research uses approximately 5,000-20,000 tokens (~$0.05-$0.20) plus API costs if enabled.

    *Stock research disclaimer: Reports are for informational purposes only, not financial advice.*
    """)


# Launch the app
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎯 RESEARCH AGENT HUB")
    print("=" * 70)
    print("Starting unified interface on port 7860...")
    print("Access at: http://0.0.0.0:7860")
    print("=" * 70 + "\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
