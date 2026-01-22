"""
Gradio Web Interface for Stock Research Agent with Progress Indicators
"""

import gradio as gr
import asyncio
import time
from threading import Thread
from queue import Queue
from stock_research_agent import stock_research_with_progress
from stock_data_models import StockProgressUpdate


def format_progress_display(update: StockProgressUpdate, total_elapsed: float) -> str:
    """Format progress update for display in status box"""
    # Stage icons
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

    # Show source status if fetching
    if update.source_status:
        source_icons = {"success": "✅", "failed": "❌", "pending": "⏳"}
        source_lines = []
        for source, status in update.source_status.items():
            source_name = source.replace('_', ' ').title()
            source_lines.append(f"  {source_icons.get(status, '⏳')} {source_name}")
        lines.append("\n**Data Sources:**\n" + "\n".join(source_lines))

    # Stage time
    if update.elapsed_time > 0:
        lines.append(f"\n**Stage time:** {update.elapsed_time:.1f}s")

    lines.append(f"\n---\n**Total elapsed:** {total_elapsed:.1f}s")

    return "\n".join(lines)


def run_stock_research(ticker: str):
    """
    Generator function that yields progress updates and final report.
    Uses a background thread to run async code while yielding updates.
    """
    # Validate input
    if not ticker or ticker.strip() == "":
        yield (
            "### ⚠️ Input Required\n\nPlease enter a stock ticker symbol!",
            "*Enter a ticker like AAPL, TSLA, or MSFT to generate a research report...*"
        )
        return

    ticker = ticker.upper().strip()

    # Queue for communication between async task and generator
    progress_queue = Queue()

    def run_async_research():
        """Run the async research in a separate thread"""
        async def async_wrapper():
            async for update in stock_research_with_progress(ticker):
                progress_queue.put(update)

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_wrapper())
        except Exception as e:
            progress_queue.put(Exception(str(e)))
        finally:
            loop.close()

    # Start async task in background thread
    thread = Thread(target=run_async_research)
    thread.start()

    # Track overall timing
    start_time = time.time()
    report = f"*Researching {ticker}...*"

    # Initial status
    yield (
        f"### ⏳ Starting Research\n\nInitializing research for **{ticker}**...",
        report
    )

    # Yield progress updates until complete
    while thread.is_alive() or not progress_queue.empty():
        try:
            # Wait for next update with timeout
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
                # Update status display
                status = format_progress_display(update, total_elapsed)
                yield status, report

        except:
            # Queue empty, continue waiting
            pass

    thread.join()


# Create the Gradio interface
with gr.Blocks(title="Stock Research Agent", theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("""
    # 📈 Stock Research Agent
    ### AI-Powered Comprehensive Stock Analysis

    Enter a stock ticker to generate a detailed research report including:
    - **Investment Thesis** (Bull/Bear case, recommendation)
    - **Financial Metrics** (Valuation, profitability, health)
    - **Analyst Opinions** (Ratings, price targets, sentiment)
    - **Latest News** (with source links)
    - **SEC Filings** (10-K, 10-Q, 8-K with direct links)
    - **Insider Activity** (Recent transactions)

    **Need general research?** [Open Deep Research Agent (port 7861)](http://localhost:7861) - Research any topic with comprehensive reports.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Input field
            ticker_input = gr.Textbox(
                label="📌 Stock Ticker",
                placeholder="e.g., AAPL, TSLA, MSFT, NVDA",
                max_lines=1,
                scale=2
            )

            # Buttons
            with gr.Row():
                submit_btn = gr.Button("🚀 Generate Report", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)

            # Example tickers
            gr.Examples(
                examples=["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
                inputs=ticker_input,
                label="Popular Tickers"
            )

        with gr.Column(scale=1):
            # Progress status panel
            status_display = gr.Markdown(
                value="### ⏳ Ready\n\nEnter a ticker symbol and click **Generate Report**",
                label="Status"
            )

    # Output area
    output = gr.Markdown(
        label="Research Report",
        value="*Your comprehensive stock research report will appear here...*"
    )

    # Button actions
    submit_btn.click(
        fn=run_stock_research,
        inputs=[ticker_input],
        outputs=[status_display, output]
    )

    clear_btn.click(
        lambda: ("", "### ⏳ Ready\n\nEnter a ticker symbol and click **Generate Report**", "*Your comprehensive stock research report will appear here...*"),
        outputs=[ticker_input, status_display, output]
    )

    # Also trigger on Enter key
    ticker_input.submit(
        fn=run_stock_research,
        inputs=[ticker_input],
        outputs=[status_display, output]
    )

    # Instructions
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
    **Note:** This agent uses multiple financial APIs and Anthropic's Claude for analysis.
    Each research uses approximately 10,000-20,000 tokens (~$0.10-$0.20).

    *Disclaimer: Reports are for informational purposes only, not financial advice.*
    """)


# Launch the app
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📈 STOCK APP - NOT THE GENERAL APP")
    print("This app uses stock_research_agent.py")
    print("You should see BUY/HOLD/SELL recommendations!")
    print("=" * 70)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )
