"""
Gradio Web Interface for Deep Research Agent with Progress Indicators
"""

import gradio as gr
import asyncio
import time
from threading import Thread
from queue import Queue
from deep_research_agent import (
    deep_research,
    deep_research_with_progress,
    ProgressUpdate,
    tavily_client
)


def format_progress_display(update: ProgressUpdate, total_elapsed: float) -> str:
    """Format progress update for display in status box"""
    # Stage icons
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

    # Format elapsed time
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
    """
    Generator function that yields progress updates and final report.
    Uses a background thread to run async code while yielding updates.
    """
    # Validate inputs
    if not query or query.strip() == "":
        yield "### ⚠️ Input Required\n\nPlease enter a research query!", "*Your research report will appear here...*"
        return

    # Queue for communication between async task and generator
    progress_queue = Queue()

    def run_async_research():
        """Run the async research in a separate thread"""
        async def async_wrapper():
            async for update in deep_research_with_progress(
                query=query.strip()
            ):
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
    report = "*Research in progress...*"

    # Initial status
    yield "### ⏳ Starting...\n\nInitializing research agents...", report

    # Yield progress updates until complete
    while thread.is_alive() or not progress_queue.empty():
        try:
            # Wait for next update with timeout
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
                # Update status display
                status = format_progress_display(update, total_elapsed)
                yield status, report

        except:
            # Queue empty, continue waiting
            pass

    thread.join()


# Create the Gradio interface
search_mode = "Tavily web search" if tavily_client else "LLM knowledge base"
with gr.Blocks(title="Deep Research Agent") as demo:

    # Header
    gr.Markdown(f"""
    # 🔬 Deep Research Agent
    ### Multi-Agent System for Comprehensive Research Reports

    This AI-powered research assistant uses multiple specialized agents to:
    1. **Plan** relevant search queries
    2. **Research** each topic via {search_mode}
    3. **Synthesize** findings into a professional report
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Input fields
            query_input = gr.Textbox(
                label="🎯 Research Query",
                placeholder="e.g., How effective is Seeking Alpha investment advice in 2025?",
                lines=2
            )

            # Buttons
            with gr.Row():
                submit_btn = gr.Button("🚀 Start Research", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)

        with gr.Column(scale=1):
            # Progress status panel
            status_display = gr.Markdown(
                value="### ⏳ Ready\n\nEnter a query and click **Start Research**",
                label="Status"
            )

    # Output area
    output = gr.Markdown(
        label="Research Report",
        value="*Your research report will appear here...*"
    )

    # Button actions - use generator for streaming updates
    submit_btn.click(
        fn=run_research_with_progress,
        inputs=[query_input],
        outputs=[status_display, output]
    )

    clear_btn.click(
        lambda: ("", "### ⏳ Ready\n\nEnter a query and click **Start Research**", "*Your research report will appear here...*"),
        outputs=[query_input, status_display, output]
    )

    # Instructions in collapsible section
    with gr.Accordion("📋 Instructions & Requirements", open=False):
        gr.Markdown("""
        ### How to Use
        1. Enter your research question
        2. Click "Start Research"
        3. Watch the progress indicators
        4. Wait for the report (typically 1-2 minutes)

        ### Requirements
        - **Anthropic API key** in `.env` (required)
        - **Tavily API key** in `.env` (optional - enables real-time web search)

        ### Example Queries
        - "AI trends in 2026"
        - "Sustainable energy investment opportunities"
        - "Remote work productivity tools comparison"
        """)

    # Footer
    gr.Markdown("""
    ---
    **Note:** This agent uses Anthropic's Claude API and optionally Tavily search.
    Each research typically uses 5,000-15,000 tokens (~$0.05-$0.15) plus Tavily API costs if enabled.
    """)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting Gradio Web Interface...")
    print("="*70)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=False,
        theme=gr.themes.Soft()
    )
