"""
Launcher - Landing page with links to both Research Agents

Port Configuration:
- Launcher:              7859 (this app)
- Deep Research Agent:   7861 (app.py)
- Stock Research Agent:  7862 (stock_app.py)
"""

import gradio as gr
from vm_ip_utils import get_vm_ip


def get_agent_urls():
    """Get URLs for both agents using VM IP"""
    try:
        vm_ip = get_vm_ip()
    except:
        vm_ip = "localhost"

    return {
        "vm_ip": vm_ip,
        "deep_research": f"http://{vm_ip}:7861",
        "stock_research": f"http://{vm_ip}:7862"
    }


# Create the Gradio interface
with gr.Blocks(title="Research Agent Hub", theme=gr.themes.Soft()) as demo:

    urls = get_agent_urls()

    gr.Markdown(f"""
    # Research Agent Hub
    ### Choose Your Research Tool

    ---
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            ## Deep Research Agent

            **For:** General research on any topic

            - AI-planned search queries
            - Web search via Tavily API
            - Comprehensive markdown reports
            - Optional email delivery

            **Example queries:**
            - "AI trends in 2026"
            - "Best productivity tools comparison"
            - "Sustainable energy investments"

            ### [{urls['deep_research']}]({urls['deep_research']})
            """)

        with gr.Column():
            gr.Markdown(f"""
            ## Stock Research Agent

            **For:** Stock analysis and trading insights

            - Buy/Hold/Sell recommendations
            - Price targets & valuation metrics
            - Analyst ratings & insider activity
            - SEC filings with direct links
            - Macro & political risk analysis

            **Example tickers:** AAPL, TSLA, NVDA, MSFT

            ### [{urls['stock_research']}]({urls['stock_research']})
            """)

    gr.Markdown(f"""
    ---

    ### Quick Links

    | Agent | Port | URL |
    |-------|------|-----|
    | Deep Research | 7861 | {urls['deep_research']} |
    | Stock Research | 7862 | {urls['stock_research']} |

    ---

    *Powered by Anthropic Claude API*
    """)


# Launch the app
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RESEARCH AGENT HUB - LAUNCHER")
    print("=" * 70)
    print("Port Configuration:")
    print("  - Launcher:            http://0.0.0.0:7859")
    print("  - Deep Research:       http://0.0.0.0:7861")
    print("  - Stock Research:      http://0.0.0.0:7862")
    print("=" * 70)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7859,
        share=False,
        show_error=True
    )
