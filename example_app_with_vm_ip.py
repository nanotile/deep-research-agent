"""
Example: How to use vm_ip_utils with the Deep Research Agent

This file demonstrates how to integrate vm_ip_utils into app.py
WITHOUT modifying the original app.py file.

This is a standalone example - copy this pattern to any project.
"""

import gradio as gr
import asyncio
from deep_research_agent import deep_research
from vm_ip_utils import configure_gradio_server, print_access_info, get_server_url

def run_research(query: str, send_email: bool, recipient: str = ""):
    """
    Gradio interface wrapper - runs the research agent
    """
    # Validate inputs
    if not query or query.strip() == "":
        return "❌ Please enter a research query!"

    if send_email and (not recipient or recipient.strip() == ""):
        return "❌ Please provide a recipient email address if you want to send via email!"

    # Run the async research function
    try:
        report = asyncio.run(deep_research(
            query=query.strip(),
            send_via_email=send_email,
            recipient=recipient.strip() if send_email else None
        ))
        return report
    except Exception as e:
        return f"❌ **Error occurred:**\n\n```{str(e)}```\n\nPlease check your API keys in the .env file."


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Deep Research Agent") as demo:

    # Header with VM IP information
    vm_ip_html = f"""
    # 🔬 Deep Research Agent
    ### Multi-Agent System for Comprehensive Research Reports

    **🌐 Access URLs:**
    - Public: `{get_server_url(7860)}`
    - Local: `http://127.0.0.1:7860`

    ---

    This AI-powered research assistant uses multiple specialized agents to:
    1. **Plan** relevant search queries
    2. **Research** each topic thoroughly
    3. **Synthesize** findings into a professional report
    4. **Email** the report (optional)
    """

    gr.Markdown(vm_ip_html)

    with gr.Row():
        with gr.Column(scale=2):
            # Input fields
            query_input = gr.Textbox(
                label="🎯 Research Query",
                placeholder="e.g., Latest AI Agent frameworks in 2025",
                lines=2
            )

            with gr.Row():
                email_checkbox = gr.Checkbox(
                    label="📧 Send via Email",
                    value=False
                )
                recipient_input = gr.Textbox(
                    label="Recipient Email",
                    placeholder="your@email.com",
                    scale=2
                )

            # Buttons
            with gr.Row():
                submit_btn = gr.Button("🚀 Start Research", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📋 Instructions

            1. Enter your research question
            2. Click "Start Research"
            3. Wait for the report (takes 1-2 minutes)
            4. Optionally send via email

            ### ⚙️ Requirements

            - OpenAI API key in `.env`
            - Resend API key (for email)
            - Internet connection

            ### 💡 Examples

            - "AI trends in 2025"
            - "Sustainable energy solutions"
            - "Remote work productivity tools"
            """)

    # Output area
    output = gr.Markdown(
        label="📊 Research Report",
        value="*Your research report will appear here...*"
    )

    # Button actions
    submit_btn.click(
        fn=run_research,
        inputs=[query_input, email_checkbox, recipient_input],
        outputs=output
    )

    clear_btn.click(
        lambda: ("", False, "", "*Your research report will appear here...*"),
        outputs=[query_input, email_checkbox, recipient_input, output]
    )

    # Footer
    gr.Markdown(f"""
    ---
    **Note:** This agent uses OpenAI's API and consumes tokens. Each research typically uses 5,000-15,000 tokens (~$0.05-$0.15).

    **Network Access:** Accessible at {get_server_url(7860)}
    """)


# Launch the app with VM IP utilities
if __name__ == "__main__":
    # Configure port
    PORT = 7860
    SERVICE_NAME = "Deep Research Agent (with VM IP)"

    print("\n" + "="*70)
    print("🌐 Starting Gradio Web Interface with VM IP Utilities...")
    print("="*70)

    # Print access information BEFORE launching
    print_access_info(
        port=PORT,
        service_name=SERVICE_NAME,
        additional_ports=[8000, 5000]  # If you have other services
    )

    # Get configuration from vm_ip_utils
    config = configure_gradio_server(port=PORT, share=False)

    print("\n🚀 Launching application...")
    print(f"📍 Main URL: {get_server_url(PORT)}")
    print("-"*70)

    # Launch with automatic configuration
    demo.launch(**config)

    # Note: The demo.launch() call blocks, so code after this won't run
    # until the server is stopped (Ctrl+C)
