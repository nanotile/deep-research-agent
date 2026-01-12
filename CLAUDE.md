# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Research Agent is a multi-agent AI system that automates research and report generation using OpenAI's API. The system uses an asynchronous architecture with four specialized agents that work in sequence: Planning, Research, Report Writing, and Email Delivery.

## Core Commands

### Running the Application

```bash
# Web Interface (Gradio UI) - opens at http://127.0.0.1:7860
python app.py

# CLI Interface
python deep_research_agent.py

# Test Resend email API
python test_email.py
```

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and RESEND_API_KEY
```

## Multi-Agent Architecture

All agents are defined in `deep_research_agent.py` and execute sequentially:

| Agent | Function | Input | Output |
|-------|----------|-------|--------|
| **Planning** | `plan_searches()` | User query | List of `WebSearchItem` (reason + search_term) |
| **Research** | `execute_searches()` | Search items | List of dicts with summaries |
| **Report Writing** | `write_report()` | Query + results | Markdown report |
| **Email** | `send_email_report()` | Report + recipient | Email via Resend API |

### Main Orchestrator

```python
async def deep_research(
    query: str,
    send_via_email: bool = False,
    recipient: str = None
) -> str:  # Returns markdown report
```

### Key Configuration (in `deep_research_agent.py`)

- `HOW_MANY_SEARCHES = 3` - Number of search queries to generate
- `MODEL = "claude-sonnet-4-20250514"` - Anthropic model (can change to `"claude-opus-4-20250514"`)

### Pydantic Models for Structured Output

- `WebSearchItem`: Single search with `reason` and `search_term` fields
- `WebSearchPlan`: Container with list of `WebSearchItem` objects

## Gradio Web Interface (`app.py`)

- Wraps `deep_research()` with sync/async bridge via `asyncio.run()`
- Input: query text, email checkbox, recipient email
- Output: Rendered markdown report
- Port 7860, localhost only by default

## API Dependencies

- **Required**: Anthropic API key (all LLM operations)
- **Optional**: Resend API key (email delivery only)

## Cost Considerations

- ~$0.05-$0.15 per research run
- 5,000-15,000 tokens per query
- Monitor at https://platform.openai.com/usage

## VM Networking Utilities (GCP)

Standalone utilities for Google Cloud VMs with non-static IPs. These are **project-agnostic** and can be copied to any project.

### Key Utilities

| File | Purpose |
|------|---------|
| `vm_ip_utils.py` | Dynamic IP management: `get_vm_ip()`, `get_server_url(port)`, `configure_gradio_server(port)` |
| `vm_firewall_utils.py` | Port diagnostics: `diagnose_port(port)`, `check_port_listening(port)` |
| `check_firewall.py` | Complete GCP firewall analysis (requires gcloud CLI) |

### Quick Diagnostics

```bash
python check_firewall.py              # Full firewall analysis
python vm_firewall_utils.py 3000      # Check specific port
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"
```

### External Access Pattern

Always bind to `0.0.0.0` (not `127.0.0.1`) for external access:

```python
# Gradio
from vm_ip_utils import configure_gradio_server, print_access_info
print_access_info(port=7860, service_name="My App")
demo.launch(**configure_gradio_server(port=7860))

# Flask/FastAPI
app.run(host="0.0.0.0", port=5000)
```

### Allowed Ports

22 (SSH), 3000, 5000, 7859-7862, 8000, 8080, 8888, 8889

### Common Issue

"ERR_CONNECTION_REFUSED" usually means nothing is listening or service is bound to 127.0.0.1 instead of 0.0.0.0. Run `python vm_firewall_utils.py <port>` for diagnosis.
