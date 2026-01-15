# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Research Agent is a multi-agent AI system that automates research and report generation using Anthropic's Claude API. The system uses an asynchronous architecture with four specialized agents that work in sequence: Planning, Research, Report Writing, and Email Delivery.

## Core Commands

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and RESEND_API_KEY
```

## Architecture

### Agent Pipeline (`deep_research_agent.py`)

All agents execute sequentially via the `deep_research()` orchestrator:

```
User Query → plan_searches() → execute_searches() → write_report() → [send_email_report()]
                  ↓                    ↓                   ↓                    ↓
           WebSearchPlan        List[summaries]      Markdown report    Email via Resend
```

1. **Planning Agent** (`plan_searches`): Uses tool calling with `SEARCH_PLAN_TOOL` to generate structured search queries
2. **Research Agent** (`execute_searches`): Uses LLM knowledge to research each topic (no external search API)
3. **Report Agent** (`write_report`): Synthesizes findings into structured markdown report
4. **Email Agent** (`send_email_report`): Converts markdown to HTML and sends via Resend API

### Key Configuration

```python
HOW_MANY_SEARCHES = 3  # Number of search queries to generate
MODEL = "claude-sonnet-4-20250514"  # or "claude-opus-4-20250514" for better quality
```

### Structured Output

Uses Pydantic models with Anthropic tool calling for type-safe planning output:
- `WebSearchItem`: `{reason: str, search_term: str}`
- `WebSearchPlan`: `{searches: List[WebSearchItem]}`

### Web Interface (`app.py`)

- Gradio Blocks UI wrapping `deep_research()` with `asyncio.run()` sync bridge
- Binds to `0.0.0.0:7860` for external access

## API Dependencies

- **Required**: `ANTHROPIC_API_KEY` - all LLM operations
- **Optional**: `RESEND_API_KEY` - email delivery only

## VM Networking Utilities (GCP)

Standalone utilities for Google Cloud VMs with non-static IPs. Project-agnostic.

| File | Purpose |
|------|---------|
| `vm_ip_utils.py` | `get_vm_ip()`, `get_server_url(port)`, `configure_gradio_server(port)` |
| `vm_firewall_utils.py` | `diagnose_port(port)`, `check_port_listening(port)` |
| `check_firewall.py` | Full GCP firewall analysis (requires gcloud CLI) |

### Quick Diagnostics

```bash
python check_firewall.py              # Full firewall analysis
python vm_firewall_utils.py 3000      # Check specific port
```

### External Access

Always bind to `0.0.0.0` (not `127.0.0.1`):

```python
demo.launch(**configure_gradio_server(port=7860))
```

**Allowed Ports**: 22, 3000, 5000, 7859-7862, 8000, 8080, 8888, 8889

**Common Issue**: "ERR_CONNECTION_REFUSED" = nothing listening or bound to 127.0.0.1. Run `python vm_firewall_utils.py <port>` to diagnose.
