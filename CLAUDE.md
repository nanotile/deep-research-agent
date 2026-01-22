# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Research Agent is a multi-agent AI system that automates research and report generation using Anthropic's Claude API. The system supports real-time web search via Tavily API (with LLM knowledge fallback) and provides live progress tracking in the web UI. Includes a specialized Stock Research Agent for financial analysis.

## Core Commands

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Web Interface (Gradio UI) - opens at http://127.0.0.1:7860
python app.py

# CLI Interface - general research
python deep_research_agent.py

# CLI Interface - stock analysis
python stock_research_agent.py

# Test Resend email API
python test_email.py
```

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Docker

```bash
docker build -t deep-research-agent .
docker run -p 7860:7860 --env-file .env deep-research-agent
```

## Architecture

### Deep Research Agent Pipeline (`deep_research_agent.py`)

Four agents execute sequentially via `deep_research()`:

```
User Query → plan_searches() → execute_searches() → write_report() → [send_email_report()]
                  ↓                    ↓                   ↓                    ↓
           WebSearchPlan        List[summaries]      Markdown report    Email via Resend
```

1. **Planning Agent** (`plan_searches`:97): Uses tool calling with `SEARCH_PLAN_TOOL` to generate structured search queries
2. **Research Agent** (`execute_searches`:216): Searches via Tavily API or falls back to LLM knowledge
3. **Report Agent** (`write_report`:250): Synthesizes findings into structured markdown report
4. **Email Agent** (`send_email_report`:294): Converts markdown to HTML and sends via Resend API

**Main orchestrators:**
- `deep_research()`:337 - Simple async function returning final report
- `deep_research_with_progress()`:376 - Async generator yielding `ProgressUpdate` objects for UI streaming

**Key configuration** (lines 63-65):
```python
HOW_MANY_SEARCHES = 3  # Number of search queries to generate
MODEL = "claude-sonnet-4-20250514"  # or "claude-opus-4-20250514" for better quality
```

### Stock Research Agent (`stock_research_agent.py`)

Specialized agent for stock analysis using multiple financial data sources:

```
Ticker → validate_ticker() → fetch_all_stock_data() → analyze_stock_data() → write_stock_report()
              ↓                       ↓                        ↓                      ↓
         Validation           StockDataBundle          StockAnalysis          Investment Thesis
```

**Data sources** (`stock_data_fetchers.py`):
- `fetch_yfinance_data()`:26 - Price, fundamentals, financials
- `fetch_finnhub_data()`:133 - News, sentiment, analyst ratings, insider transactions
- `fetch_sec_edgar_filings()`:291 - SEC filings (10-K, 10-Q, 8-K)
- `fetch_alpha_vantage_data()`:438 - Financial statements, company overview
- `fetch_tavily_news()`:580 - Real-time news via Tavily

**Key functions:**
- `stock_research()`:563 - Main entry point for stock analysis
- `stock_research_with_progress()`:595 - Async generator for UI streaming

### Web Interface (`app.py`)

- Gradio Blocks UI with live progress indicators
- Uses background thread + queue pattern for async-to-sync bridge
- Binds to `0.0.0.0:7860` for external access

### Structured Output

Both agents use Pydantic models with Anthropic tool calling for type-safe output:
- **Deep Research**: `WebSearchItem`, `WebSearchPlan` (lines 43-50)
- **Stock Research**: `StockAnalysis`, `InvestmentThesis` (`stock_data_models.py`)

## API Dependencies

| Key | Required | Purpose |
|-----|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | All LLM operations |
| `TAVILY_API_KEY` | No | Real-time web search (falls back to LLM knowledge) |
| `RESEND_API_KEY` | No | Email delivery |
| `FINNHUB_API_KEY` | No | Stock news, sentiment, analyst ratings |
| `ALPHA_VANTAGE_API_KEY` | No | Financial statements, technical indicators |
| `SEC_EDGAR_USER_AGENT` | No | SEC EDGAR API access |

## VM Networking Utilities (GCP)

Standalone utilities for Google Cloud VMs with non-static IPs.

```bash
python check_firewall.py              # Full GCP firewall analysis
python vm_firewall_utils.py 3000      # Check specific port
```

Always bind to `0.0.0.0` for external access:
```python
demo.launch(**configure_gradio_server(port=7860))
```

**Allowed Ports**: 22, 3000, 5000, 7859-7862, 8000, 8080, 8888, 8889

## Python API

```python
from deep_research_agent import deep_research, deep_research_with_progress
from stock_research_agent import stock_research, stock_research_with_progress
import asyncio

# General research
report = asyncio.run(deep_research("Your research query"))

# Stock analysis
report = asyncio.run(stock_research("AAPL"))

# With progress tracking (for custom UIs)
async def research_with_updates():
    async for update in deep_research_with_progress("Your query"):
        print(f"{update.stage}: {update.message}")
        if update.report:
            return update.report
```
