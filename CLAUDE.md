# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Research Agent is a multi-agent AI system that automates research and report generation using Anthropic's Claude API. The system supports real-time web search via Tavily API (with LLM knowledge fallback) and provides live progress tracking in the web UI. Agents cover general research, stock/financial analysis, AI domain research, sector analysis, competitor intelligence, portfolio analysis, and earnings tracking.

## Core Commands

```bash
source venv/bin/activate

# Web Interface (Gradio UI) - http://127.0.0.1:7860
python unified_app.py     # Full hub with all agents (primary entry point)
python app.py             # Deep research only
python stock_app.py       # Stock research only

# CLI Interface
python -m agents.deep_research_agent
python -m agents.stock_research_agent

# Test email integration
python test_email.py
```

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then edit with API keys
```

### Docker

```bash
docker build -t deep-research-agent .
docker run -p 7860:7860 --env-file .env deep-research-agent
```

## Architecture

### Agent Pattern (All Agents Follow This)

Every agent in `agents/` shares the same structural pattern with **no base class** — consistency is by convention:

1. **Dual entry points**: `agent_function(query) -> str` (simple) and `agent_function_with_progress(query) -> AsyncIterator[ProgressUpdate]` (UI streaming)
2. **ProgressUpdate dataclass** with fields: `stage`, `stage_display`, `current_step`, `total_steps`, `elapsed_time`, `message`, `report` (populated on completion), plus token tracking fields
3. **Tool calling for structured output**: JSON Schema tool definitions passed to `client.messages.create(tools=[...], tool_choice={"type": "tool", "name": "..."})`, results extracted via `next(block for block in response.content if block.type == "tool_use").input`
4. **Pipeline pattern**: Planning → Execution → Analysis → Synthesis

### Deep Research Agent (`agents/deep_research_agent.py`)

```
User Query → plan_searches() → execute_searches() → write_report() → [send_email_report()]
                  ↓                    ↓                   ↓                    ↓
           WebSearchPlan        List[summaries]      Markdown report    Email via Resend
```

**Recursive depth research**: Uses `DeepResearchState` to track multi-depth research. After initial searches, `extract_learnings()` identifies knowledge gaps via `LEARNING_EXTRACTION_TOOL`, generates follow-up queries, and recurses (default max_depth=2, max 15 total searches). Results tracked per-depth in `summaries_by_depth` and `gaps_by_depth` for collapsible report sections.

### Stock Research Agent (`agents/stock_research_agent.py`)

```
Ticker → validate_ticker() → fetch_all_stock_data() → analyze_stock_data() → write_stock_report()
```

Fetches 6 data sources in parallel via `asyncio.gather()`: yfinance, Finnhub, SEC EDGAR, Alpha Vantage, Tavily news, and macro sentiment. All return `DataWithSources` models with source URL tracking for citation. Has deep analysis mode with gap identification and follow-up searches.

### AI Research Agent (`agents/ai_research_agent.py`)

Specialized for AI domain research with 4-tier source prioritization (Foundational → Strategic → Policy → Practitioner), category-based parallel search execution, inline citation system, and a `CircuitBreaker` model that stops recursion when new learning ratio falls below threshold.

### Gradio Web Interface (`unified_app.py`)

Multi-tab hub with async-to-sync bridge pattern: each agent runs in a background thread with a `Queue()` passing `ProgressUpdate` objects back to the Gradio generator function. UI yields `(status_markdown, report_markdown)` tuples. Always bind to `0.0.0.0` for GCP VM access.

### Key Architectural Patterns

- **Graceful degradation**: Tavily optional (falls back to LLM knowledge), all stock data fetchers handle partial failures
- **Source attribution**: Every data model extends `DataWithSources` tracking `SourceURL` objects (url, title, source_type, accessed_at) for automatic citation in reports
- **2026 context injection**: `services/market_context_2026.py` dynamically injects current market context (BIS export controls, 2nm node race, agentic AI trends) into prompts for tech/semiconductor queries via `get_2026_analysis_prompt_injection()`
- **Tiered source quality**: AI Research Agent uses `services/ai_domain_context.py` with 4-tier domain classification and relevance scoring; managed via `manage_sources.py` CLI
- **Pydantic everywhere**: All data structures, tool outputs, and API responses use Pydantic models (`models/` directory)

## API Dependencies

| Key | Required | Purpose |
|-----|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | All LLM operations |
| `TAVILY_API_KEY` | No | Real-time web search (falls back to LLM knowledge) |
| `RESEND_API_KEY` | No | Email delivery |
| `FINNHUB_API_KEY` | No | Stock news, sentiment, analyst ratings |
| `ALPHA_VANTAGE_API_KEY` | No | Financial statements, technical indicators |
| `SEC_EDGAR_USER_AGENT` | No | SEC EDGAR API access |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | No | Google Drive report export |
| `GOOGLE_DRIVE_FOLDER_ID` | No | Target Drive folder for exports |

Additional config: `CLAUDE_MODEL` (default claude-sonnet-4-20250514), `HOW_MANY_SEARCHES` (default 3), `LOG_LEVEL` (default INFO).

## Adding a New Agent

1. Create `agents/your_agent.py` with `your_agent()` and `your_agent_with_progress()` following the dual entry point pattern
2. Define a `YourProgressUpdate` dataclass matching the standard fields
3. Create Pydantic models in `models/` if structured output is needed
4. Export from `agents/__init__.py`
5. Add a new tab in `unified_app.py` using the thread + queue async bridge pattern

## VM Networking (GCP)

```bash
python check_firewall.py              # Full GCP firewall analysis
python vm_firewall_utils.py 3000      # Check specific port
```

Allowed ports: 22, 3000, 5000, 7859-7862, 8000, 8080, 8888, 8889. Always use `configure_gradio_server(port=7860)` to bind `0.0.0.0`.
