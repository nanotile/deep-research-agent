"""
Deep Research Agent
A multi-agent system for conducting in-depth web research and generating comprehensive reports.
Supports real-time web search via Tavily API with LLM fallback.

Updated for 2026 market context with awareness of:
- BIS 25% China Surcharge on high-performance compute exports
- 2nm process node race (TSMC vs Samsung)
- Agentic AI enterprise adoption shift
"""

import os
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, AsyncIterator, Any
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
import resend

from services.market_context_2026 import DEEP_RESEARCH_2026_CONTEXT
from utils.logging_config import setup_logging, get_logger
from utils.token_tracker import extract_usage_from_response, format_token_display

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Initialize clients
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
resend.api_key = os.getenv("RESEND_API_KEY")

# Initialize Tavily client (optional - graceful fallback if not available)
tavily_client = None
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key and tavily_api_key != "your_tavily_api_key_here":
    try:
        from tavily import AsyncTavilyClient
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    except ImportError:
        print("Warning: tavily-python not installed. Using LLM knowledge fallback.")

logger.info("=" * 70)
logger.info("DEEP RESEARCH AGENT")
if tavily_client:
    logger.info("Tavily API: Enabled (real-time web search)")
else:
    logger.info("Tavily API: Disabled (using LLM knowledge)")
logger.info("=" * 70)

# Pydantic models for structured outputs
class WebSearchItem(BaseModel):
    """A single web search query"""
    reason: str = Field(description="Why this search is relevant to the research query")
    search_term: str = Field(description="The specific search term to use")

class WebSearchPlan(BaseModel):
    """Collection of web searches to perform"""
    searches: List[WebSearchItem] = Field(description="List of web searches to perform")

@dataclass
class ProgressUpdate:
    """Progress update for UI display"""
    stage: str           # "planning", "searching", "extracting", "recursing", "writing", "emailing", "complete"
    stage_display: str   # Human-readable stage name
    current_step: int    # Current step within stage (1-based)
    total_steps: int     # Total steps in stage
    elapsed_time: float  # Seconds elapsed for current stage
    message: str         # Detailed status message
    report: Optional[str] = None  # Final report (only set when complete)
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    # Depth tracking for recursive research
    current_depth: int = 0
    max_depth: int = 1
    learnings_count: int = 0
    total_searches: int = 0
    gaps_identified: Optional[List[str]] = None  # Gap descriptions for UI display


@dataclass
class DeepResearchState:
    """State management for recursive deep research."""
    query: str
    current_depth: int = 0
    max_depth: int = 2
    all_summaries: List[Dict[str, str]] = field(default_factory=list)
    learnings: List[Dict[str, Any]] = field(default_factory=list)
    completed_queries: List[str] = field(default_factory=list)
    pending_searches: List[Dict[str, str]] = field(default_factory=list)
    total_searches: int = 0
    max_total_searches: int = 15
    summaries_by_depth: Dict[int, List[Dict[str, str]]] = field(default_factory=dict)
    gaps_by_depth: Dict[int, List[str]] = field(default_factory=dict)


@dataclass
class Learning:
    """Key insight extracted from research with confidence and follow-up needs."""
    content: str
    confidence: float
    source_queries: List[str]
    follow_up_queries: List[str] = field(default_factory=list)

def add_depth_breakdown(state: DeepResearchState) -> str:
    """Generate a collapsible research depth breakdown appendix for the report."""
    sections = ["\n\n---\n\n<details>\n<summary><strong>Research Depth Breakdown</strong></summary>\n"]

    for depth_level in sorted(state.summaries_by_depth.keys()):
        summaries = state.summaries_by_depth[depth_level]
        if depth_level == 1:
            header = f"### Initial Research ({len(summaries)} searches)"
        else:
            header = f"### Follow-up Research — Depth {depth_level} ({len(summaries)} searches)"

        sections.append(header)

        # Show gaps that prompted this depth level (for depth > 1)
        prev_depth = depth_level - 1
        if prev_depth in state.gaps_by_depth and state.gaps_by_depth[prev_depth]:
            sections.append("\n**Gaps Addressed:**")
            for gap in state.gaps_by_depth[prev_depth][:5]:
                display = gap[:100] + "..." if len(gap) > 100 else gap
                sections.append(f"- {display}")

        # List search topics and brief findings
        sections.append("\n**Searches:**")
        for s in summaries:
            term = s.get('search_term', 'Unknown')
            summary_text = s.get('summary', '')
            brief = summary_text[:120] + "..." if len(summary_text) > 120 else summary_text
            sections.append(f"- **{term}**: {brief}")
        sections.append("")

    sections.append(f"\n*Research completed: {state.current_depth} depth levels, {state.total_searches} total searches, {len(state.learnings)} learnings extracted*")
    sections.append("\n</details>")

    return "\n".join(sections)


# Configuration - read from .env with defaults
HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES", "3"))
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Token tracking accumulator
class TokenAccumulator:
    """Accumulates token usage across multiple API calls."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, response):
        """Extract and add tokens from an API response."""
        if hasattr(response, 'usage'):
            self.input_tokens += getattr(response.usage, 'input_tokens', 0)
            self.output_tokens += getattr(response.usage, 'output_tokens', 0)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self):
        # Claude Sonnet: $3/1M input, $15/1M output
        return (self.input_tokens / 1_000_000) * 3.0 + (self.output_tokens / 1_000_000) * 15.0

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0

# Tool definition for structured output
SEARCH_PLAN_TOOL = {
    "name": "create_search_plan",
    "description": "Create a structured search plan with multiple search queries",
    "input_schema": {
        "type": "object",
        "properties": {
            "searches": {
                "type": "array",
                "description": "List of search queries to perform",
                "items": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why this search is relevant to the research query"
                        },
                        "search_term": {
                            "type": "string",
                            "description": "The specific search term to use"
                        }
                    },
                    "required": ["reason", "search_term"]
                }
            }
        },
        "required": ["searches"]
    }
}

# Tool for extracting learnings and identifying knowledge gaps
LEARNING_EXTRACTION_TOOL = {
    "name": "extract_learnings",
    "description": "Extract key learnings from research and identify knowledge gaps needing follow-up",
    "input_schema": {
        "type": "object",
        "properties": {
            "learnings": {
                "type": "array",
                "description": "Key insights extracted from the research",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The key insight or finding"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence score 0-1 based on source quality"
                        },
                        "source_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries that provided this learning"
                        },
                        "follow_up_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Questions raised that need further research (0-2 per learning)"
                        }
                    },
                    "required": ["content", "confidence", "source_queries"]
                }
            },
            "knowledge_gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Overall gaps in understanding that need follow-up research"
            }
        },
        "required": ["learnings"]
    }
}

async def plan_searches(query: str, tokens: TokenAccumulator = None, model: str = None) -> List[Dict[str, str]]:
    """
    Planning Agent: Generate search queries based on the research topic
    Updated for 2026 context awareness.
    """
    print(f"\n🤔 Planning searches for: {query}")

    # Add 2026 context for tech/AI queries
    context_hint = ""
    if _is_tech_ai_query(query):
        context_hint = """

    IMPORTANT: When researching tech, AI, or semiconductor topics, consider including searches for:
    - 2026 market developments and regulatory changes
    - BIS export controls and China trade policy impacts
    - Agentic AI enterprise adoption trends
    - 2nm process node competition and supply chain dynamics"""

    prompt = f"""You are a research planning assistant. Given a research query,
    generate {HOW_MANY_SEARCHES} specific web search terms that will help comprehensively
    answer the query.

    Research Query: {query}
    {context_hint}

    For each search, provide:
    1. A clear reason why this search is relevant
    2. A specific search term optimized for web search engines

    Make searches diverse to cover different aspects of the topic.

    Use the create_search_plan tool to provide your response."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1024,
        system="You are a research planning expert. Today's date is January 2026. Always use the provided tool to structure your response.",
        messages=[{"role": "user", "content": prompt}],
        tools=[SEARCH_PLAN_TOOL],
        tool_choice={"type": "tool", "name": "create_search_plan"}
    )

    # Track tokens
    if tokens:
        tokens.add(response)

    # Extract tool use result
    tool_use = next(block for block in response.content if block.type == "tool_use")
    search_plan = tool_use.input

    search_items = [
        {"reason": item["reason"], "search_term": item["search_term"]}
        for item in search_plan["searches"]
    ]

    print(f"✓ Generated {len(search_items)} search queries")
    for i, item in enumerate(search_items, 1):
        print(f"  [{i}] {item['search_term']}")
        print(f"      → {item['reason']}")

    return search_items

async def search_with_tavily(search_term: str, tokens: TokenAccumulator = None, model: str = None) -> str:
    """
    Execute a Tavily web search and synthesize results with Claude.
    Returns a summary string compatible with the existing pipeline.
    """
    # Call Tavily API
    response = await tavily_client.search(
        query=search_term,
        search_depth="basic",
        max_results=5,
        include_answer=False
    )

    # Format raw results for Claude synthesis
    raw_results = []
    for result in response.get("results", []):
        raw_results.append(
            f"Source: {result.get('title', 'Unknown')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}\n"
        )

    formatted_results = "\n---\n".join(raw_results)

    # Use Claude to synthesize the raw search results
    synthesis_prompt = f"""Based on these web search results for "{search_term}",
provide a comprehensive summary (2-3 paragraphs, max 300 words) covering:
- Key facts and current developments
- Important trends or patterns
- Relevant statistics or examples

Search Results:
{formatted_results}

Write only the summary, synthesizing information from all sources.
Be specific and factual. Do not include URLs or source citations."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1024,
        system="You are an expert researcher synthesizing web search results into concise summaries.",
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.3
    )

    # Track tokens
    if tokens:
        tokens.add(response)

    return response.content[0].text


def _is_tech_ai_query(query: str) -> bool:
    """Check if a query relates to tech, AI, or semiconductors."""
    tech_keywords = [
        "ai", "artificial intelligence", "semiconductor", "chip", "gpu", "nvidia",
        "amd", "intel", "tsmc", "samsung", "tech", "technology", "software",
        "cloud", "data center", "machine learning", "llm", "transformer",
        "agentic", "agent", "compute", "processor", "memory", "hbm"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in tech_keywords)


async def search_with_llm_knowledge(search_term: str, tokens: TokenAccumulator = None, model: str = None) -> str:
    """
    Fallback: Use LLM's knowledge base for research when Tavily is unavailable.
    Updated for January 2026 knowledge context.
    """
    # Add 2026 context for tech/AI queries
    context_injection = ""
    if _is_tech_ai_query(search_term):
        context_injection = f"""
{DEEP_RESEARCH_2026_CONTEXT}

"""

    research_prompt = f"""{context_injection}Research and provide comprehensive information about: {search_term}

Provide a detailed summary (2-3 paragraphs, max 300 words) covering:
- Key facts and current developments
- Important trends or patterns
- Relevant statistics or examples
- Recent updates (if applicable)

Be specific, factual, and cite approximate timeframes when relevant.
Write only the summary, no additional commentary."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1024,
        system="You are an expert researcher with deep knowledge across many domains. Your knowledge extends to January 2026. When discussing tech, AI, or semiconductors, consider the 2026 market context including BIS export controls, 2nm process node competition, and Agentic AI adoption trends.",
        messages=[{"role": "user", "content": research_prompt}],
        temperature=0.3
    )

    # Track tokens
    if tokens:
        tokens.add(response)

    return response.content[0].text


async def execute_searches(search_items: List[Dict[str, str]], tokens: TokenAccumulator = None, model: str = None) -> List[Dict[str, str]]:
    """
    Search Agent: Execute web searches using Tavily API, with LLM fallback
    """
    use_tavily = tavily_client is not None

    if use_tavily:
        print("\n🔍 Executing web searches via Tavily API...")
    else:
        print("\n🔍 Executing knowledge-based research (no Tavily API key)...")

    search_results = []

    for i, item in enumerate(search_items, 1):
        print(f"  [{i}/{len(search_items)}] Searching: {item['search_term']}")

        if use_tavily:
            try:
                summary = await search_with_tavily(item['search_term'], tokens=tokens, model=model)
            except Exception as e:
                print(f"      ! Tavily failed ({e}), using LLM fallback")
                summary = await search_with_llm_knowledge(item['search_term'], tokens=tokens, model=model)
        else:
            summary = await search_with_llm_knowledge(item['search_term'], tokens=tokens, model=model)

        search_results.append({
            "search_term": item['search_term'],
            "reason": item['reason'],
            "summary": summary
        })
        print(f"      ✓ Search complete")

    return search_results


async def extract_learnings(
    search_results: List[Dict[str, str]],
    query: str,
    existing_learnings: List[Dict[str, Any]] = None,
    tokens: TokenAccumulator = None,
    model: str = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract key learnings from search results and identify knowledge gaps.

    Args:
        search_results: Results from execute_searches()
        query: Original research query
        existing_learnings: Learnings from previous depths (to avoid duplicates)
        tokens: Token accumulator
        model: Claude model to use

    Returns:
        Tuple of (learnings list, follow-up queries list)
    """
    print("\n🧠 Extracting learnings and identifying gaps...")

    # Format search summaries for analysis
    formatted_results = []
    for result in search_results:
        formatted_results.append(
            f"Search: {result['search_term']}\n"
            f"Purpose: {result['reason']}\n"
            f"Findings: {result['summary']}\n"
        )

    # Format existing learnings to avoid duplicates
    existing_summary = ""
    if existing_learnings:
        existing_points = [f"- {l['content'][:100]}..." for l in existing_learnings[:10]]
        existing_summary = f"\n\nExisting learnings (avoid duplicates):\n" + "\n".join(existing_points)

    prompt = f"""Analyze these research findings for the query: "{query}"

Research Findings:
{"="*60}
{chr(10).join(formatted_results)}
{existing_summary}

Extract key learnings and identify gaps:
1. For each learning: content, confidence (0-1), source queries that provided it
2. For learnings with gaps: suggest 0-2 follow-up queries to fill the gap
3. List overall knowledge gaps that need more research

Focus on:
- Novel insights not already covered
- Factual claims with clear evidence
- Areas where information is incomplete or contradictory
- Questions raised but not answered by the research

Use the extract_learnings tool to structure your response."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=2048,
        system="You are a research analyst extracting key insights and identifying knowledge gaps. Today's date is January 2026.",
        messages=[{"role": "user", "content": prompt}],
        tools=[LEARNING_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_learnings"}
    )

    if tokens:
        tokens.add(response)

    # Extract tool use result
    tool_use = next(block for block in response.content if block.type == "tool_use")
    result_data = tool_use.input

    learnings = result_data.get("learnings", [])
    knowledge_gaps = result_data.get("knowledge_gaps", [])

    # Collect all follow-up queries from learnings and knowledge gaps
    follow_up_queries = []
    for learning in learnings:
        follow_up_queries.extend(learning.get("follow_up_queries", []))

    # Add knowledge gaps as follow-up queries if they're phrased as questions
    for gap in knowledge_gaps:
        if "?" in gap or gap.lower().startswith(("what", "how", "why", "when", "where", "who")):
            follow_up_queries.append(gap)

    # Deduplicate follow-up queries
    follow_up_queries = list(dict.fromkeys(follow_up_queries))

    print(f"✓ Extracted {len(learnings)} learnings, {len(follow_up_queries)} follow-up queries")
    return learnings, follow_up_queries


async def generate_followup_searches(
    follow_up_queries: List[str],
    completed_queries: List[str],
    max_searches: int = 3,
    tokens: TokenAccumulator = None,
    model: str = None
) -> List[Dict[str, str]]:
    """
    Generate search items for follow-up queries.

    Args:
        follow_up_queries: Questions needing follow-up research
        completed_queries: Already executed search terms (to avoid duplicates)
        max_searches: Maximum number of follow-up searches
        tokens: Token accumulator
        model: Claude model to use

    Returns:
        List of search items ready for execute_searches()
    """
    # Filter out queries too similar to completed ones
    new_queries = []
    completed_lower = [q.lower() for q in completed_queries]

    for query in follow_up_queries:
        query_lower = query.lower()
        # Check if this query is too similar to completed ones
        is_duplicate = any(
            query_lower in completed or completed in query_lower
            for completed in completed_lower
        )
        if not is_duplicate:
            new_queries.append(query)

    # Limit to max_searches
    new_queries = new_queries[:max_searches]

    if not new_queries:
        return []

    # Convert to search items format
    search_items = [
        {
            "reason": f"Follow-up research to address knowledge gap",
            "search_term": query
        }
        for query in new_queries
    ]

    print(f"📋 Generated {len(search_items)} follow-up searches")
    return search_items


async def write_report(query: str, search_results: List[Dict[str, str]], tokens: TokenAccumulator = None, model: str = None) -> str:
    """
    Report Writing Agent: Synthesize findings into comprehensive report
    Updated for 2026 context awareness.
    """
    print("\n📝 Writing comprehensive report...")

    # Format search summaries
    formatted_summaries = []
    for i, result in enumerate(search_results, 1):
        formatted_summaries.append(
            f"Research Topic {i}: {result['search_term']}\n"
            f"Purpose: {result['reason']}\n"
            f"Findings:\n{result['summary']}\n"
        )

    # Add 2026 context for tech/AI queries
    context_section = ""
    if _is_tech_ai_query(query):
        context_section = f"""
    IMPORTANT 2026 CONTEXT:
    {DEEP_RESEARCH_2026_CONTEXT}

    When writing about tech, AI, or semiconductors, incorporate this 2026 context where relevant.
    """

    report_prompt = f"""Synthesize the following research findings into a comprehensive, well-structured report.

    Research Query: {query}
    {context_section}
    Research Findings:
    {"="*60}
    {chr(10).join(formatted_summaries)}

    Create a professional report with:
    1. **Executive Summary** (2-3 sentences highlighting key insights)
    2. **Key Findings** (organized by theme with clear subheadings)
    3. **Detailed Analysis** (synthesize information from all research topics)
    4. **Conclusions and Insights** (actionable takeaways)

    Use markdown formatting with ## for main sections and ### for subsections.
    Be thorough, professional, and insightful."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=4096,
        system="You are an expert research report writer. Today's date is January 2026.",
        messages=[{"role": "user", "content": report_prompt}],
        temperature=0.5
    )

    # Track tokens
    if tokens:
        tokens.add(response)

    report = response.content[0].text
    print("✓ Report generated")
    return report

async def send_email_report(recipient: str, subject: str, report: str) -> str:
    """
    Email Agent: Send report via email using Resend
    """
    print(f"\n📧 Sending report to {recipient}...")

    # Convert markdown to HTML
    html_prompt = f"""Convert the following markdown report into clean, professional HTML
    suitable for email. Use proper HTML structure with <html>, <body>, and appropriate styling.
    Use professional colors and formatting.

    Markdown Report:
    {report}

    Provide only the complete HTML code."""

    response = await client.messages.create(
        model=DEFAULT_MODEL,  # Use default for email formatting
        max_tokens=4096,
        system="You are an HTML email formatting expert.",
        messages=[{"role": "user", "content": html_prompt}],
        temperature=0.2
    )

    html_body = response.content[0].text

    # Send email via Resend
    try:
        params = {
            "from": "onboarding@resend.dev",
            "to": [recipient],
            "subject": subject,
            "html": html_body
        }
        email = resend.Emails.send(params)
        result = f"✓ Email sent successfully! ID: {email['id']}"
        print(result)
        return result
    except Exception as e:
        error = f"✗ Failed to send email: {str(e)}"
        print(error)
        return error

async def deep_research(
    query: str,
    send_via_email: bool = False,
    recipient: str = None,
    model: str = None,
    depth: int = 1,
    max_searches: int = 15
) -> str:
    """
    Main Orchestrator: Coordinate all agents to perform deep research

    Args:
        query: Research question or topic
        send_via_email: Whether to email the report
        recipient: Email address (required if send_via_email=True)
        model: Claude model to use (defaults to DEFAULT_MODEL)
        depth: Research depth (1=quick single-pass, 2-3=recursive with follow-ups)
        max_searches: Maximum total searches across all depths

    Returns:
        Final research report as markdown string
    """
    print(f"\n🎯 Research Query: {query}")
    print(f"📊 Depth: {depth} | Max Searches: {max_searches}")
    print("="*70)

    # Initialize state for recursive research
    state = DeepResearchState(
        query=query,
        max_depth=depth,
        max_total_searches=max_searches
    )

    # Depth 1: Initial research (always performed)
    # Stage 1: Planning
    search_items = await plan_searches(query, model=model)

    # Stage 2: Execute Searches
    search_results = await execute_searches(search_items, model=model)
    state.all_summaries.extend(search_results)
    state.total_searches += len(search_results)
    state.completed_queries.extend([item['search_term'] for item in search_items])
    state.current_depth = 1
    state.summaries_by_depth[1] = list(search_results)

    # Recursive research for depth > 1
    if depth > 1:
        while state.current_depth < state.max_depth and state.total_searches < state.max_total_searches:
            print(f"\n🔄 Depth {state.current_depth + 1}/{state.max_depth}: Analyzing for follow-ups...")

            # Extract learnings and identify gaps
            new_learnings, follow_up_queries = await extract_learnings(
                search_results,
                query,
                existing_learnings=state.learnings,
                model=model
            )
            state.learnings.extend(new_learnings)
            state.gaps_by_depth[state.current_depth] = follow_up_queries

            if not follow_up_queries:
                print("✓ No knowledge gaps identified, research complete")
                break

            # Generate follow-up searches
            remaining_searches = state.max_total_searches - state.total_searches
            follow_up_items = await generate_followup_searches(
                follow_up_queries,
                state.completed_queries,
                max_searches=min(3, remaining_searches),
                model=model
            )

            if not follow_up_items:
                print("✓ No new follow-up searches needed")
                break

            # Execute follow-up searches
            print(f"\n🔍 Executing {len(follow_up_items)} follow-up searches...")
            search_results = await execute_searches(follow_up_items, model=model)
            state.all_summaries.extend(search_results)
            state.total_searches += len(search_results)
            state.completed_queries.extend([item['search_term'] for item in follow_up_items])
            state.current_depth += 1
            state.summaries_by_depth[state.current_depth] = list(search_results)

    # Stage 3: Write Report (using all gathered research)
    final_report = await write_report(query, state.all_summaries, model=model)

    # Add depth breakdown appendix if recursive research was performed
    if depth > 1 and len(state.summaries_by_depth) > 1:
        final_report += add_depth_breakdown(state)
    elif depth > 1:
        final_report += f"\n\n---\n*Research completed at depth {state.current_depth}/{depth}, {state.total_searches} total searches*"

    # Stage 4: Email (optional)
    if send_via_email and recipient:
        await send_email_report(
            recipient=recipient,
            subject=f"Research Report: {query}",
            report=final_report
        )

    return final_report


async def deep_research_with_progress(
    query: str,
    send_via_email: bool = False,
    recipient: str = None,
    model: str = None,
    depth: int = 1,
    max_searches: int = 15
) -> AsyncIterator[ProgressUpdate]:
    """
    Main Orchestrator with progress updates.
    Yields ProgressUpdate objects as research progresses.

    Args:
        query: Research question or topic
        send_via_email: Whether to email the report
        recipient: Email address (required if send_via_email=True)
        model: Claude model to use (defaults to DEFAULT_MODEL)
        depth: Research depth (1=quick single-pass, 2-3=recursive with follow-ups)
        max_searches: Maximum total searches across all depths

    Yields:
        ProgressUpdate objects tracking each stage
    """
    print(f"\n🎯 Research Query: {query}")
    print(f"📊 Depth: {depth} | Max Searches: {max_searches}")
    print("="*70)

    # Initialize token tracking and state
    tokens = TokenAccumulator()
    state = DeepResearchState(
        query=query,
        max_depth=depth,
        max_total_searches=max_searches
    )
    use_tavily = tavily_client is not None

    # Stage 1: Planning
    stage_start = time.time()
    yield ProgressUpdate(
        stage="planning",
        stage_display="Planning Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Generating search queries...",
        current_depth=1,
        max_depth=depth
    )

    search_items = await plan_searches(query, tokens, model=model)

    yield ProgressUpdate(
        stage="planning",
        stage_display="Planning Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message=f"Generated {len(search_items)} search queries",
        current_depth=1,
        max_depth=depth
    )

    # Stage 2: Initial Searching (Depth 1)
    stage_start = time.time()
    search_results = []

    for i, item in enumerate(search_items, 1):
        yield ProgressUpdate(
            stage="searching",
            stage_display=f"Research Agent (Depth 1/{depth})" + (" - Tavily" if use_tavily else " - LLM"),
            current_step=i,
            total_steps=len(search_items),
            elapsed_time=time.time() - stage_start,
            message=f"Searching: {item['search_term']}",
            current_depth=1,
            max_depth=depth,
            total_searches=state.total_searches + i
        )

        # Execute single search
        if use_tavily:
            try:
                summary = await search_with_tavily(item['search_term'], tokens, model=model)
            except Exception as e:
                print(f"      ! Tavily failed ({e}), using LLM fallback")
                summary = await search_with_llm_knowledge(item['search_term'], tokens, model=model)
        else:
            summary = await search_with_llm_knowledge(item['search_term'], tokens, model=model)

        search_results.append({
            "search_term": item['search_term'],
            "reason": item['reason'],
            "summary": summary
        })

    # Update state after initial searches
    state.all_summaries.extend(search_results)
    state.total_searches += len(search_results)
    state.completed_queries.extend([item['search_term'] for item in search_items])
    state.current_depth = 1
    state.summaries_by_depth[1] = list(search_results)

    yield ProgressUpdate(
        stage="searching",
        stage_display="Research Agent",
        current_step=len(search_items),
        total_steps=len(search_items),
        elapsed_time=time.time() - stage_start,
        message=f"Completed {len(search_items)} initial searches",
        current_depth=1,
        max_depth=depth,
        total_searches=state.total_searches
    )

    # Recursive research for depth > 1
    if depth > 1:
        while state.current_depth < state.max_depth and state.total_searches < state.max_total_searches:
            # Stage: Extract learnings
            stage_start = time.time()
            yield ProgressUpdate(
                stage="extracting",
                stage_display=f"Learning Extraction (Depth {state.current_depth}/{depth})",
                current_step=1,
                total_steps=1,
                elapsed_time=0,
                message="Analyzing findings and identifying knowledge gaps...",
                current_depth=state.current_depth,
                max_depth=depth,
                total_searches=state.total_searches,
                learnings_count=len(state.learnings)
            )

            new_learnings, follow_up_queries = await extract_learnings(
                search_results,
                query,
                existing_learnings=state.learnings,
                tokens=tokens,
                model=model
            )
            state.learnings.extend(new_learnings)
            state.gaps_by_depth[state.current_depth] = follow_up_queries

            yield ProgressUpdate(
                stage="extracting",
                stage_display="Learning Extraction",
                current_step=1,
                total_steps=1,
                elapsed_time=time.time() - stage_start,
                message=f"Extracted {len(new_learnings)} learnings, {len(follow_up_queries)} gaps identified",
                current_depth=state.current_depth,
                max_depth=depth,
                total_searches=state.total_searches,
                learnings_count=len(state.learnings),
                gaps_identified=follow_up_queries[:5]
            )

            if not follow_up_queries:
                yield ProgressUpdate(
                    stage="recursing",
                    stage_display="Recursion Check",
                    current_step=1,
                    total_steps=1,
                    elapsed_time=0,
                    message="No knowledge gaps identified, research complete",
                    current_depth=state.current_depth,
                    max_depth=depth,
                    total_searches=state.total_searches,
                    learnings_count=len(state.learnings)
                )
                break

            # Generate follow-up searches
            remaining_searches = state.max_total_searches - state.total_searches
            follow_up_items = await generate_followup_searches(
                follow_up_queries,
                state.completed_queries,
                max_searches=min(3, remaining_searches),
                tokens=tokens,
                model=model
            )

            if not follow_up_items:
                yield ProgressUpdate(
                    stage="recursing",
                    stage_display="Recursion Check",
                    current_step=1,
                    total_steps=1,
                    elapsed_time=0,
                    message="No new follow-up searches needed",
                    current_depth=state.current_depth,
                    max_depth=depth,
                    total_searches=state.total_searches,
                    learnings_count=len(state.learnings)
                )
                break

            # Execute follow-up searches
            state.current_depth += 1
            stage_start = time.time()
            search_results = []

            for i, item in enumerate(follow_up_items, 1):
                yield ProgressUpdate(
                    stage="searching",
                    stage_display=f"Follow-up Research (Depth {state.current_depth}/{depth})",
                    current_step=i,
                    total_steps=len(follow_up_items),
                    elapsed_time=time.time() - stage_start,
                    message=f"Following up: {item['search_term'][:50]}...",
                    current_depth=state.current_depth,
                    max_depth=depth,
                    total_searches=state.total_searches + i,
                    learnings_count=len(state.learnings)
                )

                if use_tavily:
                    try:
                        summary = await search_with_tavily(item['search_term'], tokens, model=model)
                    except Exception as e:
                        summary = await search_with_llm_knowledge(item['search_term'], tokens, model=model)
                else:
                    summary = await search_with_llm_knowledge(item['search_term'], tokens, model=model)

                search_results.append({
                    "search_term": item['search_term'],
                    "reason": item['reason'],
                    "summary": summary
                })

            state.all_summaries.extend(search_results)
            state.total_searches += len(search_results)
            state.completed_queries.extend([item['search_term'] for item in follow_up_items])
            state.summaries_by_depth[state.current_depth] = list(search_results)

            yield ProgressUpdate(
                stage="searching",
                stage_display=f"Follow-up Research (Depth {state.current_depth}/{depth})",
                current_step=len(follow_up_items),
                total_steps=len(follow_up_items),
                elapsed_time=time.time() - stage_start,
                message=f"Completed {len(follow_up_items)} follow-up searches",
                current_depth=state.current_depth,
                max_depth=depth,
                total_searches=state.total_searches,
                learnings_count=len(state.learnings)
            )

    # Stage 3: Writing
    stage_start = time.time()
    yield ProgressUpdate(
        stage="writing",
        stage_display="Report Writing Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message=f"Synthesizing {len(state.all_summaries)} research findings into report...",
        current_depth=state.current_depth,
        max_depth=depth,
        total_searches=state.total_searches,
        learnings_count=len(state.learnings)
    )

    final_report = await write_report(query, state.all_summaries, tokens, model=model)

    # Add depth breakdown appendix if recursive research was performed
    if depth > 1 and len(state.summaries_by_depth) > 1:
        final_report += add_depth_breakdown(state)
    elif depth > 1:
        final_report += f"\n\n---\n*Research completed at depth {state.current_depth}/{depth}, {state.total_searches} total searches, {len(state.learnings)} learnings extracted*"

    yield ProgressUpdate(
        stage="writing",
        stage_display="Report Writing Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message=f"Report generated | Tokens: {tokens.total_tokens:,}",
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
        current_depth=state.current_depth,
        max_depth=depth,
        total_searches=state.total_searches,
        learnings_count=len(state.learnings)
    )

    # Stage 4: Email (optional)
    if send_via_email and recipient:
        stage_start = time.time()
        yield ProgressUpdate(
            stage="emailing",
            stage_display="Email Agent",
            current_step=1,
            total_steps=1,
            elapsed_time=0,
            message=f"Sending report to {recipient}..."
        )

        await send_email_report(
            recipient=recipient,
            subject=f"Research Report: {query}",
            report=final_report
        )

        yield ProgressUpdate(
            stage="emailing",
            stage_display="Email Agent",
            current_step=1,
            total_steps=1,
            elapsed_time=time.time() - stage_start,
            message="Email sent successfully"
        )

    # Final complete status with report
    yield ProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Research complete!",
        report=final_report,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost,
        current_depth=state.current_depth,
        max_depth=depth,
        total_searches=state.total_searches,
        learnings_count=len(state.learnings)
    )


async def main():
    """
    Main entry point
    """
    # Customize your research query here
    query = "Latest AI Agent frameworks in 2025"

    # Option 1: Just generate and print report
    report = await deep_research(query)

    # Option 2: Generate report AND send via email (uncomment to use)
    # report = await deep_research(
    #     query=query,
    #     send_via_email=True,
    #     recipient="your@email.com"
    # )

    # Display final report
    print("\n" + "="*70)
    print("📊 FINAL REPORT")
    print("="*70)
    print(report)
    print("\n" + "="*70)
    print("✅ Research complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
