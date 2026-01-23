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
from dataclasses import dataclass
from typing import List, Dict, Optional, AsyncIterator
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
import resend

from market_context_2026 import DEEP_RESEARCH_2026_CONTEXT

# Load environment variables
load_dotenv()

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

print("="*70)
print("🔬 DEEP RESEARCH AGENT")
if tavily_client:
    print("📡 Tavily API: Enabled (real-time web search)")
else:
    print("📚 Tavily API: Disabled (using LLM knowledge)")
print("="*70)

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
    stage: str           # "planning", "searching", "writing", "emailing", "complete"
    stage_display: str   # Human-readable stage name
    current_step: int    # Current step within stage (1-based)
    total_steps: int     # Total steps in stage
    elapsed_time: float  # Seconds elapsed for current stage
    message: str         # Detailed status message
    report: Optional[str] = None  # Final report (only set when complete)

# Configuration
HOW_MANY_SEARCHES = 3
MODEL = "claude-sonnet-4-20250514"  # or "claude-opus-4-20250514" for better quality

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

async def plan_searches(query: str) -> List[Dict[str, str]]:
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
        model=MODEL,
        max_tokens=1024,
        system="You are a research planning expert. Today's date is January 2026. Always use the provided tool to structure your response.",
        messages=[{"role": "user", "content": prompt}],
        tools=[SEARCH_PLAN_TOOL],
        tool_choice={"type": "tool", "name": "create_search_plan"}
    )

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

async def search_with_tavily(search_term: str) -> str:
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
        model=MODEL,
        max_tokens=1024,
        system="You are an expert researcher synthesizing web search results into concise summaries.",
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.3
    )

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


async def search_with_llm_knowledge(search_term: str) -> str:
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
        model=MODEL,
        max_tokens=1024,
        system="You are an expert researcher with deep knowledge across many domains. Your knowledge extends to January 2026. When discussing tech, AI, or semiconductors, consider the 2026 market context including BIS export controls, 2nm process node competition, and Agentic AI adoption trends.",
        messages=[{"role": "user", "content": research_prompt}],
        temperature=0.3
    )

    return response.content[0].text


async def execute_searches(search_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
                summary = await search_with_tavily(item['search_term'])
            except Exception as e:
                print(f"      ! Tavily failed ({e}), using LLM fallback")
                summary = await search_with_llm_knowledge(item['search_term'])
        else:
            summary = await search_with_llm_knowledge(item['search_term'])

        search_results.append({
            "search_term": item['search_term'],
            "reason": item['reason'],
            "summary": summary
        })
        print(f"      ✓ Search complete")

    return search_results

async def write_report(query: str, search_results: List[Dict[str, str]]) -> str:
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
        model=MODEL,
        max_tokens=4096,
        system="You are an expert research report writer. Today's date is January 2026.",
        messages=[{"role": "user", "content": report_prompt}],
        temperature=0.5
    )

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
        model=MODEL,
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
    recipient: str = None
) -> str:
    """
    Main Orchestrator: Coordinate all agents to perform deep research

    Args:
        query: Research question or topic
        send_via_email: Whether to email the report
        recipient: Email address (required if send_via_email=True)

    Returns:
        Final research report as markdown string
    """
    print(f"\n🎯 Research Query: {query}")
    print("="*70)

    # Stage 1: Planning
    search_items = await plan_searches(query)

    # Stage 2: Execute Searches
    search_results = await execute_searches(search_items)

    # Stage 3: Write Report
    final_report = await write_report(query, search_results)

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
    recipient: str = None
) -> AsyncIterator[ProgressUpdate]:
    """
    Main Orchestrator with progress updates.
    Yields ProgressUpdate objects as research progresses.

    Args:
        query: Research question or topic
        send_via_email: Whether to email the report
        recipient: Email address (required if send_via_email=True)

    Yields:
        ProgressUpdate objects tracking each stage
    """
    print(f"\n🎯 Research Query: {query}")
    print("="*70)

    # Stage 1: Planning
    stage_start = time.time()
    yield ProgressUpdate(
        stage="planning",
        stage_display="Planning Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Generating search queries..."
    )

    search_items = await plan_searches(query)

    yield ProgressUpdate(
        stage="planning",
        stage_display="Planning Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message=f"Generated {len(search_items)} search queries"
    )

    # Stage 2: Searching
    stage_start = time.time()
    total_searches = len(search_items)
    search_results = []
    use_tavily = tavily_client is not None

    for i, item in enumerate(search_items, 1):
        yield ProgressUpdate(
            stage="searching",
            stage_display="Research Agent" + (" (Tavily)" if use_tavily else " (LLM)"),
            current_step=i,
            total_steps=total_searches,
            elapsed_time=time.time() - stage_start,
            message=f"Searching: {item['search_term']}"
        )

        # Execute single search
        if use_tavily:
            try:
                summary = await search_with_tavily(item['search_term'])
            except Exception as e:
                print(f"      ! Tavily failed ({e}), using LLM fallback")
                summary = await search_with_llm_knowledge(item['search_term'])
        else:
            summary = await search_with_llm_knowledge(item['search_term'])

        search_results.append({
            "search_term": item['search_term'],
            "reason": item['reason'],
            "summary": summary
        })

    yield ProgressUpdate(
        stage="searching",
        stage_display="Research Agent",
        current_step=total_searches,
        total_steps=total_searches,
        elapsed_time=time.time() - stage_start,
        message=f"Completed {total_searches} searches"
    )

    # Stage 3: Writing
    stage_start = time.time()
    yield ProgressUpdate(
        stage="writing",
        stage_display="Report Writing Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Synthesizing research into report..."
    )

    final_report = await write_report(query, search_results)

    yield ProgressUpdate(
        stage="writing",
        stage_display="Report Writing Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - stage_start,
        message="Report generated"
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
        report=final_report
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
