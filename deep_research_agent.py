"""
Deep Research Agent
A multi-agent system for conducting in-depth web research and generating comprehensive reports.
Works standalone without external search APIs.
"""

import os
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import resend

# Load environment variables
load_dotenv()

# Initialize clients
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
resend.api_key = os.getenv("RESEND_API_KEY")

print("="*70)
print("🔬 DEEP RESEARCH AGENT")
print("="*70)

# Pydantic models for structured outputs
class WebSearchItem(BaseModel):
    """A single web search query"""
    reason: str = Field(description="Why this search is relevant to the research query")
    search_term: str = Field(description="The specific search term to use")

class WebSearchPlan(BaseModel):
    """Collection of web searches to perform"""
    searches: List[WebSearchItem] = Field(description="List of web searches to perform")

# Configuration
HOW_MANY_SEARCHES = 3
MODEL = "gpt-4o-mini"  # or "gpt-4" for better quality

async def plan_searches(query: str) -> List[Dict[str, str]]:
    """
    Planning Agent: Generate search queries based on the research topic
    """
    print(f"\n🤔 Planning searches for: {query}")
    
    prompt = f"""You are a research planning assistant. Given a research query, 
    generate {HOW_MANY_SEARCHES} specific web search terms that will help comprehensively 
    answer the query.

    Research Query: {query}
    
    For each search, provide:
    1. A clear reason why this search is relevant
    2. A specific search term optimized for web search engines
    
    Make searches diverse to cover different aspects of the topic."""
    
    response = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a research planning expert."},
            {"role": "user", "content": prompt}
        ],
        response_format=WebSearchPlan
    )
    
    search_plan = response.choices[0].message.parsed
    search_items = [
        {"reason": item.reason, "search_term": item.search_term}
        for item in search_plan.searches
    ]
    
    print(f"✓ Generated {len(search_items)} search queries")
    for i, item in enumerate(search_items, 1):
        print(f"  [{i}] {item['search_term']}")
        print(f"      → {item['reason']}")
    
    return search_items

async def execute_searches(search_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Search Agent: Use LLM's knowledge to research topics
    """
    print("\n🔍 Executing knowledge-based research...")
    search_results = []
    
    for i, item in enumerate(search_items, 1):
        print(f"  [{i}/{len(search_items)}] Researching: {item['search_term']}")
        
        # Use LLM's knowledge base for research
        research_prompt = f"""You are a knowledgeable research assistant with access to information 
        up to January 2025. Research and provide comprehensive information about: {item['search_term']}
        
        Provide a detailed summary (2-3 paragraphs, max 300 words) covering:
        - Key facts and current developments
        - Important trends or patterns
        - Relevant statistics or examples
        - Recent updates (if applicable)
        
        Be specific, factual, and cite approximate timeframes when relevant.
        Write only the summary, no additional commentary."""
        
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert researcher with deep knowledge across many domains."},
                {"role": "user", "content": research_prompt}
            ],
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        search_results.append({
            "search_term": item['search_term'],
            "reason": item['reason'],
            "summary": summary
        })
        print(f"      ✓ Research complete")
    
    return search_results

async def write_report(query: str, search_results: List[Dict[str, str]]) -> str:
    """
    Report Writing Agent: Synthesize findings into comprehensive report
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
    
    report_prompt = f"""You are an expert analyst. Synthesize the following research findings 
    into a comprehensive, well-structured report.

    Research Query: {query}
    
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
    
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert research report writer."},
            {"role": "user", "content": report_prompt}
        ],
        temperature=0.5
    )
    
    report = response.choices[0].message.content
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
    
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an HTML email formatting expert."},
            {"role": "user", "content": html_prompt}
        ],
        temperature=0.2
    )
    
    html_body = response.choices[0].message.content
    
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

async def main():
    """
    Main entry point
    """
    # Customize your research query here
    query = "Latest AI Agent frameworks in 2025"
    
    # Option 1: Just generate and print report
    # report = await deep_research(query)
    
    # Option 2: Generate report AND send via email (uncomment to use)
    report = await deep_research(
        query=query,
        send_via_email=True,
        recipient="aidinsahneh19@gmail.com"
    )
    
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
