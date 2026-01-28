"""
AI Information Research Agent
A specialized multi-agent system for conducting comprehensive AI research with:
- Recursive research with depth control
- Domain prioritization via tiered sources
- Parallel execution of searches
- Inline citations with reference tracking

Updated for January 2026 AI landscape.
"""

import os
import asyncio
import time
from typing import List, Dict, Optional, AsyncIterator, Any
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from models.ai_research_models import (
    ResearchBrief,
    SearchQuery,
    SearchCategory,
    Learning,
    Citation,
    ResearchState,
    CircuitBreaker,
    SearchResult,
    SearchResponse,
    AIResearchProgressUpdate,
    SourceTier,
)
from services.ai_domain_context import (
    classify_source_tier,
    get_domains_for_category,
    calculate_relevance_score,
    get_ai_research_system_prompt,
    get_priority_boost,
)
from utils.logging_config import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Initialize clients
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize Tavily client (optional)
tavily_client = None
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key and tavily_api_key != "your_tavily_api_key_here":
    try:
        from tavily import AsyncTavilyClient
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    except ImportError:
        logger.warning("tavily-python not installed. Using LLM knowledge fallback.")

# Configuration
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_CONCURRENT_SEARCHES = 5
SEARCH_COOLDOWN_MS = 200


# =============================================================================
# Token Tracking
# =============================================================================

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


# =============================================================================
# Tool Definitions
# =============================================================================

RESEARCH_BRIEF_TOOL = {
    "name": "create_research_brief",
    "description": "Create a structured research brief with search queries and priorities",
    "input_schema": {
        "type": "object",
        "properties": {
            "research_objective": {
                "type": "string",
                "description": "Clarified and specific research goal"
            },
            "search_categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["technical", "business", "policy", "applications"]
                },
                "description": "Relevant topic categories"
            },
            "initial_queries": {
                "type": "array",
                "description": "List of 5-8 structured search queries",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query text"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["technical", "business", "policy", "applications"]
                        },
                        "priority": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Priority 1-10, higher = more important"
                        }
                    },
                    "required": ["query", "category", "priority"]
                }
            },
            "depth_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "Maximum recursion depth (default 2)"
            }
        },
        "required": ["research_objective", "search_categories", "initial_queries"]
    }
}

LEARNING_EXTRACTION_TOOL = {
    "name": "extract_learnings",
    "description": "Extract key learnings and identify follow-up queries from search results",
    "input_schema": {
        "type": "object",
        "properties": {
            "learnings": {
                "type": "array",
                "description": "Key insights extracted from search results",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The key insight or finding"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["technical", "business", "policy", "applications"]
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence score 0-1"
                        },
                        "source_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs supporting this learning"
                        },
                        "follow_up_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Questions raised that need further research"
                        }
                    },
                    "required": ["content", "category", "confidence", "source_urls"]
                }
            }
        },
        "required": ["learnings"]
    }
}


# =============================================================================
# Phase 1: Enhanced Planner Agent
# =============================================================================

async def create_research_brief(
    query: str,
    tokens: TokenAccumulator,
    model: str = None
) -> ResearchBrief:
    """
    Planning Agent: Generate structured research brief from a single prompt.

    Args:
        query: User's research question
        tokens: Token accumulator for tracking
        model: Claude model to use

    Returns:
        ResearchBrief with structured queries and priorities
    """
    logger.info(f"Creating research brief for: {query}")

    prompt = f"""Analyze this research query and create a comprehensive research plan.

Research Query: {query}

Create a research brief that includes:
1. A clarified research objective (be specific and actionable)
2. Relevant search categories from: technical, business, policy, applications
3. 5-8 diverse search queries covering different aspects:
   - Include both broad overview queries and specific technical queries
   - Prioritize queries that will find authoritative AI sources
   - Set priority 8-10 for core queries, 5-7 for supporting queries, 1-4 for nice-to-have
4. Recommended depth limit (1-3, where 2 is typical for comprehensive research)

For AI research, prioritize:
- Academic papers (arxiv, NeurIPS, ICML)
- Major AI lab publications (OpenAI, Anthropic, DeepMind, Google Research)
- Technical blogs from practitioners
- Business analysis from reputable sources

Use the create_research_brief tool to structure your response."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=2048,
        system=get_ai_research_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
        tools=[RESEARCH_BRIEF_TOOL],
        tool_choice={"type": "tool", "name": "create_research_brief"}
    )

    tokens.add(response)

    # Extract tool use result
    tool_use = next(block for block in response.content if block.type == "tool_use")
    brief_data = tool_use.input

    # Convert to Pydantic model
    initial_queries = []
    for q in brief_data.get("initial_queries", []):
        query_obj = SearchQuery(
            query=q["query"],
            category=SearchCategory(q["category"]),
            priority=q["priority"],
            required_domains=get_domains_for_category(SearchCategory(q["category"]))
        )
        initial_queries.append(query_obj)

    # Sort by priority (highest first)
    initial_queries.sort(key=lambda x: x.priority, reverse=True)

    brief = ResearchBrief(
        research_objective=brief_data["research_objective"],
        search_categories=[SearchCategory(c) for c in brief_data.get("search_categories", [])],
        initial_queries=initial_queries,
        depth_limit=brief_data.get("depth_limit", 2)
    )

    logger.info(f"Created brief with {len(initial_queries)} queries, depth_limit={brief.depth_limit}")
    return brief


# =============================================================================
# Phase 2: Parallel Execution
# =============================================================================

async def execute_single_search(
    query: SearchQuery,
    semaphore: asyncio.Semaphore,
    tokens: TokenAccumulator,
    model: str = None
) -> SearchResponse:
    """
    Execute a single search with rate limiting.

    Args:
        query: SearchQuery to execute
        semaphore: Asyncio semaphore for concurrency control
        tokens: Token accumulator
        model: Claude model for synthesis

    Returns:
        SearchResponse with results
    """
    async with semaphore:
        results = []

        if tavily_client:
            try:
                # Execute Tavily search with domain prioritization
                search_config = {
                    "query": query.query,
                    "search_depth": query.search_depth,
                    "max_results": query.max_results,
                    "include_raw_content": True,
                }

                # Add domain filtering if we have required domains
                if query.required_domains:
                    search_config["include_domains"] = query.required_domains[:5]  # Tavily limit

                response = await tavily_client.search(**search_config)

                for result in response.get("results", []):
                    results.append(SearchResult(
                        url=result.get("url", ""),
                        title=result.get("title", "Unknown"),
                        content=result.get("content", ""),
                        score=result.get("score"),
                        raw_content=result.get("raw_content")
                    ))

                # Cooldown between requests
                await asyncio.sleep(SEARCH_COOLDOWN_MS / 1000)

            except Exception as e:
                logger.warning(f"Tavily search failed for '{query.query}': {e}")
                # Fall back to LLM knowledge
                results = await _search_with_llm_fallback(query, tokens, model)
        else:
            # Use LLM knowledge fallback
            results = await _search_with_llm_fallback(query, tokens, model)

        return SearchResponse(
            query=query.query,
            results=results,
            category=query.category
        )


async def _search_with_llm_fallback(
    query: SearchQuery,
    tokens: TokenAccumulator,
    model: str = None
) -> List[SearchResult]:
    """Fallback to LLM knowledge when Tavily unavailable."""
    prompt = f"""Research and provide comprehensive information about: {query.query}

Provide detailed information covering:
- Key facts and current developments (as of January 2026)
- Important trends or patterns
- Relevant examples or case studies
- Notable organizations or researchers in this area

Be specific and factual. Focus on AI/ML topics."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1024,
        system=get_ai_research_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    tokens.add(response)

    # Create a synthetic result from LLM knowledge
    return [SearchResult(
        url="llm://knowledge-base",
        title=f"LLM Knowledge: {query.query}",
        content=response.content[0].text,
        score=0.7
    )]


async def execute_searches_parallel(
    queries: List[SearchQuery],
    tokens: TokenAccumulator,
    max_concurrent: int = MAX_CONCURRENT_SEARCHES,
    model: str = None
) -> List[SearchResponse]:
    """
    Execute multiple searches in parallel with rate limiting.

    Args:
        queries: List of SearchQuery objects
        tokens: Token accumulator
        max_concurrent: Maximum concurrent searches
        model: Claude model to use

    Returns:
        List of SearchResponse objects
    """
    logger.info(f"Executing {len(queries)} searches in parallel (max concurrent: {max_concurrent})")

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        execute_single_search(query, semaphore, tokens, model)
        for query in queries
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and log them
    valid_responses = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            logger.error(f"Search failed for query {i}: {resp}")
        else:
            valid_responses.append(resp)

    logger.info(f"Completed {len(valid_responses)}/{len(queries)} searches successfully")
    return valid_responses


# =============================================================================
# Phase 3: Learning Extraction & Recursive Research
# =============================================================================

async def extract_learnings(
    search_responses: List[SearchResponse],
    existing_learnings: List[Learning],
    tokens: TokenAccumulator,
    current_depth: int = 0,
    model: str = None
) -> List[Learning]:
    """
    Extract key learnings from search results and identify gaps.

    Args:
        search_responses: Results from parallel searches
        existing_learnings: Learnings from previous depths
        tokens: Token accumulator
        current_depth: Current recursion depth
        model: Claude model

    Returns:
        List of new Learning objects
    """
    logger.info(f"Extracting learnings from {len(search_responses)} search responses")

    # Format search results for Claude
    formatted_results = []
    url_to_metadata = {}  # Track metadata for citation building

    for response in search_responses:
        for result in response.results:
            formatted_results.append(
                f"Query: {response.query}\n"
                f"Category: {response.category.value}\n"
                f"Source: {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.content[:2000]}...\n"
                f"---"
            )
            # Track metadata
            url_to_metadata[result.url] = {
                "title": result.title,
                "category": response.category,
                "score": result.score
            }

    # Format existing learnings summary
    existing_summary = ""
    if existing_learnings:
        existing_points = [f"- {l.content[:100]}..." for l in existing_learnings[:10]]
        existing_summary = f"\n\nExisting learnings (avoid duplicates):\n" + "\n".join(existing_points)

    prompt = f"""Analyze these search results and extract key learnings.

Search Results:
{"".join(formatted_results)}
{existing_summary}

For each learning:
1. Extract the key insight or finding
2. Classify by category (technical, business, policy, applications)
3. Rate confidence 0-1 based on source quality and consistency
4. List source URLs that support this learning
5. Identify 0-2 follow-up questions if this learning reveals gaps

Focus on:
- Novel insights not covered by existing learnings
- Factual claims with clear evidence
- Trends or patterns across multiple sources
- Areas where sources disagree (note the disagreement)

Use the extract_learnings tool to structure your response."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=4096,
        system=get_ai_research_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
        tools=[LEARNING_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_learnings"}
    )

    tokens.add(response)

    # Extract tool use result
    tool_use = next(block for block in response.content if block.type == "tool_use")
    learnings_data = tool_use.input.get("learnings", [])

    # Convert to Pydantic models with proper citations
    new_learnings = []
    for learning_data in learnings_data:
        citations = []
        for url in learning_data.get("source_urls", []):
            metadata = url_to_metadata.get(url, {})
            tier = classify_source_tier(url)
            citations.append(Citation(
                url=url,
                title=metadata.get("title", "Unknown"),
                source_name=_extract_source_name(url),
                tier=tier,
                relevance_score=calculate_relevance_score(url, learning_data.get("category"))
            ))

        learning = Learning(
            content=learning_data["content"],
            category=SearchCategory(learning_data["category"]),
            confidence=learning_data["confidence"],
            citations=citations,
            follow_up_queries=learning_data.get("follow_up_queries", []),
            depth=current_depth
        )
        new_learnings.append(learning)

    logger.info(f"Extracted {len(new_learnings)} new learnings at depth {current_depth}")
    return new_learnings


def _extract_source_name(url: str) -> str:
    """Extract human-readable source name from URL."""
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        # Map common domains to readable names
        name_map = {
            "arxiv.org": "arXiv",
            "openai.com": "OpenAI",
            "anthropic.com": "Anthropic",
            "deepmind.com": "DeepMind",
            "deepmind.google": "DeepMind",
            "huggingface.co": "Hugging Face",
            "github.com": "GitHub",
            "bloomberg.com": "Bloomberg",
            "reuters.com": "Reuters",
        }
        return name_map.get(domain, domain.split(".")[0].title())
    except Exception:
        return "Unknown"


async def recursive_research(
    state: ResearchState,
    circuit_breaker: CircuitBreaker,
    tokens: TokenAccumulator,
    model: str = None,
    progress_callback=None
) -> ResearchState:
    """
    Recursive research controller with circuit breakers.

    Args:
        state: Current research state
        circuit_breaker: Controls for preventing runaway recursion
        tokens: Token accumulator
        model: Claude model
        progress_callback: Optional callback for progress updates

    Returns:
        Updated ResearchState
    """
    while state.pending_queries and circuit_breaker.should_continue(
        state.current_depth,
        len([l for l in state.learnings if l.depth == state.current_depth]),
        len(state.learnings)
    ):
        logger.info(f"Recursive research depth {state.current_depth}, "
                    f"{len(state.pending_queries)} pending queries")

        # Execute current batch of searches
        responses = await execute_searches_parallel(
            state.pending_queries[:10],  # Limit batch size
            tokens,
            model=model
        )

        circuit_breaker.current_search_count += len(responses)

        # Track completed queries
        for query in state.pending_queries[:10]:
            state.completed_queries.append(query.query)
        state.pending_queries = state.pending_queries[10:]

        # Store raw results
        for response in responses:
            state.search_results_raw.append({
                "query": response.query,
                "category": response.category.value,
                "results": [r.model_dump() for r in response.results]
            })

        # Extract learnings
        new_learnings = await extract_learnings(
            responses,
            state.learnings,
            tokens,
            state.current_depth,
            model
        )

        state.learnings.extend(new_learnings)

        # Generate follow-up queries for next depth
        if state.current_depth < state.brief.depth_limit - 1:
            follow_ups = []
            for learning in new_learnings:
                for fq in learning.follow_up_queries:
                    if fq not in state.completed_queries:
                        follow_ups.append(SearchQuery(
                            query=fq,
                            category=learning.category,
                            priority=5,  # Lower priority for follow-ups
                            required_domains=get_domains_for_category(learning.category)
                        ))

            # Add unique follow-ups to pending
            existing_queries = {q.query for q in state.pending_queries}
            for fq in follow_ups[:5]:  # Limit follow-ups per depth
                if fq.query not in existing_queries:
                    state.pending_queries.append(fq)

        # Move to next depth if no more queries at current depth
        if not state.pending_queries or all(
            q.query in state.completed_queries for q in state.pending_queries
        ):
            state.current_depth += 1

        # Progress callback if provided
        if progress_callback:
            progress_callback(state, circuit_breaker)

    return state


# =============================================================================
# Phase 4: Synthesis Engine
# =============================================================================

async def synthesize_report(
    state: ResearchState,
    tokens: TokenAccumulator,
    model: str = None
) -> str:
    """
    Synthesize learnings into markdown report with inline citations.

    Args:
        state: Complete research state
        tokens: Token accumulator
        model: Claude model

    Returns:
        Markdown report with [N] citation markers
    """
    logger.info(f"Synthesizing report from {len(state.learnings)} learnings")

    # Build citation index
    citation_index = state.build_citation_index()
    unique_citations = state.get_unique_citations()

    # Format learnings with citation markers
    formatted_learnings = []
    for learning in state.learnings:
        citation_markers = []
        for c in learning.citations:
            if c.url in citation_index:
                citation_markers.append(f"[{citation_index[c.url]}]")

        markers_str = "".join(citation_markers) if citation_markers else ""
        tier_emoji = {
            SourceTier.TIER_1_FOUNDATIONAL: "🎓",
            SourceTier.TIER_2_STRATEGIC: "📊",
            SourceTier.TIER_3_POLICY: "📜",
            SourceTier.TIER_4_PRACTITIONER: "💻",
            SourceTier.UNCLASSIFIED: "📄",
        }.get(learning.citations[0].tier if learning.citations else SourceTier.UNCLASSIFIED, "📄")

        formatted_learnings.append(
            f"- [{learning.category.value.upper()}] {learning.content} {markers_str} "
            f"(confidence: {learning.confidence:.0%}) {tier_emoji}"
        )

    # Format reference section
    references = []
    for citation in unique_citations:
        idx = citation_index[citation.url]
        tier_label = citation.tier.value.replace("_", " ").title()
        references.append(
            f"[{idx}] [{citation.title}]({citation.url}) - *{citation.source_name}* ({tier_label})"
        )

    prompt = f"""Create a comprehensive research report synthesizing these findings.

Research Objective: {state.brief.research_objective}

Key Learnings (with citation markers):
{chr(10).join(formatted_learnings)}

IMPORTANT: Preserve the [N] citation markers in your text. When you mention a finding, include the relevant [N] marker inline.

Structure your report as:
1. **Executive Summary** - 2-3 sentences with key takeaways
2. **Key Findings** - Organized by theme with subheadings, include citation markers
3. **Detailed Analysis** - Synthesize information across categories
4. **Implications & Outlook** - What this means for the field
5. **Research Gaps** - Areas needing further investigation

Use markdown formatting. Be thorough and insightful. Target 800-1200 words."""

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=4096,
        system=get_ai_research_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    tokens.add(response)

    report = response.content[0].text

    # Append references section
    report += "\n\n---\n\n## References\n\n" + "\n".join(references)

    # Add metadata footer
    report += f"""

---

*Research completed at depth {state.current_depth + 1}/{state.brief.depth_limit}*
*Total searches: {len(state.completed_queries)} | Learnings: {len(state.learnings)}*
*Sources: {len(unique_citations)} unique citations*
"""

    return report


# =============================================================================
# Main Orchestrators
# =============================================================================

async def ai_research(
    query: str,
    depth: int = 2,
    max_searches: int = 15,
    model: str = None
) -> str:
    """
    Main entry point for AI research.

    Args:
        query: Research question
        depth: Maximum recursion depth (1-3)
        max_searches: Maximum total searches
        model: Claude model to use

    Returns:
        Final research report as markdown
    """
    logger.info(f"Starting AI research: {query}")
    tokens = TokenAccumulator()

    # Phase 1: Create research brief
    brief = await create_research_brief(query, tokens, model)
    brief.depth_limit = min(depth, 3)

    # Initialize state and circuit breaker
    state = ResearchState(
        brief=brief,
        pending_queries=brief.initial_queries.copy(),
        current_depth=0
    )

    circuit_breaker = CircuitBreaker(
        max_depth=brief.depth_limit,
        max_total_searches=max_searches
    )

    # Phase 2-3: Recursive research
    state = await recursive_research(state, circuit_breaker, tokens, model)

    # Phase 4: Synthesize report
    report = await synthesize_report(state, tokens, model)

    logger.info(f"AI research complete. Tokens: {tokens.total_tokens}, Cost: ${tokens.estimated_cost:.4f}")
    return report


async def ai_research_with_progress(
    query: str,
    depth: int = 2,
    max_searches: int = 15,
    model: str = None
) -> AsyncIterator[AIResearchProgressUpdate]:
    """
    Main entry point with progress streaming.

    Args:
        query: Research question
        depth: Maximum recursion depth (1-3)
        max_searches: Maximum total searches
        model: Claude model to use

    Yields:
        AIResearchProgressUpdate objects
    """
    logger.info(f"Starting AI research with progress: {query}")
    tokens = TokenAccumulator()
    start_time = time.time()

    # Phase 1: Planning
    yield AIResearchProgressUpdate(
        stage="planning",
        stage_display="Planning Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=0,
        message="Creating research brief...",
        current_depth=0,
        max_depth=depth
    )

    brief = await create_research_brief(query, tokens, model)
    brief.depth_limit = min(depth, 3)

    yield AIResearchProgressUpdate(
        stage="planning",
        stage_display="Planning Agent",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message=f"Created {len(brief.initial_queries)} search queries",
        current_depth=0,
        max_depth=brief.depth_limit,
        total_searches=len(brief.initial_queries)
    )

    # Initialize state and circuit breaker
    state = ResearchState(
        brief=brief,
        pending_queries=brief.initial_queries.copy(),
        current_depth=0
    )

    circuit_breaker = CircuitBreaker(
        max_depth=brief.depth_limit,
        max_total_searches=max_searches
    )

    # Phase 2-3: Recursive research with progress
    while state.pending_queries and circuit_breaker.should_continue(
        state.current_depth,
        len([l for l in state.learnings if l.depth == state.current_depth]),
        len(state.learnings)
    ):
        batch = state.pending_queries[:10]

        yield AIResearchProgressUpdate(
            stage="searching",
            stage_display=f"Research Agent (Depth {state.current_depth + 1})",
            current_step=circuit_breaker.current_search_count + 1,
            total_steps=min(max_searches, len(state.pending_queries) + circuit_breaker.current_search_count),
            elapsed_time=time.time() - start_time,
            message=f"Searching: {batch[0].query[:50]}...",
            current_depth=state.current_depth,
            max_depth=brief.depth_limit,
            searches_completed=circuit_breaker.current_search_count,
            total_searches=max_searches,
            learnings_count=len(state.learnings)
        )

        # Execute batch
        responses = await execute_searches_parallel(batch, tokens, model=model)
        circuit_breaker.current_search_count += len(responses)

        # Track completed
        for query_obj in batch:
            state.completed_queries.append(query_obj.query)
        state.pending_queries = state.pending_queries[10:]

        # Store raw results
        for response in responses:
            state.search_results_raw.append({
                "query": response.query,
                "category": response.category.value,
                "results": [r.model_dump() for r in response.results]
            })

        yield AIResearchProgressUpdate(
            stage="extracting",
            stage_display="Learning Extraction",
            current_step=circuit_breaker.current_search_count,
            total_steps=max_searches,
            elapsed_time=time.time() - start_time,
            message="Extracting insights from search results...",
            current_depth=state.current_depth,
            max_depth=brief.depth_limit,
            searches_completed=circuit_breaker.current_search_count,
            learnings_count=len(state.learnings)
        )

        # Extract learnings
        new_learnings = await extract_learnings(
            responses, state.learnings, tokens, state.current_depth, model
        )
        state.learnings.extend(new_learnings)

        # Generate follow-ups
        if state.current_depth < brief.depth_limit - 1:
            follow_ups = []
            for learning in new_learnings:
                for fq in learning.follow_up_queries:
                    if fq not in state.completed_queries:
                        follow_ups.append(SearchQuery(
                            query=fq,
                            category=learning.category,
                            priority=5,
                            required_domains=get_domains_for_category(learning.category)
                        ))

            existing = {q.query for q in state.pending_queries}
            for fq in follow_ups[:5]:
                if fq.query not in existing:
                    state.pending_queries.append(fq)

        # Check depth progression
        if not state.pending_queries:
            state.current_depth += 1

        yield AIResearchProgressUpdate(
            stage="recursing",
            stage_display=f"Depth {state.current_depth + 1}",
            current_step=circuit_breaker.current_search_count,
            total_steps=max_searches,
            elapsed_time=time.time() - start_time,
            message=f"Extracted {len(new_learnings)} learnings, {len(state.pending_queries)} follow-ups",
            current_depth=state.current_depth,
            max_depth=brief.depth_limit,
            searches_completed=circuit_breaker.current_search_count,
            learnings_count=len(state.learnings)
        )

    # Phase 4: Synthesis
    yield AIResearchProgressUpdate(
        stage="synthesizing",
        stage_display="Synthesis Engine",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message=f"Synthesizing {len(state.learnings)} learnings into report...",
        current_depth=state.current_depth,
        max_depth=brief.depth_limit,
        searches_completed=circuit_breaker.current_search_count,
        learnings_count=len(state.learnings)
    )

    report = await synthesize_report(state, tokens, model)

    # Complete
    yield AIResearchProgressUpdate(
        stage="complete",
        stage_display="Complete",
        current_step=1,
        total_steps=1,
        elapsed_time=time.time() - start_time,
        message="AI research complete!",
        current_depth=state.current_depth,
        max_depth=brief.depth_limit,
        searches_completed=circuit_breaker.current_search_count,
        learnings_count=len(state.learnings),
        report=report,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.total_tokens,
        estimated_cost=tokens.estimated_cost
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for testing."""
    query = "Latest advances in reasoning models and their enterprise applications"

    print(f"\n🤖 AI Research Agent")
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    async for update in ai_research_with_progress(query, depth=2, max_searches=15):
        if update.stage != "complete":
            print(f"[{update.stage_display}] {update.message}")
        else:
            print("\n" + "=" * 70)
            print("📊 FINAL REPORT")
            print("=" * 70)
            print(update.report)
            print("\n" + "=" * 70)
            print(f"✅ Research complete!")
            print(f"📊 Tokens: {update.total_tokens:,} (${update.estimated_cost:.4f})")
            print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
