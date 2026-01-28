"""
Pydantic models for AI Information Research Agent.
Supports recursive research with domain prioritization and inline citations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class SearchCategory(str, Enum):
    """Categories for AI research queries"""
    TECHNICAL = "technical"
    BUSINESS = "business"
    POLICY = "policy"
    APPLICATIONS = "applications"


class SourceTier(str, Enum):
    """Source credibility tiers"""
    TIER_1_FOUNDATIONAL = "tier_1_foundational"
    TIER_2_STRATEGIC = "tier_2_strategic"
    TIER_3_POLICY = "tier_3_policy"
    TIER_4_PRACTITIONER = "tier_4_practitioner"
    UNCLASSIFIED = "unclassified"


# =============================================================================
# Search Query Models
# =============================================================================

class SearchQuery(BaseModel):
    """A structured search query with metadata"""
    query: str = Field(description="The search query text")
    category: SearchCategory = Field(description="Topic category")
    priority: int = Field(ge=1, le=10, description="Priority 1-10, higher = more important")
    required_domains: List[str] = Field(
        default_factory=list,
        description="Prioritized domains to include in search"
    )
    search_depth: str = Field(default="basic", description="basic or advanced")

    @property
    def max_results(self) -> int:
        """Return max results based on priority"""
        return 8 if self.priority >= 7 else 5


# =============================================================================
# Research Brief Models
# =============================================================================

class ResearchBrief(BaseModel):
    """Structured research brief from single prompt"""
    research_objective: str = Field(description="Clarified research goal")
    search_categories: List[SearchCategory] = Field(
        default_factory=list,
        description="Relevant topic categories"
    )
    initial_queries: List[SearchQuery] = Field(
        default_factory=list,
        description="List of 5-8 structured queries with priority"
    )
    domain_focus: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Prioritized domains per category"
    )
    depth_limit: int = Field(default=2, ge=1, le=3, description="Max recursion depth")
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Citation and Learning Models
# =============================================================================

class Citation(BaseModel):
    """Source citation with metadata"""
    url: str = Field(description="Source URL")
    title: str = Field(description="Page/article title")
    source_name: str = Field(description="Website or publication name")
    tier: SourceTier = Field(default=SourceTier.UNCLASSIFIED)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    accessed_at: datetime = Field(default_factory=datetime.now)


class Learning(BaseModel):
    """Key insight extracted from research"""
    content: str = Field(description="The key insight or finding")
    category: SearchCategory = Field(description="Topic category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source citations"
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Generated sub-queries for gaps"
    )
    depth: int = Field(default=0, description="Research depth at which this was discovered")


# =============================================================================
# Research State
# =============================================================================

class CircuitBreaker(BaseModel):
    """Controls for preventing runaway recursion"""
    max_depth: int = Field(default=2, description="Maximum recursion depth")
    max_total_searches: int = Field(default=20, description="Cap on total searches")
    min_new_learnings_ratio: float = Field(
        default=0.2,
        description="Minimum ratio of new learnings to trigger next depth"
    )
    current_search_count: int = Field(default=0)

    def should_continue(self, current_depth: int, new_learnings: int, total_learnings: int) -> bool:
        """Check if we should continue to next depth"""
        # Stop if max depth reached
        if current_depth >= self.max_depth:
            return False

        # Stop if max searches reached
        if self.current_search_count >= self.max_total_searches:
            return False

        # Stop if diminishing returns (not enough new learnings)
        if total_learnings > 0:
            ratio = new_learnings / max(total_learnings, 1)
            if ratio < self.min_new_learnings_ratio:
                return False

        return True


class ResearchState(BaseModel):
    """Complete state of a recursive research session"""
    brief: ResearchBrief
    learnings: List[Learning] = Field(default_factory=list)
    completed_queries: List[str] = Field(default_factory=list)
    pending_queries: List[SearchQuery] = Field(default_factory=list)
    current_depth: int = Field(default=0)
    search_results_raw: List[Dict[str, Any]] = Field(default_factory=list)

    def get_unique_citations(self) -> List[Citation]:
        """Get all unique citations across learnings"""
        seen_urls = set()
        unique = []
        for learning in self.learnings:
            for citation in learning.citations:
                if citation.url not in seen_urls:
                    seen_urls.add(citation.url)
                    unique.append(citation)
        return unique

    def build_citation_index(self) -> Dict[str, int]:
        """Build url -> [N] mapping for inline citations"""
        citations = self.get_unique_citations()
        return {c.url: i + 1 for i, c in enumerate(citations)}


# =============================================================================
# Search Result Models
# =============================================================================

class SearchResult(BaseModel):
    """Raw search result from Tavily"""
    url: str
    title: str
    content: str
    score: Optional[float] = None
    raw_content: Optional[str] = None


class SearchResponse(BaseModel):
    """Aggregated search response"""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    category: SearchCategory
    depth: int = 0


# =============================================================================
# Progress Tracking
# =============================================================================

class AIResearchProgressUpdate(BaseModel):
    """Progress update for UI display"""
    stage: str  # "planning", "searching", "extracting", "recursing", "synthesizing", "complete"
    stage_display: str  # Human-readable stage name
    current_step: int
    total_steps: int
    elapsed_time: float
    message: str

    # Research-specific tracking
    current_depth: int = 0
    max_depth: int = 2
    searches_completed: int = 0
    total_searches: int = 0
    learnings_count: int = 0

    # Final outputs
    report: Optional[str] = None

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
