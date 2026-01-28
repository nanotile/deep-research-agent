"""
AI Domain Context Module
Provides tiered domain configuration for AI Information Research Agent.
Domains are prioritized by credibility and relevance to AI research.
"""

from typing import Dict, List, Optional
from urllib.parse import urlparse

from models.ai_research_models import SourceTier, SearchCategory


# =============================================================================
# AI Domain Tier Configuration
# =============================================================================

AI_DOMAIN_TIERS: Dict[str, Dict] = {
    "TIER_1_FOUNDATIONAL": {
        "description": "Academic institutions, top AI labs, arxiv",
        "domains": [
            "bair.berkeley.edu",
            "news.mit.edu",
            "ai.stanford.edu",
            "cs.stanford.edu",
            "seas.harvard.edu",
            "cs.cmu.edu",
            "deepmind.google",
            "deepmind.com",
            "openai.com",
            "research.google",
            "ai.google",
            "research.facebook.com",
            "ai.meta.com",
            "anthropic.com",
            "mistral.ai",
            "arxiv.org",
            "nature.com",
            "science.org",
            "proceedings.neurips.cc",
            "proceedings.mlr.press",
            "aclanthology.org",
            "semianalysis.com",
        ],
        "priority_boost": 1.5,
        "tier": SourceTier.TIER_1_FOUNDATIONAL,
    },
    "TIER_2_STRATEGIC": {
        "description": "Business intelligence, tech news, analyst firms",
        "domains": [
            "theinformation.com",
            "stratechery.com",
            "arstechnica.com",
            "wired.com",
            "technologyreview.com",
            "spectrum.ieee.org",
            "deloitte.com",
            "mckinsey.com",
            "gartner.com",
            "forrester.com",
            "idc.com",
            "bloomberg.com",
            "reuters.com",
            "ft.com",
            "wsj.com",
        ],
        "priority_boost": 1.3,
        "tier": SourceTier.TIER_2_STRATEGIC,
    },
    "TIER_3_POLICY": {
        "description": "AI policy, ethics, governance organizations",
        "domains": [
            "caidp.org",
            "iaps.ai",
            "futureoflife.org",
            "ainowinstitute.org",
            "cset.georgetown.edu",
            "hai.stanford.edu",
            "whitehouse.gov",
            "nist.gov",
            "ec.europa.eu",
            "oecd.org",
            "brookings.edu",
            "rand.org",
            "carnegieendowment.org",
            "cfr.org",
        ],
        "priority_boost": 1.2,
        "tier": SourceTier.TIER_3_POLICY,
    },
    "TIER_4_PRACTITIONER": {
        "description": "Developer blogs, tutorials, practitioner content",
        "domains": [
            "huggingface.co",
            "paperswithcode.com",
            "distill.pub",
            "colah.github.io",
            "lilianweng.github.io",
            "blog.google",
            "aws.amazon.com/blogs",
            "azure.microsoft.com/blog",
            "cloud.google.com/blog",
            "kdnuggets.com",
            "towardsdatascience.com",
            "medium.com",
            "substack.com",
            "openai.com/blog",
            "anthropic.com/news",
            "blog.langchain.dev",
            "llamaindex.ai",
        ],
        "priority_boost": 1.0,
        "tier": SourceTier.TIER_4_PRACTITIONER,
    },
}


# =============================================================================
# Category-Specific Domain Preferences
# =============================================================================

CATEGORY_DOMAIN_PREFERENCES: Dict[SearchCategory, List[str]] = {
    SearchCategory.TECHNICAL: [
        "arxiv.org",
        "deepmind.google",
        "openai.com",
        "research.google",
        "proceedings.neurips.cc",
        "huggingface.co",
        "paperswithcode.com",
        "anthropic.com",
    ],
    SearchCategory.BUSINESS: [
        "theinformation.com",
        "bloomberg.com",
        "mckinsey.com",
        "gartner.com",
        "stratechery.com",
        "semianalysis.com",
        "reuters.com",
    ],
    SearchCategory.POLICY: [
        "futureoflife.org",
        "ainowinstitute.org",
        "cset.georgetown.edu",
        "hai.stanford.edu",
        "whitehouse.gov",
        "nist.gov",
        "brookings.edu",
    ],
    SearchCategory.APPLICATIONS: [
        "huggingface.co",
        "blog.langchain.dev",
        "aws.amazon.com",
        "azure.microsoft.com",
        "cloud.google.com",
        "towardsdatascience.com",
    ],
}


# =============================================================================
# Helper Functions
# =============================================================================

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def classify_source_tier(url: str) -> SourceTier:
    """
    Classify a URL into a source tier.

    Args:
        url: The URL to classify

    Returns:
        SourceTier enum value
    """
    domain = extract_domain(url)
    if not domain:
        return SourceTier.UNCLASSIFIED

    # Check each tier
    for tier_name, tier_config in AI_DOMAIN_TIERS.items():
        for tier_domain in tier_config["domains"]:
            # Check for exact match or subdomain match
            if domain == tier_domain or domain.endswith("." + tier_domain):
                return tier_config["tier"]
            # Also check if tier_domain is contained (for subpaths like openai.com/blog)
            if tier_domain in domain:
                return tier_config["tier"]

    return SourceTier.UNCLASSIFIED


def get_priority_boost(tier: SourceTier) -> float:
    """Get priority boost multiplier for a tier"""
    tier_boosts = {
        SourceTier.TIER_1_FOUNDATIONAL: 1.5,
        SourceTier.TIER_2_STRATEGIC: 1.3,
        SourceTier.TIER_3_POLICY: 1.2,
        SourceTier.TIER_4_PRACTITIONER: 1.0,
        SourceTier.UNCLASSIFIED: 0.8,
    }
    return tier_boosts.get(tier, 0.8)


def get_domains_for_category(category: SearchCategory) -> List[str]:
    """Get prioritized domains for a search category"""
    return CATEGORY_DOMAIN_PREFERENCES.get(category, [])


def get_all_priority_domains() -> List[str]:
    """Get all domains across all tiers (for Tavily include_domains)"""
    all_domains = []
    for tier_config in AI_DOMAIN_TIERS.values():
        all_domains.extend(tier_config["domains"])
    return list(set(all_domains))


def calculate_relevance_score(url: str, category: Optional[SearchCategory] = None) -> float:
    """
    Calculate relevance score for a URL based on tier and category match.

    Args:
        url: The URL to score
        category: Optional search category for bonus scoring

    Returns:
        Float score between 0.0 and 1.0
    """
    tier = classify_source_tier(url)
    base_score = {
        SourceTier.TIER_1_FOUNDATIONAL: 0.9,
        SourceTier.TIER_2_STRATEGIC: 0.75,
        SourceTier.TIER_3_POLICY: 0.7,
        SourceTier.TIER_4_PRACTITIONER: 0.6,
        SourceTier.UNCLASSIFIED: 0.4,
    }.get(tier, 0.4)

    # Bonus if domain matches category preference
    if category:
        domain = extract_domain(url)
        preferred = get_domains_for_category(category)
        if any(d in domain or domain in d for d in preferred):
            base_score = min(base_score + 0.1, 1.0)

    return base_score


# =============================================================================
# 2026 AI Context
# =============================================================================

AI_RESEARCH_2026_CONTEXT = """
## 2026 AI Research Context

You are conducting AI research in January 2026. Be aware of the following developments:

1. **Frontier Models**: GPT-5, Claude 4, Gemini 2.0, and open-weight models like Llama 4 have significantly advanced reasoning capabilities. Multi-modal understanding is standard.

2. **Agentic AI Maturation**: Enterprise AI agents are now deployed at scale for complex workflows. Key frameworks include LangChain, LlamaIndex, CrewAI, and AutoGen. Measurable ROI benchmarks are expected.

3. **Reasoning Models**: Chain-of-thought, tree-of-thought, and reinforcement learning from human feedback (RLHF) have evolved into sophisticated reasoning pipelines. "Thinking" models (like o1) are common.

4. **AI Governance**: The EU AI Act is in effect. US has implemented executive orders on AI safety. China has separate AI governance frameworks. International coordination remains fragmented.

5. **Compute Landscape**: H100/H200 GPUs dominate training. Inference optimization is a major focus. Custom AI accelerators (TPUs, Trainium, Inferentia) compete with NVIDIA.

6. **Open vs Closed Debate**: Tension between open-weight models (Llama, Mistral, Qwen) and proprietary frontier models continues. Hybrid approaches gaining ground.

When researching AI topics, consider these 2026-specific developments in your analysis.
"""


def get_ai_research_system_prompt() -> str:
    """Get system prompt for AI research agent with 2026 context"""
    return f"""You are an expert AI researcher conducting comprehensive research on artificial intelligence topics.

{AI_RESEARCH_2026_CONTEXT}

Key principles:
1. Prioritize authoritative sources (academic papers, major AI labs, reputable publications)
2. Distinguish between speculation and demonstrated capabilities
3. Consider both technical and business/policy implications
4. Note when information may be outdated or contested
5. Identify gaps in current knowledge that warrant further research

Today's date is January 2026. Use this temporal context when evaluating claims and developments.
"""
