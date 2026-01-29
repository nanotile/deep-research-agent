"""
2026 Commodity Market Context Module
Provides 2026-specific macro context for commodity research including:
- Fed policy and USD outlook
- Inflation dynamics
- Geopolitical supply risks
- Category-specific context (precious metals, energy, industrial, agriculture)
"""

from typing import Optional


# =============================================================================
# 2026 Commodity Macro Context
# =============================================================================

COMMODITY_MACRO_CONTEXT_2026 = {
    "fed_policy": (
        "The Federal Reserve is in a cautious easing cycle in early 2026, having "
        "cut rates modestly from the 2024 peak. Markets expect 1-2 additional 25 bp "
        "cuts through mid-2026, but sticky services inflation keeps the Fed data-dependent."
    ),
    "usd_outlook": (
        "The US Dollar Index (DXY) has softened from 2024 highs as rate differentials "
        "narrow, but remains supported by relative US economic resilience and safe-haven "
        "demand amid geopolitical tensions."
    ),
    "inflation_dynamics": (
        "Headline CPI has moderated toward 3%, but core services inflation remains "
        "above the Fed's 2% target. Energy disinflation has helped headline numbers "
        "while food prices show weather-related volatility."
    ),
    "geopolitical_risks": (
        "US-China tech decoupling continues to reshape trade flows. Middle East tensions "
        "keep energy risk premia elevated. BRICS+ de-dollarisation efforts support central "
        "bank gold accumulation. Ukraine conflict keeps European natural gas supply fragile."
    ),
}


# =============================================================================
# Category-Specific Context
# =============================================================================

_CATEGORY_CONTEXT = {
    "precious_metals": (
        "## 2026 Precious Metals Context\n\n"
        "Central bank gold buying remains at record levels as BRICS+ nations diversify "
        "reserves away from the US dollar. China and India continue to be the largest "
        "official-sector buyers. Silver benefits from dual monetary/industrial demand as "
        "solar panel installations accelerate globally. Platinum faces structural surplus "
        "as hydrogen fuel cell adoption remains below forecasts.\n\n"
        "Key drivers: real interest rates, USD direction, central bank purchases, "
        "geopolitical safe-haven demand, silver industrial demand from solar and electronics."
    ),
    "energy": (
        "## 2026 Energy Context\n\n"
        "OPEC+ voluntary production cuts have kept crude oil in a $70-90 range through "
        "late 2025. US shale production growth is slowing as tier-1 acreage depletes. "
        "Natural gas markets remain bifurcated: US Henry Hub stays low on abundant supply "
        "while European TTF and Asian LNG prices reflect ongoing pipeline/LNG constraints "
        "post-Russia.\n\n"
        "Key drivers: OPEC+ policy, US shale production trajectory, China demand recovery, "
        "EV penetration impact on gasoline demand, LNG export capacity additions, "
        "Middle East geopolitical risk premium."
    ),
    "industrial_metals": (
        "## 2026 Industrial Metals Context\n\n"
        "Copper is in a structural deficit driven by energy transition demand (EVs, grid "
        "upgrades, renewable energy). Limited new mine supply supports multi-year bull thesis. "
        "China's construction sector remains weak but manufacturing and EV demand partially "
        "offset. Green premium is emerging as electrification demand outpaces new supply.\n\n"
        "Key drivers: China manufacturing PMI, global EV sales, grid infrastructure spending, "
        "mine supply disruptions, LME inventory levels, green energy capex."
    ),
    "agriculture": (
        "## 2026 Agriculture Context\n\n"
        "Weather volatility from El Nino/La Nina transitions is the dominant price driver. "
        "Global grain stocks remain tight after multiple years of disrupted harvests. "
        "Biofuel mandates in US, Brazil, and EU compete with food demand for corn and "
        "soybeans. Ukraine's crop production has partially recovered but export logistics "
        "remain constrained.\n\n"
        "Key drivers: weather patterns (El Nino/La Nina), USDA crop reports, China import "
        "demand, biofuel policy, fertiliser costs, Black Sea export logistics."
    ),
}


def get_commodity_macro_context(category: str) -> str:
    """
    Return category-specific macro context for prompt injection.

    Args:
        category: One of precious_metals, energy, industrial_metals, agriculture

    Returns:
        Prompt text with 2026 macro + category context
    """
    parts = [
        "## 2026 COMMODITY MARKET CONTEXT (Apply to analysis)\n",
        f"**Fed Policy:** {COMMODITY_MACRO_CONTEXT_2026['fed_policy']}\n",
        f"**USD Outlook:** {COMMODITY_MACRO_CONTEXT_2026['usd_outlook']}\n",
        f"**Inflation:** {COMMODITY_MACRO_CONTEXT_2026['inflation_dynamics']}\n",
        f"**Geopolitical:** {COMMODITY_MACRO_CONTEXT_2026['geopolitical_risks']}\n",
    ]

    category_ctx = _CATEGORY_CONTEXT.get(category)
    if category_ctx:
        parts.append(f"\n{category_ctx}")

    return "\n".join(parts)
