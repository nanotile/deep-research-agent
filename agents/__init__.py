"""
Research Agents Package

Contains all specialized research agents for the Deep Research Agent system.
"""

from .deep_research_agent import (
    deep_research,
    deep_research_with_progress,
    ProgressUpdate,
    tavily_client,
)
from .stock_research_agent import (
    stock_research,
    stock_research_with_progress,
)
from .competitor_agent import (
    competitor_analysis_with_progress,
    CompetitorProgressUpdate,
)
from .sector_research_agent import (
    sector_research_with_progress,
    get_available_sectors,
    get_sector_info,
    SectorProgressUpdate,
)
from .portfolio_agent import (
    portfolio_analysis_with_progress,
    PortfolioProgressUpdate,
)
from .earnings_agent import (
    earnings_calendar_with_progress,
    EarningsProgressUpdate,
)
from .alert_system import (
    create_price_alert,
    create_earnings_alert,
    get_alert_summary,
    cancel_alert,
    check_all_alerts,
    send_test_email,
    AlertType,
)

__all__ = [
    # Deep Research
    "deep_research",
    "deep_research_with_progress",
    "ProgressUpdate",
    "tavily_client",
    # Stock Research
    "stock_research",
    "stock_research_with_progress",
    # Competitor Analysis
    "competitor_analysis_with_progress",
    "CompetitorProgressUpdate",
    # Sector Research
    "sector_research_with_progress",
    "get_available_sectors",
    "get_sector_info",
    "SectorProgressUpdate",
    # Portfolio Analysis
    "portfolio_analysis_with_progress",
    "PortfolioProgressUpdate",
    # Earnings Calendar
    "earnings_calendar_with_progress",
    "EarningsProgressUpdate",
    # Alert System
    "create_price_alert",
    "create_earnings_alert",
    "get_alert_summary",
    "cancel_alert",
    "check_all_alerts",
    "send_test_email",
    "AlertType",
]
