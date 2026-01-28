"""
Services Package

Contains data fetchers and context providers for the Deep Research Agent system.
"""

from .stock_data_fetchers import (
    validate_ticker,
    fetch_yfinance_data,
    fetch_finnhub_data,
    fetch_sec_edgar_filings,
    fetch_alpha_vantage_data,
    fetch_tavily_news,
    fetch_macro_sentiment,
)

from .market_context_2026 import (
    DEEP_RESEARCH_2026_CONTEXT,
    GLOBAL_MARKET_CONTEXT_2026,
    CORE_2026_VARIABLES,
    is_tech_semiconductor_sector,
    get_china_exposure_level,
    estimate_surcharge_eps_impact,
    get_process_node_status,
    get_software_moat,
    get_2026_analysis_prompt_injection,
)

from .google_drive_service import (
    is_drive_configured,
    create_google_doc,
    save_report_to_drive,
    DriveExportResult,
)

__all__ = [
    # Data fetchers
    "validate_ticker",
    "fetch_yfinance_data",
    "fetch_finnhub_data",
    "fetch_sec_edgar_filings",
    "fetch_alpha_vantage_data",
    "fetch_tavily_news",
    "fetch_macro_sentiment",
    # 2026 Market context
    "DEEP_RESEARCH_2026_CONTEXT",
    "GLOBAL_MARKET_CONTEXT_2026",
    "CORE_2026_VARIABLES",
    "is_tech_semiconductor_sector",
    "get_china_exposure_level",
    "estimate_surcharge_eps_impact",
    "get_process_node_status",
    "get_software_moat",
    "get_2026_analysis_prompt_injection",
    # Google Drive export
    "is_drive_configured",
    "create_google_doc",
    "save_report_to_drive",
    "DriveExportResult",
]
