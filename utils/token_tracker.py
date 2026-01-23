"""
Token Usage Tracker - Track and display Claude API token consumption.

Provides utilities to capture, store, and display token usage from API calls.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from threading import Lock

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Single API call token usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    operation: str = ""  # e.g., "planning", "analysis", "report"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD based on Claude Sonnet pricing."""
        # Claude Sonnet 4 pricing (as of 2025)
        # Input: $3 per 1M tokens, Output: $15 per 1M tokens
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


@dataclass
class SessionTokenUsage:
    """Aggregated token usage for a research session."""
    operations: List[TokenUsage] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(op.input_tokens for op in self.operations)

    @property
    def total_output_tokens(self) -> int:
        return sum(op.output_tokens for op in self.operations)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(op.estimated_cost for op in self.operations)

    def add(self, usage: TokenUsage):
        self.operations.append(usage)

    def clear(self):
        self.operations.clear()

    def format_summary(self) -> str:
        """Format a summary string for display."""
        if not self.operations:
            return ""

        lines = [
            f"**Token Usage Summary:**",
            f"- Input: {self.total_input_tokens:,} tokens",
            f"- Output: {self.total_output_tokens:,} tokens",
            f"- **Total: {self.total_tokens:,} tokens**",
            f"- Estimated Cost: **${self.total_cost:.4f}**",
        ]

        if len(self.operations) > 1:
            lines.append(f"- API Calls: {len(self.operations)}")

        return "\n".join(lines)

    def format_detailed(self) -> str:
        """Format detailed breakdown by operation."""
        if not self.operations:
            return ""

        lines = [
            "| Operation | Input | Output | Total | Cost |",
            "|-----------|-------|--------|-------|------|"
        ]

        for op in self.operations:
            lines.append(
                f"| {op.operation or 'API Call'} | {op.input_tokens:,} | {op.output_tokens:,} | {op.total_tokens:,} | ${op.estimated_cost:.4f} |"
            )

        lines.append(f"| **Total** | **{self.total_input_tokens:,}** | **{self.total_output_tokens:,}** | **{self.total_tokens:,}** | **${self.total_cost:.4f}** |")

        return "\n".join(lines)


class TokenTracker:
    """Global token usage tracker."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sessions: Dict[str, SessionTokenUsage] = {}
        self._session_lock = Lock()
        self._initialized = True

    def get_session(self, session_id: str) -> SessionTokenUsage:
        """Get or create a session's token usage tracker."""
        with self._session_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionTokenUsage()
            return self._sessions[session_id]

    def track(self, session_id: str, response, operation: str = "") -> TokenUsage:
        """
        Track token usage from an Anthropic API response.

        Args:
            session_id: Session identifier
            response: Anthropic API response object
            operation: Description of the operation (e.g., "planning", "analysis")

        Returns:
            TokenUsage object with the extracted usage data
        """
        usage = TokenUsage(operation=operation)

        try:
            if hasattr(response, 'usage'):
                usage.input_tokens = getattr(response.usage, 'input_tokens', 0)
                usage.output_tokens = getattr(response.usage, 'output_tokens', 0)
                usage.cache_read_tokens = getattr(response.usage, 'cache_read_input_tokens', 0)
                usage.cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', 0)

            if hasattr(response, 'model'):
                usage.model = response.model

            session = self.get_session(session_id)
            session.add(usage)

            logger.debug(f"Tracked {usage.total_tokens} tokens for session {session_id} ({operation})")

        except Exception as e:
            logger.warning(f"Error tracking token usage: {e}")

        return usage

    def clear_session(self, session_id: str):
        """Clear token usage for a session."""
        with self._session_lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()

    def get_summary(self, session_id: str) -> str:
        """Get formatted summary for a session."""
        session = self.get_session(session_id)
        return session.format_summary()

    def get_detailed(self, session_id: str) -> str:
        """Get detailed breakdown for a session."""
        session = self.get_session(session_id)
        return session.format_detailed()


# Global tracker instance
token_tracker = TokenTracker()


def extract_usage_from_response(response) -> TokenUsage:
    """Extract token usage from an Anthropic API response without tracking."""
    usage = TokenUsage()

    try:
        if hasattr(response, 'usage'):
            usage.input_tokens = getattr(response.usage, 'input_tokens', 0)
            usage.output_tokens = getattr(response.usage, 'output_tokens', 0)
        if hasattr(response, 'model'):
            usage.model = response.model
    except Exception as e:
        logger.warning(f"Error extracting token usage: {e}")

    return usage


def format_token_display(input_tokens: int, output_tokens: int) -> str:
    """Format tokens for inline display."""
    total = input_tokens + output_tokens
    cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0
    return f"📊 {total:,} tokens (${cost:.4f})"


__all__ = [
    'TokenUsage',
    'SessionTokenUsage',
    'TokenTracker',
    'token_tracker',
    'extract_usage_from_response',
    'format_token_display',
]
