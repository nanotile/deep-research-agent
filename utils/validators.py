"""
Input validation utilities for Research Agent Hub.
Sanitizes ticker symbols, research queries, and other user inputs.
"""

import re
from typing import Tuple


# Constants
MAX_TICKER_LENGTH = 10
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 3

# Valid ticker pattern: 1-10 alphanumeric characters, may include dots and hyphens
TICKER_PATTERN = re.compile(r'^[A-Z0-9][A-Z0-9.\-]{0,9}$')

# Characters that should be stripped from queries (potential injection vectors)
DANGEROUS_CHARS = re.compile(r'[<>{}|\[\]\\`]')


def sanitize_ticker(ticker: str) -> Tuple[str, bool, str]:
    """
    Sanitize and validate a stock ticker symbol.

    Args:
        ticker: Raw ticker input from user

    Returns:
        Tuple of (sanitized_ticker, is_valid, error_message)

    Examples:
        >>> sanitize_ticker("AAPL")
        ("AAPL", True, "")
        >>> sanitize_ticker("  aapl  ")
        ("AAPL", True, "")
        >>> sanitize_ticker("BRK.B")
        ("BRK.B", True, "")
        >>> sanitize_ticker("AAPL123456789")
        ("", False, "Ticker too long (max 10 characters)")
        >>> sanitize_ticker("<script>")
        ("", False, "Invalid ticker format")
    """
    if not ticker:
        return "", False, "Ticker is required"

    # Strip whitespace and convert to uppercase
    cleaned = ticker.strip().upper()

    # Check length
    if len(cleaned) > MAX_TICKER_LENGTH:
        return "", False, f"Ticker too long (max {MAX_TICKER_LENGTH} characters)"

    if len(cleaned) == 0:
        return "", False, "Ticker cannot be empty"

    # Validate pattern
    if not TICKER_PATTERN.match(cleaned):
        return "", False, "Invalid ticker format (use letters, numbers, dots, or hyphens)"

    return cleaned, True, ""


def validate_ticker_format(ticker: str) -> bool:
    """
    Quick validation check for ticker format.

    Args:
        ticker: Ticker string to validate

    Returns:
        True if format is valid, False otherwise
    """
    _, is_valid, _ = sanitize_ticker(ticker)
    return is_valid


def sanitize_query(query: str) -> Tuple[str, bool, str]:
    """
    Sanitize and validate a research query.

    Args:
        query: Raw query input from user

    Returns:
        Tuple of (sanitized_query, is_valid, error_message)

    Examples:
        >>> sanitize_query("AI trends in 2026")
        ("AI trends in 2026", True, "")
        >>> sanitize_query("  Research <script>alert('xss')</script>  ")
        ("Research alert('xss')", True, "")  # Dangerous chars stripped
        >>> sanitize_query("ab")
        ("", False, "Query too short (min 3 characters)")
    """
    if not query:
        return "", False, "Research query is required"

    # Strip whitespace
    cleaned = query.strip()

    # Remove dangerous characters
    cleaned = DANGEROUS_CHARS.sub('', cleaned)

    # Collapse multiple spaces
    cleaned = ' '.join(cleaned.split())

    # Check length
    if len(cleaned) < MIN_QUERY_LENGTH:
        return "", False, f"Query too short (min {MIN_QUERY_LENGTH} characters)"

    if len(cleaned) > MAX_QUERY_LENGTH:
        return "", False, f"Query too long (max {MAX_QUERY_LENGTH} characters)"

    return cleaned, True, ""


def sanitize_email(email: str) -> Tuple[str, bool, str]:
    """
    Sanitize and validate an email address.

    Args:
        email: Raw email input from user

    Returns:
        Tuple of (sanitized_email, is_valid, error_message)
    """
    if not email:
        return "", False, "Email is required"

    cleaned = email.strip().lower()

    # Basic email pattern
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    if not email_pattern.match(cleaned):
        return "", False, "Invalid email format"

    if len(cleaned) > 254:  # RFC 5321
        return "", False, "Email too long"

    return cleaned, True, ""


def escape_for_display(text: str) -> str:
    """
    Escape text for safe display in UI (prevents XSS in markdown).

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for display
    """
    if not text:
        return ""

    # Escape HTML entities
    escapes = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
    }

    for char, escaped in escapes.items():
        text = text.replace(char, escaped)

    return text
