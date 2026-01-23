"""
Rate limiting utilities for Research Agent Hub.
Provides session-based rate limiting for the web UI.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter with sliding window.
    Thread-safe for use with Gradio's threading model.
    """

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, session_id: str) -> Tuple[bool, str]:
        """
        Check if a request is allowed for the given session.

        Args:
            session_id: Unique session identifier

        Returns:
            Tuple of (is_allowed, message)
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Clean old requests
            self._requests[session_id] = [
                t for t in self._requests[session_id]
                if t > cutoff
            ]

            # Check if under limit
            if len(self._requests[session_id]) >= self.max_requests:
                wait_time = self._requests[session_id][0] - cutoff
                logger.warning(
                    f"Rate limit exceeded for session {session_id[:8]}... "
                    f"({len(self._requests[session_id])}/{self.max_requests})"
                )
                return False, f"Rate limit exceeded. Please wait {wait_time:.0f} seconds."

            # Record this request
            self._requests[session_id].append(now)
            return True, ""

    def get_remaining(self, session_id: str) -> int:
        """
        Get remaining requests for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Number of remaining requests in the current window
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Count requests in window
            valid_requests = [
                t for t in self._requests[session_id]
                if t > cutoff
            ]

            return max(0, self.max_requests - len(valid_requests))

    def reset(self, session_id: str) -> None:
        """
        Reset rate limit for a session.

        Args:
            session_id: Unique session identifier
        """
        with self._lock:
            self._requests[session_id] = []

    def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old session data to prevent memory growth.

        Args:
            max_age_seconds: Remove sessions with no activity for this long

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            now = time.time()
            cutoff = now - max_age_seconds

            sessions_to_remove = []
            for session_id, requests in self._requests.items():
                if not requests or max(requests) < cutoff:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self._requests[session_id]

            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")

            return len(sessions_to_remove)


# Global rate limiters for different resource types
research_limiter = RateLimiter(max_requests=10, window_seconds=60)
stock_limiter = RateLimiter(max_requests=15, window_seconds=60)


def check_rate_limit(
    session_id: str,
    limiter_type: str = "research"
) -> Tuple[bool, str]:
    """
    Check rate limit for a session.

    Args:
        session_id: Session identifier (can be IP, user ID, etc.)
        limiter_type: "research" or "stock"

    Returns:
        Tuple of (is_allowed, error_message)
    """
    limiter = research_limiter if limiter_type == "research" else stock_limiter
    return limiter.is_allowed(session_id or "anonymous")


def get_session_id_from_request(request) -> str:
    """
    Extract session ID from Gradio request.
    Falls back to a default if request is unavailable.

    Args:
        request: Gradio request object

    Returns:
        Session identifier string
    """
    if request is None:
        return "anonymous"

    # Try to get client IP
    if hasattr(request, 'client') and request.client:
        return f"ip:{request.client.host}"

    # Fallback to session hash if available
    if hasattr(request, 'session_hash'):
        return f"session:{request.session_hash}"

    return "anonymous"
