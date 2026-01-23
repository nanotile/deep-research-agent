"""
Report history management for Research Agent Hub.
Stores recent reports in memory with session-based access.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReportEntry:
    """A single report entry in history."""
    id: str
    title: str  # Query or ticker
    report_type: str  # "research" or "stock"
    content: str
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def display_name(self) -> str:
        """Display name for dropdown."""
        time_str = self.created_at.strftime("%H:%M")
        title_short = self.title[:40] + "..." if len(self.title) > 40 else self.title
        return f"[{time_str}] {title_short}"

    @property
    def age_str(self) -> str:
        """Human-readable age."""
        delta = datetime.now() - self.created_at
        if delta.seconds < 60:
            return "just now"
        elif delta.seconds < 3600:
            return f"{delta.seconds // 60} min ago"
        else:
            return f"{delta.seconds // 3600} hr ago"


class ReportHistory:
    """
    In-memory report history manager.
    Thread-safe for use with Gradio's threading model.
    """

    def __init__(self, max_entries: int = 10):
        """
        Initialize report history.

        Args:
            max_entries: Maximum reports to keep per session
        """
        self.max_entries = max_entries
        self._history: Dict[str, OrderedDict[str, ReportEntry]] = {}
        self._lock = Lock()
        self._counter = 0

    def add_report(
        self,
        session_id: str,
        title: str,
        content: str,
        report_type: str = "research"
    ) -> str:
        """
        Add a report to history.

        Args:
            session_id: Session identifier
            title: Report title (query or ticker)
            content: Full report content
            report_type: "research" or "stock"

        Returns:
            Report ID
        """
        with self._lock:
            self._counter += 1
            report_id = f"report_{self._counter}"

            # Initialize session history if needed
            if session_id not in self._history:
                self._history[session_id] = OrderedDict()

            # Create entry
            entry = ReportEntry(
                id=report_id,
                title=title,
                report_type=report_type,
                content=content,
            )

            # Add to history (newest first)
            self._history[session_id][report_id] = entry
            self._history[session_id].move_to_end(report_id, last=False)

            # Trim to max entries
            while len(self._history[session_id]) > self.max_entries:
                self._history[session_id].popitem(last=True)

            logger.info(
                f"Added report {report_id} for session {session_id[:8]}... "
                f"(total: {len(self._history[session_id])})"
            )

            return report_id

    def get_report(self, session_id: str, report_id: str) -> Optional[ReportEntry]:
        """Get a specific report by ID."""
        with self._lock:
            if session_id in self._history:
                return self._history[session_id].get(report_id)
            return None

    def get_latest(self, session_id: str) -> Optional[ReportEntry]:
        """Get the most recent report for a session."""
        with self._lock:
            if session_id in self._history and self._history[session_id]:
                # Get first item (most recent due to move_to_end)
                report_id = next(iter(self._history[session_id]))
                return self._history[session_id][report_id]
            return None

    def list_reports(
        self,
        session_id: str,
        report_type: Optional[str] = None
    ) -> List[ReportEntry]:
        """
        List all reports for a session.

        Args:
            session_id: Session identifier
            report_type: Optional filter by type ("research" or "stock")

        Returns:
            List of report entries, newest first
        """
        with self._lock:
            if session_id not in self._history:
                return []

            reports = list(self._history[session_id].values())

            if report_type:
                reports = [r for r in reports if r.report_type == report_type]

            return reports

    def get_dropdown_choices(
        self,
        session_id: str,
        report_type: Optional[str] = None
    ) -> List[tuple]:
        """
        Get choices for Gradio dropdown.

        Returns:
            List of (display_name, report_id) tuples
        """
        reports = self.list_reports(session_id, report_type)
        return [(r.display_name, r.id) for r in reports]

    def clear_session(self, session_id: str) -> int:
        """
        Clear all reports for a session.

        Returns:
            Number of reports cleared
        """
        with self._lock:
            if session_id in self._history:
                count = len(self._history[session_id])
                del self._history[session_id]
                return count
            return 0

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Remove sessions with no recent activity.

        Args:
            max_age_hours: Remove sessions older than this

        Returns:
            Number of sessions removed
        """
        with self._lock:
            cutoff = datetime.now()
            sessions_to_remove = []

            for session_id, reports in self._history.items():
                if not reports:
                    sessions_to_remove.append(session_id)
                else:
                    # Check age of newest report
                    newest = next(iter(reports.values()))
                    age_hours = (cutoff - newest.created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self._history[session_id]

            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

            return len(sessions_to_remove)


# Global history instance
report_history = ReportHistory(max_entries=10)


def add_to_history(
    session_id: str,
    title: str,
    content: str,
    report_type: str = "research"
) -> str:
    """Convenience function to add report to global history."""
    return report_history.add_report(session_id, title, content, report_type)


def get_history_choices(
    session_id: str,
    report_type: Optional[str] = None
) -> List[tuple]:
    """Convenience function to get dropdown choices."""
    return report_history.get_dropdown_choices(session_id, report_type)


def get_report_content(session_id: str, report_id: str) -> str:
    """Get report content by ID, returns empty string if not found."""
    entry = report_history.get_report(session_id, report_id)
    return entry.content if entry else ""
