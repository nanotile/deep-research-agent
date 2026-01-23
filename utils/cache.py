"""
Caching Layer - SQLite for persistent storage, in-memory for API responses.

Features:
- SQLite for report history persistence
- In-memory cache with TTL for API responses
- Thread-safe operations
"""

import os
import sqlite3
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from threading import Lock
from dataclasses import dataclass, asdict
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)

# Cache configuration
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
DB_PATH = CACHE_DIR / "research_hub.db"
API_CACHE_TTL = int(os.getenv("API_CACHE_TTL", 300))  # 5 minutes default
REPORT_RETENTION_DAYS = int(os.getenv("REPORT_RETENTION_DAYS", 30))


@dataclass
class CachedReport:
    """Stored report data."""
    id: int
    report_type: str  # 'deep_research', 'stock', 'sector', 'competitor', 'portfolio', 'earnings'
    query: str  # ticker, sector name, or research query
    report_content: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class SQLiteCache:
    """SQLite-based persistent cache for reports and alerts."""

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

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._db_lock = Lock()
        self._init_db()
        self._initialized = True
        logger.info(f"SQLite cache initialized at {DB_PATH}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Reports table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_type TEXT NOT NULL,
                        query TEXT NOT NULL,
                        report_content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata TEXT
                    )
                """)

                # Alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        target_value REAL,
                        email TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL,
                        triggered_at TEXT,
                        metadata TEXT
                    )
                """)

                # API cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        cache_key TEXT PRIMARY KEY,
                        response TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        ttl INTEGER NOT NULL
                    )
                """)

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_query ON reports(query)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active)")

                conn.commit()
            finally:
                conn.close()

    # ============================================================
    # Report Storage Methods
    # ============================================================

    def save_report(
        self,
        report_type: str,
        query: str,
        report_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save a report to the database. Returns the report ID."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO reports (report_type, query, report_content, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        report_type,
                        query,
                        report_content,
                        datetime.now().isoformat(),
                        json.dumps(metadata) if metadata else None
                    )
                )
                conn.commit()
                report_id = cursor.lastrowid
                logger.info(f"Saved report {report_id}: {report_type} - {query}")
                return report_id
            finally:
                conn.close()

    def get_report(self, report_id: int) -> Optional[CachedReport]:
        """Retrieve a specific report by ID."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
                row = cursor.fetchone()
                if row:
                    return CachedReport(
                        id=row['id'],
                        report_type=row['report_type'],
                        query=row['query'],
                        report_content=row['report_content'],
                        created_at=row['created_at'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                return None
            finally:
                conn.close()

    def get_recent_reports(
        self,
        report_type: Optional[str] = None,
        limit: int = 20
    ) -> List[CachedReport]:
        """Get recent reports, optionally filtered by type."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if report_type:
                    cursor.execute(
                        """
                        SELECT * FROM reports
                        WHERE report_type = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (report_type, limit)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM reports
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )

                return [
                    CachedReport(
                        id=row['id'],
                        report_type=row['report_type'],
                        query=row['query'],
                        report_content=row['report_content'],
                        created_at=row['created_at'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    def search_reports(self, search_term: str, limit: int = 10) -> List[CachedReport]:
        """Search reports by query text."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM reports
                    WHERE query LIKE ? OR report_content LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{search_term}%", f"%{search_term}%", limit)
                )

                return [
                    CachedReport(
                        id=row['id'],
                        report_type=row['report_type'],
                        query=row['query'],
                        report_content=row['report_content'],
                        created_at=row['created_at'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    def delete_old_reports(self, days: int = None) -> int:
        """Delete reports older than specified days. Returns count deleted."""
        days = days or REPORT_RETENTION_DAYS
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - days) if cutoff.day > days else cutoff

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM reports WHERE created_at < ?",
                    (cutoff.isoformat(),)
                )
                conn.commit()
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Deleted {deleted} old reports")
                return deleted
            finally:
                conn.close()

    # ============================================================
    # Alert Storage Methods
    # ============================================================

    def save_alert(
        self,
        alert_type: str,
        ticker: str,
        condition: str,
        target_value: Optional[float],
        email: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save a new alert. Returns alert ID."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO alerts (alert_type, ticker, condition, target_value, email, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_type,
                        ticker.upper(),
                        condition,
                        target_value,
                        email,
                        datetime.now().isoformat(),
                        json.dumps(metadata) if metadata else None
                    )
                )
                conn.commit()
                alert_id = cursor.lastrowid
                logger.info(f"Created alert {alert_id}: {alert_type} for {ticker}")
                return alert_id
            finally:
                conn.close()

    def get_active_alerts(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active alerts, optionally filtered by ticker."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if ticker:
                    cursor.execute(
                        "SELECT * FROM alerts WHERE is_active = 1 AND ticker = ?",
                        (ticker.upper(),)
                    )
                else:
                    cursor.execute("SELECT * FROM alerts WHERE is_active = 1")

                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def trigger_alert(self, alert_id: int) -> bool:
        """Mark an alert as triggered."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE alerts
                    SET triggered_at = ?, is_active = 0
                    WHERE id = ?
                    """,
                    (datetime.now().isoformat(), alert_id)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def deactivate_alert(self, alert_id: int) -> bool:
        """Deactivate an alert without triggering."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE alerts SET is_active = 0 WHERE id = ?",
                    (alert_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ============================================================
    # API Response Cache Methods
    # ============================================================

    def cache_api_response(
        self,
        cache_key: str,
        response: Any,
        ttl: int = None
    ):
        """Cache an API response."""
        ttl = ttl or API_CACHE_TTL
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO api_cache (cache_key, response, created_at, ttl)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cache_key, json.dumps(response), time.time(), ttl)
                )
                conn.commit()
            finally:
                conn.close()

    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get a cached API response if not expired."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response, created_at, ttl FROM api_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    if time.time() - row['created_at'] < row['ttl']:
                        return json.loads(row['response'])
                    else:
                        # Expired - delete it
                        cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                        conn.commit()
                return None
            finally:
                conn.close()

    def clear_expired_cache(self) -> int:
        """Clear all expired cache entries. Returns count cleared."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM api_cache WHERE (created_at + ttl) < ?",
                    (time.time(),)
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()


class MemoryCache:
    """Fast in-memory cache with TTL for frequently accessed data."""

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, tuple[Any, float, int]] = {}  # key -> (value, created_at, ttl)
        self._lock = Lock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value if not expired."""
        with self._lock:
            if key in self._cache:
                value, created_at, ttl = self._cache[key]
                if time.time() - created_at < ttl:
                    return value
                else:
                    del self._cache[key]
            return None

    def set(self, key: str, value: Any, ttl: int = None):
        """Set a cache value with TTL."""
        ttl = ttl or self._default_ttl
        with self._lock:
            self._cache[key] = (value, time.time(), ttl)

    def delete(self, key: str):
        """Delete a cached value."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

    def cleanup(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            now = time.time()
            expired = [
                k for k, (_, created_at, ttl) in self._cache.items()
                if now - created_at >= ttl
            ]
            for k in expired:
                del self._cache[k]
            return len(expired)


def make_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = json.dumps((args, kwargs), sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


# Singleton instances
db_cache = SQLiteCache()
memory_cache = MemoryCache()


# Decorator for caching function results
def cached(ttl: int = None, use_db: bool = False):
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds (default: 300)
        use_db: Whether to use SQLite (persistent) or memory cache
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{make_cache_key(*args, **kwargs)}"

            if use_db:
                cached_value = db_cache.get_cached_response(cache_key)
            else:
                cached_value = memory_cache.get(cache_key)

            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value

            result = func(*args, **kwargs)

            if use_db:
                db_cache.cache_api_response(cache_key, result, ttl)
            else:
                memory_cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


# Export for utils package
__all__ = [
    'SQLiteCache',
    'MemoryCache',
    'db_cache',
    'memory_cache',
    'cached',
    'make_cache_key',
    'CachedReport',
]
