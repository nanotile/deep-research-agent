"""
Alert System - Price targets, earnings reminders, and sentiment alerts.

Features:
- Price target alerts (above/below threshold)
- Earnings date reminders
- News sentiment alerts
- Email notifications via Resend
"""

import os
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import yfinance as yf

from utils.cache import db_cache
from utils import get_logger, sanitize_ticker

logger = get_logger(__name__)

# Email configuration
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "alerts@resend.dev")


class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    EARNINGS_REMINDER = "earnings_reminder"
    SENTIMENT_CHANGE = "sentiment_change"


@dataclass
class Alert:
    """Alert definition."""
    id: int
    alert_type: str
    ticker: str
    condition: str
    target_value: Optional[float]
    email: str
    is_active: bool
    created_at: str
    triggered_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertCheckResult:
    """Result of checking an alert."""
    alert: Alert
    triggered: bool
    current_value: Optional[float] = None
    message: str = ""


def create_price_alert(
    ticker: str,
    condition: str,  # 'above' or 'below'
    target_price: float,
    email: str
) -> int:
    """
    Create a price target alert.

    Args:
        ticker: Stock ticker symbol
        condition: 'above' or 'below'
        target_price: Price threshold
        email: Email address for notification

    Returns:
        Alert ID
    """
    clean_ticker, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        raise ValueError(f"Invalid ticker: {error}")

    alert_type = AlertType.PRICE_ABOVE.value if condition == 'above' else AlertType.PRICE_BELOW.value

    # Get current price for reference
    try:
        stock = yf.Ticker(clean_ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
    except:
        current_price = None

    alert_id = db_cache.save_alert(
        alert_type=alert_type,
        ticker=clean_ticker,
        condition=f"Price {condition} ${target_price:.2f}",
        target_value=target_price,
        email=email,
        metadata={
            'created_price': current_price,
            'target_price': target_price,
            'condition': condition
        }
    )

    logger.info(f"Created price alert {alert_id}: {clean_ticker} {condition} ${target_price}")
    return alert_id


def create_earnings_alert(
    ticker: str,
    days_before: int,
    email: str
) -> int:
    """
    Create an earnings reminder alert.

    Args:
        ticker: Stock ticker symbol
        days_before: Days before earnings to send reminder
        email: Email address for notification

    Returns:
        Alert ID
    """
    clean_ticker, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        raise ValueError(f"Invalid ticker: {error}")

    # Get earnings date
    try:
        stock = yf.Ticker(clean_ticker)
        calendar = stock.calendar
        earnings_date = None
        if calendar is not None and not calendar.empty:
            if 'Earnings Date' in calendar.index:
                earnings_dates = calendar.loc['Earnings Date']
                if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
                    earnings_date = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
                else:
                    earnings_date = earnings_dates
    except:
        earnings_date = None

    alert_id = db_cache.save_alert(
        alert_type=AlertType.EARNINGS_REMINDER.value,
        ticker=clean_ticker,
        condition=f"Earnings reminder {days_before} days before",
        target_value=days_before,
        email=email,
        metadata={
            'days_before': days_before,
            'earnings_date': str(earnings_date) if earnings_date else None
        }
    )

    logger.info(f"Created earnings alert {alert_id}: {clean_ticker} {days_before} days before earnings")
    return alert_id


def create_sentiment_alert(
    ticker: str,
    sentiment_threshold: str,  # 'positive', 'negative', 'any_change'
    email: str
) -> int:
    """
    Create a sentiment change alert.

    Args:
        ticker: Stock ticker symbol
        sentiment_threshold: 'positive', 'negative', or 'any_change'
        email: Email address for notification

    Returns:
        Alert ID
    """
    clean_ticker, is_valid, error = sanitize_ticker(ticker)
    if not is_valid:
        raise ValueError(f"Invalid ticker: {error}")

    alert_id = db_cache.save_alert(
        alert_type=AlertType.SENTIMENT_CHANGE.value,
        ticker=clean_ticker,
        condition=f"Sentiment turns {sentiment_threshold}",
        target_value=None,
        email=email,
        metadata={
            'sentiment_threshold': sentiment_threshold,
            'baseline_sentiment': 'neutral'  # Would be set by actual sentiment check
        }
    )

    logger.info(f"Created sentiment alert {alert_id}: {clean_ticker} -> {sentiment_threshold}")
    return alert_id


def check_price_alert(alert: Dict[str, Any]) -> AlertCheckResult:
    """Check if a price alert should trigger."""
    ticker = alert['ticker']
    target_value = alert['target_value']
    alert_type = alert['alert_type']

    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')

        if current_price is None:
            return AlertCheckResult(
                alert=Alert(**{**alert, 'metadata': None}),
                triggered=False,
                message=f"Could not get current price for {ticker}"
            )

        triggered = False
        if alert_type == AlertType.PRICE_ABOVE.value and current_price >= target_value:
            triggered = True
        elif alert_type == AlertType.PRICE_BELOW.value and current_price <= target_value:
            triggered = True

        return AlertCheckResult(
            alert=Alert(**{**alert, 'metadata': None}),
            triggered=triggered,
            current_value=current_price,
            message=f"{ticker} is now ${current_price:.2f} (target: ${target_value:.2f})"
        )

    except Exception as e:
        logger.error(f"Error checking price alert for {ticker}: {e}")
        return AlertCheckResult(
            alert=Alert(**{**alert, 'metadata': None}),
            triggered=False,
            message=f"Error checking {ticker}: {e}"
        )


def check_earnings_alert(alert: Dict[str, Any]) -> AlertCheckResult:
    """Check if an earnings reminder should trigger."""
    ticker = alert['ticker']
    days_before = int(alert['target_value'] or 3)

    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar

        if calendar is None or calendar.empty:
            return AlertCheckResult(
                alert=Alert(**{**alert, 'metadata': None}),
                triggered=False,
                message=f"No earnings date found for {ticker}"
            )

        if 'Earnings Date' not in calendar.index:
            return AlertCheckResult(
                alert=Alert(**{**alert, 'metadata': None}),
                triggered=False,
                message=f"No earnings date in calendar for {ticker}"
            )

        earnings_dates = calendar.loc['Earnings Date']
        if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
            next_earnings = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
        else:
            next_earnings = earnings_dates

        if next_earnings is None:
            return AlertCheckResult(
                alert=Alert(**{**alert, 'metadata': None}),
                triggered=False,
                message=f"Could not parse earnings date for {ticker}"
            )

        if hasattr(next_earnings, 'to_pydatetime'):
            next_earnings = next_earnings.to_pydatetime()

        days_until = (next_earnings.date() - datetime.now().date()).days

        triggered = days_until <= days_before and days_until >= 0

        return AlertCheckResult(
            alert=Alert(**{**alert, 'metadata': None}),
            triggered=triggered,
            current_value=days_until,
            message=f"{ticker} earnings in {days_until} days ({next_earnings.strftime('%Y-%m-%d')})"
        )

    except Exception as e:
        logger.error(f"Error checking earnings alert for {ticker}: {e}")
        return AlertCheckResult(
            alert=Alert(**{**alert, 'metadata': None}),
            triggered=False,
            message=f"Error checking {ticker}: {e}"
        )


def send_test_email(to_email: str) -> tuple[bool, str]:
    """Send a test email to verify Resend configuration."""
    if not RESEND_API_KEY:
        return False, "RESEND_API_KEY not set in .env file"

    try:
        import resend
        resend.api_key = RESEND_API_KEY

        params = {
            "from": ALERT_FROM_EMAIL,
            "to": [to_email],
            "subject": "Research Agent Hub - Test Email",
            "html": """
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #2563eb;">Test Email Successful!</h2>
                <p>Your email configuration is working correctly.</p>
                <p>You will receive alert notifications at this address.</p>
                <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 20px 0;">
                <p style="color: #9ca3af; font-size: 11px;">
                    Sent from Research Agent Hub
                </p>
            </body>
            </html>
            """
        }

        response = resend.Emails.send(params)
        logger.info(f"Test email sent to {to_email}: {response}")
        return True, f"Test email sent to {to_email}"

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to send test email: {error_msg}")
        # Provide helpful error messages
        if "validation" in error_msg.lower() or "from" in error_msg.lower():
            return False, f"Email error: The 'from' address ({ALERT_FROM_EMAIL}) may not be verified. With Resend free tier, you can only send to your own verified email."
        return False, f"Email error: {error_msg}"


async def send_alert_email(alert: Alert, result: AlertCheckResult) -> bool:
    """Send alert notification email via Resend."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set, cannot send alert email")
        return False

    try:
        import resend
        resend.api_key = RESEND_API_KEY

        # Build email content
        subject = f"Alert Triggered: {alert.ticker} - {alert.condition}"

        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #2563eb;">Alert Triggered</h2>

            <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin-top: 0;">{alert.ticker}</h3>
                <p><strong>Condition:</strong> {alert.condition}</p>
                <p><strong>Status:</strong> {result.message}</p>
                {f'<p><strong>Current Value:</strong> ${result.current_value:.2f}</p>' if result.current_value else ''}
            </div>

            <p style="color: #6b7280; font-size: 12px;">
                Alert created: {alert.created_at}<br>
                Triggered: {datetime.now().isoformat()}
            </p>

            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 20px 0;">

            <p style="color: #9ca3af; font-size: 11px;">
                This alert was sent by Research Agent Hub.<br>
                To manage your alerts, visit the Alerts tab in the app.
            </p>
        </body>
        </html>
        """

        params = {
            "from": ALERT_FROM_EMAIL,
            "to": [alert.email],
            "subject": subject,
            "html": html_content
        }

        resend.Emails.send(params)
        logger.info(f"Sent alert email to {alert.email} for {alert.ticker}")
        return True

    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False


async def check_all_alerts() -> List[AlertCheckResult]:
    """
    Check all active alerts and trigger any that meet conditions.

    Returns:
        List of AlertCheckResult for triggered alerts
    """
    active_alerts = db_cache.get_active_alerts()
    triggered_results = []

    for alert_data in active_alerts:
        alert_type = alert_data['alert_type']

        if alert_type in [AlertType.PRICE_ABOVE.value, AlertType.PRICE_BELOW.value]:
            result = check_price_alert(alert_data)
        elif alert_type == AlertType.EARNINGS_REMINDER.value:
            result = check_earnings_alert(alert_data)
        else:
            continue

        if result.triggered:
            # Mark as triggered in database
            db_cache.trigger_alert(alert_data['id'])

            # Send email notification
            alert = Alert(
                id=alert_data['id'],
                alert_type=alert_data['alert_type'],
                ticker=alert_data['ticker'],
                condition=alert_data['condition'],
                target_value=alert_data['target_value'],
                email=alert_data['email'],
                is_active=False,
                created_at=alert_data['created_at'],
                triggered_at=datetime.now().isoformat()
            )
            await send_alert_email(alert, result)

            triggered_results.append(result)

    return triggered_results


def get_user_alerts(email: str) -> List[Dict[str, Any]]:
    """Get all alerts for a specific email address."""
    all_alerts = db_cache.get_active_alerts()
    return [a for a in all_alerts if a['email'] == email]


def cancel_alert(alert_id: int) -> bool:
    """Cancel an active alert."""
    return db_cache.deactivate_alert(alert_id)


def get_alert_summary() -> Dict[str, Any]:
    """Get summary of all active alerts."""
    alerts = db_cache.get_active_alerts()

    summary = {
        'total_active': len(alerts),
        'by_type': {},
        'by_ticker': {}
    }

    for alert in alerts:
        # Count by type
        alert_type = alert['alert_type']
        summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1

        # Count by ticker
        ticker = alert['ticker']
        summary['by_ticker'][ticker] = summary['by_ticker'].get(ticker, 0) + 1

    return summary


# Background alert checker (for use with scheduler)
async def run_alert_checker_once():
    """Run the alert checker once. Call this from a scheduler."""
    logger.info("Running alert check...")
    triggered = await check_all_alerts()
    logger.info(f"Alert check complete. {len(triggered)} alerts triggered.")
    return triggered


# CLI for manual testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python alert_system.py create_price TICKER above|below PRICE EMAIL")
        print("  python alert_system.py create_earnings TICKER DAYS_BEFORE EMAIL")
        print("  python alert_system.py check")
        print("  python alert_system.py list")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create_price" and len(sys.argv) >= 6:
        ticker = sys.argv[2]
        condition = sys.argv[3]
        price = float(sys.argv[4])
        email = sys.argv[5]
        alert_id = create_price_alert(ticker, condition, price, email)
        print(f"Created alert {alert_id}")

    elif command == "create_earnings" and len(sys.argv) >= 5:
        ticker = sys.argv[2]
        days = int(sys.argv[3])
        email = sys.argv[4]
        alert_id = create_earnings_alert(ticker, days, email)
        print(f"Created alert {alert_id}")

    elif command == "check":
        triggered = asyncio.run(run_alert_checker_once())
        print(f"Triggered {len(triggered)} alerts")
        for result in triggered:
            print(f"  - {result.alert.ticker}: {result.message}")

    elif command == "list":
        summary = get_alert_summary()
        print(f"Active alerts: {summary['total_active']}")
        print(f"By type: {summary['by_type']}")
        print(f"By ticker: {summary['by_ticker']}")

    else:
        print("Unknown command")
