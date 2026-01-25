#!/bin/bash
# Auto-shutdown script for GCP VM
# Shuts down the VM after a period of inactivity to save costs
#
# Usage:
#   ./auto_shutdown.sh              # Run once (check and shutdown if idle)
#   ./auto_shutdown.sh --install    # Install as cron job (checks every 5 min)
#   ./auto_shutdown.sh --uninstall  # Remove cron job
#   ./auto_shutdown.sh --status     # Show current idle time and settings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# === Configuration ===
IDLE_THRESHOLD_MINUTES=30       # Shutdown after this many minutes of inactivity
ACTIVITY_LOG="logs/activity.log"
LAST_ACTIVITY_FILE="logs/.last_activity"
CHECK_PORT=7860                 # Gradio app port

# Ensure logs directory exists
mkdir -p logs

# === Helper Functions ===

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$ACTIVITY_LOG"
    echo "$1"
}

get_last_activity() {
    if [[ -f "$LAST_ACTIVITY_FILE" ]]; then
        cat "$LAST_ACTIVITY_FILE"
    else
        echo 0
    fi
}

update_activity() {
    date +%s > "$LAST_ACTIVITY_FILE"
}

check_app_activity() {
    # Check 1: Is the app running?
    if ! pgrep -f "unified_app.py" > /dev/null; then
        return 1  # App not running = no activity
    fi

    # Check 2: Recent network connections to the app port
    # Count established connections in the last check
    CONNECTIONS=$(ss -tn state established "( sport = :$CHECK_PORT )" 2>/dev/null | wc -l)
    if [[ "$CONNECTIONS" -gt 1 ]]; then  # Header line counts as 1
        return 0  # Active connections
    fi

    # Check 3: Check if unified_app.log was modified recently (within 5 min)
    if [[ -f "logs/unified_app.log" ]]; then
        LOG_MOD_TIME=$(stat -c %Y "logs/unified_app.log" 2>/dev/null || stat -f %m "logs/unified_app.log" 2>/dev/null)
        NOW=$(date +%s)
        LOG_AGE=$(( (NOW - LOG_MOD_TIME) / 60 ))
        if [[ "$LOG_AGE" -lt 5 ]]; then
            return 0  # Recent log activity
        fi
    fi

    # Check 4: Any SSH sessions? (user might be working)
    SSH_SESSIONS=$(who | grep -c pts 2>/dev/null || echo 0)
    if [[ "$SSH_SESSIONS" -gt 0 ]]; then
        return 0  # Active SSH session
    fi

    return 1  # No activity detected
}

get_idle_minutes() {
    LAST=$(get_last_activity)
    NOW=$(date +%s)
    if [[ "$LAST" -eq 0 ]]; then
        echo 0
    else
        echo $(( (NOW - LAST) / 60 ))
    fi
}

# === Commands ===

do_check() {
    if check_app_activity; then
        update_activity
        log_message "Activity detected - reset idle timer"
    else
        IDLE_MINS=$(get_idle_minutes)

        if [[ "$IDLE_MINS" -ge "$IDLE_THRESHOLD_MINUTES" ]]; then
            log_message "SHUTDOWN: Idle for ${IDLE_MINS} minutes (threshold: ${IDLE_THRESHOLD_MINUTES})"
            log_message "Stopping app before shutdown..."
            ./stop.sh 2>/dev/null
            sleep 2
            log_message "Initiating VM shutdown..."
            sudo shutdown -h now
        else
            REMAINING=$((IDLE_THRESHOLD_MINUTES - IDLE_MINS))
            log_message "Idle for ${IDLE_MINS} min - shutdown in ${REMAINING} min if no activity"
        fi
    fi
}

do_install() {
    # Remove existing cron job if any
    crontab -l 2>/dev/null | grep -v "auto_shutdown.sh" | crontab -

    # Add new cron job - runs every 5 minutes
    (crontab -l 2>/dev/null; echo "*/5 * * * * $SCRIPT_DIR/auto_shutdown.sh >> $SCRIPT_DIR/logs/auto_shutdown_cron.log 2>&1") | crontab -

    # Initialize last activity to now
    update_activity

    log_message "INSTALLED: Auto-shutdown cron job (checks every 5 min, shuts down after ${IDLE_THRESHOLD_MINUTES} min idle)"
    echo ""
    echo "Current cron jobs:"
    crontab -l
    echo ""
    echo "To modify idle threshold, edit IDLE_THRESHOLD_MINUTES in this script"
    echo "To uninstall: ./auto_shutdown.sh --uninstall"
}

do_uninstall() {
    crontab -l 2>/dev/null | grep -v "auto_shutdown.sh" | crontab -
    log_message "UNINSTALLED: Auto-shutdown cron job removed"
    echo "Auto-shutdown disabled."
}

do_status() {
    echo "=== Auto-Shutdown Status ==="
    echo ""

    # Check if cron job is installed
    if crontab -l 2>/dev/null | grep -q "auto_shutdown.sh"; then
        echo "Cron job: INSTALLED (checking every 5 min)"
    else
        echo "Cron job: NOT INSTALLED"
    fi
    echo ""

    echo "Settings:"
    echo "  Idle threshold: ${IDLE_THRESHOLD_MINUTES} minutes"
    echo "  Monitor port:   ${CHECK_PORT}"
    echo ""

    # Current idle time
    if [[ -f "$LAST_ACTIVITY_FILE" ]]; then
        IDLE_MINS=$(get_idle_minutes)
        REMAINING=$((IDLE_THRESHOLD_MINUTES - IDLE_MINS))
        echo "Current status:"
        echo "  Idle time:      ${IDLE_MINS} minutes"
        if [[ "$REMAINING" -gt 0 ]]; then
            echo "  Shutdown in:    ${REMAINING} minutes (if no activity)"
        else
            echo "  Shutdown in:    IMMINENT (next check)"
        fi
    else
        echo "Current status:   No activity tracked yet"
    fi
    echo ""

    # App status
    if pgrep -f "unified_app.py" > /dev/null; then
        echo "App status:       RUNNING"
    else
        echo "App status:       NOT RUNNING"
    fi

    # SSH sessions
    SSH_COUNT=$(who | grep -c pts 2>/dev/null || echo 0)
    echo "SSH sessions:     ${SSH_COUNT}"
    echo ""

    # Recent activity log
    if [[ -f "$ACTIVITY_LOG" ]]; then
        echo "Recent activity (last 5 entries):"
        tail -5 "$ACTIVITY_LOG" | sed 's/^/  /'
    fi
}

do_reset() {
    update_activity
    log_message "RESET: Idle timer reset manually"
    echo "Idle timer reset. Shutdown postponed."
}

# === Main ===

case "${1:-}" in
    --install|-i)
        do_install
        ;;
    --uninstall|-u)
        do_uninstall
        ;;
    --status|-s)
        do_status
        ;;
    --reset|-r)
        do_reset
        ;;
    --help|-h)
        echo "Auto-shutdown script for GCP VM cost savings"
        echo ""
        echo "Usage:"
        echo "  ./auto_shutdown.sh              Check activity and shutdown if idle"
        echo "  ./auto_shutdown.sh --install    Install as cron job (every 5 min)"
        echo "  ./auto_shutdown.sh --uninstall  Remove cron job"
        echo "  ./auto_shutdown.sh --status     Show idle time and settings"
        echo "  ./auto_shutdown.sh --reset      Reset idle timer (postpone shutdown)"
        echo ""
        echo "Configuration (edit script to change):"
        echo "  IDLE_THRESHOLD_MINUTES=${IDLE_THRESHOLD_MINUTES}"
        ;;
    *)
        do_check
        ;;
esac
