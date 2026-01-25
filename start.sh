#!/bin/bash
# Start the Research Agent Hub
# With restart protection to prevent DoS from crash loops

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# === Restart Protection Configuration ===
CRASH_LOG="logs/crash_history.log"
MAX_CRASHES_PER_HOUR=3
COOLDOWN_MINUTES=10

# Ensure logs directory exists
mkdir -p logs

# === Check for recent crashes (restart loop protection) ===
check_crash_cooldown() {
    if [[ -f "$CRASH_LOG" ]]; then
        # Count crashes in the last hour
        ONE_HOUR_AGO=$(date -d '1 hour ago' +%s 2>/dev/null || date -v-1H +%s)
        RECENT_CRASHES=$(awk -v cutoff="$ONE_HOUR_AGO" '$1 > cutoff {count++} END {print count+0}' "$CRASH_LOG")

        if [[ "$RECENT_CRASHES" -ge "$MAX_CRASHES_PER_HOUR" ]]; then
            LAST_CRASH=$(tail -1 "$CRASH_LOG" | cut -d' ' -f1)
            COOLDOWN_END=$((LAST_CRASH + COOLDOWN_MINUTES * 60))
            NOW=$(date +%s)

            if [[ "$NOW" -lt "$COOLDOWN_END" ]]; then
                REMAINING=$(( (COOLDOWN_END - NOW) / 60 ))
                echo "⚠️  RESTART PROTECTION ACTIVE"
                echo "   App crashed $RECENT_CRASHES times in the last hour."
                echo "   Cooldown: ${REMAINING} minutes remaining."
                echo "   To bypass: rm $CRASH_LOG && ./start.sh"
                exit 1
            fi
        fi
    fi
}

# === Log a crash event ===
log_crash() {
    echo "$(date +%s) $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$CRASH_LOG"
    # Keep only last 50 entries
    tail -50 "$CRASH_LOG" > "$CRASH_LOG.tmp" && mv "$CRASH_LOG.tmp" "$CRASH_LOG"
}

# === Main startup logic ===
check_crash_cooldown

# Check if already running
if pgrep -f "unified_app.py" > /dev/null; then
    echo "Research Agent Hub is already running."
    echo "Use ./stop.sh to stop it first."
    exit 1
fi

# Kill anything on port 7860
fuser -k 7860/tcp 2>/dev/null

echo "Starting Research Agent Hub..."
echo "Restart protection: max $MAX_CRASHES_PER_HOUR crashes/hour, ${COOLDOWN_MINUTES}min cooldown"

nohup ./venv/bin/python unified_app.py > logs/unified_app.log 2>&1 &
APP_PID=$!

# Wait for startup
sleep 3

# Check if started successfully
if pgrep -f "unified_app.py" > /dev/null; then
    IP=$(curl -s --connect-timeout 5 ifconfig.me 2>/dev/null || echo "localhost")
    echo "✓ Research Agent Hub started successfully! (PID: $APP_PID)"
    echo "  Access at: http://${IP}:7860"
    echo "  Logs: tail -f logs/unified_app.log"
else
    log_crash "startup_failure"
    echo "✗ Failed to start. Check logs/unified_app.log"
    echo "  Crash logged for restart protection."
    exit 1
fi
