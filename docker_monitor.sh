#!/bin/bash
# Usage: ./docker_monitor.sh [--verbose]
# Or: MONITOR_MODE=verbose ./docker_monitor.sh

CONTAINER_NAME="research-agent"
MONITOR_MODE="${MONITOR_MODE:-normal}"
[[ "$1" == "--verbose" || "$1" == "-v" ]] && MONITOR_MODE="verbose"

# Alert thresholds
RATE_LIMIT_THRESHOLD=5
ERROR_THRESHOLD=10

echo "=== Docker Monitor (Mode: $MONITOR_MODE) ==="
echo "Ctrl+C = Emergency stop"

# Follow logs with pattern highlighting
docker logs -f --since 1m "$CONTAINER_NAME" 2>&1 | while read -r line; do
    # Always show: stage progress, completions, errors
    if echo "$line" | grep -qE "STAGE:|Complete|ERROR|FAILED|Rate limit"; then
        echo "$line"
    fi

    # Verbose: show API timing
    if [[ "$MONITOR_MODE" == "verbose" ]]; then
        echo "$line" | grep -E "API:|ms$|429|503" && echo "[API] $line"
    fi

    # Check restart count
    RESTARTS=$(docker inspect "$CONTAINER_NAME" --format='{{.RestartCount}}' 2>/dev/null)
    if [[ "$RESTARTS" -ge 3 ]]; then
        echo "!!! ALERT: Container restarted $RESTARTS times - crash loop detected !!!"
    fi
done
