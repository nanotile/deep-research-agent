#!/bin/bash
# Stop the Research Agent Hub
# With graceful shutdown (SIGTERM first, then SIGKILL)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GRACEFUL_TIMEOUT=10  # Seconds to wait for graceful shutdown

echo "Stopping Research Agent Hub..."

# Get PID(s) before stopping
PIDS=$(pgrep -f "unified_app.py" 2>/dev/null)

if [[ -z "$PIDS" ]]; then
    echo "✓ Research Agent Hub is not running."
    # Clean up port just in case
    fuser -k 7860/tcp 2>/dev/null
    exit 0
fi

echo "  Found process(es): $PIDS"

# Step 1: Graceful shutdown with SIGTERM
echo "  Sending SIGTERM (graceful shutdown)..."
pkill -TERM -f "unified_app.py" 2>/dev/null

# Wait for graceful shutdown
for i in $(seq 1 $GRACEFUL_TIMEOUT); do
    if ! pgrep -f "unified_app.py" > /dev/null; then
        echo "✓ Gracefully stopped after ${i}s"
        break
    fi
    sleep 1
done

# Step 2: Force kill if still running
if pgrep -f "unified_app.py" > /dev/null; then
    echo "  Process didn't stop gracefully, sending SIGKILL..."
    pkill -9 -f "unified_app.py" 2>/dev/null
    sleep 1
fi

# Step 3: Clean up port
fuser -k 7860/tcp 2>/dev/null

# Final verification
if pgrep -f "unified_app.py" > /dev/null; then
    echo "✗ Failed to stop. Try: sudo pkill -9 -f unified_app.py"
    exit 1
else
    echo "✓ Research Agent Hub stopped."
fi
