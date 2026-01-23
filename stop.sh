#!/bin/bash
# Stop the Research Agent Hub

echo "Stopping Research Agent Hub..."

# Kill by process name
pkill -f "unified_app.py" 2>/dev/null

# Also kill anything on port 7860
fuser -k 7860/tcp 2>/dev/null

sleep 1

# Verify stopped
if pgrep -f "unified_app.py" > /dev/null; then
    echo "Warning: Process still running, trying force kill..."
    pkill -9 -f "unified_app.py" 2>/dev/null
    sleep 1
fi

if pgrep -f "unified_app.py" > /dev/null; then
    echo "Failed to stop. Try: sudo pkill -9 -f unified_app.py"
    exit 1
else
    echo "Research Agent Hub stopped."
fi
