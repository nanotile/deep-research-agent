#!/bin/bash
# Start the Research Agent Hub

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if already running
if pgrep -f "unified_app.py" > /dev/null; then
    echo "Research Agent Hub is already running."
    echo "Use ./stop.sh to stop it first."
    exit 1
fi

# Kill anything on port 7860
fuser -k 7860/tcp 2>/dev/null

echo "Starting Research Agent Hub..."
nohup ./venv/bin/python unified_app.py > logs/unified_app.log 2>&1 &

# Wait for startup
sleep 3

# Check if started successfully
if pgrep -f "unified_app.py" > /dev/null; then
    IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
    echo "Research Agent Hub started successfully!"
    echo "Access at: http://${IP}:7860"
    echo "Logs: tail -f logs/unified_app.log"
else
    echo "Failed to start. Check logs/unified_app.log"
    exit 1
fi
