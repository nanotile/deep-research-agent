#!/bin/bash
# Safe Start - Testing mode for Research Agent Hub
# Runs in foreground with monitoring, auto-stops on issues

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# === Configuration ===
TIMEOUT_SECONDS=300      # Auto-stop after 5 minutes (for testing)
MAX_CPU_PERCENT=80       # Warn if CPU exceeds this
MAX_MEM_MB=2048          # Warn if memory exceeds 2GB
CHECK_INTERVAL=5         # Resource check interval

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  SAFE START - Testing Mode"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Auto-timeout: ${TIMEOUT_SECONDS}s"
echo "  • CPU warning: >${MAX_CPU_PERCENT}%"
echo "  • Memory warning: >${MAX_MEM_MB}MB"
echo "  • Press Ctrl+C to stop anytime"
echo ""

# === Pre-flight checks ===
echo "Running pre-flight checks..."

# Check 1: Test cache module
echo -n "  [1/3] Testing cache module... "
if timeout 10 ./venv/bin/python -c "from utils.cache import db_cache; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "  Cache module test failed. Check utils/cache.py"
    exit 1
fi

# Check 2: Test VM IP utils
echo -n "  [2/3] Testing VM IP utils... "
if timeout 15 ./venv/bin/python -c "from vm_ip_utils import get_vm_public_ip; ip=get_vm_public_ip(); print(ip or 'localhost')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ (timeout/failed, will use localhost)${NC}"
fi

# Check 3: Check port availability
echo -n "  [3/3] Checking port 7860... "
if ss -tlnp 2>/dev/null | grep -q ":7860 "; then
    echo -e "${RED}✗ PORT IN USE${NC}"
    echo "  Run ./stop.sh first to free the port"
    exit 1
else
    echo -e "${GREEN}✓ available${NC}"
fi

echo ""
echo "=========================================="
echo "  Starting app in foreground..."
echo "  (Output below, Ctrl+C to stop)"
echo "=========================================="
echo ""

# === Start with timeout ===
# Run in foreground so user can see all output
# Timeout ensures it doesn't run forever during testing
timeout $TIMEOUT_SECONDS ./venv/bin/python unified_app.py
EXIT_CODE=$?

echo ""
echo "=========================================="

if [[ $EXIT_CODE -eq 124 ]]; then
    echo -e "${YELLOW}⏱ Timeout reached (${TIMEOUT_SECONDS}s) - app auto-stopped${NC}"
    echo "  This is normal for testing. Use ./start.sh for production."
elif [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ App exited cleanly${NC}"
elif [[ $EXIT_CODE -eq 130 ]]; then
    echo -e "${GREEN}✓ Stopped by user (Ctrl+C)${NC}"
else
    echo -e "${RED}✗ App crashed with exit code: $EXIT_CODE${NC}"
    echo "  Check the output above for errors."
fi

echo "=========================================="
