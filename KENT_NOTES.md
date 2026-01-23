
# Session Notes - Deep Research Agent

## Current App Configuration

| App | Port | Purpose |
|-----|------|---------|
| **unified_app.py** | **7860** | **Research Agent Hub (both agents in tabs)** |
| app.py | 7861 | Deep Research Agent (standalone fallback) |
| stock_app.py | 7862 | Stock Research Agent (standalone fallback) |

## To Start Next Session

**Step 1: Kill any old processes**
```bash
sudo pkill -9 -f python
```

**Step 2: Start the unified app (single terminal)**
```bash
cd ~/deep-research-agent && ./venv/bin/python unified_app.py
```

That's it! One command, one app, both research agents.

## Access URL (use external IP, NOT localhost)

Get your external IP first:
```bash
curl -s ifconfig.me
```

Then access (replace IP if changed):
- **Research Agent Hub:** http://34.69.26.110:7860

## Important Notes

1. **DO NOT use localhost** - VS Code port forwarding causes blank pages
2. **Use external IP** - Always access via http://EXTERNAL_IP:PORT
3. **Single app** - The unified app has tabs for both agents

## What the Unified App Contains

### Tab 1: Deep Research
- General research on any topic
- AI-planned search queries
- Web search via Tavily API
- Comprehensive markdown reports

### Tab 2: Stock Research
- Buy/Hold/Sell recommendations with confidence %
- Price targets & valuation metrics
- Macro & Political Risk analysis (VIX, sector sensitivity)
- SEC filings with direct links
- Analyst ratings & insider activity

## Troubleshooting

**Port already in use:**
```bash
sudo fuser -k 7860/tcp  # Kill process on port 7860
```

**Check what's running:**
```bash
ss -tlnp | grep 7860
```

**Check external IP:**
```bash
curl -s ifconfig.me
```

## Standalone Apps (Fallback)

If you need to run the apps separately for any reason:

```bash
# Deep Research only
cd ~/deep-research-agent && ./venv/bin/python app.py
# Access at http://EXTERNAL_IP:7861

# Stock Research only
cd ~/deep-research-agent && ./venv/bin/python stock_app.py
# Access at http://EXTERNAL_IP:7862
```
