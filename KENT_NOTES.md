
# Session Notes - Deep Research Agent

## Current App Configuration

| App | Port | Purpose |
|-----|------|---------|
| launcher.py | 7859 | Landing page with links to both agents |
| app.py | 7861 | Deep Research Agent |
| stock_app.py | 7862 | Stock Research Agent |

## To Start Next Session

**Step 1: Kill any old processes**
```bash
sudo pkill -9 -f python
```

**Step 2: Start all 3 apps (each in its own terminal)**

Terminal 1 - Launcher:
```bash
cd ~/deep-research-agent && ./venv/bin/python launcher.py
```

Terminal 2 - Deep Research:
```bash
cd ~/deep-research-agent && ./venv/bin/python app.py
```

Terminal 3 - Stock Research:
```bash
cd ~/deep-research-agent && ./venv/bin/python stock_app.py
```

## Access URLs (use external IP, NOT localhost)

Get your external IP first:
```bash
curl -s ifconfig.me
```

Then access (replace IP if changed):
- **Launcher Hub:** http://34.69.26.110:7859
- **Deep Research:** http://34.69.26.110:7861
- **Stock Research:** http://34.69.26.110:7862

## Important Notes

1. **DO NOT use localhost** - VS Code port forwarding causes blank pages
2. **Use external IP** - Always access via http://EXTERNAL_IP:PORT
3. **All 3 apps must be running** - Launcher just has links, the actual apps run separately

## What Each App Does

### Deep Research Agent (app.py - port 7861)
- General research on any topic
- AI-planned search queries
- Web search via Tavily API
- Comprehensive markdown reports

### Stock Research Agent (stock_app.py - port 7862)
- Buy/Hold/Sell recommendations with confidence %
- Price targets & valuation metrics
- Macro & Political Risk analysis (VIX, sector sensitivity)
- SEC filings with direct links
- Analyst ratings & insider activity

### Launcher (launcher.py - port 7859)
- Landing page with links to both agents
- Just a menu - doesn't do research itself

## Troubleshooting

**Port already in use:**
```bash
sudo fuser -k 7859/tcp  # Kill process on port 7859
sudo fuser -k 7861/tcp  # Kill process on port 7861
sudo fuser -k 7862/tcp  # Kill process on port 7862
```

**Check what's running:**
```bash
ss -tlnp | grep -E '(7859|7861|7862)'
```

**Check external IP:**
```bash
curl -s ifconfig.me
```
