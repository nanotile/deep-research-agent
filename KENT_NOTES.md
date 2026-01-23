
# Session Notes - Deep Research Agent

---

## 🚀 Session Completed: January 23, 2026

### What We Built Today
✅ **2026 Global Market Context Enhancement** - Committed & Pushed

For tech/semiconductor stocks (NVDA, AMD, MSFT, GOOGL, etc.), reports now include:
- **2026 Strategic Outlook** header
- **Key 2026 Insights** (Agentic AI positioning + Geopolitical risk)
- **Technical & Competitive Moat** table (Process Node, Software Moat, Supply Chain)
- **2026 Geopolitical Impact Table** (China Exposure 🔴/🟡/🟢, Surcharge %, Tech Sovereignty)
- **Export Control Analysis** with BIS 25% surcharge EPS impact

Non-tech stocks (JNJ, healthcare, etc.) use standard template.

---

## 💡 Enhancement Suggestions for Next Session

### HIGH PRIORITY - Quick Wins

| Enhancement | Effort | Impact | Description |
|-------------|--------|--------|-------------|
| **Add API Keys** | 5 min | High | Add FINNHUB_API_KEY and ALPHA_VANTAGE_API_KEY to .env for richer data (analyst ratings, financials) |
| **Peer Comparison** | 2 hrs | High | Add dropdown to compare 2-3 stocks side-by-side in one report |
| **PDF Export** | 1 hr | Medium | Add "Download PDF" button using markdown-pdf or weasyprint |
| **Email Report** | 30 min | Medium | Enable email delivery (Resend API already integrated) |

### MEDIUM PRIORITY - Feature Additions

| Enhancement | Effort | Impact | Description |
|-------------|--------|--------|-------------|
| **Watchlist Mode** | 3 hrs | High | Save favorite tickers, run batch analysis, track changes |
| **Earnings Calendar** | 2 hrs | Medium | Show upcoming earnings dates, historical beats/misses |
| **Technical Charts** | 3 hrs | Medium | Add price charts with moving averages (plotly/matplotlib) |
| **Custom Alerts** | 4 hrs | Medium | Price alerts, earnings alerts, news alerts via email |
| **Sector Reports** | 4 hrs | Medium | Generate reports for entire sectors (all semiconductor stocks) |

### LOWER PRIORITY - Advanced Features

| Enhancement | Effort | Impact | Description |
|-------------|--------|--------|-------------|
| **Portfolio Analysis** | 6 hrs | High | Upload portfolio CSV, get aggregate risk/exposure analysis |
| **Historical Backtest** | 8 hrs | Medium | "What would our model have said 6 months ago?" |
| **Voice Summary** | 4 hrs | Low | TTS audio summary of key findings |
| **Competitor Auto-Detect** | 3 hrs | Medium | Automatically identify and compare top 3 competitors |
| **News Sentiment Trend** | 4 hrs | Medium | Chart sentiment over time (requires historical data) |

### 2026 Context Enhancements

| Enhancement | Effort | Description |
|-------------|--------|-------------|
| **More Sector Templates** | 2 hrs | Add 2026 templates for: Energy (oil prices), Finance (Fed rates), Defense (govt contracts) |
| **Dynamic Surcharge Calc** | 1 hr | Calculate EPS impact from actual China revenue % in filings |
| **CHIPS Act Tracker** | 2 hrs | Track which companies have received CHIPS Act funding |
| **AI Capex Monitor** | 2 hrs | Track Big Tech AI spending announcements |

---

## 🔧 Technical Debt / Fixes

- [ ] Cache yfinance sector lookups (currently fetches twice)
- [ ] Add retry logic for failed API calls
- [ ] Rate limiting for Tavily API (avoid 429 errors)
- [ ] Add loading spinners for each data source in UI
- [ ] Unit tests for market_context_2026.py functions

---

## Current App Configuration

| App | Port | Purpose |
|-----|------|---------|
| **unified_app.py** | **7860** | **Research Agent Hub (both agents in tabs)** |
| app.py | 7861 | Deep Research Agent (standalone fallback) |
| stock_app.py | 7862 | Stock Research Agent (standalone fallback) |

## To Start Next Session

```bash
cd ~/deep-research-agent
./stop.sh   # Stop any running instance
./start.sh  # Start the Research Agent Hub
```

That's it! The scripts handle everything automatically.

**View logs:**
```bash
tail -f logs/unified_app.log
```

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

**Stop the app:**
```bash
./stop.sh
```

**Port still in use after stop.sh:**
```bash
sudo fuser -k 7860/tcp
```

**Check what's running:**
```bash
ss -tlnp | grep 7860
```

**Check external IP:**
```bash
curl -s ifconfig.me
```

**View logs:**
```bash
tail -f logs/unified_app.log
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
