# Birds Eye View

A high-level overview of the Deep Research Agent project and its infrastructure.

---

## What Is This?

A **production-ready AI research assistant** running on Google Cloud Platform, accessible via custom domain with automatic HTTPS.

```
User visits research.kentbenson.net
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLOUDFLARE EDGE                          │
│            (SSL termination, DDoS protection)               │
└─────────────────────────────────────────────────────────────┘
         │
         │ Encrypted tunnel (outbound-only)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      GCP VM                                  │
│                  Ubuntu 22.04 LTS                           │
│                  4 CPUs, 22GB RAM                           │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    Docker                               │ │
│  │  ┌──────────────────────┐  ┌─────────────────────────┐ │ │
│  │  │  deep-research       │  │  (future apps)          │ │ │
│  │  │  :7860               │  │  :8080, etc.            │ │ │
│  │  └──────────────────────┘  └─────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  cloudflared (system service)                          │ │
│  │  Routes: research.kentbenson.net → localhost:7860      │ │
│  │          digital_mind.kentbenson.net → localhost:8080  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## The Application

### Deep Research Agent

A **multi-agent AI system** that automates comprehensive research and report generation.

**What it does:**
1. Takes a research question from the user
2. AI plans optimal search queries
3. Executes web searches (Tavily API) or uses LLM knowledge
4. Synthesizes findings into a professional report
5. Optionally emails the report

**Agent Pipeline:**
```
User Query
    │
    ▼
┌─────────────────┐
│ Planning Agent  │  → Generates 3 strategic search queries
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Research Agent  │  → Executes searches, gathers information
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Agent    │  → Synthesizes into structured markdown
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Email Agent     │  → (Optional) Sends via Resend API
└─────────────────┘
```

**Tech Stack:**
- **LLM:** Claude (Anthropic API) - claude-sonnet-4 or claude-opus-4
- **Web Search:** Tavily API (real-time web results)
- **UI:** Gradio 6.x (web interface)
- **Email:** Resend API (optional)
- **Language:** Python 3.x

---

## Infrastructure

### Google Cloud Platform VM

| Spec | Value |
|------|-------|
| OS | Ubuntu 22.04.5 LTS |
| CPUs | 4 |
| RAM | 22 GB |
| Location | (GCP zone) |

### Docker Containers

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| deep-research | deep-research-agent:latest | 7860 | Main application |
| (future) | TBD | 8080 | Digital Mind app |

### Cloudflare Tunnel

| Component | Value |
|-----------|-------|
| Tunnel Name | kent-vm |
| Tunnel ID | af39f562-8246-4699-98e6-7215069fc101 |
| Service | systemd (cloudflared.service) |
| Config | ~/.cloudflared/config.yml |

**Routing:**
| Subdomain | Port | Status |
|-----------|------|--------|
| research.kentbenson.net | 7860 | ✅ Live |
| digital_mind.kentbenson.net | 8080 | ⏳ Pending |

---

## Why This Architecture?

### Cloudflare Tunnel Benefits

| Traditional Approach | Cloudflare Tunnel |
|---------------------|-------------------|
| Open firewall ports | Zero open ports |
| Manage SSL certs | Automatic HTTPS |
| Expose server IP | IP hidden |
| Manual DDoS protection | Built-in protection |
| Port forwarding | Simple config file |

### Docker Benefits

| Bare Metal | Docker |
|------------|--------|
| Dependency conflicts | Isolated environments |
| Manual setup | Reproducible builds |
| Difficult updates | Easy container replacement |
| Version chaos | Tagged images |

---

## File Structure

```
/home/kent_benson/
│
├── deep-research-agent/          # Main application
│   ├── app.py                    # Gradio web UI
│   ├── deep_research_agent.py    # Core agent logic
│   ├── Dockerfile                # Container build
│   ├── requirements.txt          # Python dependencies
│   ├── CLAUDE.md                 # Claude Code instructions
│   ├── BIRDS_EYE_VIEW.md         # This file
│   └── venv/                     # Local development
│
└── .cloudflared/                 # Tunnel configuration
    ├── config.yml                # Routing rules
    ├── cert.pem                  # Cloudflare auth
    ├── <tunnel-id>.json          # Tunnel credentials
    ├── QUICK_START.md            # Quick reference
    └── CI_CLOUDFLARE.md          # Detailed setup guide
```

---

## Common Operations

### Check System Status

```bash
# All services at a glance
docker ps                         # Running containers
systemctl status cloudflared      # Tunnel service
cloudflared tunnel list           # Tunnel connections
```

### Restart the App

```bash
docker restart deep-research
```

### View Logs

```bash
# Application logs
docker logs -f deep-research

# Tunnel logs
journalctl -u cloudflared -f
```

### Update the App

```bash
cd ~/deep-research-agent
git pull
docker build -t deep-research-agent .
docker stop deep-research
docker rm deep-research
docker run -d --name deep-research \
  -p 7860:7860 \
  --env-file .env \
  deep-research-agent
```

### Add a New Subdomain

1. Start container on new port
2. Edit `~/.cloudflared/config.yml`
3. Run `cloudflared tunnel route dns kent-vm newapp.kentbenson.net`
4. Restart: `sudo systemctl restart cloudflared`

---

## API Keys Required

| Key | Purpose | Required |
|-----|---------|----------|
| ANTHROPIC_API_KEY | Claude LLM | ✅ Yes |
| TAVILY_API_KEY | Web search | Optional (falls back to LLM) |
| RESEND_API_KEY | Email delivery | Optional |

Store in `.env` file (not committed to git).

---

## Cost Estimates

### Per Research Query
- **Claude API:** ~5,000-15,000 tokens → $0.05-$0.15
- **Tavily API:** Free tier available, then usage-based

### Infrastructure
- **GCP VM:** Varies by instance type
- **Cloudflare Tunnel:** Free
- **Domain:** ~$10-15/year

---

## Security Model

```
┌─────────────────────────────────────────────────────┐
│                   INTERNET                          │
│                                                     │
│  ✅ HTTPS only (Cloudflare terminates SSL)         │
│  ✅ DDoS protection (Cloudflare)                   │
│  ✅ Server IP hidden                               │
└─────────────────────────────────────────────────────┘
                        │
                        │ Encrypted tunnel
                        │ (outbound connection only)
                        ▼
┌─────────────────────────────────────────────────────┐
│                     GCP VM                          │
│                                                     │
│  ✅ No inbound ports open (except SSH)             │
│  ✅ Containers isolated from host                  │
│  ✅ API keys in .env (not in images)               │
│  ✅ Credentials files have restricted permissions  │
└─────────────────────────────────────────────────────┘
```

---

## Quick Links

| Resource | URL |
|----------|-----|
| Live App | https://research.kentbenson.net |
| Cloudflare Dashboard | https://one.dash.cloudflare.com |
| Anthropic Console | https://console.anthropic.com |
| Tavily Dashboard | https://tavily.com |

---

## Documentation Index

| File | Description |
|------|-------------|
| `CLAUDE.md` | Instructions for Claude Code AI assistant |
| `BIRDS_EYE_VIEW.md` | This file - high-level overview |
| `README.md` | Project readme |
| `~/.cloudflared/QUICK_START.md` | Tunnel quick reference |
| `~/.cloudflared/CI_CLOUDFLARE.md` | Detailed tunnel setup guide |
