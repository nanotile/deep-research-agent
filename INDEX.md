# Deep Research Agent - File Index

Quick reference for all files in this repository.

## 📚 Start Here

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** | Project overview, installation, basic usage | First time setup |
| **CLAUDE.md** | Guidance for Claude Code sessions | Every new session |
| **SESSION_SUMMARY.md** | Summary of VM utilities session | Understanding what was built |
| **QUICK_REFERENCE.md** | One-page command reference | Quick lookups |

## 🚀 Core Application Files

| File | Description |
|------|-------------|
| `deep_research_agent.py` | Main research logic with 4 agents (Planning, Research, Report, Email) |
| `app.py` | Gradio web interface (runs on port 7860) |
| `test_email.py` | Test Resend email API integration |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables |

## 🛠️ VM Utilities (Reusable)

### Production Utilities (Copy to Other Projects)

| File | Lines | Purpose |
|------|-------|---------|
| **vm_ip_utils.py** | 540 | Dynamic IP management for VMs with non-static IPs |
| **vm_firewall_utils.py** | 353 | Port & firewall diagnostics |
| **check_firewall.py** | 395 | Complete GCP firewall analysis via gcloud |

### Test & Example Files

| File | Purpose |
|------|---------|
| `test_port_3000.py` | Test server with success page for port 3000 |
| `vm_ip_example_usage.py` | 10 usage examples (Gradio, Flask, FastAPI, etc.) |
| `example_app_with_vm_ip.py` | Deep Research Agent with VM utilities integrated |

## 📖 Documentation

### VM Utilities Documentation

| File | What It Covers |
|------|----------------|
| **UTILITIES_README.md** | Master documentation for all utilities |
| **VM_IP_README.md** | Complete vm_ip_utils.py reference |
| **FIREWALL_GUIDE.md** | GCP firewall rules explained, persistence across IP changes |
| **FIREWALL_ANALYSIS.md** | Detailed analysis of this specific VM |
| **PORT_3000_SOLUTION.md** | Port 3000 troubleshooting guide |
| **FINAL_SUMMARY.md** | Complete summary answering all session questions |

### This File

| File | Purpose |
|------|---------|
| **INDEX.md** | This file - navigation guide |

## 🎯 Common Tasks

### Run the Application

```bash
# Web interface (Gradio)
python app.py

# CLI mode
python deep_research_agent.py

# Test email
python test_email.py
```

### Diagnose Port Issues

```bash
# Complete analysis
python check_firewall.py

# Check specific port
python vm_firewall_utils.py 3000

# Test port 3000
python test_port_3000.py
```

### Get VM Information

```bash
# Get current IP
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"

# Show access info for a service
python -c "from vm_ip_utils import print_access_info; print_access_info(7860, 'My App')"
```

## 📂 File Organization

```
deep-research-agent/
├── Core Application
│   ├── deep_research_agent.py    # Main research logic
│   ├── app.py                     # Gradio web UI
│   ├── test_email.py              # Email testing
│   └── requirements.txt           # Dependencies
│
├── VM Utilities (Reusable)
│   ├── vm_ip_utils.py             # IP management
│   ├── vm_firewall_utils.py       # Port diagnostics
│   ├── check_firewall.py          # Firewall analysis
│   ├── test_port_3000.py          # Test server
│   ├── vm_ip_example_usage.py     # Usage examples
│   └── example_app_with_vm_ip.py  # Integration example
│
├── Documentation
│   ├── CLAUDE.md                  # For Claude Code sessions
│   ├── README.md                  # Project README
│   ├── SESSION_SUMMARY.md         # Session summary
│   ├── UTILITIES_README.md        # Utilities master doc
│   ├── VM_IP_README.md            # IP utils reference
│   ├── FIREWALL_GUIDE.md          # Firewall guide
│   ├── FIREWALL_ANALYSIS.md       # VM analysis
│   ├── PORT_3000_SOLUTION.md      # Port troubleshooting
│   ├── FINAL_SUMMARY.md           # Complete summary
│   ├── QUICK_REFERENCE.md         # Command reference
│   └── INDEX.md                   # This file
│
└── Configuration
    ├── .env                       # Your API keys (not in git)
    └── .env.example               # Template for .env
```

## 🔍 Finding Information

### "How do I..."

| Question | File to Check |
|----------|---------------|
| Run the Deep Research Agent? | README.md, CLAUDE.md |
| Fix port connection issues? | QUICK_REFERENCE.md, PORT_3000_SOLUTION.md |
| Get my VM's current IP? | VM_IP_README.md, QUICK_REFERENCE.md |
| Check firewall rules? | FIREWALL_GUIDE.md, check_firewall.py |
| Use utilities in another project? | UTILITIES_README.md, vm_ip_example_usage.py |
| Understand the agent architecture? | CLAUDE.md, deep_research_agent.py |

### "What is..."

| Question | File to Check |
|----------|---------------|
| The multi-agent architecture? | CLAUDE.md (Multi-Agent Architecture section) |
| Port 8000 working but not 3000? | FINAL_SUMMARY.md, FIREWALL_ANALYSIS.md |
| The firewall configuration? | FIREWALL_ANALYSIS.md, check_firewall.py output |
| Available utilities? | UTILITIES_README.md, INDEX.md |

### "Why does..."

| Question | File to Check |
|----------|---------------|
| Port show ERR_CONNECTION_REFUSED? | PORT_3000_SOLUTION.md, FIREWALL_GUIDE.md |
| Only port 8000 return data? | FINAL_SUMMARY.md, FIREWALL_ANALYSIS.md |
| Firewall persist across IP changes? | FIREWALL_GUIDE.md, FINAL_SUMMARY.md |

## 🎓 Learning Path

**For New Users:**
1. README.md - Understand the project
2. CLAUDE.md - Learn the architecture
3. QUICK_REFERENCE.md - Common commands

**For Port/Network Issues:**
1. QUICK_REFERENCE.md - Quick diagnosis
2. FIREWALL_GUIDE.md - Understand concepts
3. Run: `python check_firewall.py` - Analyze your setup

**For Using Utilities:**
1. UTILITIES_README.md - Overview
2. VM_IP_README.md - IP utilities reference
3. vm_ip_example_usage.py - See examples

**For Integration:**
1. example_app_with_vm_ip.py - See working example
2. vm_ip_example_usage.py - Framework-specific patterns
3. Copy utilities to your project

## 📊 Quick Stats

- **Total Documentation**: 12 markdown files
- **Utility Modules**: 3 (vm_ip_utils, vm_firewall_utils, check_firewall)
- **Test/Example Files**: 3
- **Total Lines of Code**: ~2,500+
- **Frameworks Covered**: Gradio, Flask, FastAPI, Node.js

## 🔗 External Links

- **GCP Console**: https://console.cloud.google.com
- **Firewall Rules**: https://console.cloud.google.com/networking/firewalls/list
- **OpenAI Usage**: https://platform.openai.com/usage
- **Resend Dashboard**: https://resend.com/emails

## ✅ All Tests Passing

- ✓ vm_ip_utils.py - IP detection working (34.66.155.187)
- ✓ vm_firewall_utils.py - Port diagnostics working
- ✓ check_firewall.py - Firewall analysis working
- ✓ test_port_3000.py - Test server working
- ✓ Port 3000 - Accessible externally

---

**Last Updated**: Session ending 2025-10-30
**VM IP**: 34.66.155.187 (non-static)
**Status**: All utilities operational
