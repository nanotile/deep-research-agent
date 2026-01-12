# Quick Reference Card

## Your VM Setup
```
Instance:    ai-development-workstation
External IP: 34.66.155.187
Zone:        us-central1-a
Tags:        ai-development, jupyter-server
```

## Firewall Status
```
✓ Firewall allows ALL ports (0-65535)
✓ Rules persist when IP changes
✓ No firewall issues
```

## Working Services
```
✓ Port 22    (SSH)        http://34.66.155.187:22
✓ Port 5432  (PostgreSQL) http://34.66.155.187:5432
✓ Port 8000  (Python)     http://34.66.155.187:8000
```

## Common Commands

### Diagnose Port
```bash
python vm_firewall_utils.py 3000        # Check specific port
python check_firewall.py                # Full analysis
```

### Get Current IP
```bash
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"
```

### Start Service (Correct Way)
```python
# Gradio
demo.launch(server_name="0.0.0.0", server_port=7860)

# Flask
app.run(host="0.0.0.0", port=5000)

# FastAPI
uvicorn.run("main:app", host="0.0.0.0", port=8000)

# HTTP Server
python3 -m http.server 3000 --bind 0.0.0.0
```

### Check What's Listening
```bash
ss -tlnp | grep LISTEN
# Look for 0.0.0.0:PORT (good) or 127.0.0.1:PORT (bad)
```

### View Firewall Rules
```bash
gcloud compute firewall-rules list
```

## Why Port 8000 Works (Others Don't)

| Port | Running? | Bound to 0.0.0.0? | Firewall? | Result |
|------|----------|-------------------|-----------|--------|
| 8000 | ✓        | ✓                 | ✓         | ✓ Works |
| 3000 | ❌       | ❌                | ✓         | ❌ Fails |
| Others | ✓      | ❌ (127.0.0.1)    | ✓         | ❌ Fails |

## The Fix

**Always bind to 0.0.0.0 (not 127.0.0.1)**

❌ Wrong: `host="127.0.0.1"`
✓ Right: `host="0.0.0.0"`

## Test Any Port Quickly
```bash
# Start test server
python test_port_3000.py

# Visit in browser
http://34.66.155.187:3000
```

## Files You Can Use in Any Project
- `vm_ip_utils.py` - IP management
- `vm_firewall_utils.py` - Port diagnostics
- `check_firewall.py` - Complete analysis

## Remember
1. Firewall is NOT the problem (allows all ports)
2. Services must bind to 0.0.0.0 (not 127.0.0.1)
3. Firewall rules persist when IP changes
4. Port 8000 works because it's configured correctly
