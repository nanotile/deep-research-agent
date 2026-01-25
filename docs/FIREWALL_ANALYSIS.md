# Complete Firewall & Port Analysis

## Your VM Configuration

**Instance**: `ai-development-workstation`
**External IP**: `34.66.155.187`
**Zone**: `us-central1-a`
**Network Tags**: `ai-development`, `jupyter-server`

## Firewall Rules (from gcloud)

Your VM has these firewall rules applied:

### 1. allow-ai-development
- **Ports**: 22, 3000, 8000, 8080, 8888, 8889
- **Source**: 0.0.0.0/0 (anywhere)
- **Target Tags**: `ai-development` ✓ (YOUR VM HAS THIS TAG)
- **Status**: ✓ ACTIVE for your VM

### 2. allow-web-apps
- **Ports**: 3000, 8000
- **Source**: 0.0.0.0/0 (anywhere)
- **Target Tags**: None (applies to all instances)
- **Status**: ✓ ACTIVE for your VM

### 3. allow-week2-llm
- **Ports**: 5000, 7859-7862
- **Source**: 0.0.0.0/0 (anywhere)
- **Target Tags**: `ai-development`, `jupyter-server` ✓ (YOUR VM HAS BOTH)
- **Status**: ✓ ACTIVE for your VM

### 4. Default Rules
- **SSH (22)**: ✓ Allowed from anywhere
- **RDP (3389)**: ✓ Allowed from anywhere
- **Internal traffic**: ✓ Allowed within VPC

## Summary: Ports ALLOWED by Firewall

Your firewall is correctly configured and allows:
- ✓ 22 (SSH)
- ✓ 3000 (web apps)
- ✓ 5000 (week2-llm)
- ✓ 7859-7862 (week2-llm range)
- ✓ 8000 (web apps, ai-development)
- ✓ 8080 (ai-development)
- ✓ 8888 (ai-development)
- ✓ 8889 (ai-development)

**The firewall is NOT your problem!**

## The REAL Problem: Bind Addresses

Here's what's actually listening on your VM:

### ✓ Services Bound to 0.0.0.0 (ACCESSIBLE EXTERNALLY)

| Port | Service | External Access |
|------|---------|-----------------|
| 22   | SSH     | ✓ http://34.66.155.187:22 |
| 8000 | Python  | ✓ http://34.66.155.187:8000 ✓ THIS WORKS! |
| 5432 | PostgreSQL | ✓ http://34.66.155.187:5432 |

### ❌ Services Bound to 127.0.0.1 (NOT ACCESSIBLE EXTERNALLY)

| Port  | Service | Why It Fails |
|-------|---------|--------------|
| 36483 | VS Code | Bound to localhost only |
| 35431 | Node.js | Bound to localhost only |
| 5284  | Node.js | Bound to localhost only |
| 6379  | Redis   | Bound to localhost only |
| 53    | DNS     | Bound to localhost only |
| 44421 | Node.js | Bound to localhost only |
| 11434 | Ollama  | Bound to localhost only |
| 28383 | Node.js | Bound to localhost only |
| 44575 | VS Code | Bound to localhost only |
| 48307 | Node.js | Bound to localhost only |

## Why Only Port 8000 Works

Port 8000 is the ONLY service (besides SSH and PostgreSQL) that:
1. ✓ Is actually running
2. ✓ Is bound to 0.0.0.0 (all interfaces)
3. ✓ Has firewall rules allowing it

**The firewall rules are perfect!** The problem is that other services are bound to `127.0.0.1` instead of `0.0.0.0`.

## How to Make Other Ports Accessible

### For Port 3000:
```bash
# Wrong (current - doesn't work)
npm start  # Usually binds to localhost

# Right (works externally)
npm start -- --host 0.0.0.0
# or
HOST=0.0.0.0 npm start
```

### For Port 7860 (Gradio):
```python
# Wrong
demo.launch(server_name="127.0.0.1", server_port=7860)

# Right
demo.launch(server_name="0.0.0.0", server_port=7860)
# or using the utility
from vm_ip_utils import configure_gradio_server
demo.launch(**configure_gradio_server(port=7860))
```

### For Port 5000 (Flask):
```python
# Wrong
app.run(port=5000)  # Defaults to 127.0.0.1

# Right
app.run(host="0.0.0.0", port=5000)
```

### For Port 8080 (General HTTP):
```bash
# Wrong
python -m http.server 8080  # Defaults to localhost

# Right
python -m http.server 8080 --bind 0.0.0.0
```

## Quick Test: Make Port 7860 Work

Since port 7860 is allowed by firewall (in range 7859-7862), test it:

```bash
python test_port_3000.py  # Edit to use port 7860
```

Or quick test:
```bash
python3 -m http.server 7860 --bind 0.0.0.0
```

Then visit: http://34.66.155.187:7860

## Redis Example (Port 6379)

Redis is currently bound to localhost. To make it externally accessible:

```bash
# Edit Redis config
sudo nano /etc/redis/redis.conf

# Change this line:
bind 127.0.0.1 ::1

# To:
bind 0.0.0.0

# Restart Redis
sudo systemctl restart redis
```

**Warning**: Only expose Redis externally if you have authentication configured!

## PostgreSQL (Port 5432)

PostgreSQL is already bound to 0.0.0.0, but you may need to configure authentication:

```bash
# Edit pg_hba.conf
sudo nano /etc/postgresql/*/main/pg_hba.conf

# Add line to allow external connections
host    all             all             0.0.0.0/0               md5
```

**Warning**: Make sure you have strong passwords!

## Commands to Check Your Setup

### Check what's listening and how:
```bash
ss -tlnp | grep LISTEN
```

Look for:
- `0.0.0.0:PORT` = Good! Accessible externally
- `127.0.0.1:PORT` = Bad! Localhost only

### Check firewall rules:
```bash
gcloud compute firewall-rules list --format="table(name,allowed,targetTags)"
```

### Check your VM tags:
```bash
gcloud compute instances describe ai-development-workstation --zone=us-central1-a --format="get(tags.items)"
```

### Test external connectivity:
```bash
# From your local machine (not the VM)
curl http://34.66.155.187:8000  # Should work
curl http://34.66.155.187:3000  # Will fail (nothing listening)
curl http://34.66.155.187:7860  # Will fail (nothing listening)
```

## Create Additional Firewall Rules (if needed)

You already have excellent coverage, but if you need more ports:

```bash
# Add port 80 and 443 for web servers
gcloud compute firewall-rules create allow-http-https \
    --allow tcp:80,tcp:443 \
    --source-ranges 0.0.0.0/0 \
    --target-tags ai-development \
    --description "Allow HTTP and HTTPS"

# Add specific port
gcloud compute firewall-rules create allow-port-XXXX \
    --allow tcp:XXXX \
    --source-ranges 0.0.0.0/0 \
    --target-tags ai-development \
    --description "Allow port XXXX"
```

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Firewall Rules | ✓ Excellent | Allows 22, 3000, 5000, 7859-7862, 8000, 8080, 8888, 8889 |
| VM Tags | ✓ Correct | Has ai-development and jupyter-server tags |
| Port 8000 | ✓ Working | Bound to 0.0.0.0 correctly |
| Port 5432 | ✓ Working | PostgreSQL bound to 0.0.0.0 |
| Other Ports | ❌ Not accessible | Services bound to 127.0.0.1 instead of 0.0.0.0 |

**Bottom Line**: Your firewall is perfectly configured! To make other services accessible, you need to configure them to bind to `0.0.0.0` instead of `127.0.0.1`.
