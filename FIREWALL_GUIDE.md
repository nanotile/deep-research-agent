# GCP Firewall Rules & VM IP Changes - Complete Guide

## Your Question: Do Firewall Rules Reset When IP Changes?

**Answer: NO** - Firewall rules are **NOT reset** when your VM's IP address changes.

### How GCP Firewall Rules Work

Firewall rules in Google Cloud Platform are attached to:
1. **The VPC Network** (not the IP address)
2. **Target tags** (applied to instances)
3. **Service accounts**
4. **Specific VM instances** (by name)

When your VM gets a new IP address (because it's non-static), the firewall rules **remain attached** to the VM instance and continue working.

## Your Current Issue: Port 3000

### Problem Diagnosis

```
Current Status on Your VM:
✓ Port 80   - LISTENING (accessible externally)
✓ Port 8000 - LISTENING (accessible externally)
❌ Port 3000 - NOT LISTENING (no service running)
```

**The issue:** Nothing is running on port 3000!

### ERR_CONNECTION_REFUSED means:
1. The port is reachable (firewall is OK)
2. BUT no application is listening on that port
3. The connection is actively refused by the VM

## Solutions

### If you want to test port 3000:

#### Option 1: Start a Simple Test Server

**Python HTTP Server:**
```bash
# Start a simple web server on port 3000
python3 -m http.server 3000 --bind 0.0.0.0
```

**Node.js Express Server:**
```bash
# Create a quick test server
npx http-server -p 3000 -a 0.0.0.0
```

#### Option 2: Use Your Deep Research Agent on Port 3000

Modify your Gradio app to run on port 3000:

```python
from vm_ip_utils import configure_gradio_server, print_access_info

# Configure for port 3000
print_access_info(port=3000, service_name="Deep Research Agent")
demo.launch(**configure_gradio_server(port=3000))
```

Run it:
```bash
python app.py
```

### Firewall Rule Setup (One-Time, Persists Across IP Changes)

#### Check if firewall rule exists for port 3000:
```bash
gcloud compute firewall-rules list | grep 3000
```

#### Create firewall rule if it doesn't exist:
```bash
gcloud compute firewall-rules create allow-port-3000 \
    --allow tcp:3000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow port 3000 for development"
```

#### Alternative: Allow multiple ports at once
```bash
gcloud compute firewall-rules create allow-dev-ports \
    --allow tcp:3000,tcp:5000,tcp:7860,tcp:8000,tcp:8080 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow common development ports"
```

#### Using GCP Console (Web Interface):
1. Go to: https://console.cloud.google.com/networking/firewalls/list
2. Click **CREATE FIREWALL RULE**
3. Fill in:
   - **Name**: `allow-dev-ports`
   - **Targets**: All instances in the network
   - **Source IP ranges**: `0.0.0.0/0`
   - **Protocols and ports**: `tcp:3000,5000,7860,8000,8080`
4. Click **CREATE**

**Important:** This rule will persist even when your IP changes!

## Using the Diagnostic Tools

### Quick Port Check
```bash
# Check specific port
python vm_firewall_utils.py 3000

# Check all common ports
python vm_firewall_utils.py
```

### See What's Currently Running
```bash
# See all listening ports
ss -tlnp | grep LISTEN

# Or using netstat
netstat -tlnp | grep LISTEN
```

### Check Firewall Rules
```bash
# List all firewall rules
gcloud compute firewall-rules list

# List rules in detail
gcloud compute firewall-rules list --format=json
```

## Common Port Binding Mistakes

### ❌ WRONG - Binds to localhost only (not accessible externally)
```python
# Flask
app.run(host='127.0.0.1', port=3000)  # Won't work externally!

# Gradio
demo.launch(server_name='127.0.0.1', server_port=3000)  # Won't work!

# FastAPI
uvicorn.run('main:app', host='127.0.0.1', port=3000)  # Won't work!
```

### ✓ CORRECT - Binds to all interfaces (accessible externally)
```python
# Flask
app.run(host='0.0.0.0', port=3000)  # ✓ Works!

# Gradio
demo.launch(server_name='0.0.0.0', server_port=3000)  # ✓ Works!

# FastAPI
uvicorn.run('main:app', host='0.0.0.0', port=3000)  # ✓ Works!
```

## Troubleshooting Checklist

When you can't connect to a port, check in order:

1. **Is the service actually running?**
   ```bash
   python vm_firewall_utils.py 3000
   ```

2. **Is it bound to 0.0.0.0 (not 127.0.0.1)?**
   ```bash
   ss -tlnp | grep :3000
   # Look for 0.0.0.0:3000 or *:3000 (good)
   # NOT 127.0.0.1:3000 (bad - localhost only)
   ```

3. **Does the firewall rule exist?**
   ```bash
   gcloud compute firewall-rules list | grep 3000
   ```

4. **Can you connect locally?**
   ```bash
   curl http://127.0.0.1:3000
   ```

5. **Can you connect via public IP?**
   ```bash
   curl http://34.66.155.187:3000
   ```

## Your Working Services

You currently have these services accessible:

- **Port 80**: http://34.66.155.187:80
- **Port 8000**: http://34.66.155.187:8000

Both are correctly:
- ✓ Listening on port
- ✓ Bound to 0.0.0.0 (all interfaces)
- ✓ Firewall rules allow access

## Quick Reference Commands

```bash
# Diagnose specific port
python vm_firewall_utils.py 3000

# See all active ports
python vm_firewall_utils.py

# Start test server on port 3000
python3 -m http.server 3000 --bind 0.0.0.0

# Check firewall rules
gcloud compute firewall-rules list

# Create firewall rule for port 3000
gcloud compute firewall-rules create allow-port-3000 \
    --allow tcp:3000 \
    --source-ranges 0.0.0.0/0

# View your VM IP
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"
```

## Important Notes

1. **Firewall rules are persistent** - They survive:
   - VM restarts
   - IP address changes
   - VM stops/starts

2. **Services are NOT persistent** - When VM restarts:
   - You need to restart your applications
   - They don't auto-start unless configured

3. **Bind address matters**:
   - `0.0.0.0` = accessible from anywhere
   - `127.0.0.1` = localhost only, not accessible externally

4. **Test locally first**:
   - If `curl http://127.0.0.1:3000` fails, fix the service
   - If it works locally but not externally, check firewall
