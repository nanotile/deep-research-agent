# Port 3000 Connection Issue - Solution

## Problem Summary

You got `ERR_CONNECTION_REFUSED` when trying to access `http://34.66.155.187:3000`

## Root Cause

**Nothing is running on port 3000!**

Your diagnostic shows:
- ✓ Port 80 is active (service running)
- ✓ Port 8000 is active (service running)
- ❌ Port 3000 is NOT active (no service running)

## Quick Answer to Your Question

**"Are firewalls reset for every new VM instance / IP change?"**

**NO** - Firewall rules are **NOT** reset when your IP changes. They persist because they're attached to:
- The VM instance (by name/tag)
- The VPC network
- NOT to the IP address itself

See `FIREWALL_GUIDE.md` for full details.

## How to Fix It

### Step 1: Check if Firewall Rule Exists

```bash
gcloud compute firewall-rules list | grep 3000
```

If nothing shows up, create the rule:

```bash
gcloud compute firewall-rules create allow-port-3000 \
    --allow tcp:3000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow port 3000"
```

### Step 2: Test Port 3000

Run the test server:

```bash
python test_port_3000.py
```

Then visit: http://34.66.155.187:3000

You should see a success page!

### Step 3: Run Your Actual Application

Once the test works, run your real app on port 3000.

**For Gradio:**
```python
from vm_ip_utils import configure_gradio_server, print_access_info

print_access_info(port=3000, service_name="My App")
demo.launch(**configure_gradio_server(port=3000))
```

**For Flask:**
```python
from vm_ip_utils import print_access_info

print_access_info(port=3000, service_name="Flask App")
app.run(host='0.0.0.0', port=3000)
```

## Diagnostic Tools

### Quick port check:
```bash
python vm_firewall_utils.py 3000
```

### See all active ports:
```bash
python vm_firewall_utils.py
```

### Get your current IP:
```bash
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"
```

## Current Status

Your VM currently has:
- **Public IP**: 34.66.155.187
- **Active Services**:
  - Port 80: Running ✓
  - Port 8000: Running ✓
  - Port 3000: Not running ❌

## Files Created for You

1. **vm_ip_utils.py** - Dynamic IP management for any project
2. **vm_firewall_utils.py** - Firewall and port diagnostics
3. **test_port_3000.py** - Quick test server for port 3000
4. **FIREWALL_GUIDE.md** - Complete firewall reference guide
5. **vm_ip_example_usage.py** - Usage examples for all frameworks

## Quick Start

```bash
# 1. Test if port 3000 can work
python test_port_3000.py

# 2. Visit in browser
# http://34.66.155.187:3000

# 3. If it works, stop test server (Ctrl+C)
# and run your actual app on port 3000
```

## Remember

Always bind to `0.0.0.0` not `127.0.0.1`:

❌ Wrong: `app.run(host='127.0.0.1', port=3000)`
✓ Right: `app.run(host='0.0.0.0', port=3000)`
