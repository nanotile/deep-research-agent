# VM Utilities Documentation

Complete toolkit for managing Google Cloud VM networking, firewall diagnostics, and dynamic IP handling.

## 🎯 Problem Solved

You asked about port 3000 connection issues and whether firewalls reset when VM IP changes.

**Findings:**
- ✓ Firewall allows ALL ports (0-65535)
- ✓ Firewall rules do NOT reset when IP changes
- ✓ Port 8000 works perfectly
- ❌ Port 3000 fails because nothing is listening
- ❌ Other services are bound to localhost (127.0.0.1) instead of 0.0.0.0

## 📦 Utilities Created

### Core Utilities (Copy to Any Project)

#### 1. `vm_ip_utils.py`
**Dynamic IP management for VMs with non-static IPs**

Functions:
- `get_vm_ip()` - Get current VM public IP
- `get_server_url(port)` - Generate full URL with current IP
- `print_access_info(port, service_name)` - Print formatted access info
- `configure_gradio_server(port)` - Get Gradio config dict
- `get_all_ips()` - Get both public and local IPs
- `save_ip_config()` / `load_ip_config()` - Save/load IP config

```python
from vm_ip_utils import get_vm_ip, get_server_url, print_access_info

ip = get_vm_ip()  # Returns: "34.66.155.187"
url = get_server_url(7860)  # Returns: "http://34.66.155.187:7860"
print_access_info(port=7860, service_name="My App")
```

#### 2. `vm_firewall_utils.py`
**Port and firewall diagnostics**

Functions:
- `check_port_listening(port)` - Check if port is listening
- `get_all_listening_ports()` - Get all active ports
- `diagnose_port(port, vm_ip)` - Complete port diagnosis
- `test_port_connection(port, ip)` - Test connectivity
- `list_common_ports()` - Show status of common ports

```bash
# Check specific port
python vm_firewall_utils.py 3000

# Show all common ports
python vm_firewall_utils.py
```

#### 3. `check_firewall.py`
**Complete GCP firewall analysis using gcloud**

Uses gcloud CLI to:
- Get actual firewall rules from GCP
- Check VM network tags
- Compare firewall rules vs listening services
- Provide actionable recommendations

```bash
python check_firewall.py
```

### Test & Example Files

#### 4. `test_port_3000.py`
**Test server with visual success page**

Starts a test HTTP server on port 3000 with a styled success page showing connection info.

```bash
python test_port_3000.py
# Visit: http://34.66.155.187:3000
```

#### 5. `vm_ip_example_usage.py`
**10 comprehensive usage examples**

Shows how to integrate `vm_ip_utils` with:
- Gradio
- Flask
- FastAPI
- Node.js
- Frontend configs
- Multiple services

```bash
python vm_ip_example_usage.py
```

#### 6. `example_app_with_vm_ip.py`
**Full Gradio app example with VM IP utilities**

Complete working example of the Deep Research Agent with VM IP utilities integrated.

```bash
python example_app_with_vm_ip.py
```

### Documentation

#### 7. `VM_IP_README.md`
Complete documentation for `vm_ip_utils.py` with examples and integration guides.

#### 8. `FIREWALL_GUIDE.md`
Complete guide explaining:
- How GCP firewall rules work
- Why they don't reset with IP changes
- Common bind address mistakes
- Troubleshooting checklist

#### 9. `FIREWALL_ANALYSIS.md`
Detailed analysis of your specific VM:
- Current firewall rules
- What's listening
- Why only port 8000 works
- How to fix other ports

#### 10. `PORT_3000_SOLUTION.md`
Specific solution for port 3000 connection issues.

#### 11. `FINAL_SUMMARY.md`
Complete summary answering all your questions with key takeaways.

#### 12. `QUICK_REFERENCE.md`
One-page quick reference card with commands and common patterns.

## 🚀 Quick Start

### Check Your Current Setup
```bash
python check_firewall.py
```

### Test a Specific Port
```bash
python vm_firewall_utils.py 3000
```

### Get Your Current IP
```bash
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"
```

### Start Test Server
```bash
python test_port_3000.py
```

## 📋 Common Use Cases

### 1. Run Gradio App on Any Port
```python
from vm_ip_utils import configure_gradio_server, print_access_info
import gradio as gr

demo = gr.Interface(...)

if __name__ == "__main__":
    PORT = 7860
    print_access_info(port=PORT, service_name="My Gradio App")
    demo.launch(**configure_gradio_server(port=PORT))
```

### 2. Check Why Port Isn't Working
```bash
# Full diagnosis
python vm_firewall_utils.py 3000

# Shows:
# - Is service running?
# - What's it bound to (0.0.0.0 or 127.0.0.1)?
# - Firewall rules
# - How to fix it
```

### 3. See All Active Services
```bash
python vm_firewall_utils.py
# Shows all common ports and their status
```

### 4. Complete Firewall Analysis
```bash
python check_firewall.py
# Uses gcloud to show:
# - VM info and tags
# - All applicable firewall rules
# - What's accessible vs not
# - Recommendations
```

## 🔑 Key Insights

### Your VM Status (Current)
```
VM: ai-development-workstation
IP: 34.66.155.187
Zone: us-central1-a
Tags: ai-development, jupyter-server

Firewall: ✓ Allows ALL ports (0-65535)

Working Services:
✓ Port 22   (SSH)
✓ Port 8000 (Python) ← Only this returns data because bound to 0.0.0.0
✓ Port 5432 (PostgreSQL)

Not Working:
❌ Port 3000 (nothing listening)
❌ Other ports (bound to 127.0.0.1)
```

### Why Port 8000 Works
1. ✓ Service is running
2. ✓ Bound to 0.0.0.0 (all interfaces)
3. ✓ Firewall allows it

### Why Other Ports Don't Work
1. Port 3000: Nothing listening
2. Other ports: Bound to 127.0.0.1 (localhost only)

### Do Firewall Rules Reset?
**NO!** Firewall rules are attached to:
- VM instance name (persists)
- Network tags (persists)
- VPC network (persists)

NOT attached to IP address, so they persist when IP changes.

## 📂 Files to Copy to Other Projects

Essential files for any project:
1. `vm_ip_utils.py` - IP management (standalone)
2. `vm_firewall_utils.py` - Port diagnostics (standalone)
3. `check_firewall.py` - Complete analysis (requires vm_ip_utils.py)

Just copy these files to your project directory and import:
```python
from vm_ip_utils import get_vm_ip, get_server_url, print_access_info
from vm_firewall_utils import diagnose_port, check_port_listening
```

## 🛠️ Requirements

All utilities require:
```bash
pip install requests  # For vm_ip_utils.py
```

For `check_firewall.py` only:
```bash
# Requires gcloud CLI to be installed and configured
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## 💡 Remember

**Always bind to 0.0.0.0 for external access:**

❌ Wrong:
```python
app.run(host='127.0.0.1', port=3000)
demo.launch(server_name='127.0.0.1', server_port=7860)
```

✓ Right:
```python
app.run(host='0.0.0.0', port=3000)
demo.launch(server_name='0.0.0.0', server_port=7860)
```

Or use the utility:
```python
from vm_ip_utils import configure_gradio_server
demo.launch(**configure_gradio_server(port=7860))
```

## 📞 Quick Help

Problem: "Can't connect to my service"
```bash
1. python vm_firewall_utils.py YOUR_PORT
2. Look at the output - it will tell you exactly what's wrong
3. Follow the provided solution steps
```

Problem: "What's my current IP?"
```bash
python -c "from vm_ip_utils import get_vm_ip; print(get_vm_ip())"
```

Problem: "Do I have firewall issues?"
```bash
python check_firewall.py
# Will show complete firewall analysis
```

## 🎓 Learn More

See the documentation files:
- `FIREWALL_GUIDE.md` - Complete firewall guide
- `VM_IP_README.md` - IP utilities documentation
- `FINAL_SUMMARY.md` - Summary of your specific issues
- `QUICK_REFERENCE.md` - Quick command reference
