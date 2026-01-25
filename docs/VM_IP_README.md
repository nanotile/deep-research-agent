# VM IP Utilities

A standalone Python module for managing dynamic VM IP addresses on Google Cloud Platform (or any VM with non-static IPs). Use this across any project for frontend/backend testing.

## Current VM Status

- **Public IP**: 34.66.155.187
- **Local IP**: 10.128.0.2

## Quick Start

### Basic Usage

```python
from vm_ip_utils import get_vm_ip, get_server_url, print_access_info

# Get current public IP
ip = get_vm_ip()
print(f"VM IP: {ip}")

# Get full server URL
url = get_server_url(port=7860)
print(f"Access at: {url}")

# Print complete access information
print_access_info(port=7860, service_name="My App")
```

### Run the Test

```bash
# Test all functions
python vm_ip_utils.py

# See usage examples
python vm_ip_example_usage.py
```

## Key Functions

### `get_vm_ip(prefer_public=True)`
Returns the VM's public IP address (or local IP as fallback).

```python
ip = get_vm_ip()  # Returns: "34.66.155.187"
```

### `get_server_url(port, protocol="http", path="")`
Generates a complete URL with the current VM IP.

```python
url = get_server_url(port=7860)
# Returns: "http://34.66.155.187:7860"

api_url = get_server_url(port=8000, path="/api/v1")
# Returns: "http://34.66.155.187:8000/api/v1"
```

### `print_access_info(port, service_name, additional_ports=None)`
Prints formatted access information for your service.

```python
print_access_info(
    port=7860,
    service_name="Deep Research Agent",
    additional_ports=[8000, 5000]
)
```

Output:
```
======================================================================
🌐 Deep Research Agent Access Information
======================================================================

📍 Public IP:  34.66.155.187
🔗 Public URL: http://34.66.155.187:7860

🏠 Local IP:   10.128.0.2
🔗 Local URL:  http://10.128.0.2:7860

🔗 Localhost:  http://127.0.0.1:7860

📌 Additional Ports:
   Port 8000: http://34.66.155.187:8000
   Port 5000: http://34.66.155.187:5000

💡 Tips:
   - Use Public URL for external access
   - Ensure firewall rules allow traffic on these ports
   - Check GCP firewall settings if unable to connect
======================================================================
```

### `get_all_ips()`
Returns both public and local IP addresses.

```python
ips = get_all_ips()
# Returns: {'public': '34.66.155.187', 'local': '10.128.0.2'}
```

### `configure_gradio_server(port, share=False)`
Generates Gradio configuration dictionary.

```python
config = configure_gradio_server(port=7860)
demo.launch(**config)
```

### `save_ip_config(filename, port)`
Saves current IP configuration to JSON file.

```python
save_ip_config("vm_config.json", port=7860)
```

### `load_ip_config(filename)`
Loads IP configuration from JSON file.

```python
config = load_ip_config("vm_config.json")
```

## Integration Examples

### Gradio App

```python
import gradio as gr
from vm_ip_utils import configure_gradio_server, print_access_info

demo = gr.Interface(...)

if __name__ == "__main__":
    print_access_info(port=7860, service_name="My Gradio App")
    demo.launch(**configure_gradio_server(port=7860))
```

### Flask App

```python
from flask import Flask
from vm_ip_utils import print_access_info

app = Flask(__name__)

if __name__ == '__main__':
    print_access_info(port=5000, service_name='Flask API')
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI App

```python
from fastapi import FastAPI
from vm_ip_utils import print_access_info, get_server_url
import uvicorn

app = FastAPI()

if __name__ == '__main__':
    print_access_info(port=8000, service_name='FastAPI Server')
    print(f"API Docs: {get_server_url(8000, path='/docs')}")
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
```

### Frontend Configuration

```python
from vm_ip_utils import get_server_url
import json

# Generate config for React/Vue/etc
config = {
    "API_BASE_URL": get_server_url(8000, path="/api"),
    "WS_URL": get_server_url(8000, protocol="ws"),
}

# Save to config file
with open("frontend/config.json", "w") as f:
    json.dump(config, f)
```

## How It Works

The module uses multiple methods to determine your VM's IP:

1. **GCP Metadata Server** (primary): Queries Google Cloud's internal metadata service
2. **External IP Services** (fallback): Uses ipify.org, ifconfig.me, or icanhazip.com

This ensures it works reliably even if one method fails.

## Firewall Configuration

To access services from external machines, ensure your GCP firewall allows traffic:

```bash
# Allow specific port (e.g., 7860 for Gradio)
gcloud compute firewall-rules create allow-gradio \
    --allow tcp:7860 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow Gradio access"

# Allow multiple ports
gcloud compute firewall-rules create allow-web-services \
    --allow tcp:7860,tcp:8000,tcp:5000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow web services"
```

Or use the GCP Console: **VPC Network → Firewall → Create Firewall Rule**

## Copying to Other Projects

Simply copy `vm_ip_utils.py` to any project:

```bash
# Copy to another project
cp vm_ip_utils.py /path/to/other/project/

# Or create a symlink
ln -s /home/kent_benson/deep-research-agent/vm_ip_utils.py /path/to/other/project/
```

## Requirements

```python
# Already included in most Python installations
import socket
import requests  # pip install requests
import json
import os
```

If `requests` is not installed:
```bash
pip install requests
```

## Troubleshooting

### Can't connect to public IP

1. Check firewall rules (see above)
2. Ensure server binds to `0.0.0.0` not `127.0.0.1`
3. Verify the port isn't blocked by OS firewall
4. Check if the service is actually running: `netstat -tuln | grep <port>`

### Wrong IP returned

- The module tries multiple methods to get the IP
- If on GCP, it should correctly use the metadata server
- If not, it falls back to external IP services

### Module can't find IP

```python
from vm_ip_utils import get_vm_ip

try:
    ip = get_vm_ip()
except RuntimeError as e:
    print(f"Unable to determine IP: {e}")
    # Handle error or use manual IP
```

## License

Free to use in any project.
