"""
VM Firewall and Port Diagnostic Utilities

Helps diagnose connection issues and check firewall/port status on Google Cloud VMs.
"""

import subprocess
import json
from typing import List, Dict, Optional
import socket


def check_port_listening(port: int) -> Dict[str, any]:
    """
    Check if a port is listening and what process is using it.

    Args:
        port: Port number to check

    Returns:
        dict: Status information about the port
    """
    result = {
        'port': port,
        'listening': False,
        'process': None,
        'bind_address': None
    }

    try:
        # Check using netstat or ss
        cmd = f"ss -tlnp | grep ':{port}' || netstat -tlnp | grep ':{port}' 2>/dev/null"
        output = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        if output.stdout:
            result['listening'] = True
            lines = output.stdout.strip().split('\n')
            for line in lines:
                # Parse the output to get bind address
                if '127.0.0.1' in line:
                    result['bind_address'] = '127.0.0.1 (localhost only - NOT accessible externally!)'
                elif '0.0.0.0' in line or '*:' in line:
                    result['bind_address'] = '0.0.0.0 (all interfaces - accessible externally)'
                elif ':::' in line:
                    result['bind_address'] = ':::(IPv6 all interfaces)'

                # Try to extract process info
                if 'users:' in line:
                    result['process'] = line.split('users:')[1].strip()

    except Exception as e:
        result['error'] = str(e)

    return result


def get_all_listening_ports() -> List[Dict]:
    """
    Get all listening ports on the system.

    Returns:
        list: List of dictionaries with port information
    """
    try:
        cmd = "ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null"
        output = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        ports = []
        for line in output.stdout.split('\n'):
            if ':' in line and ('LISTEN' in line or 'LISTEN' in line.upper()):
                # Extract port number
                parts = line.split()
                for part in parts:
                    if ':' in part:
                        try:
                            port = int(part.split(':')[-1])
                            if 1 <= port <= 65535:
                                ports.append({'port': port, 'line': line})
                                break
                        except ValueError:
                            continue

        return ports
    except Exception as e:
        return [{'error': str(e)}]


def check_gcp_firewall_rules(port: Optional[int] = None) -> Dict:
    """
    Check GCP firewall rules using gcloud command.

    Args:
        port: Specific port to check (optional)

    Returns:
        dict: Firewall rule information
    """
    result = {
        'gcloud_available': False,
        'rules': []
    }

    try:
        # Check if gcloud is available
        test_cmd = subprocess.run(
            ['which', 'gcloud'],
            capture_output=True
        )

        if test_cmd.returncode != 0:
            result['message'] = 'gcloud CLI not found. Install with: curl https://sdk.cloud.google.com | bash'
            return result

        result['gcloud_available'] = True

        # Get firewall rules
        cmd = ['gcloud', 'compute', 'firewall-rules', 'list', '--format=json']
        output = subprocess.run(cmd, capture_output=True, text=True)

        if output.returncode == 0:
            rules = json.loads(output.stdout)

            for rule in rules:
                if port:
                    # Check if this rule allows the specific port
                    allowed = rule.get('allowed', [])
                    for allow in allowed:
                        ports = allow.get('ports', [])
                        if str(port) in ports or f"{port}-{port}" in str(ports):
                            result['rules'].append(rule)
                else:
                    result['rules'].append(rule)
        else:
            result['error'] = output.stderr

    except Exception as e:
        result['error'] = str(e)

    return result


def test_port_connection(port: int, ip: str = "127.0.0.1", timeout: int = 2) -> bool:
    """
    Test if a port is accessible by attempting to connect.

    Args:
        port: Port to test
        ip: IP address to test (default: localhost)
        timeout: Connection timeout in seconds

    Returns:
        bool: True if connection successful
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def diagnose_port(port: int, vm_ip: str) -> None:
    """
    Comprehensive port diagnosis with recommendations.

    Args:
        port: Port number to diagnose
        vm_ip: Public IP of the VM
    """
    print("="*70)
    print(f"🔍 PORT {port} DIAGNOSTIC REPORT")
    print("="*70)

    # Step 1: Check if port is listening
    print(f"\n1️⃣ Checking if port {port} is listening...")
    port_status = check_port_listening(port)

    if not port_status['listening']:
        print(f"   ❌ Nothing is listening on port {port}")
        print(f"\n   💡 SOLUTION:")
        print(f"   → Start your application/service on port {port}")
        print(f"   → Make sure it binds to 0.0.0.0 (not 127.0.0.1)")
        print(f"\n   Example for common frameworks:")
        print(f"   - Flask: app.run(host='0.0.0.0', port={port})")
        print(f"   - Gradio: demo.launch(server_name='0.0.0.0', server_port={port})")
        print(f"   - FastAPI: uvicorn.run('main:app', host='0.0.0.0', port={port})")
        return
    else:
        print(f"   ✓ Port {port} is listening")
        if port_status['bind_address']:
            print(f"   📍 Bind address: {port_status['bind_address']}")

            if '127.0.0.1' in port_status['bind_address']:
                print(f"\n   ⚠️  WARNING: Service is bound to localhost only!")
                print(f"   💡 SOLUTION:")
                print(f"   → Change your service to bind to 0.0.0.0 instead of 127.0.0.1")
                print(f"   → This allows external access")
                return

        if port_status['process']:
            print(f"   📦 Process: {port_status['process']}")

    # Step 2: Test local connection
    print(f"\n2️⃣ Testing local connection...")
    local_works = test_port_connection(port, "127.0.0.1")
    if local_works:
        print(f"   ✓ Port {port} is accessible locally")
    else:
        print(f"   ❌ Cannot connect to port {port} locally")
        print(f"   💡 Service may be starting up or having issues")

    # Step 3: Check GCP firewall
    print(f"\n3️⃣ Checking GCP firewall rules...")
    fw_result = check_gcp_firewall_rules(port)

    if not fw_result['gcloud_available']:
        print(f"   ⚠️  Cannot check firewall (gcloud not available)")
        print(f"   💡 Check manually in GCP Console:")
        print(f"   → https://console.cloud.google.com/networking/firewalls/list")
    else:
        matching_rules = [r for r in fw_result['rules'] if str(port) in str(r.get('allowed', []))]

        if matching_rules:
            print(f"   ✓ Found {len(matching_rules)} firewall rule(s) for port {port}")
            for rule in matching_rules:
                print(f"      - {rule['name']}: {rule.get('allowed', [])}")
        else:
            print(f"   ❌ No firewall rule found allowing port {port}")
            print(f"\n   💡 SOLUTION - Create firewall rule:")
            print(f"\n   Option A - Using gcloud CLI:")
            print(f"   gcloud compute firewall-rules create allow-port-{port} \\")
            print(f"       --allow tcp:{port} \\")
            print(f"       --source-ranges 0.0.0.0/0 \\")
            print(f"       --description 'Allow port {port}'")
            print(f"\n   Option B - Using GCP Console:")
            print(f"   1. Go to: https://console.cloud.google.com/networking/firewalls/list")
            print(f"   2. Click 'CREATE FIREWALL RULE'")
            print(f"   3. Name: allow-port-{port}")
            print(f"   4. Target: All instances in the network")
            print(f"   5. Source IP ranges: 0.0.0.0/0")
            print(f"   6. Protocols/ports: tcp:{port}")
            print(f"   7. Click 'CREATE'")

    # Step 4: Summary
    print(f"\n4️⃣ Access URLs:")
    print(f"   🔗 Public:  http://{vm_ip}:{port}")
    print(f"   🔗 Local:   http://127.0.0.1:{port}")

    # Step 5: Common issues checklist
    print(f"\n5️⃣ Troubleshooting Checklist:")
    checklist = [
        ("Service running on port", port_status['listening']),
        ("Bound to 0.0.0.0 (not 127.0.0.1)", '0.0.0.0' in str(port_status.get('bind_address', ''))),
        ("Local connection works", local_works),
        ("Firewall rule exists", len(matching_rules) > 0 if fw_result['gcloud_available'] else None),
    ]

    for item, status in checklist:
        if status is True:
            print(f"   ✓ {item}")
        elif status is False:
            print(f"   ❌ {item}")
        else:
            print(f"   ⚠️  {item} (unable to verify)")

    print("\n" + "="*70)


def create_firewall_rule(port: int, rule_name: Optional[str] = None) -> None:
    """
    Helper to create a GCP firewall rule for a specific port.

    Args:
        port: Port number to allow
        rule_name: Custom rule name (optional)
    """
    if rule_name is None:
        rule_name = f"allow-port-{port}"

    cmd = [
        'gcloud', 'compute', 'firewall-rules', 'create', rule_name,
        '--allow', f'tcp:{port}',
        '--source-ranges', '0.0.0.0/0',
        '--description', f'Allow TCP traffic on port {port}'
    ]

    print(f"Creating firewall rule: {rule_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Firewall rule created successfully!")
            print(result.stdout)
        else:
            print("❌ Failed to create firewall rule:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error: {e}")


def list_common_ports() -> None:
    """
    Display common ports and their status.
    """
    common_ports = {
        80: "HTTP",
        443: "HTTPS",
        3000: "React/Node.js dev server",
        5000: "Flask default",
        7860: "Gradio default",
        8000: "FastAPI/Django default",
        8080: "Alternative HTTP",
        8888: "Jupyter Notebook",
        9090: "Prometheus",
    }

    print("="*70)
    print("🔌 COMMON PORTS STATUS")
    print("="*70)

    for port, description in common_ports.items():
        status = check_port_listening(port)
        if status['listening']:
            print(f"✓ {port:5d} - {description:30s} [LISTENING]")
            if status['bind_address']:
                print(f"          Bind: {status['bind_address']}")
        else:
            print(f"  {port:5d} - {description:30s} [not active]")

    print("="*70)


# Main execution
if __name__ == "__main__":
    import sys
    from vm_ip_utils import get_vm_ip

    print("🔥 VM Firewall & Port Diagnostics")
    print("="*70)

    vm_ip = get_vm_ip()
    print(f"VM Public IP: {vm_ip}\n")

    # If port specified as argument, diagnose it
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            diagnose_port(port, vm_ip)
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
    else:
        # Show common ports status
        list_common_ports()

        print("\n💡 Usage:")
        print(f"   python {sys.argv[0]} <port>     - Diagnose specific port")
        print(f"   Example: python {sys.argv[0]} 3000")
