"""
VM IP Utilities for Google Cloud VM with Non-Static IP
A standalone module for managing dynamic VM IP addresses across projects.

Usage:
    from vm_ip_utils import get_vm_ip, get_server_url, print_access_info

    # Get current public IP
    ip = get_vm_ip()

    # Get full server URL for a port
    url = get_server_url(port=7860)

    # Print all access information
    print_access_info(port=7860)
"""

import socket
import requests
import time
import logging
from typing import Optional, Dict
import json
import os

logger = logging.getLogger(__name__)


def get_vm_public_ip(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0) -> Optional[str]:
    """
    Get the public IP address of the VM using multiple methods with exponential backoff.

    Args:
        max_retries: Maximum retry attempts per method (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 30.0)

    Returns:
        str: Public IP address or None if unable to determine
    """
    methods = [
        # Method 1: GCP metadata server (works on Google Cloud VMs)
        ("GCP metadata", lambda: requests.get(
            'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip',
            headers={'Metadata-Flavor': 'Google'},
            timeout=5
        ).text),

        # Method 2: External IP services (fallback)
        ("ipify", lambda: requests.get('https://api.ipify.org', timeout=5).text),
        ("ifconfig.me", lambda: requests.get('https://ifconfig.me/ip', timeout=5).text),
        ("icanhazip", lambda: requests.get('https://icanhazip.com', timeout=5).text.strip()),
    ]

    for method_name, method in methods:
        retry_delay = base_delay

        for attempt in range(max_retries):
            try:
                ip = method()
                if ip and _is_valid_ip(ip):
                    return ip
                # Invalid IP, try next method
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"IP fetch timeout from {method_name}, retrying in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                # After max retries, fall through to next method
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"IP fetch failed from {method_name}: {e}, retrying in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
            except Exception:
                # Non-retryable error, try next method immediately
                break

    return None


def get_vm_local_ip() -> Optional[str]:
    """
    Get the local/private IP address of the VM.

    Returns:
        str: Local IP address or None if unable to determine
    """
    try:
        # Create a socket connection to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None


def _is_valid_ip(ip: str) -> bool:
    """
    Validate if string is a valid IP address.

    Args:
        ip: IP address string to validate

    Returns:
        bool: True if valid IP address
    """
    try:
        parts = ip.split('.')
        return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
    except (ValueError, AttributeError):
        return False


def get_vm_ip(prefer_public: bool = True) -> str:
    """
    Get VM IP address (public or local).

    Args:
        prefer_public: If True, returns public IP first, falls back to local

    Returns:
        str: IP address (public or local)

    Raises:
        RuntimeError: If unable to determine IP address
    """
    if prefer_public:
        ip = get_vm_public_ip()
        if ip:
            return ip
        ip = get_vm_local_ip()
        if ip:
            return ip
    else:
        ip = get_vm_local_ip()
        if ip:
            return ip
        ip = get_vm_public_ip()
        if ip:
            return ip

    raise RuntimeError("Unable to determine VM IP address")


def get_server_url(port: int, protocol: str = "http", path: str = "") -> str:
    """
    Generate full server URL with current VM IP.

    Args:
        port: Port number for the server
        protocol: Protocol to use (http or https)
        path: Optional path to append (e.g., "/api/endpoint")

    Returns:
        str: Full URL (e.g., "http://34.66.155.187:7860/api")
    """
    ip = get_vm_ip()
    path = path if path.startswith('/') or not path else f'/{path}'
    return f"{protocol}://{ip}:{port}{path}"


def get_all_ips() -> Dict[str, Optional[str]]:
    """
    Get both public and local IP addresses.

    Returns:
        dict: Dictionary with 'public' and 'local' IP addresses
    """
    return {
        'public': get_vm_public_ip(),
        'local': get_vm_local_ip()
    }


def save_ip_config(filename: str = "vm_ip_config.json", port: int = 7860) -> None:
    """
    Save current IP configuration to a JSON file.

    Args:
        filename: Name of the JSON file to save
        port: Default port to include in config
    """
    config = {
        'public_ip': get_vm_public_ip(),
        'local_ip': get_vm_local_ip(),
        'default_port': port,
        'server_url': get_server_url(port)
    }

    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"IP configuration saved to {filename}")


def load_ip_config(filename: str = "vm_ip_config.json") -> Dict:
    """
    Load IP configuration from a JSON file.

    Args:
        filename: Name of the JSON file to load

    Returns:
        dict: IP configuration
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Config file {filename} not found")

    with open(filename, 'r') as f:
        return json.load(f)


def print_access_info(
    port: int = 7860,
    service_name: str = "Server",
    additional_ports: Optional[list] = None
) -> None:
    """
    Print formatted access information for the VM service.

    Args:
        port: Primary port number
        service_name: Name of the service (for display)
        additional_ports: List of additional ports to display
    """
    print("\n" + "="*70)
    print(f"🌐 {service_name} Access Information")
    print("="*70)

    ips = get_all_ips()

    if ips['public']:
        print(f"\n📍 Public IP:  {ips['public']}")
        print(f"🔗 Public URL: {get_server_url(port)}")

    if ips['local']:
        print(f"\n🏠 Local IP:   {ips['local']}")
        print(f"🔗 Local URL:  http://{ips['local']}:{port}")

    print(f"\n🔗 Localhost:  http://127.0.0.1:{port}")

    if additional_ports:
        print(f"\n📌 Additional Ports:")
        for p in additional_ports:
            if ips['public']:
                print(f"   Port {p}: {get_server_url(p)}")

    print("\n💡 Tips:")
    print("   - Use Public URL for external access")
    print("   - Ensure firewall rules allow traffic on these ports")
    print("   - Check GCP firewall settings if unable to connect")
    print("="*70 + "\n")


def configure_gradio_server(port: int = 7860, share: bool = False) -> Dict[str, any]:
    """
    Generate Gradio server configuration using VM IP.

    Args:
        port: Port number for Gradio
        share: Whether to create a public share link

    Returns:
        dict: Configuration dict for demo.launch()
    """
    return {
        'server_name': "0.0.0.0",  # Listen on all interfaces
        'server_port': port,
        'share': share,
        'show_error': True
    }


# Example usage and testing
if __name__ == "__main__":
    print("VM IP Utilities - Testing Module")
    print("="*70)

    # Test 1: Get all IPs
    print("\n1. Testing get_all_ips():")
    ips = get_all_ips()
    print(f"   Public IP: {ips['public']}")
    print(f"   Local IP:  {ips['local']}")

    # Test 2: Get server URL
    print("\n2. Testing get_server_url():")
    try:
        url = get_server_url(port=7860)
        print(f"   Server URL: {url}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Print access info
    print("\n3. Testing print_access_info():")
    try:
        print_access_info(
            port=7860,
            service_name="Test Server",
            additional_ports=[8080, 5000]
        )
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Gradio config
    print("\n4. Testing configure_gradio_server():")
    config = configure_gradio_server(port=7860)
    print(f"   Config: {config}")

    # Test 5: Save config
    print("\n5. Testing save_ip_config():")
    try:
        save_ip_config("test_vm_config.json", port=7860)
        print("   ✓ Config saved successfully")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*70)
    print("✅ Testing complete!")
