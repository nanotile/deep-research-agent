"""
Example Usage of vm_ip_utils.py

This file demonstrates how to use the VM IP utilities module
in different scenarios and projects.
"""

from vm_ip_utils import (
    get_vm_ip,
    get_server_url,
    print_access_info,
    configure_gradio_server,
    save_ip_config,
    get_all_ips
)


# ============================================================================
# Example 1: Basic IP Retrieval
# ============================================================================
def example_basic_ip():
    """Get the current VM IP address."""
    print("\n📌 Example 1: Basic IP Retrieval")
    print("-" * 60)

    ip = get_vm_ip()
    print(f"Current VM IP: {ip}")


# ============================================================================
# Example 2: Generate Server URLs
# ============================================================================
def example_server_urls():
    """Generate URLs for different services."""
    print("\n📌 Example 2: Generate Server URLs")
    print("-" * 60)

    # Web app on port 7860
    web_url = get_server_url(port=7860)
    print(f"Web App URL:     {web_url}")

    # API on port 8000
    api_url = get_server_url(port=8000, path="/api/v1")
    print(f"API Endpoint:    {api_url}")

    # HTTPS server on port 443
    secure_url = get_server_url(port=443, protocol="https", path="/docs")
    print(f"Secure Docs URL: {secure_url}")


# ============================================================================
# Example 3: Print Complete Access Information
# ============================================================================
def example_access_info():
    """Print formatted access information."""
    print("\n📌 Example 3: Print Access Information")
    print("-" * 60)

    print_access_info(
        port=7860,
        service_name="Deep Research Agent",
        additional_ports=[8000, 5000]
    )


# ============================================================================
# Example 4: Gradio Integration
# ============================================================================
def example_gradio_integration():
    """
    How to use with Gradio apps.
    Replace the hardcoded demo.launch() with dynamic configuration.
    """
    print("\n📌 Example 4: Gradio Integration")
    print("-" * 60)

    # Get Gradio configuration
    config = configure_gradio_server(port=7860, share=False)

    print("Use this configuration in your Gradio app:")
    print(f"demo.launch(**{config})")

    # Before launching, print access info
    print("\nBefore launching your Gradio app, run:")
    print("print_access_info(port=7860, service_name='My Gradio App')")


# ============================================================================
# Example 5: Flask Integration
# ============================================================================
def example_flask_integration():
    """How to use with Flask apps."""
    print("\n📌 Example 5: Flask Integration")
    print("-" * 60)

    port = 5000
    ip = get_vm_ip()

    print("For Flask apps, use this configuration:")
    print(f"""
from vm_ip_utils import get_vm_ip, print_access_info

app = Flask(__name__)

if __name__ == '__main__':
    # Print access information before starting
    print_access_info(port={port}, service_name='Flask API')

    # Run on all interfaces so external access works
    app.run(host='0.0.0.0', port={port}, debug=True)
    """)

    print(f"\nYour Flask app will be accessible at: {get_server_url(port)}")


# ============================================================================
# Example 6: FastAPI Integration
# ============================================================================
def example_fastapi_integration():
    """How to use with FastAPI apps."""
    print("\n📌 Example 6: FastAPI Integration")
    print("-" * 60)

    port = 8000

    print("For FastAPI apps, use this configuration:")
    print(f"""
from vm_ip_utils import get_vm_ip, print_access_info
import uvicorn

if __name__ == '__main__':
    # Print access information before starting
    print_access_info(port={port}, service_name='FastAPI Server')

    # Run on all interfaces
    uvicorn.run('main:app', host='0.0.0.0', port={port}, reload=True)
    """)

    print(f"\nYour FastAPI app will be accessible at: {get_server_url(port)}")
    print(f"API docs: {get_server_url(port, path='/docs')}")


# ============================================================================
# Example 7: Save and Load IP Configuration
# ============================================================================
def example_save_load_config():
    """Save and load IP configuration to/from file."""
    print("\n📌 Example 7: Save/Load Configuration")
    print("-" * 60)

    # Save current configuration
    print("Saving IP configuration...")
    save_ip_config("my_vm_config.json", port=7860)

    # You can later load this in another script
    print("\nIn another script, load with:")
    print("""
from vm_ip_utils import load_ip_config

config = load_ip_config('my_vm_config.json')
print(f"Server URL: {config['server_url']}")
    """)


# ============================================================================
# Example 8: Environment Variables Integration
# ============================================================================
def example_env_integration():
    """Use with environment variables for configuration."""
    print("\n📌 Example 8: Environment Variables Integration")
    print("-" * 60)

    import os

    # Get current IP
    current_ip = get_vm_ip()

    print("Set environment variables in your .env file:")
    print(f"VM_IP={current_ip}")
    print(f"SERVER_URL={get_server_url(7860)}")
    print(f"API_URL={get_server_url(8000, path='/api')}")

    print("\nOr export them in your shell:")
    print(f"export VM_IP={current_ip}")


# ============================================================================
# Example 9: Frontend Configuration Generator
# ============================================================================
def example_frontend_config():
    """Generate configuration for frontend apps."""
    print("\n📌 Example 9: Frontend Configuration")
    print("-" * 60)

    import json

    # Generate frontend config
    frontend_config = {
        "API_BASE_URL": get_server_url(8000, path="/api"),
        "WS_URL": get_server_url(8000, protocol="ws", path="/ws"),
        "STATIC_URL": get_server_url(8080, path="/static"),
        "VM_IP": get_vm_ip()
    }

    print("JavaScript/TypeScript config:")
    print(f"""
// config.js or config.ts
export const config = {json.dumps(frontend_config, indent=2)}
    """)

    print("\nReact .env.local file:")
    print(f"REACT_APP_API_URL={frontend_config['API_BASE_URL']}")
    print(f"REACT_APP_WS_URL={frontend_config['WS_URL']}")


# ============================================================================
# Example 10: Multiple Services on Different Ports
# ============================================================================
def example_multiple_services():
    """Manage multiple services on different ports."""
    print("\n📌 Example 10: Multiple Services")
    print("-" * 60)

    services = {
        "Frontend": 3000,
        "Backend API": 8000,
        "Gradio App": 7860,
        "Database Admin": 5432,
        "Monitoring": 9090
    }

    print("All service URLs:")
    for name, port in services.items():
        url = get_server_url(port)
        print(f"  {name:20} → {url}")


# ============================================================================
# Main - Run All Examples
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🚀 VM IP Utilities - Usage Examples")
    print("="*70)

    examples = [
        example_basic_ip,
        example_server_urls,
        example_access_info,
        example_gradio_integration,
        example_flask_integration,
        example_fastapi_integration,
        example_save_load_config,
        example_env_integration,
        example_frontend_config,
        example_multiple_services
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")

    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
