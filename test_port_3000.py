#!/usr/bin/env python3
"""
Quick Test Server for Port 3000

Run this to test if port 3000 is accessible from external connections.
"""

import http.server
import socketserver
from vm_ip_utils import get_vm_ip, print_access_info

PORT = 3000

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with a test page"""

    def do_GET(self):
        """Handle GET requests with a test page"""
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            vm_ip = get_vm_ip()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Port 3000 Test - SUCCESS!</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }}
                    .container {{
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        padding: 30px;
                        backdrop-filter: blur(10px);
                    }}
                    h1 {{
                        font-size: 3em;
                        margin: 0;
                    }}
                    .success {{
                        color: #4ade80;
                        font-size: 2em;
                    }}
                    .info {{
                        background: rgba(255, 255, 255, 0.2);
                        padding: 15px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                    code {{
                        background: rgba(0, 0, 0, 0.3);
                        padding: 5px 10px;
                        border-radius: 3px;
                        font-family: 'Courier New', monospace;
                    }}
                    ul {{
                        list-style: none;
                        padding: 0;
                    }}
                    li {{
                        padding: 8px 0;
                        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                    }}
                    .emoji {{
                        font-size: 1.5em;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1><span class="emoji">🎉</span> SUCCESS!</h1>
                    <p class="success">Port 3000 is Working!</p>

                    <div class="info">
                        <h2><span class="emoji">📍</span> Connection Info</h2>
                        <ul>
                            <li><strong>VM IP:</strong> <code>{vm_ip}</code></li>
                            <li><strong>Port:</strong> <code>{PORT}</code></li>
                            <li><strong>URL:</strong> <code>http://{vm_ip}:{PORT}</code></li>
                        </ul>
                    </div>

                    <div class="info">
                        <h2><span class="emoji">✅</span> What This Means</h2>
                        <ul>
                            <li>✓ Service is running on port {PORT}</li>
                            <li>✓ Service is bound to 0.0.0.0 (externally accessible)</li>
                            <li>✓ Firewall rules allow traffic on port {PORT}</li>
                            <li>✓ Your VM network configuration is correct</li>
                        </ul>
                    </div>

                    <div class="info">
                        <h2><span class="emoji">🚀</span> Next Steps</h2>
                        <p>Now you can run your actual application on port {PORT}:</p>
                        <ul>
                            <li><strong>Gradio:</strong> <code>demo.launch(server_name='0.0.0.0', server_port={PORT})</code></li>
                            <li><strong>Flask:</strong> <code>app.run(host='0.0.0.0', port={PORT})</code></li>
                            <li><strong>FastAPI:</strong> <code>uvicorn.run('main:app', host='0.0.0.0', port={PORT})</code></li>
                        </ul>
                    </div>

                    <div class="info">
                        <h2><span class="emoji">💡</span> Remember</h2>
                        <p>Always use <code>host='0.0.0.0'</code> or <code>server_name='0.0.0.0'</code> to allow external access!</p>
                    </div>
                </div>
            </body>
            </html>
            """

            self.wfile.write(html_content.encode('utf-8'))
        else:
            # Serve files normally for other paths
            super().do_GET()

    def log_message(self, format, *args):
        """Custom log message"""
        print(f"📥 {self.address_string()} - {format % args}")


def main():
    """Main function to start the test server"""
    print("\n" + "="*70)
    print("🧪 Starting Test Server on Port 3000")
    print("="*70)

    # Print access information
    print_access_info(port=PORT, service_name="Test Server")

    print("\n🚀 Starting server...")
    print("-"*70)

    # Create server
    with socketserver.TCPServer(("0.0.0.0", PORT), CustomHandler) as httpd:
        vm_ip = get_vm_ip()

        print(f"✓ Server is running!")
        print(f"\n🌐 Access the test page:")
        print(f"   Public:  http://{vm_ip}:{PORT}")
        print(f"   Local:   http://127.0.0.1:{PORT}")

        print(f"\n💡 If you can see the success page, port {PORT} is working correctly!")
        print(f"\n⏹️  Press Ctrl+C to stop the server")
        print("="*70 + "\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
            print("="*70)


if __name__ == "__main__":
    main()
