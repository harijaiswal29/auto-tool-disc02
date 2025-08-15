#!/usr/bin/env python3
"""
Launch script for the Autonomous Tool Discovery Web Demo.

This script provides an easy way to start the web demonstration interface
for presenting the system to the dissertation committee.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['fastapi', 'uvicorn', 'jinja2']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:", ", ".join(missing_packages))
        print("\nPlease install them using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def init_demo_tools():
    """Initialize demo tools if needed."""
    try:
        print("\n🔧 Initializing demo tools...")
        init_script = Path(__file__).parent / "src" / "web" / "init_demo_tools.py"
        if init_script.exists():
            subprocess.run([sys.executable, str(init_script)], check=True, capture_output=True)
            print("✅ Demo tools initialized")
        return True
    except Exception as e:
        print(f"⚠️  Could not initialize demo tools: {e}")
        print("   The demo will still work with mock tools")
        return False


def find_available_port(start_port=8000):
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + 10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def is_wsl():
    """Check if running in WSL."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def main():
    """Main function to launch the demo."""
    print("=" * 60)
    print("🚀 Autonomous Tool Discovery - Web Demo Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies are installed")
    
    # Initialize demo tools
    init_demo_tools()
    
    # Find available port
    port = find_available_port(8000)
    if not port:
        print("\n❌ Could not find an available port (8000-8009)")
        print("   Please close other applications using these ports")
        sys.exit(1)
    
    # Change to the web directory
    web_dir = Path(__file__).parent / "src" / "web"
    os.chdir(web_dir)
    
    # Launch the web server
    print("\n🌐 Starting web server...")
    print(f"📍 Server will be available at: http://localhost:{port}")
    
    if is_wsl():
        print("\n🔍 Detected WSL environment")
        print(f"   Access from Windows browser: http://localhost:{port}")
    
    print("\n" + "=" * 60)
    print("ℹ️  Note: The system will use mock tools if MCP servers are not available")
    print("ℹ️  This is perfect for demonstration purposes!")
    print("\n👉 Open your browser and navigate to the URL above")
    print("\n Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Don't auto-open browser in WSL or if it fails
    if not is_wsl():
        try:
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
            print("🌐 Browser opened automatically")
        except:
            pass
    
    # Run the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "demo_app:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("Thank you for using the demo!")
    except Exception as e:
        print(f"\n❌ Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()