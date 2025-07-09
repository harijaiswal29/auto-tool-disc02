#!/usr/bin/env python3
"""
Debug script for GitHub MCP server connection issues
"""

import subprocess
import json
import os
import sys
import asyncio
import time
import select
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_direct_npx():
    """Test running the GitHub MCP server directly with npx"""
    print("\n=== Test 1: Direct npx execution ===")
    
    env = os.environ.copy()
    env['GITHUB_TOKEN'] = os.environ.get('GITHUB_TOKEN', '')
    
    try:
        # Try to run the server and capture any output
        result = subprocess.run(
            ['npx', '@modelcontextprotocol/server-github', '--help'],
            capture_output=True,
            text=True,
            env=env,
            timeout=10
        )
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error: {e}")

def test_node_direct():
    """Test running the server directly with node"""
    print("\n=== Test 2: Direct node execution ===")
    
    # Find the actual server file
    server_path = Path("node_modules/@modelcontextprotocol/server-github")
    if server_path.exists():
        print(f"Server path exists: {server_path}")
        
        # Look for the main file
        package_json_path = server_path / "package.json"
        if package_json_path.exists():
            with open(package_json_path) as f:
                package_data = json.load(f)
                print(f"Package main: {package_data.get('main', 'Not found')}")
                print(f"Package bin: {package_data.get('bin', 'Not found')}")

def test_stdio_communication():
    """Test stdio communication with the server"""
    print("\n=== Test 3: STDIO Communication ===")
    
    env = os.environ.copy()
    env['GITHUB_TOKEN'] = os.environ.get('GITHUB_TOKEN', '')
    
    # Try different command variations
    commands = [
        ['npx', '@modelcontextprotocol/server-github'],
        ['node', 'node_modules/.bin/mcp-server-github'],
        ['./node_modules/.bin/mcp-server-github']
    ]
    
    for cmd in commands:
        print(f"\nTrying command: {' '.join(cmd)}")
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env=env
            )
            
            # Check if process started successfully
            time.sleep(0.5)
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print(f"Process exited immediately with code: {proc.returncode}")
                print(f"Stdout: {stdout}")
                print(f"Stderr: {stderr}")
                continue
            
            # Try reading any initial output
            print("Checking for initial output...")
            
            # Use select to check if there's data available
            readable, _, _ = select.select([proc.stdout, proc.stderr], [], [], 1.0)
            
            if proc.stderr in readable:
                stderr_line = proc.stderr.readline()
                if stderr_line:
                    print(f"Stderr: {stderr_line.strip()}")
            
            if proc.stdout in readable:
                stdout_line = proc.stdout.readline()
                if stdout_line:
                    print(f"Stdout: {stdout_line.strip()}")
            
            # Send initialization
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "debug-client",
                        "version": "1.0"
                    }
                },
                "id": 1
            }
            
            print(f"Sending: {json.dumps(init_request)}")
            proc.stdin.write(json.dumps(init_request) + '\n')
            proc.stdin.flush()
            
            # Try to read response with timeout
            print("Waiting for response...")
            readable, _, _ = select.select([proc.stdout], [], [], 5.0)
            
            if readable:
                response = proc.stdout.readline()
                if response:
                    print(f"Response: {response.strip()}")
                    try:
                        parsed = json.loads(response)
                        print(f"Parsed response: {json.dumps(parsed, indent=2)}")
                    except:
                        pass
            else:
                print("No response received within 5 seconds")
            
            # Clean up
            proc.terminate()
            proc.wait(timeout=2)
            
        except Exception as e:
            print(f"Error with command {cmd}: {e}")

async def test_async_communication():
    """Test async communication with the server"""
    print("\n=== Test 4: Async Communication ===")
    
    env = os.environ.copy()
    env['GITHUB_TOKEN'] = os.environ.get('GITHUB_TOKEN', '')
    
    try:
        # Create the process
        proc = await asyncio.create_subprocess_exec(
            'npx', '@modelcontextprotocol/server-github',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        print("Process started, PID:", proc.pid)
        
        # Check for any stderr output
        try:
            stderr_data = await asyncio.wait_for(proc.stderr.read(1024), timeout=1.0)
            if stderr_data:
                print(f"Stderr: {stderr_data.decode()}")
        except asyncio.TimeoutError:
            print("No stderr output")
        
        # Send initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {
                    "name": "async-debug-client",
                    "version": "1.0"
                }
            },
            "id": 1
        }
        
        message = json.dumps(init_request) + '\n'
        print(f"Sending: {message.strip()}")
        proc.stdin.write(message.encode())
        await proc.stdin.drain()
        
        # Try to read response
        try:
            response = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            if response:
                print(f"Response: {response.decode().strip()}")
        except asyncio.TimeoutError:
            print("No response received (timeout)")
        
        # Terminate
        proc.terminate()
        await proc.wait()
        
    except Exception as e:
        print(f"Async error: {e}")

def check_environment():
    """Check environment and dependencies"""
    print("\n=== Environment Check ===")
    
    # Check token
    token = os.environ.get('GITHUB_TOKEN', '')
    if token:
        print(f"GITHUB_TOKEN is set (length: {len(token)})")
        print(f"Token prefix: {token[:8]}...")
    else:
        print("GITHUB_TOKEN is NOT set")
    
    # Check npm/node
    try:
        node_version = subprocess.run(['node', '--version'], capture_output=True, text=True)
        print(f"Node version: {node_version.stdout.strip()}")
    except:
        print("Node.js not found")
    
    try:
        npm_version = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        print(f"NPM version: {npm_version.stdout.strip()}")
    except:
        print("NPM not found")
    
    # Check if npx can find the package
    try:
        which_result = subprocess.run(['npx', 'which', '@modelcontextprotocol/server-github'], 
                                    capture_output=True, text=True)
        print(f"NPX which result: {which_result.stdout.strip()}")
    except:
        print("Could not locate package with npx")

def main():
    """Run all tests"""
    print("GitHub MCP Server Debug Script")
    print("=" * 50)
    
    check_environment()
    test_direct_npx()
    test_node_direct()
    test_stdio_communication()
    
    # Run async test
    print("\nRunning async test...")
    asyncio.run(test_async_communication())
    
    print("\n" + "=" * 50)
    print("Debug tests completed")

if __name__ == "__main__":
    main()