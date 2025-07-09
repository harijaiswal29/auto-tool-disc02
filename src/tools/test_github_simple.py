#!/usr/bin/env python3
"""
Simple test to verify GitHub MCP server starts correctly
"""

import subprocess
import os
import time

# Set environment
env = os.environ.copy()
env['GITHUB_TOKEN'] = os.environ.get('GITHUB_TOKEN', '')

print("Starting GitHub MCP server...")
print(f"Token present: {'Yes' if env.get('GITHUB_TOKEN') else 'No'}")

# Start the server
proc = subprocess.Popen(
    ['node', 'node_modules/.bin/mcp-server-github'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env
)

# Give it time to start
time.sleep(2)

# Check if it's still running
if proc.poll() is None:
    print("✅ Server is running!")
    
    # Check for any stderr output
    try:
        proc.stderr.flush()
        stderr_output = proc.stderr.read()
        if stderr_output:
            print(f"Stderr: {stderr_output}")
    except:
        pass
    
    # Terminate
    proc.terminate()
    proc.wait()
    print("Server terminated")
else:
    # Process exited, get output
    stdout, stderr = proc.communicate()
    print(f"❌ Server exited with code: {proc.returncode}")
    if stdout:
        print(f"Stdout:\n{stdout}")
    if stderr:
        print(f"Stderr:\n{stderr}")