#!/usr/bin/env python3
"""
Direct test of GitHub MCP server to understand the protocol
"""

import subprocess
import json
import os

# Set environment
env = os.environ.copy()
# Use environment variable for token - do not hardcode
github_token = os.environ.get('GITHUB_TOKEN', '')
if not github_token:
    print("ERROR: GITHUB_TOKEN environment variable not set")
    print("Please set GITHUB_TOKEN before running this test")
    exit(1)
env['GITHUB_TOKEN'] = github_token

# Start the server
proc = subprocess.Popen(
    ['npx', '@modelcontextprotocol/server-github'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env
)

# Send initialization
init_request = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "1.0",
        "clientInfo": {
            "name": "test-client",
            "version": "1.0"
        }
    },
    "id": 1
}

print("Sending:", json.dumps(init_request))
proc.stdin.write(json.dumps(init_request) + '\n')
proc.stdin.flush()

# Read responses
print("\nReading responses...")
for i in range(5):  # Try to read up to 5 lines
    line = proc.stdout.readline()
    if line:
        print(f"Response {i+1}: {line.strip()}")
    else:
        break

# Clean up
proc.terminate()