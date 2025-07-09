#!/usr/bin/env python3
"""
Direct test of GitHub MCP server to understand the protocol
"""

import subprocess
import json
import os

# Set environment
env = os.environ.copy()
env['GITHUB_TOKEN'] = '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'

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