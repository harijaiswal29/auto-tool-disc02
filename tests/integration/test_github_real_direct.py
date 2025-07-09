#!/usr/bin/env python3
"""
Direct test of real GitHub MCP server with detailed debugging
"""

import asyncio
import json
import subprocess
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def test_real_server():
    """Test the real GitHub MCP server with detailed debugging"""
    
    # Check if token is available
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        logger.error("No GITHUB_TOKEN found in environment")
        return
    
    logger.info(f"Token found: {token[:8]}...")
    
    # Set up environment
    env = os.environ.copy()
    env['GITHUB_TOKEN'] = token
    
    # Try to start the server
    logger.info("Starting GitHub MCP server...")
    
    try:
        # First, check if the binary exists and is executable
        binary_path = Path("node_modules/.bin/mcp-server-github")
        if not binary_path.exists():
            logger.error(f"Binary not found at {binary_path}")
            return
        
        logger.info(f"Binary exists at {binary_path}")
        
        # Start the process with detailed error capture
        process = await asyncio.create_subprocess_exec(
            'node',
            str(binary_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        logger.info(f"Process started with PID: {process.pid}")
        
        # Give it a moment to start
        await asyncio.sleep(1)
        
        # Check if process is still running
        if process.returncode is not None:
            logger.error(f"Process exited immediately with code: {process.returncode}")
            
            # Get any output
            stdout, stderr = await process.communicate()
            if stdout:
                logger.info(f"Stdout: {stdout.decode()}")
            if stderr:
                logger.error(f"Stderr: {stderr.decode()}")
            return
        
        logger.info("Process is running, checking for initial output...")
        
        # Try to read any initial stderr (non-blocking)
        try:
            stderr_data = await asyncio.wait_for(process.stderr.read(1024), timeout=1.0)
            if stderr_data:
                logger.info(f"Initial stderr: {stderr_data.decode()}")
        except asyncio.TimeoutError:
            logger.info("No initial stderr output")
        
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
        
        message = json.dumps(init_request) + '\n'
        logger.info(f"Sending: {message.strip()}")
        
        process.stdin.write(message.encode())
        await process.stdin.drain()
        
        logger.info("Waiting for response...")
        
        # Try to read response
        try:
            # Read with a longer timeout
            response_data = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            if response_data:
                response_text = response_data.decode().strip()
                logger.info(f"Response: {response_text}")
                
                # Try to parse as JSON
                try:
                    response_json = json.loads(response_text)
                    logger.info(f"Parsed response: {json.dumps(response_json, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning("Response is not valid JSON")
            else:
                logger.warning("No response data received")
        except asyncio.TimeoutError:
            logger.error("Response timed out after 10 seconds")
            
            # Check if process is still alive
            if process.returncode is None:
                logger.info("Process is still running despite timeout")
            else:
                logger.error(f"Process exited with code: {process.returncode}")
        
        # Clean up
        logger.info("Terminating process...")
        process.terminate()
        await process.wait()
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)

def main():
    """Run the test"""
    logger.info("=" * 60)
    logger.info("Real GitHub MCP Server Test")
    logger.info("=" * 60)
    
    asyncio.run(test_real_server())
    
    logger.info("=" * 60)
    logger.info("Test completed")

if __name__ == "__main__":
    main()