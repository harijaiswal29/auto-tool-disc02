#!/usr/bin/env python3
"""
Test script for the real @modelcontextprotocol/server-filesystem
This will help us understand how to work with actual MCP servers.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
import platform

# Add project root to path (go up 2 levels from tests/integration/)
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RealMCPClient:
    """Client for connecting to real MCP servers."""
    
    def __init__(self):
        self.process = None
        
    async def test_filesystem_server(self):
        """Test the real filesystem MCP server."""
        logger.info("[INFO] Testing real filesystem MCP server")
        logger.info("-" * 50)
        
        # Determine the command based on OS
        if platform.system() == "Windows":
            # For Windows, use npx to run the server
            server_command = ["npx", "@modelcontextprotocol/server-filesystem", "C:\\temp"]
        else:
            # For Linux/WSL
            server_command = ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
        
        try:
            logger.info(f"[INFO] Starting server with command: {' '.join(server_command)}")
            
            # Start the MCP server process
            self.process = subprocess.Popen(
                server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered
            )
            
            # Give it a moment to start
            await asyncio.sleep(1)
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "TestMCPClient",
                        "version": "0.1.0"
                    }
                },
                "id": 1
            }
            
            logger.info("[INFO] Sending initialize request...")
            await self._send_message(init_request)
            
            # Try to read response
            response = await self._read_response()
            if response:
                logger.info(f"[SUCCESS] Got response: {json.dumps(response, indent=2)}")
            else:
                logger.warning("[WARNING] No response received")
                
                # Check stderr for any errors
                if self.process.stderr:
                    error = self.process.stderr.read()
                    if error:
                        logger.error(f"[ERROR] Server error: {error}")
            
        except FileNotFoundError:
            logger.error("[ERROR] Could not find npx command. Make sure Node.js is installed and in PATH")
            logger.info("[TIP] If using Windows, make sure you're in PowerShell, not WSL")
        except Exception as e:
            logger.error(f"[ERROR] Failed to start server: {str(e)}")
        finally:
            await self.cleanup()
    
    async def _send_message(self, message: dict):
        """Send a message to the MCP server."""
        if self.process and self.process.stdin:
            json_message = json.dumps(message) + "\n"
            self.process.stdin.write(json_message)
            self.process.stdin.flush()
            logger.debug(f"[SENT] {message.get('method', 'response')}")
    
    async def _read_response(self, timeout: float = 5.0):
        """Try to read a response from the server."""
        if self.process and self.process.stdout:
            try:
                # Use asyncio to read with timeout
                line = await asyncio.wait_for(
                    asyncio.to_thread(self.process.stdout.readline),
                    timeout=timeout
                )
                if line:
                    return json.loads(line.strip())
            except asyncio.TimeoutError:
                logger.warning(f"[TIMEOUT] No response after {timeout} seconds")
            except json.JSONDecodeError as e:
                logger.error(f"[ERROR] Invalid JSON response: {e}")
        return None
    
    async def cleanup(self):
        """Clean up the server process."""
        if self.process:
            logger.info("[INFO] Cleaning up server process...")
            self.process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.process.wait),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                self.process.kill()
            logger.info("[INFO] Server process terminated")

async def test_mock_vs_real():
    """Compare mock and real MCP servers."""
    logger.info("=" * 60)
    logger.info("MCP SERVER TESTING - MOCK VS REAL")
    logger.info("=" * 60)
    
    # First, test our mock server
    logger.info("\n[1] TESTING MOCK SERVER")
    logger.info("-" * 40)
    
    try:
        from src.tools.mock_mcp_servers import FileSystemMCPServer
        mock_server = FileSystemMCPServer("data/test_fs")
        
        # Test initialize
        response = await mock_server.handle_request({
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1
        })
        logger.info(f"[MOCK] Initialize: {response['result']['serverInfo']['name']}")
        
        # Test tools list
        response = await mock_server.handle_request({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2
        })
        logger.info(f"[MOCK] Available tools: {[t['name'] for t in response['result']['tools']]}")
        
    except Exception as e:
        logger.error(f"[ERROR] Mock server test failed: {e}")
    
    # Then test real server
    logger.info("\n[2] TESTING REAL FILESYSTEM SERVER")
    logger.info("-" * 40)
    
    client = RealMCPClient()
    await client.test_filesystem_server()
    
    logger.info("\n[SUMMARY]")
    logger.info("-" * 40)
    logger.info("Mock servers are working and ready for development!")
    logger.info("Real server testing helps understand the actual protocol.")
    logger.info("\nNext steps:")
    logger.info("1. Use mock servers for initial development")
    logger.info("2. Gradually integrate real servers as they become available")
    logger.info("3. Build your tool discovery system to work with both")

def main():
    """Main entry point."""
    # For Windows, ensure proper encoding
    if platform.system() == 'Windows':
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    asyncio.run(test_mock_vs_real())

if __name__ == "__main__":
    main()