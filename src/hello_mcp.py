#!/usr/bin/env python3
"""
Hello MCP - Your first MCP (Model Context Protocol) test script.

This script demonstrates the basic concepts of MCP by creating a simple
client that can discover and interact with MCP servers.

Real-world analogy: This is like a universal remote control learning
to communicate with different devices in your home.
"""

import asyncio
import json
import subprocess
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import our logger
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_milestone

# Create logger for this module
logger = get_logger(__name__)

class SimpleMCPClient:
    """
    A simplified MCP client for learning purposes.
    
    Think of this as a translator that helps AI agents talk to various tools
    using a common language (the MCP protocol).
    """
    
    def __init__(self, server_name: str, server_command: List[str]):
        self.server_name = server_name
        self.server_command = server_command
        self.process: Optional[subprocess.Popen] = None
        self.capabilities: Dict[str, Any] = {}
        
        logger.info(f"Initializing MCP client for: {server_name}")
    
    async def connect(self):
        """
        Establish connection to an MCP server.
        
        Real-world analogy: Like plugging in a USB device and waiting
        for the computer to recognize it.
        """
        try:
            logger.info(f"[CONNECTING] Connecting to {self.server_name}...")
            
            # Start the MCP server process
            self.process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "HelloMCPClient",
                        "version": "0.1.0"
                    }
                },
                "id": 1
            }
            
            await self._send_message(init_request)
            response = await self._receive_message()
            
            if response and "result" in response:
                self.capabilities = response["result"].get("capabilities", {})
                logger.info(f"[SUCCESS] Connected! Capabilities: {self.capabilities}")
                return True
            else:
                logger.error(f"[FAILED] Failed to connect: {response}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Connection error: {str(e)}")
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Discover available tools from the MCP server.
        
        Real-world analogy: Like asking a Swiss Army knife to show
        you all its available tools.
        """
        logger.info("🔍 Discovering available tools...")
        
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        await self._send_message(list_request)
        response = await self._receive_message()
        
        if response and "result" in response:
            tools = response["result"].get("tools", [])
            logger.info(f"📦 Found {len(tools)} tools:")
            for tool in tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
            return tools
        
        return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a specific tool with given arguments.
        
        Real-world analogy: Like using a specific function on your
        smartphone - you select the app and provide the input.
        """
        logger.info(f"🔧 Calling tool: {tool_name}")
        logger.debug(f"Arguments: {arguments}")
        
        call_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 3
        }
        
        await self._send_message(call_request)
        response = await self._receive_message()
        
        if response and "result" in response:
            logger.info(f"✅ Tool executed successfully")
            return response["result"]
        else:
            logger.error(f"❌ Tool execution failed: {response}")
            return None
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send a message to the MCP server."""
        if self.process and self.process.stdin:
            json_message = json.dumps(message) + "\n"
            self.process.stdin.write(json_message)
            self.process.stdin.flush()
            logger.debug(f"→ Sent: {message.get('method', 'response')}")
    
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive a message from the MCP server."""
        if self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    response = json.loads(line.strip())
                    logger.debug(f"← Received: {response}")
                    return response
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
        return None
    
    async def disconnect(self):
        """Clean up and disconnect from the MCP server."""
        logger.info(f"🔌 Disconnecting from {self.server_name}")
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)  # Give it time to clean up
            if self.process.poll() is None:
                self.process.kill()
            logger.info("✅ Disconnected")

class MockMCPServer:
    """
    A mock MCP server for testing when real servers aren't available.
    
    Real-world analogy: Like a flight simulator for pilots - lets you
    practice without needing a real plane.
    """
    
    def __init__(self):
        self.tools = [
            {
                "name": "get_time",
                "description": "Get the current time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone (e.g., 'UTC', 'EST')"}
                    }
                }
            },
            {
                "name": "calculate",
                "description": "Perform basic calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]
        logger.info("🎭 Mock MCP Server initialized")
    
    async def simulate_interaction(self):
        """Simulate MCP server responses for testing."""
        logger.info("🎭 Running mock server simulation...")
        
        # Simulate initialize response
        logger.info("← Server: Initialize response")
        logger.info("  Capabilities: tools, resources")
        
        # Simulate tools list
        logger.info("← Server: Tools list")
        for tool in self.tools:
            logger.info(f"  - {tool['name']}: {tool['description']}")
        
        # Simulate tool execution
        logger.info("← Server: Tool execution (get_time)")
        logger.info("  Result: 2024-01-15 10:30:00 UTC")

async def main():
    """
    Main function demonstrating basic MCP concepts.
    """
    logger.info("🚀 Starting Hello MCP Test Script")
    logger.info("=" * 50)
    
    # Log milestone
    log_milestone(logger, "Hello MCP Script Started", {
        "objective": "Test basic MCP client functionality",
        "components": "SimpleMCPClient, MockMCPServer"
    })
    
    # First, let's use our mock server to understand the protocol
    logger.info("\n📚 PART 1: Understanding MCP with Mock Server")
    logger.info("-" * 40)
    mock_server = MockMCPServer()
    await mock_server.simulate_interaction()
    
    # Now let's try to connect to a real MCP server
    # (This will fail if no server is running, which is expected)
    logger.info("\n📚 PART 2: Attempting Real MCP Connection")
    logger.info("-" * 40)
    
    # Example: Trying to connect to a filesystem MCP server
    # You would need to have the actual MCP server installed
    client = SimpleMCPClient(
        server_name="filesystem",
        server_command=["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    
    try:
        connected = await client.connect()
        if connected:
            # List available tools
            tools = await client.list_tools()
            
            # Try calling a tool (example)
            if tools:
                result = await client.call_tool(
                    "read_file",
                    {"path": "/tmp/test.txt"}
                )
                logger.info(f"Tool result: {result}")
        else:
            logger.warning("⚠️  Could not connect to real MCP server")
            logger.info("💡 This is expected if you haven't installed MCP servers yet")
            logger.info("💡 For now, the mock server demonstration shows how MCP works")
    
    except Exception as e:
        logger.error(f"Error during real server test: {e}")
        logger.info("💡 Don't worry! This is expected for initial testing")
    
    finally:
        await client.disconnect()
    
    # Summary
    logger.info("\n📊 Summary")
    logger.info("-" * 40)
    logger.info("✅ Successfully demonstrated MCP concepts")
    logger.info("✅ Mock server showed protocol structure")
    logger.info("📝 Next steps:")
    logger.info("  1. Install actual MCP servers")
    logger.info("  2. Implement tool discovery")
    logger.info("  3. Build the learning system")
    
    log_milestone(logger, "Hello MCP Test Complete", {
        "status": "Success",
        "learned": "MCP protocol basics",
        "next": "Integrate real MCP servers"
    })

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())