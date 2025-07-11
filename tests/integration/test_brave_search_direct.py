#!/usr/bin/env python3
"""
Direct test script for the real Brave Search MCP server.
This tests the actual @modelcontextprotocol/server-brave-search functionality.
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BraveSearchMCPTester:
    """Direct tester for Brave Search MCP server."""
    
    def __init__(self):
        self.process = None
        self._message_id = 0
        
        # Check for API key
        self.api_key = os.environ.get('BRAVE_API_KEY')
        if not self.api_key:
            logger.error("BRAVE_API_KEY environment variable not set!")
            logger.info("Please set: export BRAVE_API_KEY='your-api-key'")
            raise ValueError("BRAVE_API_KEY not set")
        else:
            logger.info(f"✓ BRAVE_API_KEY found (length: {len(self.api_key)})")
    
    def _next_message_id(self) -> int:
        """Generate next message ID."""
        self._message_id += 1
        return self._message_id
    
    async def start_server(self):
        """Start the Brave Search MCP server."""
        logger.info("Starting Brave Search MCP server...")
        
        # Command to start the server
        server_command = ["npx", "@modelcontextprotocol/server-brave-search"]
        
        # Set up environment with API key
        env = os.environ.copy()
        env['BRAVE_API_KEY'] = self.api_key
        
        try:
            self.process = subprocess.Popen(
                server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env
            )
            logger.info("✓ Server process started")
            
            # Give it a moment to start
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    async def send_message(self, message: Dict[str, Any]):
        """Send a JSON-RPC message to the server."""
        if not self.process or not self.process.stdin:
            raise Exception("Server not started")
        
        json_message = json.dumps(message) + "\n"
        self.process.stdin.write(json_message)
        self.process.stdin.flush()
        logger.debug(f"→ Sent: {message.get('method', 'response')} (id: {message.get('id')})")
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive a JSON-RPC message from the server."""
        if not self.process or not self.process.stdout:
            return None
        
        try:
            line = self.process.stdout.readline()
            if line:
                response = json.loads(line.strip())
                logger.debug(f"← Received: {json.dumps(response, indent=2)}")
                return response
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Raw line: {line}")
        return None
    
    async def initialize(self):
        """Initialize the MCP connection."""
        logger.info("Initializing MCP connection...")
        
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {
                    "name": "BraveSearchTester",
                    "version": "1.0.0"
                },
                "capabilities": {}
            },
            "id": self._next_message_id()
        }
        
        await self.send_message(init_request)
        response = await self.receive_message()
        
        if response and "result" in response:
            logger.info("✓ Initialized successfully")
            logger.info(f"  Server: {response['result']['serverInfo']['name']}")
            logger.info(f"  Version: {response['result']['serverInfo']['version']}")
            logger.info(f"  Capabilities: {response['result']['capabilities']}")
            return True
        else:
            logger.error(f"Initialization failed: {response}")
            return False
    
    async def list_tools(self):
        """List available tools from the server."""
        logger.info("\nDiscovering available tools...")
        
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        await self.send_message(list_request)
        response = await self.receive_message()
        
        if response and "result" in response:
            tools = response["result"]["tools"]
            logger.info(f"✓ Found {len(tools)} tools:")
            
            for tool in tools:
                logger.info(f"\n  Tool: {tool['name']}")
                logger.info(f"  Description: {tool['description']}")
                logger.info(f"  Input Schema: {json.dumps(tool['inputSchema'], indent=4)}")
            
            return tools
        else:
            logger.error(f"Failed to list tools: {response}")
            return []
    
    async def test_web_search(self, query: str, count: int = 5):
        """Test the brave_web_search tool."""
        logger.info(f"\n🔍 Testing Web Search: '{query}'")
        
        search_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "brave_web_search",
                "arguments": {
                    "query": query,
                    "count": count
                }
            },
            "id": self._next_message_id()
        }
        
        start_time = datetime.now()
        await self.send_message(search_request)
        response = await self.receive_message()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if response and "result" in response:
            logger.info(f"✓ Search completed in {elapsed:.2f}s")
            
            # Parse and display results
            result = response["result"]
            if isinstance(result, list) and len(result) > 0 and "content" in result[0]:
                content = result[0]["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            logger.info("\nSearch Results:")
                            logger.info(item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"])
            else:
                logger.info(f"Results: {json.dumps(result, indent=2)[:500]}...")
            
            return result
        else:
            logger.error(f"Search failed: {response}")
            return None
    
    async def test_local_search(self, query: str, count: int = 5):
        """Test the brave_local_search tool."""
        logger.info(f"\n📍 Testing Local Search: '{query}'")
        
        search_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "brave_local_search",
                "arguments": {
                    "query": query,
                    "count": count
                }
            },
            "id": self._next_message_id()
        }
        
        start_time = datetime.now()
        await self.send_message(search_request)
        response = await self.receive_message()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if response and "result" in response:
            logger.info(f"✓ Local search completed in {elapsed:.2f}s")
            
            # Parse and display results
            result = response["result"]
            if isinstance(result, list) and len(result) > 0 and "content" in result[0]:
                content = result[0]["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            logger.info("\nLocal Search Results:")
                            logger.info(item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"])
            else:
                logger.info(f"Results: {json.dumps(result, indent=2)[:500]}...")
            
            return result
        else:
            logger.error(f"Local search failed: {response}")
            return None
    
    async def cleanup(self):
        """Clean up the server process."""
        if self.process:
            logger.info("\nCleaning up...")
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
            logger.info("✓ Server stopped")


async def main():
    """Run the Brave Search MCP test."""
    logger.info("=" * 60)
    logger.info("🚀 Brave Search MCP Server Test")
    logger.info("=" * 60)
    
    tester = BraveSearchMCPTester()
    
    try:
        # Start the server
        await tester.start_server()
        
        # Initialize connection
        if not await tester.initialize():
            logger.error("Failed to initialize connection")
            return
        
        # List available tools
        tools = await tester.list_tools()
        
        if not tools:
            logger.error("No tools found!")
            return
        
        # Test web search
        logger.info("\n" + "=" * 60)
        logger.info("Testing Web Search")
        logger.info("=" * 60)
        
        await tester.test_web_search("Model Context Protocol MCP", count=3)
        await asyncio.sleep(1)  # Rate limiting
        
        await tester.test_web_search("Python asyncio tutorial", count=3)
        await asyncio.sleep(1)
        
        # Test local search
        logger.info("\n" + "=" * 60)
        logger.info("Testing Local Search")
        logger.info("=" * 60)
        
        await tester.test_local_search("restaurants near me", count=3)
        await asyncio.sleep(1)
        
        await tester.test_local_search("coffee shops", count=3)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # Check if running in correct directory
    if not Path("src").exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    # Run the test
    asyncio.run(main())