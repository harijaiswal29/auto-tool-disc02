"""
Search MCP Client Wrapper

This module provides a client wrapper for Search MCP servers,
enabling various search operations through the Model Context Protocol.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.tools.mock_search_mcp import MockSearchMCPServer

logger = get_logger(__name__)

class SearchMCPClient:
    """
    Client wrapper for Search MCP server.
    
    This client enables search operations like:
    - Web search with filters
    - Code repository search
    - Documentation search
    - News search
    - Academic/scholarly search
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, server_command: Optional[List[str]] = None):
        """
        Initialize Search MCP client.
        
        Args:
            config: Configuration including API keys, endpoints, etc.
            server_command: Command to start the MCP server (if not provided, uses default)
        """
        self.config = config or {}
        self.server_name = "search"
        
        # Default to using npx to run a search MCP server
        self.server_command = server_command or [
            "npx", "@modelcontextprotocol/server-search"
        ]
        
        self.process: Optional[subprocess.Popen] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.mock_server: Optional[MockSearchMCPServer] = None
        self.use_mock = False
        
        logger.info(f"[INIT] Search MCP Client initialized")
    
    def _next_message_id(self) -> int:
        """Generate next message ID for JSON-RPC."""
        self._message_id += 1
        return self._message_id
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Connect to the Search MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if use_mock:
                logger.info(f"[CONNECTING] Using mock Search MCP server...")
                self.mock_server = MockSearchMCPServer(self.config)
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "SearchMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to mock Search MCP server")
                    
                    # Get available tools
                    await self._discover_tools_mock()
                    self.use_mock = True
                    return True
                else:
                    logger.error(f"[FAILED] Failed to initialize mock: {response}")
                    return False
            else:
                logger.info(f"[CONNECTING] Starting Search MCP server...")
                
                # Start the MCP server process
                self.process = subprocess.Popen(
                    self.server_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Send initialization request
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "SearchMCPClient",
                            "version": "0.1.0"
                        },
                        "config": self.config
                    },
                    "id": self._next_message_id()
                }
                
                await self._send_message(init_request)
                response = await self._receive_message()
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to Search MCP server")
                    logger.debug(f"Capabilities: {self.capabilities}")
                    
                    # Get available tools
                    await self._discover_tools()
                    return True
                else:
                    logger.error(f"[FAILED] Failed to initialize: {response}")
                    return False
                
        except Exception as e:
            logger.error(f"[ERROR] Connection error: {str(e)}")
            return False
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the Search MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        await self._send_message(list_request)
        response = await self._receive_message()
        
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            logger.info(f"[TOOLS] Discovered {len(self.tools)} Search tools:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def _discover_tools_mock(self) -> None:
        """Discover available tools from the mock Search MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        response = await self.mock_server.handle_request(list_request)
        
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            logger.info(f"[TOOLS] Discovered {len(self.tools)} Search tools from mock:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def web_search(self, query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a web search.
        
        Args:
            query: Search query
            options: Search options (num_results, date_range, language, etc.)
            
        Returns:
            Search results
        """
        logger.info(f"[WEB_SEARCH] Query: {query}")
        
        arguments = {"query": query}
        if options:
            arguments.update(options)
        
        return await self.call_tool("web_search", arguments)
    
    async def code_search(self, query: str, language: Optional[str] = None, 
                         repository: Optional[str] = None) -> Dict[str, Any]:
        """
        Search code repositories.
        
        Args:
            query: Search query
            language: Programming language filter
            repository: Specific repository to search
            
        Returns:
            Code search results
        """
        logger.info(f"[CODE_SEARCH] Query: {query}, Language: {language}")
        
        arguments = {"query": query}
        if language:
            arguments["language"] = language
        if repository:
            arguments["repository"] = repository
        
        return await self.call_tool("code_search", arguments)
    
    async def documentation_search(self, query: str, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Search technical documentation.
        
        Args:
            query: Search query
            source: Documentation source (e.g., "python", "javascript")
            
        Returns:
            Documentation search results
        """
        logger.info(f"[DOC_SEARCH] Query: {query}, Source: {source}")
        
        arguments = {"query": query}
        if source:
            arguments["source"] = source
        
        return await self.call_tool("doc_search", arguments)
    
    async def news_search(self, query: str, date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Search news articles.
        
        Args:
            query: Search query
            date_range: Date range filter {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}
            
        Returns:
            News search results
        """
        logger.info(f"[NEWS_SEARCH] Query: {query}")
        
        arguments = {"query": query}
        if date_range:
            arguments["date_range"] = date_range
        
        return await self.call_tool("news_search", arguments)
    
    async def scholarly_search(self, query: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search academic papers and research.
        
        Args:
            query: Search query
            fields: Academic fields to filter by
            
        Returns:
            Academic search results
        """
        logger.info(f"[SCHOLARLY_SEARCH] Query: {query}")
        
        arguments = {"query": query}
        if fields:
            arguments["fields"] = fields
        
        return await self.call_tool("scholarly_search", arguments)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific Search MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution results
        """
        logger.debug(f"[TOOL] Calling {tool_name} with args: {arguments}")
        
        # Check if tool exists
        tool_exists = any(tool.get("name") == tool_name for tool in self.tools)
        if not tool_exists:
            logger.error(f"[ERROR] Tool not found: {tool_name}")
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        
        call_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_message_id()
        }
        
        start_time = datetime.now()
        
        if self.use_mock:
            response = await self.mock_server.handle_request(call_request)
        else:
            await self._send_message(call_request)
            response = await self._receive_message()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if response and "result" in response:
            logger.info(f"[SUCCESS] Tool {tool_name} executed in {execution_time:.2f}s")
            return {
                "success": True,
                "result": response["result"],
                "execution_time": execution_time
            }
        else:
            error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
            logger.error(f"[ERROR] Tool execution failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time
            }
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message to the MCP server."""
        if self.process and self.process.stdin:
            json_message = json.dumps(message) + "\n"
            self.process.stdin.write(json_message)
            self.process.stdin.flush()
            logger.debug(f"→ Sent: {message.get('method', 'response')} (id: {message.get('id')})")
    
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive a JSON-RPC message from the MCP server."""
        if self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    response = json.loads(line.strip())
                    logger.debug(f"← Received: {response.get('id', 'notification')}")
                    return response
            except json.JSONDecodeError as e:
                logger.error(f"[ERROR] JSON decode error: {e}")
                logger.debug(f"Raw line: {line}")
        return None
    
    async def disconnect(self) -> None:
        """Disconnect from the Search MCP server."""
        logger.info("[DISCONNECTING] Closing Search MCP connection...")
        
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            
            if self.process.poll() is None:
                self.process.kill()
            
            logger.info("[SUCCESS] Disconnected from Search MCP server")
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register Search tools with the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        for tool in self.tools:
            tool_info = {
                'id': f'search.{tool["name"]}',
                'name': tool['name'],
                'server_type': 'search',
                'endpoint': self.server_command[1] if len(self.server_command) > 1 else 'search',
                'description': tool.get('description', ''),
                'capabilities': self.capabilities,
                'input_schema': tool.get('inputSchema', {})
            }
            registry.register_tool(tool_info)
            logger.info(f"[REGISTERED] Search tool: {tool_info['id']}")


async def test_search_mcp():
    """Test the Search MCP client implementation."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Search MCP Client")
    logger.info("=" * 60)
    
    # Initialize client with mock config
    config = {
        "api_key": "test_key",
        "max_results": 10
    }
    client = SearchMCPClient(config)
    
    try:
        # Try to connect to real server first, fall back to mock if it fails
        connected = await client.connect()
        if not connected:
            logger.warning("⚠️  Could not connect to Search MCP server")
            logger.info("💡 Trying with mock server instead...")
            connected = await client.connect(use_mock=True)
            if not connected:
                logger.error("[ERROR] Could not connect to mock server either")
                return
        
        # Test web search
        web_result = await client.web_search(
            "Model Context Protocol MCP",
            {"num_results": 5, "language": "en"}
        )
        logger.info(f"[WEB_SEARCH] Result: {web_result}")
        
        # Test code search
        code_result = await client.code_search(
            "async def connect",
            language="python"
        )
        logger.info(f"[CODE_SEARCH] Result: {code_result}")
        
        # Test documentation search
        doc_result = await client.documentation_search(
            "asyncio python",
            source="python"
        )
        logger.info(f"[DOC_SEARCH] Result: {doc_result}")
        
        # Test news search
        news_result = await client.news_search(
            "artificial intelligence",
            {"from": "2024-01-01", "to": "2024-12-31"}
        )
        logger.info(f"[NEWS_SEARCH] Result: {news_result}")
        
        # Test scholarly search
        scholar_result = await client.scholarly_search(
            "machine learning",
            fields=["computer science", "artificial intelligence"]
        )
        logger.info(f"[SCHOLARLY_SEARCH] Result: {scholar_result}")
        
        # Test with tool registry
        registry = ToolRegistry("data/test_search_registry.db")
        client.register_tools_to_registry(registry)
        
        # List registered Search tools
        search_tools = registry.list_tools("search")
        logger.info(f"[REGISTRY] Registered {len(search_tools)} Search tools")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()
    
    logger.info("[TEST] Search MCP test complete!")


if __name__ == "__main__":
    asyncio.run(test_search_mcp())