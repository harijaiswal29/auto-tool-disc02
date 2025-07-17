"""
Notion MCP Client Wrapper

This module provides a client wrapper for the remote Notion MCP server,
enabling Notion workspace operations through the Model Context Protocol.
"""

import asyncio
import json
import aiohttp
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.tools.mock_notion_mcp import MockNotionMCPServer

logger = get_logger(__name__)

class NotionMCPClient:
    """
    Client wrapper for remote Notion MCP server.
    
    This client enables Notion workspace operations like:
    - Creating, reading, updating, and deleting pages
    - Managing databases and records
    - Block-level content manipulation
    - Searching across workspace
    - Exporting content as Markdown
    """
    
    def __init__(self, integration_token: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize Notion MCP client.
        
        Args:
            integration_token: Notion integration token
            endpoint: Remote MCP server endpoint
        """
        self.integration_token = integration_token or os.environ.get('NOTION_INTEGRATION_TOKEN')
        self.endpoint = endpoint or os.environ.get('NOTION_MCP_ENDPOINT', 'https://api.notion.com/mcp/v1')
        self.server_name = "notion"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.connected = False
        self.mock_server: Optional[MockNotionMCPServer] = None
        self.use_mock = False
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes for read operations
        
        logger.info(f"[INIT] Notion MCP Client initialized with endpoint: {self.endpoint}")
    
    def _next_message_id(self) -> int:
        """Generate next message ID for JSON-RPC."""
        self._message_id += 1
        return self._message_id
    
    def _get_cache_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key for a request."""
        key_data = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        if not cached_data:
            return False
        cached_time = cached_data.get('timestamp', 0)
        return (time.time() - cached_time) < self._cache_ttl
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Connect to the Notion MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.use_mock = use_mock
            
            if use_mock:
                logger.info("[CONNECTING] Using mock Notion MCP server...")
                self.mock_server = MockNotionMCPServer()
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "NotionMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info("[SUCCESS] Connected to mock Notion MCP server")
                    
                    # Discover tools
                    await self.discover_tools()
                    self.connected = True
                    return True
            else:
                # Connect to real remote server
                if not self.integration_token:
                    logger.error("[ERROR] No integration token provided for Notion")
                    return False
                
                logger.info(f"[CONNECTING] Connecting to remote Notion MCP server at {self.endpoint}")
                
                # Create aiohttp session with proper headers
                self.session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.integration_token}",
                        "Content-Type": "application/json",
                        "Notion-Version": "2022-06-28"  # Latest API version
                    }
                )
                
                # Send initialization request
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "NotionMCPClient",
                            "version": "0.1.0"
                        },
                        "capabilities": {
                            "markdown": True,
                            "blocks": True
                        }
                    },
                    "id": self._next_message_id()
                }
                
                async with self.session.post(self.endpoint, json=init_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "result" in result:
                            self.capabilities = result["result"].get("capabilities", {})
                            logger.info("[SUCCESS] Connected to remote Notion MCP server")
                            
                            # Discover tools
                            await self.discover_tools()
                            self.connected = True
                            return True
                    else:
                        error_text = await response.text()
                        logger.error(f"[ERROR] Failed to connect: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Notion MCP server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.connected = False
            logger.info("[DISCONNECT] Disconnected from Notion MCP server")
        except Exception as e:
            logger.error(f"[ERROR] Error during disconnect: {e}")
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover available tools from the Notion MCP server.
        
        Returns:
            List of tool definitions
        """
        try:
            if self.use_mock and self.mock_server:
                # Use mock server
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": self._next_message_id()
                }
                response = await self.mock_server.handle_request(request)
                
                if response and "result" in response:
                    self.tools = response["result"]["tools"]
                    logger.info(f"[DISCOVER] Found {len(self.tools)} tools in mock server")
                    return self.tools
            else:
                # Use real server
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": self._next_message_id()
                }
                
                async with self.session.post(self.endpoint, json=request) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "result" in result:
                            self.tools = result["result"]["tools"]
                            logger.info(f"[DISCOVER] Found {len(self.tools)} tools")
                            return self.tools
                    else:
                        logger.error(f"[ERROR] Failed to discover tools: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"[ERROR] Failed to discover tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool with given arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            # Check cache for read operations
            cache_key = self._get_cache_key(tool_name, arguments)
            if tool_name in ['get_page', 'search_pages', 'query_database', 'list_workspace_pages']:
                cached = self._cache.get(cache_key)
                if cached and self._is_cache_valid(cached):
                    logger.info(f"[CACHE] Returning cached result for {tool_name}")
                    return cached['data']
            
            if self.use_mock and self.mock_server:
                # Use mock server
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": self._next_message_id()
                }
                response = await self.mock_server.handle_request(request)
                
                if response and "result" in response:
                    result = response["result"]
                    # Cache the result for read operations
                    if tool_name in ['get_page', 'search_pages', 'query_database', 'list_workspace_pages']:
                        self._cache[cache_key] = {
                            'data': result,
                            'timestamp': time.time()
                        }
                    return result
                elif response and "error" in response:
                    raise Exception(f"Tool error: {response['error']['message']}")
            else:
                # Use real server
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": self._next_message_id()
                }
                
                async with self.session.post(self.endpoint, json=request) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "result" in result:
                            tool_result = result["result"]
                            # Cache the result for read operations
                            if tool_name in ['get_page', 'search_pages', 'query_database', 'list_workspace_pages']:
                                self._cache[cache_key] = {
                                    'data': tool_result,
                                    'timestamp': time.time()
                                }
                            return tool_result
                        elif "error" in result:
                            raise Exception(f"Tool error: {result['error']['message']}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"[ERROR] Failed to call tool {tool_name}: {e}")
            raise
    
    # Convenience methods for common Notion operations
    
    async def create_page(self, title: str, content: str = "", 
                         parent_id: Optional[str] = None, 
                         properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new page in Notion."""
        args = {
            "title": title,
            "content": content
        }
        if parent_id:
            args["parent_id"] = parent_id
        if properties:
            args["properties"] = properties
        
        return await self.call_tool("create_page", args)
    
    async def get_page(self, page_id: str, format: str = "markdown") -> Dict[str, Any]:
        """Get page content and properties."""
        return await self.call_tool("get_page", {
            "page_id": page_id,
            "format": format
        })
    
    async def update_page(self, page_id: str, title: Optional[str] = None,
                         content: Optional[str] = None,
                         properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update page content or properties."""
        args = {"page_id": page_id}
        if title:
            args["title"] = title
        if content:
            args["content"] = content
        if properties:
            args["properties"] = properties
        
        return await self.call_tool("update_page", args)
    
    async def delete_page(self, page_id: str) -> Dict[str, Any]:
        """Archive/delete a page."""
        return await self.call_tool("delete_page", {"page_id": page_id})
    
    async def search_pages(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for pages in the workspace."""
        return await self.call_tool("search_pages", {
            "query": query,
            "limit": limit
        })
    
    async def create_database(self, title: str, properties: Dict[str, Any],
                            parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new database."""
        args = {
            "title": title,
            "properties": properties
        }
        if parent_id:
            args["parent_id"] = parent_id
        
        return await self.call_tool("create_database", args)
    
    async def query_database(self, database_id: str, filter: Optional[Dict[str, Any]] = None,
                           sorts: Optional[List[Dict[str, Any]]] = None,
                           limit: int = 10) -> Dict[str, Any]:
        """Query a database with filters."""
        args = {
            "database_id": database_id,
            "limit": limit
        }
        if filter:
            args["filter"] = filter
        if sorts:
            args["sorts"] = sorts
        
        return await self.call_tool("query_database", args)
    
    async def create_database_record(self, database_id: str, 
                                   properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record in a database."""
        return await self.call_tool("create_database_record", {
            "database_id": database_id,
            "properties": properties
        })
    
    async def append_block(self, page_id: str, block_type: str, content: str) -> Dict[str, Any]:
        """Append a block to a page."""
        return await self.call_tool("append_block", {
            "page_id": page_id,
            "block_type": block_type,
            "content": content
        })
    
    async def list_workspace_pages(self, limit: int = 20) -> Dict[str, Any]:
        """List all pages in the workspace."""
        return await self.call_tool("list_workspace_pages", {"limit": limit})
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register Notion tools to the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        try:
            for tool in self.tools:
                tool_data = {
                    "id": f"notion_{tool['name']}",
                    "name": tool["name"],
                    "type": "mcp",
                    "endpoint": self.endpoint if not self.use_mock else "mock://notion",
                    "capabilities": {
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("inputSchema", {}),
                        "category": "productivity",
                        "domain": "documentation",
                        "semantic_tags": ["notion", "notes", "documentation", "knowledge-base"]
                    },
                    "server_id": self.server_name,
                    "client": self
                }
                
                # Use synchronous add_tool method
                registry.add_tool(tool_data)
                logger.info(f"[REGISTER] Registered tool: {tool_data['id']}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to register tools: {e}")
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        logger.info("[CACHE] Cache cleared")


async def main():
    """Test the Notion MCP client."""
    # Test with mock server
    client = NotionMCPClient()
    
    try:
        # Connect to mock server
        connected = await client.connect(use_mock=True)
        if not connected:
            logger.error("Failed to connect")
            return
        
        # Test create page
        logger.info("\n=== Testing Create Page ===")
        page = await client.create_page(
            title="My Test Page",
            content="# Welcome\n\nThis is a test page created via Notion MCP.\n\n## Features\n- Easy integration\n- Full API access\n- Markdown support",
            properties={"Tags": ["test", "mcp"]}
        )
        logger.info(f"Created page: {page['id']}")
        
        # Test get page
        logger.info("\n=== Testing Get Page ===")
        content = await client.get_page(page['id'])
        logger.info(f"Page content: {content['content'][:100]}...")
        
        # Test update page
        logger.info("\n=== Testing Update Page ===")
        updated = await client.update_page(
            page['id'],
            content=content['content'] + "\n\n### Updated\nThis content was updated!"
        )
        logger.info(f"Page updated: {updated['success']}")
        
        # Test search
        logger.info("\n=== Testing Search ===")
        results = await client.search_pages("test")
        logger.info(f"Found {len(results['results'])} pages")
        
        # Test create database
        logger.info("\n=== Testing Create Database ===")
        db = await client.create_database(
            title="Project Tasks",
            properties={
                "Name": {"type": "title"},
                "Status": {"type": "select", "options": ["Todo", "In Progress", "Done"]},
                "Due Date": {"type": "date"}
            }
        )
        logger.info(f"Created database: {db['id']}")
        
        # Test create database record
        logger.info("\n=== Testing Create Record ===")
        record = await client.create_database_record(
            db['id'],
            properties={
                "Name": "Implement Notion integration",
                "Status": "In Progress",
                "Due Date": "2024-12-31"
            }
        )
        logger.info(f"Created record: {record['id']}")
        
        # Test append block
        logger.info("\n=== Testing Append Block ===")
        block = await client.append_block(
            page['id'],
            "heading_2",
            "Additional Section"
        )
        logger.info(f"Appended block: {block['id']}")
        
        # Test list workspace pages
        logger.info("\n=== Testing List Pages ===")
        pages = await client.list_workspace_pages(limit=5)
        logger.info(f"Found {len(pages['pages'])} pages in workspace")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())