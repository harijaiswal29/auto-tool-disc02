"""
Financial Datasets MCP Client Wrapper

This module provides a client wrapper for the remote Financial Datasets MCP server,
enabling financial data operations through the Model Context Protocol.

IMPORTANT: The real Financial Datasets MCP server uses OAuth 2.1 authentication,
not API keys. The current implementation attempts Bearer token authentication
which is not supported by the real server. Use the mock server for testing.

For real server integration, OAuth 2.1 flow needs to be implemented:
1. Redirect to authorization URL
2. Handle OAuth callback
3. Exchange authorization code for access token
4. Use access token for API requests

See: https://docs.financialdatasets.ai/mcp-server
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
from src.tools.mock_financial_datasets_mcp import MockFinancialDatasetsMCPServer

logger = get_logger(__name__)

class FinancialDatasetsMCPClient:
    """
    Client wrapper for remote Financial Datasets MCP server.
    
    This client enables financial data operations like:
    - Retrieving income statements
    - Fetching balance sheets
    - Getting cash flow statements
    - Accessing stock prices (current and historical)
    - Getting company news
    - Cryptocurrency data access
    """
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize Financial Datasets MCP client.
        
        Args:
            api_key: Financial Datasets API key (Note: Real server requires OAuth, not API keys)
            endpoint: Remote MCP server endpoint
            
        OAuth Implementation Guide:
        To implement OAuth 2.1 for real server:
        1. Add oauth_client_id and oauth_client_secret parameters
        2. Add methods:
           - get_authorization_url() -> str: Generate OAuth authorization URL
           - handle_oauth_callback(code: str) -> dict: Exchange code for token
           - refresh_access_token() -> str: Refresh expired tokens
        3. Store access_token and refresh_token
        4. Update session headers to use OAuth access token:
           headers["Authorization"] = f"Bearer {self.access_token}"
        5. Implement token refresh logic when token expires
        """
        self.api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY')
        self.endpoint = endpoint or "https://mcp.financialdatasets.ai/sse"
        self.server_name = "financial_datasets"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.connected = False
        self.mock_server: Optional[MockFinancialDatasetsMCPServer] = None
        self.use_mock = False
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info(f"[INIT] Financial Datasets MCP Client initialized with endpoint: {self.endpoint}")
    
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
        Connect to the Financial Datasets MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.use_mock = use_mock
            
            if use_mock:
                logger.info("[CONNECTING] Using mock Financial Datasets MCP server...")
                self.mock_server = MockFinancialDatasetsMCPServer()
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "FinancialDatasetsMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info("[SUCCESS] Connected to mock Financial Datasets MCP server")
                    
                    # Discover tools
                    await self.discover_tools()
                    self.connected = True
                    return True
            else:
                # Connect to real remote server
                if not self.api_key:
                    logger.error("[ERROR] No API key provided for Financial Datasets")
                    return False
                
                logger.info(f"[CONNECTING] Connecting to remote Financial Datasets MCP server at {self.endpoint}")
                
                # Create aiohttp session
                # NOTE: This Bearer token approach doesn't work with the real server
                # Real server requires OAuth 2.1 authentication flow
                # Current implementation will result in 401 "Invalid token format" errors
                self.session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                # Send initialization request
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "FinancialDatasetsMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                async with self.session.post(self.endpoint, json=init_request) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "result" in result:
                            self.capabilities = result["result"].get("capabilities", {})
                            logger.info("[SUCCESS] Connected to remote Financial Datasets MCP server")
                            
                            # Discover tools
                            await self.discover_tools()
                            self.connected = True
                            return True
                    else:
                        error_text = await response.text()
                        logger.error(f"[ERROR] Failed to connect: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Financial Datasets MCP server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.connected = False
            logger.info("[DISCONNECT] Disconnected from Financial Datasets MCP server")
        except Exception as e:
            logger.error(f"[ERROR] Error during disconnect: {e}")
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover available tools from the Financial Datasets MCP server.
        
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
            # Check cache for GET-like operations
            cache_key = self._get_cache_key(tool_name, arguments)
            if tool_name in ['get_stock_price', 'get_income_statement', 'get_balance_sheet']:
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
                    # Cache the result
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
                            # Cache the result
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
    
    # Convenience methods for common financial data operations
    
    async def get_stock_price(self, symbol: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get stock price for a symbol."""
        args = {"symbol": symbol}
        if date:
            args["date"] = date
        return await self.call_tool("get_stock_price", args)
    
    async def get_income_statement(self, symbol: str, period: Optional[str] = "latest") -> Dict[str, Any]:
        """Get income statement for a company."""
        return await self.call_tool("get_income_statement", {
            "symbol": symbol,
            "period": period
        })
    
    async def get_balance_sheet(self, symbol: str, period: Optional[str] = "latest") -> Dict[str, Any]:
        """Get balance sheet for a company."""
        return await self.call_tool("get_balance_sheet", {
            "symbol": symbol,
            "period": period
        })
    
    async def get_cash_flow(self, symbol: str, period: Optional[str] = "latest") -> Dict[str, Any]:
        """Get cash flow statement for a company."""
        return await self.call_tool("get_cash_flow_statement", {
            "symbol": symbol,
            "period": period
        })
    
    async def get_company_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest news for a company."""
        result = await self.call_tool("get_company_news", {
            "symbol": symbol,
            "limit": limit
        })
        return result.get("news", [])
    
    async def get_crypto_price(self, symbol: str, currency: str = "USD") -> Dict[str, Any]:
        """Get cryptocurrency price."""
        return await self.call_tool("get_crypto_price", {
            "symbol": symbol,
            "currency": currency
        })
    
    async def search_companies(self, query: str) -> List[Dict[str, Any]]:
        """Search for companies by name or ticker."""
        result = await self.call_tool("search_companies", {"query": query})
        return result.get("companies", [])
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register Financial Datasets tools to the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        try:
            for tool in self.tools:
                tool_data = {
                    "id": f"financial_datasets_{tool['name']}",
                    "name": tool["name"],
                    "type": "mcp",
                    "endpoint": self.endpoint if not self.use_mock else "mock://financial_datasets",
                    "capabilities": {
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("inputSchema", {}),
                        "category": "finance",
                        "domain": "financial_data"
                    },
                    "server_id": self.server_name,
                    "server_type": "financial_datasets",
                    "client": self
                }
                
                # Use synchronous register_tool method
                registry.register_tool(tool_data)
                logger.info(f"[REGISTER] Registered tool: {tool_data['id']}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to register tools: {e}")


async def main():
    """Test the Financial Datasets MCP client."""
    # Test with mock server
    client = FinancialDatasetsMCPClient()
    
    try:
        # Connect
        connected = await client.connect(use_mock=True)
        if not connected:
            logger.error("Failed to connect")
            return
        
        # Test stock price
        logger.info("\n=== Testing Stock Price ===")
        price = await client.get_stock_price("AAPL")
        logger.info(f"Apple stock price: ${price.get('price', 'N/A')}")
        
        # Test income statement
        logger.info("\n=== Testing Income Statement ===")
        income = await client.get_income_statement("MSFT")
        logger.info(f"Microsoft revenue: ${income.get('revenue', 'N/A')}")
        
        # Test company search
        logger.info("\n=== Testing Company Search ===")
        companies = await client.search_companies("Tesla")
        logger.info(f"Found {len(companies)} companies")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())