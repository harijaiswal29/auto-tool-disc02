"""
Zerodha MCP Client Wrapper

This module provides a client wrapper for the remote Zerodha Kite MCP server,
enabling trading operations through the Model Context Protocol.
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
from src.tools.mock_zerodha_mcp import MockZerodhaMCPServer

logger = get_logger(__name__)

class ZerodhaMCPClient:
    """
    Client wrapper for remote Zerodha Kite MCP server.
    
    This client enables trading operations like:
    - Portfolio management (holdings, positions)
    - Order management (place, modify, cancel orders)
    - Market data (quotes, LTP, historical data)
    - Account information (margins, funds)
    - Trade and order book access
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 access_token: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize Zerodha MCP client.
        
        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
            access_token: User's access token
            endpoint: Remote MCP server endpoint
        """
        self.api_key = api_key or os.environ.get('ZERODHA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ZERODHA_API_SECRET')
        self.access_token = access_token or os.environ.get('ZERODHA_ACCESS_TOKEN')
        self.endpoint = endpoint or "https://mcp.kite.trade/sse" 
        self.server_name = "zerodha"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.connected = False
        self.mock_server: Optional[MockZerodhaMCPServer] = None
        self.use_mock = False
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 60  # 1 minute for market data
        
        logger.info(f"[INIT] Zerodha MCP Client initialized with endpoint: {self.endpoint}")
    
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
        Connect to the Zerodha MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.use_mock = use_mock
            
            if use_mock:
                logger.info("[CONNECTING] Using mock Zerodha MCP server...")
                self.mock_server = MockZerodhaMCPServer()
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "ZerodhaMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info("[SUCCESS] Connected to mock Zerodha MCP server")
                    
                    # Discover tools
                    await self.discover_tools()
                    self.connected = True
                    return True
            else:
                # Connect to real remote server
                if not self.api_key or not self.access_token:
                    logger.error("[ERROR] No API key or access token provided for Zerodha")
                    return False
                
                logger.info(f"[CONNECTING] Connecting to remote Zerodha MCP server at {self.endpoint}")
                
                # Create aiohttp session
                self.session = aiohttp.ClientSession(
                    headers={
                        "X-Kite-Version": "3",
                        "Authorization": f"token {self.api_key}:{self.access_token}",
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
                            "name": "ZerodhaMCPClient",
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
                            logger.info("[SUCCESS] Connected to remote Zerodha MCP server")
                            
                            # Discover tools
                            await self.discover_tools()
                            self.connected = True
                            return True
                    else:
                        error_text = await response.text()
                        logger.error(f"[ERROR] Failed to connect: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Zerodha MCP server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.connected = False
            logger.info("[DISCONNECT] Disconnected from Zerodha MCP server")
        except Exception as e:
            logger.error(f"[ERROR] Error during disconnect: {e}")
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover available tools from the Zerodha MCP server.
        
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
            # Check cache for market data operations
            cache_key = self._get_cache_key(tool_name, arguments)
            if tool_name in ['get_quote', 'get_ltp', 'get_margins']:
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
    
    # Convenience methods for common trading operations
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get user's equity holdings."""
        return await self.call_tool("get_holdings", {})
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get user's open positions."""
        return await self.call_tool("get_positions", {})
    
    async def place_order(self, exchange: str, symbol: str, transaction_type: str,
                         quantity: int, product: str, order_type: str,
                         price: Optional[float] = None, trigger_price: Optional[float] = None) -> Dict[str, Any]:
        """Place a buy/sell order."""
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": product,
            "order_type": order_type
        }
        
        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
            
        return await self.call_tool("place_order", params)
    
    async def modify_order(self, order_id: str, quantity: Optional[int] = None,
                          price: Optional[float] = None, trigger_price: Optional[float] = None) -> Dict[str, Any]:
        """Modify an existing order."""
        params = {"order_id": order_id}
        
        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
            
        return await self.call_tool("modify_order", params)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        return await self.call_tool("cancel_order", {"order_id": order_id})
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get list of all orders for the day."""
        return await self.call_tool("get_orders", {})
    
    async def get_trades(self) -> List[Dict[str, Any]]:
        """Get list of all executed trades."""
        return await self.call_tool("get_trades", {})
    
    async def get_quote(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get market quote for an instrument."""
        return await self.call_tool("get_quote", {
            "exchange": exchange,
            "symbol": symbol
        })
    
    async def get_ltp(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """Get last traded price for instruments."""
        return await self.call_tool("get_ltp", {"instruments": instruments})
    
    async def get_historical_data(self, instrument_token: str, from_date: str,
                                 to_date: str, interval: str) -> Dict[str, Any]:
        """Get historical candle data."""
        return await self.call_tool("get_historical_data", {
            "instrument_token": instrument_token,
            "from_date": from_date,
            "to_date": to_date,
            "interval": interval
        })
    
    async def get_margins(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """Get account margins and funds."""
        params = {}
        if segment:
            params["segment"] = segment
        return await self.call_tool("get_margins", params)
    
    async def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of tradeable instruments."""
        params = {}
        if exchange:
            params["exchange"] = exchange
        return await self.call_tool("get_instruments", params)
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register Zerodha tools to the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        try:
            for tool in self.tools:
                tool_data = {
                    "id": f"zerodha_{tool['name']}",
                    "name": tool["name"],
                    "type": "mcp",
                    "endpoint": self.endpoint if not self.use_mock else "mock://zerodha",
                    "capabilities": {
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("inputSchema", {}),
                        "category": "trading",
                        "domain": "finance"
                    },
                    "server_id": self.server_name,
                    "server_type": "zerodha",
                    "client": self
                }
                
                # Use synchronous register_tool method
                registry.register_tool(tool_data)
                logger.info(f"[REGISTER] Registered tool: {tool_data['id']}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to register tools: {e}")


async def main():
    """Test the Zerodha MCP client."""
    # Test with mock server
    client = ZerodhaMCPClient()
    
    try:
        # Connect
        connected = await client.connect(use_mock=True)
        if not connected:
            logger.error("Failed to connect")
            return
        
        # Test holdings
        logger.info("\n=== Testing Holdings ===")
        holdings = await client.get_holdings()
        logger.info(f"Found {len(holdings)} holdings")
        for holding in holdings[:2]:  # Show first 2
            logger.info(f"  {holding['tradingsymbol']}: {holding['quantity']} @ {holding['average_price']}")
        
        # Test positions
        logger.info("\n=== Testing Positions ===")
        positions = await client.get_positions()
        logger.info(f"Net positions: {len(positions.get('net', []))}")
        logger.info(f"Day positions: {len(positions.get('day', []))}")
        
        # Test quote
        logger.info("\n=== Testing Quote ===")
        quote = await client.get_quote("NSE", "RELIANCE")
        logger.info(f"RELIANCE LTP: {quote.get('last_price', 'N/A')}")
        
        # Test order placement
        logger.info("\n=== Testing Order Placement ===")
        order = await client.place_order(
            exchange="NSE",
            symbol="INFY",
            transaction_type="BUY",
            quantity=10,
            product="CNC",
            order_type="LIMIT",
            price=1450.00
        )
        logger.info(f"Order placed: {order.get('order_id', 'N/A')}")
        
        # Test margins
        logger.info("\n=== Testing Margins ===")
        margins = await client.get_margins()
        equity_margins = margins.get("equity", {})
        available = equity_margins.get("available", {})
        logger.info(f"Available cash: {available.get('cash', 'N/A')}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())