#!/usr/bin/env python3
"""
Unit tests for Zerodha MCP Client.

Tests individual components and methods of the Zerodha MCP client
with extensive mocking to ensure isolation from external dependencies.
"""

import pytest
import asyncio
import json
import time
import hashlib
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime, timedelta
import aiohttp
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.zerodha_mcp import ZerodhaMCPClient
from src.tools.mock_zerodha_mcp import MockZerodhaMCPServer
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry

logger = get_logger(__name__)


class TestZerodhaMCPClient:
    """Unit tests for Zerodha MCP Client."""
    
    @pytest.fixture
    def client(self):
        """Create Zerodha MCP client instance."""
        return ZerodhaMCPClient(
            api_key="test_api_key",
            api_secret="test_api_secret",
            access_token="test_access_token",
            endpoint="https://test.kite.trade/sse"
        )
    
    @pytest.fixture
    def mock_aiohttp_session(self):
        """Create mock aiohttp session."""
        session = Mock(spec=aiohttp.ClientSession)
        session.closed = False
        session.close = AsyncMock()
        
        # Create a mock that returns an async context manager
        post_context = AsyncMock()
        post_context.__aenter__ = AsyncMock()
        post_context.__aexit__ = AsyncMock()
        session.post = Mock(return_value=post_context)
        
        return session
    
    @pytest.fixture
    def mock_response(self):
        """Create mock aiohttp response."""
        response = Mock()
        response.status = 200
        response.text = AsyncMock(return_value="Mock error")
        response.json = AsyncMock()
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock()
        return response
    
    @pytest.fixture
    def mock_server(self):
        """Create mock Zerodha MCP server."""
        return MockZerodhaMCPServer()
    
    def test_client_initialization(self):
        """Test client initialization with various parameter combinations."""
        # Test with all parameters
        client = ZerodhaMCPClient(
            api_key="key1",
            api_secret="secret1",
            access_token="token1",
            endpoint="https://custom.endpoint"
        )
        assert client.api_key == "key1"
        assert client.api_secret == "secret1"
        assert client.access_token == "token1"
        assert client.endpoint == "https://custom.endpoint"
        assert client.server_name == "zerodha"
        assert client.connected is False
        assert client._message_id == 0
        
        # Test with environment variables
        with patch.dict(os.environ, {
            'ZERODHA_API_KEY': 'env_key',
            'ZERODHA_API_SECRET': 'env_secret',
            'ZERODHA_ACCESS_TOKEN': 'env_token'
        }):
            client = ZerodhaMCPClient()
            assert client.api_key == "env_key"
            assert client.api_secret == "env_secret"
            assert client.access_token == "env_token"
            assert client.endpoint == "https://mcp.kite.trade/sse"
    
    def test_next_message_id(self, client):
        """Test message ID generation."""
        assert client._next_message_id() == 1
        assert client._next_message_id() == 2
        assert client._next_message_id() == 3
        assert client._message_id == 3
    
    def test_cache_key_generation(self, client):
        """Test cache key generation for requests."""
        # Test with simple params
        key1 = client._get_cache_key("get_quote", {"exchange": "NSE", "symbol": "RELIANCE"})
        key2 = client._get_cache_key("get_quote", {"symbol": "RELIANCE", "exchange": "NSE"})
        assert key1 == key2  # Order shouldn't matter
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
        
        # Test with different params
        key3 = client._get_cache_key("get_quote", {"exchange": "BSE", "symbol": "RELIANCE"})
        assert key1 != key3
        
        # Test with different methods
        key4 = client._get_cache_key("get_ltp", {"exchange": "NSE", "symbol": "RELIANCE"})
        assert key1 != key4
    
    def test_cache_validation(self, client):
        """Test cache validation logic."""
        # Test with no cached data
        assert client._is_cache_valid({}) is False
        assert client._is_cache_valid(None) is False
        
        # Test with fresh cached data
        fresh_data = {
            'timestamp': time.time(),
            'data': {'test': 'value'}
        }
        assert client._is_cache_valid(fresh_data) is True
        
        # Test with expired cached data
        expired_data = {
            'timestamp': time.time() - 120,  # 2 minutes old
            'data': {'test': 'value'}
        }
        assert client._is_cache_valid(expired_data) is False
        
        # Test with missing timestamp
        invalid_data = {'data': {'test': 'value'}}
        assert client._is_cache_valid(invalid_data) is False
    
    @pytest.mark.asyncio
    async def test_connect_with_mock_server(self, client):
        """Test connecting with mock server."""
        with patch('src.tools.zerodha_mcp.MockZerodhaMCPServer') as mock_server_class:
            mock_instance = Mock()
            mock_server_class.return_value = mock_instance
            
            # Mock the handle_request method to return proper initialization response
            mock_instance.handle_request = AsyncMock()
            mock_instance.handle_request.side_effect = [
                # First call: initialization
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "1.0",
                        "serverInfo": {"name": "Mock Zerodha", "version": "0.1.0"},
                        "capabilities": {"tools": True, "resources": False, "prompts": False}
                    },
                    "id": 1
                },
                # Second call: tools/list for discover_tools
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "tools": [{"name": "get_holdings", "description": "Get holdings"}]
                    },
                    "id": 2
                }
            ]
            
            connected = await client.connect(use_mock=True)
            
            assert connected is True
            assert client.connected is True
            assert client.use_mock is True
            assert client.mock_server is not None
            assert client.capabilities.get("tools") is True
            
            # Verify initialization request
            assert mock_instance.handle_request.call_count >= 1
            call_args = mock_instance.handle_request.call_args_list[0][0][0]
            assert call_args["method"] == "initialize"
            assert call_args["params"]["protocolVersion"] == "1.0"
    
    @pytest.mark.asyncio
    async def test_connect_with_real_server(self, client, mock_aiohttp_session, mock_response):
        """Test connecting with real server."""
        # Set up response side effects for initialization and tools discovery
        mock_response.status = 200
        mock_response.json.side_effect = [
            # First call: initialization
            {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "1.0",
                    "capabilities": {"tools": True}
                },
                "id": 1
            },
            # Second call: tools/list
            {
                "jsonrpc": "2.0",
                "result": {
                    "tools": [{"name": "get_holdings", "description": "Get holdings"}]
                },
                "id": 2
            }
        ]
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = mock_aiohttp_session
            
            # Set up the async context manager for post
            post_context = mock_aiohttp_session.post.return_value
            post_context.__aenter__.return_value = mock_response
            
            connected = await client.connect(use_mock=False)
            
            assert connected is True
            assert client.connected is True
            assert client.use_mock is False
            assert client.session == mock_aiohttp_session
            assert client.capabilities.get("tools") is True
            
            # Verify session headers
            session_call = mock_session_class.call_args
            headers = session_call[1]['headers']
            assert headers['X-Kite-Version'] == "3"
            assert headers['Authorization'] == "token test_api_key:test_access_token"
            assert headers['Content-Type'] == "application/json"
    
    @pytest.mark.asyncio
    async def test_connect_without_credentials(self):
        """Test connecting without API credentials."""
        client = ZerodhaMCPClient(api_key=None, access_token=None)
        connected = await client.connect(use_mock=False)
        
        assert connected is False
        assert client.connected is False
    
    @pytest.mark.asyncio
    async def test_connect_server_error(self, client, mock_aiohttp_session, mock_response):
        """Test connection failure with server error."""
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = mock_aiohttp_session
            mock_aiohttp_session.post.return_value = mock_response
            
            connected = await client.connect(use_mock=False)
            
            assert connected is False
            assert client.connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_aiohttp_session):
        """Test disconnecting from server."""
        client.session = mock_aiohttp_session
        client.connected = True
        
        await client.disconnect()
        
        assert client.connected is False
        mock_aiohttp_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_tools_mock(self, client, mock_server):
        """Test tool discovery with mock server."""
        client.use_mock = True
        client.mock_server = mock_server
        
        mock_tools = [
            {"name": "get_holdings", "description": "Get holdings"},
            {"name": "place_order", "description": "Place order"}
        ]
        
        mock_server.handle_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "result": {"tools": mock_tools},
            "id": 1
        })
        
        tools = await client.discover_tools()
        
        assert len(tools) == 2
        assert tools == mock_tools
        assert client.tools == mock_tools
        
        # Verify request
        mock_server.handle_request.assert_called_once()
        call_args = mock_server.handle_request.call_args[0][0]
        assert call_args["method"] == "tools/list"
    
    @pytest.mark.asyncio
    async def test_discover_tools_real(self, client, mock_aiohttp_session, mock_response):
        """Test tool discovery with real server."""
        client.session = mock_aiohttp_session
        client.use_mock = False
        
        mock_tools = [
            {"name": "get_quote", "description": "Get quote"},
            {"name": "get_margins", "description": "Get margins"}
        ]
        
        mock_response.status = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"tools": mock_tools},
            "id": 1
        }
        
        # Set up the async context manager for post
        post_context = mock_aiohttp_session.post.return_value
        post_context.__aenter__.return_value = mock_response
        
        tools = await client.discover_tools()
        
        assert len(tools) == 2
        assert tools == mock_tools
        assert client.tools == mock_tools
    
    @pytest.mark.asyncio
    async def test_call_tool_with_cache_hit(self, client):
        """Test calling tool with cache hit."""
        client.use_mock = True
        tool_name = "get_quote"
        arguments = {"exchange": "NSE", "symbol": "RELIANCE"}
        
        # Pre-populate cache
        cache_key = client._get_cache_key(tool_name, arguments)
        cached_data = {
            'data': {'last_price': 2500.50},
            'timestamp': time.time()
        }
        client._cache[cache_key] = cached_data
        
        result = await client.call_tool(tool_name, arguments)
        
        assert result == {'last_price': 2500.50}
    
    @pytest.mark.asyncio
    async def test_call_tool_with_cache_miss(self, client, mock_server):
        """Test calling tool with cache miss."""
        client.use_mock = True
        client.mock_server = mock_server
        
        tool_name = "get_quote"
        arguments = {"exchange": "NSE", "symbol": "INFY"}
        expected_result = {'last_price': 1450.75}
        
        mock_server.handle_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "result": expected_result,
            "id": 1
        })
        
        result = await client.call_tool(tool_name, arguments)
        
        assert result == expected_result
        
        # Verify cache was populated
        cache_key = client._get_cache_key(tool_name, arguments)
        assert cache_key in client._cache
        assert client._cache[cache_key]['data'] == expected_result
    
    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, client, mock_server):
        """Test tool call error handling."""
        client.use_mock = True
        client.mock_server = mock_server
        
        mock_server.handle_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "Invalid parameters"},
            "id": 1
        })
        
        with pytest.raises(Exception) as exc_info:
            await client.call_tool("invalid_tool", {})
        
        assert "Tool error: Invalid parameters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_holdings(self, client):
        """Test get_holdings convenience method."""
        expected_holdings = [
            {"tradingsymbol": "RELIANCE", "quantity": 10}
        ]
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_holdings)):
            holdings = await client.get_holdings()
            
            assert holdings == expected_holdings
            client.call_tool.assert_called_once_with("get_holdings", {})
    
    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test get_positions convenience method."""
        expected_positions = {
            "net": [{"tradingsymbol": "INFY", "quantity": 20}],
            "day": []
        }
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_positions)):
            positions = await client.get_positions()
            
            assert positions == expected_positions
            client.call_tool.assert_called_once_with("get_positions", {})
    
    @pytest.mark.asyncio
    async def test_place_order(self, client):
        """Test place_order convenience method."""
        expected_result = {"order_id": "231229001234"}
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_result)):
            result = await client.place_order(
                exchange="NSE",
                symbol="TCS",
                transaction_type="BUY",
                quantity=5,
                product="CNC",
                order_type="LIMIT",
                price=3200.00
            )
            
            assert result == expected_result
            client.call_tool.assert_called_once_with("place_order", {
                "exchange": "NSE",
                "symbol": "TCS",
                "transaction_type": "BUY",
                "quantity": 5,
                "product": "CNC",
                "order_type": "LIMIT",
                "price": 3200.00
            })
    
    @pytest.mark.asyncio
    async def test_place_order_market(self, client):
        """Test place_order for market orders without price."""
        expected_result = {"order_id": "231229001235"}
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_result)):
            result = await client.place_order(
                exchange="NSE",
                symbol="WIPRO",
                transaction_type="SELL",
                quantity=50,
                product="MIS",
                order_type="MARKET"
            )
            
            assert result == expected_result
            # Verify price is not included for market orders
            call_args = client.call_tool.call_args[0][1]
            assert "price" not in call_args
            assert "trigger_price" not in call_args
    
    @pytest.mark.asyncio
    async def test_modify_order(self, client):
        """Test modify_order convenience method."""
        expected_result = {"order_id": "231229001234"}
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_result)):
            result = await client.modify_order(
                order_id="231229001234",
                quantity=10,
                price=3195.00
            )
            
            assert result == expected_result
            client.call_tool.assert_called_once_with("modify_order", {
                "order_id": "231229001234",
                "quantity": 10,
                "price": 3195.00
            })
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test cancel_order convenience method."""
        expected_result = {"order_id": "231229001234"}
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_result)):
            result = await client.cancel_order("231229001234")
            
            assert result == expected_result
            client.call_tool.assert_called_once_with("cancel_order", {
                "order_id": "231229001234"
            })
    
    @pytest.mark.asyncio
    async def test_get_orders(self, client):
        """Test get_orders convenience method."""
        expected_orders = [
            {"order_id": "123", "status": "OPEN"},
            {"order_id": "124", "status": "COMPLETE"}
        ]
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_orders)):
            orders = await client.get_orders()
            
            assert orders == expected_orders
            client.call_tool.assert_called_once_with("get_orders", {})
    
    @pytest.mark.asyncio
    async def test_get_trades(self, client):
        """Test get_trades convenience method."""
        expected_trades = [
            {"trade_id": "T123", "order_id": "123"}
        ]
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_trades)):
            trades = await client.get_trades()
            
            assert trades == expected_trades
            client.call_tool.assert_called_once_with("get_trades", {})
    
    @pytest.mark.asyncio
    async def test_get_quote(self, client):
        """Test get_quote convenience method."""
        expected_quote = {
            "last_price": 2520.75,
            "volume": 1234567
        }
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_quote)):
            quote = await client.get_quote("NSE", "RELIANCE")
            
            assert quote == expected_quote
            client.call_tool.assert_called_once_with("get_quote", {
                "exchange": "NSE",
                "symbol": "RELIANCE"
            })
    
    @pytest.mark.asyncio
    async def test_get_ltp(self, client):
        """Test get_ltp convenience method."""
        instruments = ["NSE:INFY", "NSE:TCS"]
        expected_ltp = {
            "NSE:INFY": {"last_price": 1455.30},
            "NSE:TCS": {"last_price": 3180.50}
        }
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_ltp)):
            ltp = await client.get_ltp(instruments)
            
            assert ltp == expected_ltp
            client.call_tool.assert_called_once_with("get_ltp", {
                "instruments": instruments
            })
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, client):
        """Test get_historical_data convenience method."""
        expected_data = {
            "candles": [
                {"date": "2023-12-25", "open": 2500, "high": 2520, "low": 2490, "close": 2510}
            ]
        }
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_data)):
            data = await client.get_historical_data(
                instrument_token="738561",
                from_date="2023-12-25",
                to_date="2023-12-29",
                interval="day"
            )
            
            assert data == expected_data
            client.call_tool.assert_called_once_with("get_historical_data", {
                "instrument_token": "738561",
                "from_date": "2023-12-25",
                "to_date": "2023-12-29",
                "interval": "day"
            })
    
    @pytest.mark.asyncio
    async def test_get_margins(self, client):
        """Test get_margins convenience method."""
        expected_margins = {
            "equity": {
                "available": {"cash": 145678.50}
            }
        }
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_margins)):
            # Test without segment
            margins = await client.get_margins()
            assert margins == expected_margins
            client.call_tool.assert_called_with("get_margins", {})
            
            # Test with segment
            margins = await client.get_margins(segment="equity")
            client.call_tool.assert_called_with("get_margins", {"segment": "equity"})
    
    @pytest.mark.asyncio
    async def test_get_instruments(self, client):
        """Test get_instruments convenience method."""
        expected_instruments = [
            {"tradingsymbol": "RELIANCE", "exchange": "NSE"},
            {"tradingsymbol": "INFY", "exchange": "NSE"}
        ]
        
        with patch.object(client, 'call_tool', AsyncMock(return_value=expected_instruments)):
            # Test without exchange
            instruments = await client.get_instruments()
            assert instruments == expected_instruments
            client.call_tool.assert_called_with("get_instruments", {})
            
            # Test with exchange
            instruments = await client.get_instruments(exchange="NSE")
            client.call_tool.assert_called_with("get_instruments", {"exchange": "NSE"})
    
    def test_register_tools_to_registry(self, client):
        """Test registering tools to registry."""
        # Mock tools
        client.tools = [
            {
                "name": "get_holdings",
                "description": "Get user holdings",
                "inputSchema": {"type": "object"}
            },
            {
                "name": "place_order",
                "description": "Place an order",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
        client.use_mock = False
        client.endpoint = "https://test.endpoint"
        
        # Create mock registry
        mock_registry = Mock(spec=ToolRegistry)
        mock_registry.add_tool = Mock()
        
        # Register tools
        client.register_tools_to_registry(mock_registry)
        
        # Verify calls
        assert mock_registry.add_tool.call_count == 2
        
        # Check first tool registration
        first_call = mock_registry.add_tool.call_args_list[0][0][0]
        assert first_call["id"] == "zerodha_get_holdings"
        assert first_call["name"] == "get_holdings"
        assert first_call["type"] == "mcp"
        assert first_call["endpoint"] == "https://test.endpoint"
        assert first_call["server_id"] == "zerodha"
        assert first_call["client"] == client
        assert first_call["capabilities"]["category"] == "trading"
        assert first_call["capabilities"]["domain"] == "finance"
        
        # Check second tool registration
        second_call = mock_registry.add_tool.call_args_list[1][0][0]
        assert second_call["id"] == "zerodha_place_order"
        assert second_call["name"] == "place_order"
    
    def test_register_tools_with_mock_endpoint(self, client):
        """Test tool registration with mock server endpoint."""
        client.tools = [{"name": "test_tool", "description": "Test"}]
        client.use_mock = True
        
        mock_registry = Mock(spec=ToolRegistry)
        mock_registry.add_tool = Mock()
        
        client.register_tools_to_registry(mock_registry)
        
        # Verify mock endpoint is used
        call_args = mock_registry.add_tool.call_args[0][0]
        assert call_args["endpoint"] == "mock://zerodha"
    
    @pytest.mark.asyncio
    async def test_exception_handling_in_connect(self, client):
        """Test exception handling during connection."""
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            connected = await client.connect(use_mock=False)
            
            assert connected is False
            assert client.connected is False
    
    @pytest.mark.asyncio
    async def test_exception_handling_in_discover_tools(self, client):
        """Test exception handling during tool discovery."""
        client.use_mock = True
        client.mock_server = Mock()
        client.mock_server.handle_request = AsyncMock(side_effect=Exception("Discovery failed"))
        
        tools = await client.discover_tools()
        
        assert tools == []
        assert client.tools == []
    
    @pytest.mark.asyncio
    async def test_exception_handling_in_call_tool(self, client):
        """Test exception handling during tool call."""
        client.use_mock = True
        client.mock_server = Mock()
        client.mock_server.handle_request = AsyncMock(side_effect=Exception("Tool call failed"))
        
        with pytest.raises(Exception) as exc_info:
            await client.call_tool("test_tool", {})
        
        assert "Tool call failed" in str(exc_info.value)
    
    def test_exception_handling_in_register_tools(self, client):
        """Test exception handling during tool registration."""
        client.tools = [{"name": "test_tool"}]
        
        mock_registry = Mock(spec=ToolRegistry)
        mock_registry.add_tool = Mock(side_effect=Exception("Registration failed"))
        
        # Should not raise, just log error
        client.register_tools_to_registry(mock_registry)
        
        # Verify attempt was made
        assert mock_registry.add_tool.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])