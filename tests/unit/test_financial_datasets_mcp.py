#!/usr/bin/env python3
"""
Unit tests for Financial Datasets MCP Client.

Tests individual components and methods of the Financial Datasets MCP client
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

from src.tools.financial_datasets_mcp import FinancialDatasetsMCPClient
from src.tools.mock_financial_datasets_mcp import MockFinancialDatasetsMCPServer
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry

logger = get_logger(__name__)


class TestFinancialDatasetsMCPClient:
    """Unit tests for Financial Datasets MCP Client."""
    
    @pytest.fixture
    def client(self):
        """Create Financial Datasets MCP client instance."""
        return FinancialDatasetsMCPClient(
            api_key="test_api_key",
            endpoint="https://test.financialdatasets.ai/sse"
        )
    
    @pytest.fixture
    def mock_aiohttp_session(self):
        """Create mock aiohttp session."""
        session = Mock(spec=aiohttp.ClientSession)
        session.closed = False
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_response(self):
        """Create mock aiohttp response."""
        response = Mock()
        response.status = 200
        response.text = AsyncMock(return_value="Mock response")
        response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "Financial Datasets MCP Server",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": True
                }
            },
            "id": 1
        })
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock()
        return response
    
    @pytest.fixture
    def mock_registry(self, tmp_path):
        """Create mock tool registry."""
        registry_path = tmp_path / "test_registry.db"
        return ToolRegistry(str(registry_path))
    
    def test_initialization(self, client):
        """Test client initialization."""
        assert client.api_key == "test_api_key"
        assert client.endpoint == "https://test.financialdatasets.ai/sse"
        assert client.server_name == "financial_datasets"
        assert not client.connected
        assert client.session is None
        assert len(client.tools) == 0
        assert client._cache_ttl == 300
        assert client._message_id == 0
    
    def test_initialization_with_env_vars(self, monkeypatch):
        """Test initialization with environment variables."""
        monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "env_api_key")
        
        client = FinancialDatasetsMCPClient()
        
        assert client.api_key == "env_api_key"
        assert client.endpoint == "https://mcp.financialdatasets.ai/sse"
    
    def test_next_message_id(self, client):
        """Test message ID generation."""
        id1 = client._next_message_id()
        id2 = client._next_message_id()
        id3 = client._next_message_id()
        
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
    
    def test_cache_key_generation(self, client):
        """Test cache key generation."""
        method = "get_stock_price"
        params = {"symbol": "AAPL", "date": "2024-01-01"}
        
        key1 = client._get_cache_key(method, params)
        key2 = client._get_cache_key(method, params)
        
        # Same inputs should produce same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
        
        # Different params should produce different key
        params2 = {"symbol": "MSFT", "date": "2024-01-01"}
        key3 = client._get_cache_key(method, params2)
        assert key3 != key1
    
    def test_cache_validity(self, client):
        """Test cache validity checking."""
        # No cache data
        assert not client._is_cache_valid({})
        assert not client._is_cache_valid(None)
        
        # Fresh cache
        fresh_cache = {
            'data': {'test': 'data'},
            'timestamp': time.time()
        }
        assert client._is_cache_valid(fresh_cache)
        
        # Expired cache
        expired_cache = {
            'data': {'test': 'data'},
            'timestamp': time.time() - 400  # More than 300s TTL
        }
        assert not client._is_cache_valid(expired_cache)
    
    @pytest.mark.asyncio
    async def test_connect_with_mock_server(self, client):
        """Test connecting with mock server."""
        result = await client.connect(use_mock=True)
        
        assert result is True
        assert client.connected is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert isinstance(client.mock_server, MockFinancialDatasetsMCPServer)
        assert client.capabilities is not None
        assert "tools" in client.capabilities
    
    @pytest.mark.asyncio
    async def test_connect_with_real_server(self, client, mock_aiohttp_session, mock_response):
        """Test connecting with real server (mocked)."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            mock_aiohttp_session.post = Mock(return_value=mock_response)
            
            result = await client.connect(use_mock=False)
            
            assert result is True
            assert client.connected is True
            assert client.use_mock is False
            assert client.session is not None
            assert client.capabilities is not None
    
    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        """Test connecting to real server without API key."""
        client = FinancialDatasetsMCPClient(api_key=None)
        
        result = await client.connect(use_mock=False)
        
        assert result is False
        assert not client.connected
    
    @pytest.mark.asyncio
    async def test_connect_with_server_error(self, client, mock_aiohttp_session):
        """Test connection failure with server error."""
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            mock_aiohttp_session.post = Mock(return_value=mock_response)
            
            result = await client.connect(use_mock=False)
            
            assert result is False
            assert not client.connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_aiohttp_session):
        """Test disconnecting from server."""
        # First connect
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            client.session = mock_aiohttp_session
            client.connected = True
            
            await client.disconnect()
            
            assert not client.connected
            mock_aiohttp_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_tools_mock_server(self, client):
        """Test discovering tools from mock server."""
        await client.connect(use_mock=True)
        
        tools = await client.discover_tools()
        
        assert len(tools) > 0
        assert len(tools) == 7  # Mock server has 7 tools
        
        tool_names = [tool["name"] for tool in tools]
        assert "get_stock_price" in tool_names
        assert "get_income_statement" in tool_names
        assert "get_balance_sheet" in tool_names
        assert "get_cash_flow_statement" in tool_names
        assert "get_company_news" in tool_names
        assert "get_crypto_price" in tool_names
        assert "search_companies" in tool_names
    
    @pytest.mark.asyncio
    async def test_discover_tools_real_server(self, client, mock_aiohttp_session):
        """Test discovering tools from real server (mocked)."""
        tools_response = {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {"name": "get_stock_price", "description": "Get stock price"},
                    {"name": "get_income_statement", "description": "Get income statement"}
                ]
            },
            "id": 2
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=tools_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        with patch.object(client, 'session', mock_aiohttp_session):
            mock_aiohttp_session.post = Mock(return_value=mock_response)
            client.connected = True
            client.use_mock = False
            
            tools = await client.discover_tools()
            
            assert len(tools) == 2
            assert tools[0]["name"] == "get_stock_price"
            assert tools[1]["name"] == "get_income_statement"
    
    @pytest.mark.asyncio
    async def test_call_tool_mock_server(self, client):
        """Test calling a tool with mock server."""
        await client.connect(use_mock=True)
        
        result = await client.call_tool("get_stock_price", {"symbol": "AAPL"})
        
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "price" in result
        assert isinstance(result["price"], (int, float))
    
    @pytest.mark.asyncio
    async def test_call_tool_real_server(self, client, mock_aiohttp_session):
        """Test calling a tool with real server (mocked)."""
        tool_response = {
            "jsonrpc": "2.0",
            "result": {
                "symbol": "AAPL",
                "price": 185.50,
                "change": 2.34,
                "change_percent": 1.28
            },
            "id": 3
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=tool_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        with patch.object(client, 'session', mock_aiohttp_session):
            mock_aiohttp_session.post = Mock(return_value=mock_response)
            client.connected = True
            client.use_mock = False
            
            result = await client.call_tool("get_stock_price", {"symbol": "AAPL"})
            
            assert result is not None
            assert result["symbol"] == "AAPL"
            assert result["price"] == 185.50
    
    @pytest.mark.asyncio
    async def test_call_tool_with_cache(self, client):
        """Test tool calling with cache functionality."""
        await client.connect(use_mock=True)
        
        # First call - should hit server
        result1 = await client.call_tool("get_stock_price", {"symbol": "AAPL"})
        
        # Second call - should hit cache
        result2 = await client.call_tool("get_stock_price", {"symbol": "AAPL"})
        
        # Results should be identical
        assert result1 == result2
        
        # Check cache
        cache_key = client._get_cache_key("get_stock_price", {"symbol": "AAPL"})
        assert cache_key in client._cache
    
    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, client):
        """Test error handling in tool calls."""
        # Test with mock server - calling unknown tool
        await client.connect(use_mock=True)
        
        # Call an unknown tool
        with pytest.raises(Exception) as exc_info:
            await client.call_tool("unknown_tool", {"param": "value"})
        
        assert "Unknown tool" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_stock_price(self, client):
        """Test get_stock_price convenience method."""
        await client.connect(use_mock=True)
        
        # Test without date
        result = await client.get_stock_price("MSFT")
        assert result["symbol"] == "MSFT"
        assert "price" in result
        
        # Test with date
        result = await client.get_stock_price("MSFT", "2024-01-01")
        assert result["symbol"] == "MSFT"
        assert "price" in result
    
    @pytest.mark.asyncio
    async def test_get_income_statement(self, client):
        """Test get_income_statement method."""
        await client.connect(use_mock=True)
        
        result = await client.get_income_statement("AAPL")
        
        assert result["symbol"] == "AAPL"
        assert "revenue" in result
        assert "gross_profit" in result
        assert "operating_income" in result
        assert "net_income" in result
        assert "eps" in result
    
    @pytest.mark.asyncio
    async def test_get_balance_sheet(self, client):
        """Test get_balance_sheet method."""
        await client.connect(use_mock=True)
        
        result = await client.get_balance_sheet("GOOGL")
        
        assert result["symbol"] == "GOOGL"
        assert "total_assets" in result
        assert "total_liabilities" in result
        assert "total_equity" in result
        assert "cash" in result
        assert "debt" in result
    
    @pytest.mark.asyncio
    async def test_get_cash_flow(self, client):
        """Test get_cash_flow method."""
        await client.connect(use_mock=True)
        
        result = await client.get_cash_flow("TSLA")
        
        assert result["symbol"] == "TSLA"
        assert "operating_cash_flow" in result
        assert "investing_cash_flow" in result
        assert "financing_cash_flow" in result
        assert "free_cash_flow" in result
    
    @pytest.mark.asyncio
    async def test_get_company_news(self, client):
        """Test get_company_news method."""
        await client.connect(use_mock=True)
        
        # Test with default limit
        news = await client.get_company_news("AMZN")
        assert isinstance(news, list)
        assert len(news) == 10  # Default limit
        
        # Test with custom limit
        news = await client.get_company_news("AMZN", limit=5)
        assert len(news) == 5
        
        # Check news item structure
        if news:
            item = news[0]
            assert "title" in item
            assert "summary" in item
            assert "source" in item
            assert "timestamp" in item
            assert "url" in item
            assert "sentiment" in item
    
    @pytest.mark.asyncio
    async def test_get_crypto_price(self, client):
        """Test get_crypto_price method."""
        await client.connect(use_mock=True)
        
        # Test with default currency
        result = await client.get_crypto_price("BTC")
        assert result["symbol"] == "BTC"
        assert result["currency"] == "USD"
        assert "price" in result
        assert "change_24h" in result
        
        # Test with custom currency
        result = await client.get_crypto_price("ETH", "EUR")
        assert result["symbol"] == "ETH"
        assert result["currency"] == "EUR"
        assert "price" in result
    
    @pytest.mark.asyncio
    async def test_search_companies(self, client):
        """Test search_companies method."""
        await client.connect(use_mock=True)
        
        companies = await client.search_companies("Apple")
        
        assert isinstance(companies, list)
        assert len(companies) > 0
        
        # Check company structure
        if companies:
            company = companies[0]
            assert "symbol" in company
            assert "name" in company
            assert "exchange" in company
    
    def test_register_tools_to_registry(self, client):
        """Test registering tools to registry."""
        # Use a mock registry
        mock_registry = Mock(spec=ToolRegistry)
        
        # Set up mock tools
        client.tools = [
            {
                "name": "get_stock_price",
                "description": "Get stock price",
                "inputSchema": {"type": "object"}
            },
            {
                "name": "get_income_statement",
                "description": "Get income statement",
                "inputSchema": {"type": "object"}
            }
        ]
        client.use_mock = True
        
        # Register tools
        client.register_tools_to_registry(mock_registry)
        
        # Check that register_tool was called for each tool
        assert mock_registry.register_tool.call_count == 2
        
        # Check first tool registration
        first_call = mock_registry.register_tool.call_args_list[0][0][0]
        assert first_call["id"] == "financial_datasets_get_stock_price"
        assert first_call["name"] == "get_stock_price"
        assert first_call["type"] == "mcp"
        assert first_call["server_id"] == "financial_datasets"
        assert first_call["capabilities"]["category"] == "finance"
        assert first_call["capabilities"]["domain"] == "financial_data"
        
        # Check second tool registration
        second_call = mock_registry.register_tool.call_args_list[1][0][0]
        assert second_call["id"] == "financial_datasets_get_income_statement"
        assert second_call["name"] == "get_income_statement"


class TestMockFinancialDatasetsMCPServer:
    """Unit tests for Mock Financial Datasets MCP Server."""
    
    @pytest.fixture
    def mock_server(self):
        """Create mock server instance."""
        return MockFinancialDatasetsMCPServer()
    
    def test_mock_server_initialization(self, mock_server):
        """Test mock server initialization."""
        assert mock_server.server_name == "mock_financial_datasets"
        assert len(mock_server.tools) == 7
        assert mock_server.mock_data is not None
        assert "stock_prices" in mock_server.mock_data
        assert "income_statements" in mock_server.mock_data
        assert "balance_sheets" in mock_server.mock_data
        assert "cash_flows" in mock_server.mock_data
        assert "crypto_prices" in mock_server.mock_data
        assert "companies" in mock_server.mock_data
    
    @pytest.mark.asyncio
    async def test_handle_initialize(self, mock_server):
        """Test initialization request handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {
                    "name": "TestClient",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert response["result"]["protocolVersion"] == "1.0"
        assert "serverInfo" in response["result"]
        assert "capabilities" in response["result"]
        assert response["result"]["capabilities"]["tools"] is True
    
    @pytest.mark.asyncio
    async def test_handle_tools_list(self, mock_server):
        """Test tools list request handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 7
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_stock_price(self, mock_server):
        """Test stock price tool call."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_stock_price",
                "arguments": {"symbol": "AAPL"}
            },
            "id": 3
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        result = response["result"]
        assert result["symbol"] == "AAPL"
        assert result["price"] == 182.52  # Mock data value
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, mock_server):
        """Test handling unknown method."""
        request = {
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "id": 4
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, mock_server):
        """Test handling unknown tool."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {}
            },
            "id": 5
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "Unknown tool" in response["error"]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])