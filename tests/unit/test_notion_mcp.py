"""
Unit tests for Notion MCP client.

Tests individual components and methods of the Notion MCP client.
"""

import pytest
import asyncio
import json
import hashlib
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.notion_mcp import NotionMCPClient
from src.tools.mock_notion_mcp import MockNotionMCPServer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestNotionMCPClient:
    """Unit tests for Notion MCP client."""
    
    @pytest.fixture
    def client(self):
        """Create a Notion MCP client."""
        return NotionMCPClient(
            integration_token="test-token",
            endpoint="https://test.notion.com/mcp/v1"
        )
    
    def test_initialization(self, client):
        """Test client initialization."""
        assert client.integration_token == "test-token"
        assert client.endpoint == "https://test.notion.com/mcp/v1"
        assert client.server_name == "notion"
        assert not client.connected
        assert client.session is None
        assert len(client.tools) == 0
        assert client._cache_ttl == 300
    
    def test_initialization_with_env_vars(self, monkeypatch):
        """Test initialization with environment variables."""
        monkeypatch.setenv("NOTION_INTEGRATION_TOKEN", "env-token")
        monkeypatch.setenv("NOTION_MCP_ENDPOINT", "https://env.notion.com/mcp/v1")
        
        client = NotionMCPClient()
        
        assert client.integration_token == "env-token"
        assert client.endpoint == "https://env.notion.com/mcp/v1"
    
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
        method = "get_page"
        params = {"page_id": "123", "format": "markdown"}
        
        key1 = client._get_cache_key(method, params)
        key2 = client._get_cache_key(method, params)
        
        # Same inputs should produce same key
        assert key1 == key2
        
        # Different params should produce different key
        params2 = {"page_id": "456", "format": "markdown"}
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
    async def test_connect_mock_server(self, client):
        """Test connecting to mock server."""
        result = await client.connect(use_mock=True)
        
        assert result is True
        assert client.connected is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert isinstance(client.mock_server, MockNotionMCPServer)
        assert len(client.tools) > 0
    
    @pytest.mark.asyncio
    async def test_connect_without_token(self, client):
        """Test connecting without integration token."""
        client.integration_token = None
        result = await client.connect(use_mock=False)
        
        assert result is False
        assert not client.connected
    
    @pytest.mark.asyncio
    async def test_connect_real_server_success(self, client):
        """Test successful connection to real server."""
        # Mock the aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "capabilities": {
                    "tools": True,
                    "notion": {"pages": True, "databases": True}
                }
            }
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Mock discover_tools to return sample tools
            with patch.object(client, 'discover_tools', return_value=[
                {"name": "create_page", "description": "Create a page"}
            ]):
                result = await client.connect(use_mock=False)
        
        assert result is True
        assert client.connected is True
        assert client.session is not None
    
    @pytest.mark.asyncio
    async def test_connect_real_server_failure(self, client):
        """Test failed connection to real server."""
        # Mock the aiohttp session to return error
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await client.connect(use_mock=False)
        
        assert result is False
        assert not client.connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnecting from server."""
        # Create a mock session
        client.session = AsyncMock()
        client.session.closed = False
        client.connected = True
        
        await client.disconnect()
        
        assert not client.connected
        client.session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_tools_mock(self, client):
        """Test tool discovery from mock server."""
        await client.connect(use_mock=True)
        
        tools = await client.discover_tools()
        
        assert len(tools) > 0
        assert all('name' in tool for tool in tools)
        assert all('description' in tool for tool in tools)
        assert any(tool['name'] == 'create_page' for tool in tools)
    
    @pytest.mark.asyncio
    async def test_call_tool_with_cache(self, client):
        """Test calling tool with caching."""
        await client.connect(use_mock=True)
        
        # First call - not cached
        result1 = await client.call_tool("get_page", {"page_id": "test-123"})
        
        # Second call - should be cached
        result2 = await client.call_tool("get_page", {"page_id": "test-123"})
        
        assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, client):
        """Test error handling in tool calls."""
        await client.connect(use_mock=True)
        
        # Mock server to return error
        client.mock_server.handle_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32602,
                "message": "Invalid parameters"
            }
        })
        
        with pytest.raises(Exception) as exc_info:
            await client.call_tool("invalid_tool", {})
        
        assert "Invalid parameters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_page_convenience_method(self, client):
        """Test create_page convenience method."""
        await client.connect(use_mock=True)
        
        # Mock call_tool
        expected_result = {"id": "page-123", "title": "Test"}
        client.call_tool = AsyncMock(return_value=expected_result)
        
        result = await client.create_page(
            title="Test",
            content="Content",
            parent_id="parent-123",
            properties={"Tag": "test"}
        )
        
        assert result == expected_result
        client.call_tool.assert_called_once_with("create_page", {
            "title": "Test",
            "content": "Content",
            "parent_id": "parent-123",
            "properties": {"Tag": "test"}
        })
    
    @pytest.mark.asyncio
    async def test_get_page_convenience_method(self, client):
        """Test get_page convenience method."""
        await client.connect(use_mock=True)
        
        expected_result = {"id": "page-123", "content": "Test content"}
        client.call_tool = AsyncMock(return_value=expected_result)
        
        result = await client.get_page("page-123", format="json")
        
        assert result == expected_result
        client.call_tool.assert_called_once_with("get_page", {
            "page_id": "page-123",
            "format": "json"
        })
    
    @pytest.mark.asyncio
    async def test_search_pages_convenience_method(self, client):
        """Test search_pages convenience method."""
        await client.connect(use_mock=True)
        
        expected_result = {"results": [{"id": "page-1", "title": "Test"}]}
        client.call_tool = AsyncMock(return_value=expected_result)
        
        result = await client.search_pages("test query", limit=5)
        
        assert result == expected_result
        client.call_tool.assert_called_once_with("search_pages", {
            "query": "test query",
            "limit": 5
        })
    
    def test_register_tools_to_registry(self, client):
        """Test registering tools to registry."""
        # Set up mock tools
        client.tools = [
            {
                "name": "create_page",
                "description": "Create a page",
                "inputSchema": {"type": "object"}
            },
            {
                "name": "get_page",
                "description": "Get a page",
                "inputSchema": {"type": "object"}
            }
        ]
        client.use_mock = True
        
        # Mock registry
        mock_registry = Mock()
        
        # Register tools
        client.register_tools_to_registry(mock_registry)
        
        # Verify add_tool was called for each tool
        assert mock_registry.add_tool.call_count == 2
        
        # Check first call
        first_call = mock_registry.add_tool.call_args_list[0][0][0]
        assert first_call["id"] == "notion_create_page"
        assert first_call["name"] == "create_page"
        assert first_call["type"] == "mcp"
        assert "productivity" in first_call["capabilities"]["category"]
    
    def test_clear_cache(self, client):
        """Test clearing cache."""
        # Add some data to cache
        client._cache = {
            "key1": {"data": "test1", "timestamp": time.time()},
            "key2": {"data": "test2", "timestamp": time.time()}
        }
        
        assert len(client._cache) == 2
        
        client.clear_cache()
        
        assert len(client._cache) == 0
    
    @pytest.mark.asyncio
    async def test_database_operations(self, client):
        """Test database-related convenience methods."""
        await client.connect(use_mock=True)
        
        # Mock call_tool
        client.call_tool = AsyncMock()
        
        # Test create_database
        await client.create_database(
            title="Test DB",
            properties={"Name": {"type": "title"}},
            parent_id="parent-123"
        )
        
        client.call_tool.assert_called_with("create_database", {
            "title": "Test DB",
            "properties": {"Name": {"type": "title"}},
            "parent_id": "parent-123"
        })
        
        # Test query_database
        await client.query_database(
            "db-123",
            filter={"Status": "Active"},
            sorts=[{"property": "Name", "direction": "ascending"}],
            limit=20
        )
        
        client.call_tool.assert_called_with("query_database", {
            "database_id": "db-123",
            "limit": 20,
            "filter": {"Status": "Active"},
            "sorts": [{"property": "Name", "direction": "ascending"}]
        })
    
    @pytest.mark.asyncio
    async def test_real_server_call_tool(self, client):
        """Test calling tool on real server."""
        client.connected = True
        client.use_mock = False
        
        # Mock session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"success": True, "data": "test"}
        })
        
        client.session = AsyncMock()
        client.session.post.return_value.__aenter__.return_value = mock_response
        
        result = await client.call_tool("test_tool", {"param": "value"})
        
        assert result == {"success": True, "data": "test"}
    
    @pytest.mark.asyncio
    async def test_real_server_call_tool_http_error(self, client):
        """Test handling HTTP errors from real server."""
        client.connected = True
        client.use_mock = False
        
        # Mock session to return HTTP error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        client.session = AsyncMock()
        client.session.post.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(Exception) as exc_info:
            await client.call_tool("test_tool", {})
        
        assert "HTTP 500" in str(exc_info.value)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])