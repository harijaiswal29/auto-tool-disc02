"""
Comprehensive tests for Search MCP Client
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path (go up 2 levels from tests/unit/)
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.search_mcp import SearchMCPClient
from src.tools.mock_search_mcp import MockSearchMCPServer
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestSearchMCPClient:
    """Test suite for Search MCP Client."""
    
    @pytest.fixture
    async def client(self):
        """Create a Search MCP client instance."""
        config = {
            "api_key": "test_key",
            "max_results": 10
        }
        return SearchMCPClient(config)
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock Search MCP server instance."""
        return MockSearchMCPServer({"api_key": "test_key"})
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.server_name == "search"
        assert client.config["api_key"] == "test_key"
        assert client.tools == []
        assert not client.use_mock
    
    @pytest.mark.asyncio
    async def test_connect_with_mock(self, client):
        """Test connecting to mock server."""
        connected = await client.connect(use_mock=True)
        assert connected
        assert client.use_mock
        assert client.mock_server is not None
        assert len(client.tools) == 5  # Should have 5 search tools
        
        # Verify tool names
        tool_names = [tool['name'] for tool in client.tools]
        assert 'web_search' in tool_names
        assert 'code_search' in tool_names
        assert 'doc_search' in tool_names
        assert 'news_search' in tool_names
        assert 'scholarly_search' in tool_names
    
    @pytest.mark.asyncio
    async def test_web_search(self, client):
        """Test web search functionality."""
        await client.connect(use_mock=True)
        
        result = await client.web_search(
            "Model Context Protocol",
            {"num_results": 5, "language": "en"}
        )
        
        assert result['success']
        assert 'result' in result
        assert 'total_results' in result['result']
        assert 'results' in result['result']
        assert len(result['result']['results']) <= 5
    
    @pytest.mark.asyncio
    async def test_code_search(self, client):
        """Test code search functionality."""
        await client.connect(use_mock=True)
        
        result = await client.code_search(
            "async def connect",
            language="python"
        )
        
        assert result['success']
        assert 'result' in result
        assert 'language' in result['result']
        assert result['result']['language'] == "python"
    
    @pytest.mark.asyncio
    async def test_documentation_search(self, client):
        """Test documentation search functionality."""
        await client.connect(use_mock=True)
        
        result = await client.documentation_search(
            "asyncio python",
            source="python"
        )
        
        assert result['success']
        assert 'result' in result
        assert 'source' in result['result']
        assert result['result']['source'] == "python"
    
    @pytest.mark.asyncio
    async def test_news_search(self, client):
        """Test news search functionality."""
        await client.connect(use_mock=True)
        
        result = await client.news_search(
            "artificial intelligence",
            {"from": "2024-01-01", "to": "2024-12-31"}
        )
        
        assert result['success']
        assert 'result' in result
        assert 'date_range' in result['result']
    
    @pytest.mark.asyncio
    async def test_scholarly_search(self, client):
        """Test scholarly search functionality."""
        await client.connect(use_mock=True)
        
        result = await client.scholarly_search(
            "machine learning",
            fields=["computer science", "artificial intelligence"]
        )
        
        assert result['success']
        assert 'result' in result
        assert 'fields' in result['result']
        assert len(result['result']['fields']) == 2
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self, client):
        """Test calling a non-existent tool."""
        await client.connect(use_mock=True)
        
        result = await client.call_tool("non_existent_tool", {"query": "test"})
        
        assert not result['success']
        assert 'error' in result
        assert 'Tool not found' in result['error']
    
    @pytest.mark.asyncio
    async def test_register_tools_to_registry(self, client, tmp_path):
        """Test registering Search tools to the tool registry."""
        await client.connect(use_mock=True)
        
        # Create a temporary registry
        registry_path = tmp_path / "test_registry.db"
        registry = ToolRegistry(str(registry_path))
        
        # Register tools
        client.register_tools_to_registry(registry)
        
        # Verify all tools are registered
        search_tools = registry.list_tools("search")
        assert len(search_tools) == 5
        
        # Verify tool IDs
        tool_ids = [tool['id'] for tool in search_tools]
        assert 'search.web_search' in tool_ids
        assert 'search.code_search' in tool_ids
        assert 'search.doc_search' in tool_ids
        assert 'search.news_search' in tool_ids
        assert 'search.scholarly_search' in tool_ids
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnecting from server."""
        await client.connect(use_mock=True)
        await client.disconnect()
        # Should complete without errors
    
    @pytest.mark.asyncio
    async def test_fallback_to_mock(self, client):
        """Test automatic fallback to mock server."""
        # Mock the real server connection to fail
        with patch.object(client, '_send_message') as mock_send:
            mock_send.side_effect = Exception("Connection failed")
            
            # Should fail with real server
            connected = await client.connect(use_mock=False)
            assert not connected
    
    @pytest.mark.asyncio
    async def test_execution_time_tracking(self, client):
        """Test that execution time is tracked."""
        await client.connect(use_mock=True)
        
        result = await client.web_search("test query")
        
        assert 'execution_time' in result
        assert result['execution_time'] > 0


class TestMockSearchMCPServer:
    """Test suite for Mock Search MCP Server."""
    
    @pytest.fixture
    async def server(self):
        """Create a mock server instance."""
        return MockSearchMCPServer({"api_key": "test_key"})
    
    @pytest.mark.asyncio
    async def test_initialization_request(self, server):
        """Test handling initialization request."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {"name": "TestClient", "version": "0.1.0"}
            },
            "id": 1
        }
        
        response = await server.handle_request(request)
        
        assert response['id'] == 1
        assert 'result' in response
        assert response['result']['protocolVersion'] == "1.0"
        assert response['result']['capabilities']['tools'] == True
    
    @pytest.mark.asyncio
    async def test_tools_list_request(self, server):
        """Test handling tools list request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        response = await server.handle_request(request)
        
        assert response['id'] == 2
        assert 'result' in response
        assert len(response['result']['tools']) == 5
    
    @pytest.mark.asyncio
    async def test_web_search_tool_call(self, server):
        """Test web search tool execution."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {
                    "query": "Model Context Protocol",
                    "num_results": 3
                }
            },
            "id": 3
        }
        
        response = await server.handle_request(request)
        
        assert response['id'] == 3
        assert 'result' in response
        assert 'total_results' in response['result']
        assert response['result']['query'] == "Model Context Protocol"
    
    @pytest.mark.asyncio
    async def test_unknown_method(self, server):
        """Test handling unknown method."""
        request = {
            "jsonrpc": "2.0",
            "method": "unknown_method",
            "params": {},
            "id": 4
        }
        
        response = await server.handle_request(request)
        
        assert response['id'] == 4
        assert 'error' in response
        assert 'Unknown method' in response['error']['message']
    
    @pytest.mark.asyncio
    async def test_unknown_tool(self, server):
        """Test handling unknown tool call."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {}
            },
            "id": 5
        }
        
        response = await server.handle_request(request)
        
        assert response['id'] == 5
        assert 'error' in response
        assert 'Unknown tool' in response['error']['message']


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a complete integration scenario."""
    logger.info("=" * 60)
    logger.info("[INTEGRATION TEST] Testing Search MCP Integration")
    logger.info("=" * 60)
    
    # Create client
    client = SearchMCPClient({
        "api_key": "integration_test_key",
        "max_results": 5
    })
    
    # Connect with mock
    connected = await client.connect(use_mock=True)
    assert connected
    
    # Perform various searches
    searches = [
        ("web", client.web_search("Python programming", {"num_results": 3})),
        ("code", client.code_search("def main", language="python")),
        ("docs", client.documentation_search("unittest", source="python")),
        ("news", client.news_search("technology trends")),
        ("scholar", client.scholarly_search("neural networks"))
    ]
    
    results = {}
    for search_type, search_coro in searches:
        result = await search_coro
        assert result['success'], f"{search_type} search failed"
        results[search_type] = result
        logger.info(f"[{search_type.upper()}] Success: {result['success']}")
    
    # Test with registry
    registry = ToolRegistry("data/test_integration_registry.db")  # Use file database
    client.register_tools_to_registry(registry)
    
    # Verify registration
    tools = registry.list_tools("search")
    assert len(tools) == 5
    
    # Record some usage
    for tool in tools:
        registry.record_usage(tool['id'], True, 0.5, "integration_test")
    
    # Check performance
    for tool in tools:
        perf = registry.get_tool_performance(tool['id'])
        assert perf['overall_score'] == 1.0  # All successful
    
    await client.disconnect()
    
    logger.info("[INTEGRATION TEST] All tests passed!")


if __name__ == "__main__":
    # Run integration test directly
    asyncio.run(test_integration_scenario())