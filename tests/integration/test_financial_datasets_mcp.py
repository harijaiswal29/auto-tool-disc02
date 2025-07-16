"""
Integration tests for Financial Datasets MCP Server
"""

import pytest
import asyncio
import os
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.financial_datasets_mcp import FinancialDatasetsMCPClient
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TestFinancialDatasetsMCPIntegration:
    """Integration tests for Financial Datasets MCP client."""
    
    @pytest.fixture
    async def client(self):
        """Create a Financial Datasets MCP client instance."""
        client = FinancialDatasetsMCPClient()
        yield client
        # Cleanup
        if client.connected:
            await client.disconnect()
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create a temporary tool registry."""
        registry_path = tmp_path / "test_registry.db"
        return ToolRegistry(str(registry_path))
    
    @pytest.mark.asyncio
    async def test_connect_mock_server(self, client):
        """Test connecting to mock Financial Datasets server."""
        # Connect using mock
        connected = await client.connect(use_mock=True)
        assert connected is True
        assert client.connected is True
        assert client.use_mock is True
        assert client.mock_server is not None
        
        # Check capabilities
        assert client.capabilities is not None
        assert "tools" in client.capabilities
    
    @pytest.mark.asyncio
    async def test_discover_tools(self, client):
        """Test discovering available financial tools."""
        # Connect first
        await client.connect(use_mock=True)
        
        # Discover tools
        tools = await client.discover_tools()
        assert len(tools) > 0
        
        # Check for expected tools
        tool_names = [tool["name"] for tool in tools]
        assert "get_stock_price" in tool_names
        assert "get_income_statement" in tool_names
        assert "get_balance_sheet" in tool_names
        assert "get_cash_flow_statement" in tool_names
        assert "get_company_news" in tool_names
        assert "get_crypto_price" in tool_names
        assert "search_companies" in tool_names
    
    @pytest.mark.asyncio
    async def test_get_stock_price(self, client):
        """Test getting stock price."""
        await client.connect(use_mock=True)
        
        # Test current price
        result = await client.get_stock_price("AAPL")
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "price" in result
        assert isinstance(result["price"], (int, float))
        assert result["price"] > 0
        assert "change" in result
        assert "change_percent" in result
        assert "volume" in result
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_get_income_statement(self, client):
        """Test getting income statement."""
        await client.connect(use_mock=True)
        
        result = await client.get_income_statement("MSFT")
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "MSFT"
        assert "revenue" in result
        assert "gross_profit" in result
        assert "operating_income" in result
        assert "net_income" in result
        assert "eps" in result
        assert "period" in result
        assert "currency" in result
    
    @pytest.mark.asyncio
    async def test_get_balance_sheet(self, client):
        """Test getting balance sheet."""
        await client.connect(use_mock=True)
        
        result = await client.get_balance_sheet("AAPL")
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "total_assets" in result
        assert "total_liabilities" in result
        assert "total_equity" in result
        assert "cash" in result
        assert "debt" in result
        assert "period" in result
        assert "currency" in result
    
    @pytest.mark.asyncio
    async def test_get_cash_flow(self, client):
        """Test getting cash flow statement."""
        await client.connect(use_mock=True)
        
        result = await client.get_cash_flow("GOOGL")
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "GOOGL"
        assert "operating_cash_flow" in result
        assert "investing_cash_flow" in result
        assert "financing_cash_flow" in result
        assert "free_cash_flow" in result
        assert "period" in result
        assert "currency" in result
    
    @pytest.mark.asyncio
    async def test_get_company_news(self, client):
        """Test getting company news."""
        await client.connect(use_mock=True)
        
        news = await client.get_company_news("TSLA", limit=5)
        assert isinstance(news, list)
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
        """Test getting cryptocurrency price."""
        await client.connect(use_mock=True)
        
        # Test BTC price in USD
        result = await client.get_crypto_price("BTC", "USD")
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "BTC"
        assert "currency" in result
        assert result["currency"] == "USD"
        assert "price" in result
        assert isinstance(result["price"], (int, float))
        assert result["price"] > 0
        assert "change_24h" in result
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_search_companies(self, client):
        """Test searching for companies."""
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
    
    @pytest.mark.asyncio
    async def test_register_tools_to_registry(self, client, registry):
        """Test registering tools to the tool registry."""
        await client.connect(use_mock=True)
        await client.discover_tools()
        
        # Register tools
        client.register_tools_to_registry(registry)
        
        # Check tools were registered
        tools = registry.get_all_tools()
        assert len(tools) > 0
        
        # Check a specific tool
        stock_tool = next((t for t in tools if t["name"] == "get_stock_price"), None)
        assert stock_tool is not None
        assert stock_tool["type"] == "mcp"
        assert stock_tool["server_id"] == "financial_datasets"
        assert "finance" in stock_tool["capabilities"]["category"]
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, client):
        """Test that caching works for financial data."""
        await client.connect(use_mock=True)
        
        # First call - should hit the server
        result1 = await client.get_stock_price("AAPL")
        
        # Second call - should hit cache
        result2 = await client.get_stock_price("AAPL")
        
        # Results should be identical
        assert result1 == result2
        
        # Different symbol - should not hit cache
        result3 = await client.get_stock_price("MSFT")
        assert result3["symbol"] != result1["symbol"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling for invalid requests."""
        await client.connect(use_mock=True)
        
        # Test with empty symbol
        with pytest.raises(Exception):
            await client.call_tool("get_stock_price", {})
    
    @pytest.mark.asyncio
    async def test_real_server_connection_without_api_key(self, client):
        """Test that real server connection fails without API key."""
        # Remove any API key from environment
        with patch.dict(os.environ, {'FINANCIAL_DATASETS_API_KEY': ''}, clear=True):
            client = FinancialDatasetsMCPClient(api_key=None)
            connected = await client.connect(use_mock=False)
            assert connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnecting from server."""
        await client.connect(use_mock=True)
        assert client.connected is True
        
        await client.disconnect()
        assert client.connected is False


@pytest.mark.asyncio
async def test_financial_datasets_with_mcp_integration():
    """Test Financial Datasets integration with MCP Integration class."""
    from src.core.mcp_integration import MCPIntegration
    
    # Create MCP integration
    integration = MCPIntegration()
    
    # Add Financial Datasets server using mock
    success = await integration.add_financial_datasets_server(
        server_id="test_financial",
        use_mock=True
    )
    assert success is True
    
    # Check server was added
    assert "test_financial" in integration.servers
    assert integration.servers["test_financial"]["type"] == "financial_datasets"
    assert integration.servers["test_financial"]["is_mock"] is True
    
    # Check tools were registered
    tools = integration.registry.get_all_tools()
    financial_tools = [t for t in tools if t["server_id"] == "financial_datasets"]
    assert len(financial_tools) > 0
    
    # Execute a tool through integration
    result = await integration.execute_tool(
        "financial_datasets_get_stock_price",
        {"symbol": "AAPL"}
    )
    assert result is not None
    assert "symbol" in result
    assert result["symbol"] == "AAPL"
    
    # Cleanup
    await integration.shutdown()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])