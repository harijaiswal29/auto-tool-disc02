"""
Integration tests for Zerodha MCP Server
"""

import pytest
import asyncio
import os
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add the src directory to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.zerodha_mcp import ZerodhaMCPClient
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TestZerodhaMCPIntegration:
    """Integration tests for Zerodha MCP client."""
    
    @pytest.fixture
    async def client(self):
        """Create a Zerodha MCP client instance."""
        client = ZerodhaMCPClient()
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
        """Test connecting to mock Zerodha server."""
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
        """Test discovering available trading tools."""
        # Connect first
        await client.connect(use_mock=True)
        
        # Discover tools
        tools = await client.discover_tools()
        assert len(tools) > 0
        
        # Check for expected tools
        tool_names = [tool["name"] for tool in tools]
        assert "get_holdings" in tool_names
        assert "get_positions" in tool_names
        assert "place_order" in tool_names
        assert "modify_order" in tool_names
        assert "cancel_order" in tool_names
        assert "get_orders" in tool_names
        assert "get_trades" in tool_names
        assert "get_quote" in tool_names
        assert "get_ltp" in tool_names
        assert "get_historical_data" in tool_names
        assert "get_margins" in tool_names
        assert "get_instruments" in tool_names
    
    @pytest.mark.asyncio
    async def test_get_holdings(self, client):
        """Test getting user holdings."""
        await client.connect(use_mock=True)
        
        holdings = await client.get_holdings()
        assert isinstance(holdings, list)
        assert len(holdings) > 0
        
        # Check holding structure
        holding = holdings[0]
        assert "tradingsymbol" in holding
        assert "exchange" in holding
        assert "quantity" in holding
        assert "average_price" in holding
        assert "last_price" in holding
        assert "pnl" in holding
        assert "product" in holding
    
    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test getting user positions."""
        await client.connect(use_mock=True)
        
        positions = await client.get_positions()
        assert isinstance(positions, dict)
        assert "net" in positions
        assert "day" in positions
        
        # Check net positions
        net_positions = positions["net"]
        assert isinstance(net_positions, list)
        if len(net_positions) > 0:
            position = net_positions[0]
            assert "tradingsymbol" in position
            assert "quantity" in position
            assert "average_price" in position
            assert "pnl" in position
    
    @pytest.mark.asyncio
    async def test_place_order(self, client):
        """Test placing an order."""
        await client.connect(use_mock=True)
        
        # Place a limit buy order
        result = await client.place_order(
            exchange="NSE",
            symbol="RELIANCE",
            transaction_type="BUY",
            quantity=10,
            product="CNC",
            order_type="LIMIT",
            price=2500.00
        )
        
        assert result is not None
        assert "order_id" in result
        assert isinstance(result["order_id"], str)
        assert len(result["order_id"]) > 0
    
    @pytest.mark.asyncio
    async def test_modify_order(self, client):
        """Test modifying an order."""
        await client.connect(use_mock=True)
        
        # First place an order
        order_result = await client.place_order(
            exchange="NSE",
            symbol="INFY",
            transaction_type="BUY",
            quantity=20,
            product="CNC",
            order_type="LIMIT",
            price=1450.00
        )
        
        order_id = order_result["order_id"]
        
        # Modify the order
        modify_result = await client.modify_order(
            order_id=order_id,
            quantity=25,
            price=1445.00
        )
        
        assert modify_result is not None
        assert "order_id" in modify_result
        assert modify_result["order_id"] == order_id
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test canceling an order."""
        await client.connect(use_mock=True)
        
        # First place an order
        order_result = await client.place_order(
            exchange="NSE",
            symbol="TCS",
            transaction_type="SELL",
            quantity=5,
            product="CNC",
            order_type="LIMIT",
            price=3200.00
        )
        
        order_id = order_result["order_id"]
        
        # Cancel the order
        cancel_result = await client.cancel_order(order_id)
        
        assert cancel_result is not None
        assert "order_id" in cancel_result
        assert cancel_result["order_id"] == order_id
    
    @pytest.mark.asyncio
    async def test_get_orders(self, client):
        """Test getting order list."""
        await client.connect(use_mock=True)
        
        # Place some orders first
        await client.place_order(
            exchange="NSE",
            symbol="WIPRO",
            transaction_type="BUY",
            quantity=50,
            product="MIS",
            order_type="MARKET"
        )
        
        # Get orders
        orders = await client.get_orders()
        assert isinstance(orders, list)
        assert len(orders) > 0
        
        # Check order structure
        order = orders[0]
        assert "order_id" in order
        assert "tradingsymbol" in order
        assert "status" in order
        assert "quantity" in order
    
    @pytest.mark.asyncio
    async def test_get_quote(self, client):
        """Test getting market quote."""
        await client.connect(use_mock=True)
        
        quote = await client.get_quote("NSE", "RELIANCE")
        assert quote is not None
        assert "last_price" in quote
        assert "volume" in quote
        assert "ohlc" in quote
        
        # Check OHLC structure
        ohlc = quote["ohlc"]
        assert "open" in ohlc
        assert "high" in ohlc
        assert "low" in ohlc
        assert "close" in ohlc
        
        # Check market depth if available
        if "depth" in quote:
            depth = quote["depth"]
            assert "buy" in depth
            assert "sell" in depth
            assert isinstance(depth["buy"], list)
            assert isinstance(depth["sell"], list)
    
    @pytest.mark.asyncio
    async def test_get_ltp(self, client):
        """Test getting last traded prices."""
        await client.connect(use_mock=True)
        
        instruments = ["NSE:RELIANCE", "NSE:INFY", "NSE:TCS"]
        ltps = await client.get_ltp(instruments)
        
        assert isinstance(ltps, dict)
        assert len(ltps) == len(instruments)
        
        for instrument in instruments:
            assert instrument in ltps
            assert "last_price" in ltps[instrument]
            assert isinstance(ltps[instrument]["last_price"], (int, float))
            assert ltps[instrument]["last_price"] > 0
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, client):
        """Test getting historical data."""
        await client.connect(use_mock=True)
        
        # Get data for last 5 days
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=5)
        
        result = await client.get_historical_data(
            instrument_token="738561",  # RELIANCE
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            interval="day"
        )
        
        assert result is not None
        assert "candles" in result
        assert isinstance(result["candles"], list)
        assert len(result["candles"]) > 0
        
        # Check candle structure
        candle = result["candles"][0]
        assert "date" in candle
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle
        assert "volume" in candle
    
    @pytest.mark.asyncio
    async def test_get_margins(self, client):
        """Test getting account margins."""
        await client.connect(use_mock=True)
        
        margins = await client.get_margins()
        assert margins is not None
        
        # Check for equity segment by default
        assert "equity" in margins or "available" in margins
        
        if "equity" in margins:
            equity = margins["equity"]
            assert "available" in equity
            assert "used" in equity
            
            # Check available funds
            available = equity["available"]
            assert "cash" in available
            assert isinstance(available["cash"], (int, float))
    
    @pytest.mark.asyncio
    async def test_get_instruments(self, client):
        """Test getting instruments list."""
        await client.connect(use_mock=True)
        
        # Get all instruments
        instruments = await client.get_instruments()
        assert isinstance(instruments, list)
        assert len(instruments) > 0
        
        # Check instrument structure
        instrument = instruments[0]
        assert "instrument_token" in instrument
        assert "tradingsymbol" in instrument
        assert "exchange" in instrument
        assert "instrument_type" in instrument
        
        # Get instruments for specific exchange
        nse_instruments = await client.get_instruments("NSE")
        assert isinstance(nse_instruments, list)
        for inst in nse_instruments:
            assert inst["exchange"] == "NSE"
    
    @pytest.mark.asyncio
    async def test_caching(self, client):
        """Test that market data is cached."""
        await client.connect(use_mock=True)
        
        # First call - should hit the server
        start_time = asyncio.get_event_loop().time()
        quote1 = await client.get_quote("NSE", "RELIANCE")
        first_call_time = asyncio.get_event_loop().time() - start_time
        
        # Second call - should be cached
        start_time = asyncio.get_event_loop().time()
        quote2 = await client.get_quote("NSE", "RELIANCE")
        second_call_time = asyncio.get_event_loop().time() - start_time
        
        # Cached call should be faster (no network delay)
        assert second_call_time < first_call_time
        
        # Data should be the same (from cache)
        assert quote1["last_price"] == quote2["last_price"]
    
    @pytest.mark.asyncio
    async def test_tool_registration(self, client, registry):
        """Test registering tools to the registry."""
        await client.connect(use_mock=True)
        
        # Register tools
        client.register_tools_to_registry(registry)
        
        # Check tools are registered
        all_tools = registry.get_all_tools()
        zerodha_tools = [t for t in all_tools if t.get('id', '').startswith('zerodha_')]
        
        assert len(zerodha_tools) == len(client.tools)
        
        # Check tool properties
        for tool in zerodha_tools:
            assert tool['type'] == 'mcp'
            assert tool['server_id'] == 'zerodha'
            assert 'capabilities' in tool
            assert tool['capabilities']['category'] == 'trading'
            assert tool['capabilities']['domain'] == 'finance'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling for invalid operations."""
        await client.connect(use_mock=True)
        
        # Test invalid order modification (non-existent order)
        with pytest.raises(Exception) as exc_info:
            await client.modify_order(order_id="INVALID_ORDER_ID", quantity=100)
        assert "Order not found" in str(exc_info.value)
        
        # Test invalid order cancellation
        with pytest.raises(Exception) as exc_info:
            await client.cancel_order("INVALID_ORDER_ID")
        assert "Order not found" in str(exc_info.value)
    
    @pytest.mark.asyncio 
    async def test_concurrent_operations(self, client):
        """Test concurrent tool calls."""
        await client.connect(use_mock=True)
        
        # Execute multiple operations concurrently
        tasks = [
            client.get_holdings(),
            client.get_positions(),
            client.get_margins(),
            client.get_quote("NSE", "RELIANCE"),
            client.get_ltp(["NSE:INFY", "NSE:TCS"])
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert len(results) == 5
        assert all(result is not None for result in results)
        
        # Verify result types
        assert isinstance(results[0], list)  # holdings
        assert isinstance(results[1], dict)  # positions
        assert isinstance(results[2], dict)  # margins
        assert isinstance(results[3], dict)  # quote
        assert isinstance(results[4], dict)  # ltp