"""
Integration tests for Zerodha MCP Server
"""

import pytest
import asyncio
import os
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock
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
    
    @pytest.mark.asyncio
    async def test_real_server_mock_response(self, client):
        """Test real server connection with mocked HTTP responses."""
        # Mock successful initialization response
        init_response = {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "Zerodha Kite MCP Server",
                    "version": "1.0.0"
                },
                "capabilities": {"tools": True}
            },
            "id": 1
        }
        
        # Mock tools list response
        tools_response = {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "get_holdings",
                        "description": "Get user holdings",
                        "inputSchema": {"type": "object"}
                    },
                    {
                        "name": "place_order",
                        "description": "Place an order",
                        "inputSchema": {"type": "object"}
                    }
                ]
            },
            "id": 2
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Create async mock for post method
            async def mock_post(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.status = 200
                
                # Return different responses based on call count
                if not hasattr(mock_post, 'call_count'):
                    mock_post.call_count = 0
                mock_post.call_count += 1
                
                async def mock_json():
                    if mock_post.call_count == 1:
                        return init_response
                    else:
                        return tools_response
                
                mock_response.json = mock_json
                
                async def mock_aenter(self):
                    return mock_response
                
                async def mock_aexit(self, *args):
                    return None
                
                mock_response.__aenter__ = mock_aenter
                mock_response.__aexit__ = mock_aexit
                return mock_response
            
            mock_session.post = mock_post
            mock_session.close = AsyncMock()
            
            # Create client with API credentials
            test_client = ZerodhaMCPClient(
                api_key="test_api_key",
                api_secret="test_api_secret", 
                access_token="test_access_token"
            )
            
            # Connect to "real" server (mocked)
            connected = await test_client.connect(use_mock=False)
            assert connected is True
            assert test_client.connected is True
            assert test_client.use_mock is False
            
            # Verify initialization was called
            assert hasattr(mock_post, 'call_count')
            assert mock_post.call_count >= 1
            
            # Clean up
            await test_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_reconnection_scenario(self, client):
        """Test reconnection after disconnect."""
        # First connection
        await client.connect(use_mock=True)
        assert client.connected is True
        
        # Get some data
        holdings = await client.get_holdings()
        assert holdings is not None
        assert len(holdings) > 0
        
        # Disconnect
        await client.disconnect()
        assert client.connected is False
        
        # Reconnect
        await client.connect(use_mock=True)
        assert client.connected is True
        
        # Get data again
        holdings2 = await client.get_holdings()
        assert holdings2 is not None
        assert len(holdings2) > 0
    
    @pytest.mark.asyncio
    async def test_load_testing(self, client):
        """Test Zerodha MCP under load."""
        await client.connect(use_mock=True)
        
        try:
            # Simulate multiple concurrent users
            async def simulate_user_requests(user_id: int, num_requests: int):
                results = []
                symbols = ["RELIANCE", "INFY", "TCS", "WIPRO", "HDFC"]
                
                for i in range(num_requests):
                    symbol = symbols[i % len(symbols)]
                    result = await client.get_quote("NSE", symbol)
                    results.append(result)
                
                return results
            
            # Run 5 concurrent users, each making 10 requests
            user_tasks = [
                simulate_user_requests(user_id, 10)
                for user_id in range(5)
            ]
            
            import time
            start_time = time.time()
            all_results = await asyncio.gather(*user_tasks)
            end_time = time.time()
            
            # Verify all requests completed
            assert len(all_results) == 5
            for user_results in all_results:
                assert len(user_results) == 10
                assert all(r is not None for r in user_results)
            
            # Check performance
            total_time = end_time - start_time
            total_requests = 50
            avg_time_per_request = total_time / total_requests
            
            logger.info(f"Load test completed: {total_requests} requests in {total_time:.2f}s")
            logger.info(f"Average time per request: {avg_time_per_request:.3f}s")
            
            # With caching, should be very fast
            assert avg_time_per_request < 0.1  # Less than 100ms per request
            
        finally:
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_zerodha_real_api_simulation(self, client):
        """Test Zerodha with simulated real API behavior."""
        # This test simulates what would happen with a real API
        
        class SimulatedRealAPIClient(ZerodhaMCPClient):
            """Client that simulates real API delays and behaviors."""
            
            async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                # Add realistic delay
                await asyncio.sleep(0.05)  # 50ms API latency
                
                # Simulate rate limiting occasionally
                if hasattr(self, '_request_count'):
                    self._request_count += 1
                    if self._request_count % 10 == 0:
                        await asyncio.sleep(0.2)  # Rate limit delay
                else:
                    self._request_count = 1
                
                # Call parent method
                return await super().call_tool(tool_name, arguments)
        
        sim_client = SimulatedRealAPIClient()
        await sim_client.connect(use_mock=True)
        
        try:
            import time
            start_time = time.time()
            
            # Morning market check workflow
            tasks = [
                sim_client.get_holdings(),
                sim_client.get_positions(),
                sim_client.get_margins()
            ]
            portfolio = await asyncio.gather(*tasks)
            
            # Check specific stocks
            quotes = await asyncio.gather(
                sim_client.get_quote("NSE", "RELIANCE"),
                sim_client.get_quote("NSE", "INFY"),
                sim_client.get_quote("NSE", "TCS")
            )
            
            # Place an order
            order = await sim_client.place_order(
                exchange="NSE",
                symbol="RELIANCE",
                transaction_type="BUY",
                quantity=10,
                product="CNC",
                order_type="LIMIT",
                price=2500.00
            )
            
            end_time = time.time()
            
            # Verify all data retrieved
            assert len(portfolio) == 3
            assert len(quotes) == 3
            assert order is not None
            
            total_time = end_time - start_time
            logger.info(f"Simulated real API workflow completed in {total_time:.2f}s")
            
        finally:
            await sim_client.disconnect()


@pytest.mark.asyncio
async def test_zerodha_with_mcp_integration():
    """Test Zerodha integration with MCP Integration class."""
    from src.core.mcp_integration import MCPIntegration
    
    # Create MCP integration
    integration = MCPIntegration()
    
    # Add Zerodha server using mock
    success = await integration.add_zerodha_server(
        server_id="test_zerodha",
        use_mock=True
    )
    assert success is True
    
    # Check server was added
    assert "test_zerodha" in integration.servers
    assert integration.servers["test_zerodha"]["type"] == "zerodha"
    assert integration.servers["test_zerodha"]["is_mock"] is True
    
    # Check tools were registered
    tools = integration.registry.get_all_tools()
    logger.info(f"All registered tools: {[t.get('id', 'unknown') for t in tools]}")
    
    # Filter by id pattern since server_id might not be in all tools
    zerodha_tools = [t for t in tools if "zerodha" in t.get("id", "")]
    assert len(zerodha_tools) > 0, f"No Zerodha tools found. Available tools: {[t.get('id', '') for t in tools]}"
    
    # Execute a tool through integration
    result = await integration.execute_tool(
        "zerodha_get_holdings",
        {}
    )
    assert result is not None
    assert isinstance(result, list)
    
    # Test another tool with parameters
    quote = await integration.execute_tool(
        "zerodha_get_quote",
        {"exchange": "NSE", "symbol": "RELIANCE"}
    )
    assert quote is not None
    assert "last_price" in quote
    
    # Cleanup
    await integration.shutdown()


@pytest.mark.asyncio
async def test_zerodha_integration_with_multiple_tools():
    """Test Zerodha integration with other MCP tools."""
    from src.core.mcp_integration import MCPIntegration
    
    integration = MCPIntegration()
    
    try:
        # Add multiple servers
        success = await integration.add_zerodha_server(
            server_id="zerodha",
            use_mock=True
        )
        assert success is True, "Failed to add Zerodha server"
        
        # Add filesystem server for storing trading data
        fs_success = await integration.add_filesystem_server(
            server_id="filesystem",
            use_mock=True
        )
        assert fs_success is True, "Failed to add Filesystem server"
        
        # Get holdings from Zerodha
        holdings = await integration.execute_tool(
            "zerodha_get_holdings",
            {}
        )
        
        # Save to file (simulated - we won't actually write)
        # This demonstrates multi-tool integration
        assert holdings is not None
        assert isinstance(holdings, list)
        
        # Get market quote
        quote = await integration.execute_tool(
            "zerodha_get_quote",
            {"exchange": "NSE", "symbol": "INFY"}
        )
        assert quote is not None
        
        # Verify both tools are available
        tools = integration.registry.get_all_tools()
        tool_ids = {t.get("id", "") for t in tools}
        zerodha_tools = [id for id in tool_ids if "zerodha" in id]
        filesystem_tools = [id for id in tool_ids if "filesystem" in id]
        assert len(zerodha_tools) > 0
        assert len(filesystem_tools) > 0
        
    finally:
        await integration.shutdown()


@pytest.mark.asyncio
async def test_zerodha_error_recovery():
    """Test error recovery in Zerodha integration."""
    import aiohttp
    
    client = ZerodhaMCPClient()
    
    # Test recovery from connection failure
    with patch('aiohttp.ClientSession') as mock_session_class:
        # Simulate connection failure
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
        
        # Should fail to connect to real server
        connected = await client.connect(use_mock=False)
        assert connected is False
    
    # Now connect with mock - should work
    connected = await client.connect(use_mock=True)
    assert connected is True
    
    # Should work normally now
    holdings = await client.get_holdings()
    assert holdings is not None
    assert isinstance(holdings, list)
    
    # Test error in tool execution
    with pytest.raises(Exception) as exc_info:
        await client.call_tool("invalid_tool", {})
    assert "Tool error" in str(exc_info.value) or "Unknown tool" in str(exc_info.value)
    
    # Should still be connected and functional
    assert client.connected is True
    positions = await client.get_positions()
    assert positions is not None
    
    await client.disconnect()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])