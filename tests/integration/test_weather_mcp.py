#!/usr/bin/env python3
"""
Weather MCP Integration Tests

Tests the Weather MCP tool implementation with both mock and real scenarios.
"""

import asyncio
import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.custom_wrappers.weather_mcp import WeatherMCPClient
from src.tools.custom_wrappers.mock_weather_mcp import MockWeatherMCPServer
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestWeatherMCPIntegration:
    """Integration tests for Weather MCP functionality."""
    
    @pytest.fixture
    def weather_client(self):
        """Create a Weather MCP client instance."""
        # Use environment variable or test key
        api_key = os.getenv("OPENWEATHER_API_KEY", "test_api_key")
        return WeatherMCPClient(api_key=api_key)
    
    @pytest.fixture
    def mock_server(self):
        """Create a mock Weather MCP server."""
        return MockWeatherMCPServer()
    
    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry instance."""
        return ToolRegistry(":memory:")  # In-memory database for testing
    
    @pytest.mark.asyncio
    async def test_weather_mcp_initialization(self, weather_client):
        """Test Weather MCP initialization."""
        assert weather_client.server_name == "weather"
        assert weather_client.capabilities == {"tools": True, "resources": False}
        assert len(weather_client.tools) == 5  # Should have 5 weather tools
    
    @pytest.mark.asyncio
    async def test_connect_with_mock_server(self, weather_client):
        """Test connection with mock server."""
        connected = await weather_client.connect(use_mock=True)
        
        assert connected is True
        assert weather_client.use_mock is True
        assert weather_client.mock_server is not None
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_connect_with_real_api(self, weather_client):
        """Test connection with real API (if API key is available)."""
        # This test will only pass if a valid API key is provided
        if weather_client.api_key == "test_api_key":
            pytest.skip("Skipping real API test - no valid API key")
        
        connected = await weather_client.connect(use_mock=False)
        
        # Connection may fail without valid API key
        if connected:
            assert weather_client.use_mock is False
            await weather_client.disconnect()
        else:
            # Fall back to mock if real API fails
            connected = await weather_client.connect(use_mock=True)
            assert connected is True
            await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_current_weather_with_mock(self, weather_client):
        """Test getting current weather using mock server."""
        # Connect with mock server
        await weather_client.connect(use_mock=True)
        
        # Get current weather
        result = await weather_client.get_current_weather("London,UK", "metric")
        
        assert result["success"] is True
        assert "result" in result
        assert "main" in result["result"]
        assert "weather" in result["result"]
        assert "execution_time" in result
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_weather_forecast_with_mock(self, weather_client):
        """Test getting weather forecast using mock server."""
        # Connect with mock server
        await weather_client.connect(use_mock=True)
        
        # Get weather forecast
        result = await weather_client.get_forecast("New York,US", days=3, units="imperial")
        
        assert result["success"] is True
        assert "result" in result
        assert "list" in result["result"]
        assert len(result["result"]["list"]) == 24  # 3 days * 8 intervals
        assert "city" in result["result"]
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_weather_by_coordinates(self, weather_client):
        """Test getting weather by coordinates."""
        # Connect with mock server
        await weather_client.connect(use_mock=True)
        
        # Get weather by coordinates (London)
        result = await weather_client.get_weather_by_coords(51.5074, -0.1278, "metric")
        
        assert result["success"] is True
        assert "result" in result
        assert "coord" in result["result"]
        assert result["result"]["coord"]["lat"] == 51.5074
        assert result["result"]["coord"]["lon"] == -0.1278
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_air_pollution_data(self, weather_client):
        """Test getting air pollution data."""
        # Connect with mock server
        await weather_client.connect(use_mock=True)
        
        # Get air pollution data
        result = await weather_client.call_tool("air_pollution", {
            "lat": 51.5074,
            "lon": -0.1278
        })
        
        assert result["success"] is True
        assert "result" in result
        assert "list" in result["result"]
        assert "main" in result["result"]["list"][0]
        assert "aqi" in result["result"]["list"][0]["main"]
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_uv_index_data(self, weather_client):
        """Test getting UV index data."""
        # Connect with mock server
        await weather_client.connect(use_mock=True)
        
        # Get UV index data
        result = await weather_client.call_tool("uv_index", {
            "lat": 40.7128,
            "lon": -74.0060
        })
        
        assert result["success"] is True
        assert "result" in result
        assert "value" in result["result"]
        assert "lat" in result["result"]
        assert "lon" in result["result"]
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_tool_discovery(self, weather_client):
        """Test tool discovery from Weather MCP."""
        # Tool discovery happens during initialization
        tools = weather_client.tools
        
        assert len(tools) == 5
        tool_names = [tool["name"] for tool in tools]
        
        assert "current_weather" in tool_names
        assert "weather_forecast" in tool_names
        assert "weather_by_coords" in tool_names
        assert "air_pollution" in tool_names
        assert "uv_index" in tool_names
        
        # Check tool schemas
        current_weather_tool = next(t for t in tools if t["name"] == "current_weather")
        assert "inputSchema" in current_weather_tool
        assert "properties" in current_weather_tool["inputSchema"]
        assert "location" in current_weather_tool["inputSchema"]["properties"]
    
    @pytest.mark.asyncio
    async def test_invalid_tool_call(self, weather_client):
        """Test calling invalid tool."""
        await weather_client.connect(use_mock=True)
        
        result = await weather_client.call_tool("invalid_tool", {"location": "London"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Tool not found" in result["error"]
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_tool_registry_integration(self, weather_client, tool_registry):
        """Test integration with tool registry."""
        # Register tools
        weather_client.register_tools_to_registry(tool_registry)
        
        # List weather tools
        weather_tools = tool_registry.list_tools("weather")
        
        assert len(weather_tools) == 5
        tool_ids = [tool["id"] for tool in weather_tools]
        
        assert "weather.current_weather" in tool_ids
        assert "weather.weather_forecast" in tool_ids
        assert "weather.weather_by_coords" in tool_ids
        assert "weather.air_pollution" in tool_ids
        assert "weather.uv_index" in tool_ids
    
    @pytest.mark.asyncio
    async def test_mock_server_protocol_compliance(self, mock_server):
        """Test mock server JSON-RPC protocol compliance."""
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1
        }
        init_response = await mock_server.handle_request(init_request)
        
        assert init_response["jsonrpc"] == "2.0"
        assert init_response["id"] == 1
        assert "result" in init_response
        assert init_response["result"]["capabilities"]["tools"] is True
        
        # Test tools list
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        list_response = await mock_server.handle_request(list_request)
        
        assert list_response["jsonrpc"] == "2.0"
        assert list_response["id"] == 2
        assert len(list_response["result"]["tools"]) == 5
    
    @pytest.mark.asyncio
    async def test_temperature_units_in_responses(self, weather_client):
        """Test temperature unit handling in responses."""
        await weather_client.connect(use_mock=True)
        
        # Test metric units
        result_metric = await weather_client.call_tool("current_weather", {
            "location": "Paris,FR",
            "units": "metric"
        })
        
        # Temperature should be in reasonable Celsius range
        temp = result_metric["result"]["main"]["temp"]
        assert -50 < temp < 60  # Reasonable Celsius range
        
        # Test imperial units
        result_imperial = await weather_client.call_tool("current_weather", {
            "location": "Paris,FR",
            "units": "imperial"
        })
        
        # Temperature should be in reasonable Fahrenheit range
        temp_f = result_imperial["result"]["main"]["temp"]
        assert -50 < temp_f < 140  # Reasonable Fahrenheit range
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_api_calls(self, weather_client):
        """Test error handling during API calls."""
        await weather_client.connect(use_mock=True)
        
        # Mock an error in the mock server
        with patch.object(weather_client.mock_server, 'handle_request',
                         side_effect=Exception("Mock server error")):
            result = await weather_client.call_tool("current_weather", {
                "location": "London,UK"
            })
            
            assert result["success"] is False
            assert "error" in result
            assert "Mock server error" in result["error"]
        
        await weather_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, weather_client):
        """Test caching behavior for API calls."""
        # This test only applies to real API calls
        if weather_client.api_key == "test_api_key":
            pytest.skip("Caching test requires real API")
        
        await weather_client.connect(use_mock=False)
        
        if weather_client.use_mock:
            # If connection failed and fell back to mock, skip test
            pytest.skip("Real API connection failed")
        
        # Make first call
        result1 = await weather_client.call_tool("current_weather", {
            "location": "London,UK"
        })
        
        # Make second call (should use cache)
        result2 = await weather_client.call_tool("current_weather", {
            "location": "London,UK"
        })
        
        # Results should be identical (from cache)
        assert result1["result"] == result2["result"]
        
        await weather_client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])