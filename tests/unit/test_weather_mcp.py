#!/usr/bin/env python3
"""
Unit tests for Weather MCP Client.

Tests individual components and methods of the Weather MCP client
with extensive mocking to ensure isolation from external dependencies.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime, timedelta
import aiohttp
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.custom_wrappers.weather_mcp import WeatherMCPClient
from src.tools.custom_wrappers.mock_weather_mcp import MockWeatherMCPServer
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry

logger = get_logger(__name__)


class TestWeatherMCPClient:
    """Unit tests for Weather MCP Client."""
    
    @pytest.fixture
    def client(self):
        """Create Weather MCP client instance."""
        return WeatherMCPClient(api_key="test_api_key")
    
    @pytest.fixture
    def mock_aiohttp_session(self):
        """Create mock aiohttp session."""
        session = Mock(spec=aiohttp.ClientSession)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_response(self):
        """Create mock aiohttp response."""
        response = Mock()
        response.status = 200
        response.json = AsyncMock()
        response.text = AsyncMock(return_value="Error message")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock()
        return response
    
    def test_initialization(self):
        """Test client initialization."""
        # Test with API key
        client = WeatherMCPClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.server_name == "weather"
        assert client.capabilities == {"tools": True, "resources": False}
        assert len(client.tools) == 5  # Should have 5 tools defined
        assert client._cache == {}
        assert client._cache_ttl == 600
        
        # Test without API key
        client2 = WeatherMCPClient()
        assert client2.api_key is None
        
        # Test with config
        config = {"cache_ttl": 300}
        client3 = WeatherMCPClient(api_key="test", config=config)
        assert client3.config == config
    
    def test_message_id_generation(self, client):
        """Test message ID generation."""
        id1 = client._next_message_id()
        id2 = client._next_message_id()
        id3 = client._next_message_id()
        
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
    
    def test_tool_definitions(self, client):
        """Test tool definitions."""
        tools = client.tools
        
        # Check all tools are defined
        tool_names = [tool["name"] for tool in tools]
        assert "current_weather" in tool_names
        assert "weather_forecast" in tool_names
        assert "weather_by_coords" in tool_names
        assert "air_pollution" in tool_names
        assert "uv_index" in tool_names
        
        # Check tool schemas
        current_weather = next(t for t in tools if t["name"] == "current_weather")
        assert "location" in current_weather["inputSchema"]["properties"]
        assert "units" in current_weather["inputSchema"]["properties"]
        assert current_weather["inputSchema"]["required"] == ["location"]
    
    @pytest.mark.asyncio
    async def test_connect_with_mock_server(self, client):
        """Test connection with mock server."""
        result = await client.connect(use_mock=True)
        
        assert result is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert isinstance(client.mock_server, MockWeatherMCPServer)
    
    @pytest.mark.asyncio
    async def test_connect_with_real_api_success(self, client, mock_aiohttp_session, mock_response):
        """Test successful connection with real API."""
        mock_response.status = 200
        mock_aiohttp_session.get.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await client.connect(use_mock=False)
            
            assert result is True
            assert client.use_mock is False
            # Verify API key validation call
            mock_aiohttp_session.get.assert_called_once()
            call_args = mock_aiohttp_session.get.call_args[0][0]
            assert "London" in call_args
            assert client.api_key in call_args
    
    @pytest.mark.asyncio
    async def test_connect_with_real_api_failure(self, client, mock_aiohttp_session, mock_response):
        """Test failed connection with real API."""
        mock_response.status = 401  # Unauthorized
        mock_aiohttp_session.get.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await client.connect(use_mock=False)
            
            assert result is False
            assert client.use_mock is False
    
    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        """Test connection without API key."""
        client = WeatherMCPClient()  # No API key
        result = await client.connect(use_mock=False)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_current_weather(self, client):
        """Test get_current_weather method."""
        expected_result = {"success": True, "result": {"temp": 20}}
        
        with patch.object(client, 'call_tool', return_value=expected_result) as mock_call:
            result = await client.get_current_weather("London,UK", "metric")
            
            assert result == expected_result
            mock_call.assert_called_once_with("current_weather", {
                "location": "London,UK",
                "units": "metric"
            })
    
    @pytest.mark.asyncio
    async def test_get_forecast(self, client):
        """Test get_forecast method."""
        expected_result = {"success": True, "result": {"forecast": []}}
        
        with patch.object(client, 'call_tool', return_value=expected_result) as mock_call:
            result = await client.get_forecast("New York,US", days=5, units="imperial")
            
            assert result == expected_result
            mock_call.assert_called_once_with("weather_forecast", {
                "location": "New York,US",
                "days": 5,
                "units": "imperial"
            })
    
    @pytest.mark.asyncio
    async def test_get_weather_by_coords(self, client):
        """Test get_weather_by_coords method."""
        expected_result = {"success": True, "result": {"temp": 25}}
        
        with patch.object(client, 'call_tool', return_value=expected_result) as mock_call:
            result = await client.get_weather_by_coords(51.5074, -0.1278, "kelvin")
            
            assert result == expected_result
            mock_call.assert_called_once_with("weather_by_coords", {
                "lat": 51.5074,
                "lon": -0.1278,
                "units": "kelvin"
            })
    
    @pytest.mark.asyncio
    async def test_call_tool_invalid_tool(self, client):
        """Test calling non-existent tool."""
        result = await client.call_tool("invalid_tool", {})
        
        assert result["success"] is False
        assert "Tool not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_call_tool_with_mock_server(self, client):
        """Test tool execution with mock server."""
        await client.connect(use_mock=True)
        
        # Mock server response
        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "main": {"temp": 20},
                "weather": [{"main": "Clear"}]
            }
        }
        
        with patch.object(client.mock_server, 'handle_request', return_value=mock_response):
            result = await client.call_tool("current_weather", {"location": "London,UK"})
            
            assert result["success"] is True
            assert result["result"]["main"]["temp"] == 20
            assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_execute_api_call_current_weather(self, client, mock_aiohttp_session, mock_response):
        """Test API call for current weather."""
        mock_data = {
            "main": {"temp": 20},
            "weather": [{"main": "Clear"}]
        }
        mock_response.json.return_value = mock_data
        mock_aiohttp_session.get.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await client._execute_api_call("current_weather", {
                "location": "London,UK",
                "units": "metric"
            })
            
            assert result == mock_data
            # Verify API call
            mock_aiohttp_session.get.assert_called_once()
            call_args = mock_aiohttp_session.get.call_args
            assert call_args[0][0] == f"{client.BASE_URL}/weather"
            assert call_args[1]["params"]["q"] == "London,UK"
            assert call_args[1]["params"]["units"] == "metric"
    
    @pytest.mark.asyncio
    async def test_execute_api_call_forecast(self, client, mock_aiohttp_session, mock_response):
        """Test API call for weather forecast."""
        mock_data = {"list": [], "cnt": 24}
        mock_response.json.return_value = mock_data
        mock_aiohttp_session.get.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await client._execute_api_call("weather_forecast", {
                "location": "Paris,FR",
                "days": 3,
                "units": "metric"
            })
            
            assert result == mock_data
            # Verify API call
            call_args = mock_aiohttp_session.get.call_args
            assert call_args[0][0] == f"{client.BASE_URL}/forecast"
            assert call_args[1]["params"]["cnt"] == 24  # 3 days * 8 intervals
    
    @pytest.mark.asyncio
    async def test_execute_api_call_air_pollution(self, client, mock_aiohttp_session, mock_response):
        """Test API call for air pollution."""
        mock_data = {"list": [{"main": {"aqi": 2}}]}
        mock_response.json.return_value = mock_data
        mock_aiohttp_session.get.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await client._execute_api_call("air_pollution", {
                "lat": 51.5074,
                "lon": -0.1278
            })
            
            assert result == mock_data
            # Verify API call
            call_args = mock_aiohttp_session.get.call_args
            assert call_args[0][0] == f"{client.BASE_URL}/air_pollution"
    
    @pytest.mark.asyncio
    async def test_execute_api_call_with_cache(self, client, mock_aiohttp_session, mock_response):
        """Test API call caching."""
        mock_data = {"temp": 20}
        mock_response.json.return_value = mock_data
        mock_aiohttp_session.get.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            # First call - should hit API
            result1 = await client._execute_api_call("current_weather", {
                "location": "London,UK"
            })
            assert result1 == mock_data
            assert mock_aiohttp_session.get.call_count == 1
            
            # Second call - should use cache
            result2 = await client._execute_api_call("current_weather", {
                "location": "London,UK"
            })
            assert result2 == mock_data
            assert mock_aiohttp_session.get.call_count == 1  # No additional API call
    
    @pytest.mark.asyncio
    async def test_execute_api_call_cache_expiry(self, client, mock_aiohttp_session, mock_response):
        """Test cache expiry."""
        mock_data = {"temp": 20}
        mock_response.json.return_value = mock_data
        mock_aiohttp_session.get.return_value = mock_response
        
        # Set cache with expired timestamp
        cache_key = "current_weather:{\"location\":\"London,UK\"}"
        client._cache[cache_key] = {
            'data': {"temp": 15},  # Old data
            'timestamp': datetime.now() - timedelta(seconds=700)  # Expired
        }
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await client._execute_api_call("current_weather", {
                "location": "London,UK"
            })
            
            assert result == mock_data  # Should return fresh data
            assert mock_aiohttp_session.get.call_count == 1  # API called
    
    @pytest.mark.asyncio
    async def test_api_error_through_mock_server(self, client):
        """Test API error handling through mock server error response."""
        # Connect with mock server
        await client.connect(use_mock=True)
        
        # Mock the mock server to return an error
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": "API error simulation"
            }
        }
        
        with patch.object(client.mock_server, 'handle_request', return_value=error_response):
            result = await client.call_tool("current_weather", {
                "location": "InvalidCity"
            })
            
            # When mock server returns error response, call_tool should handle it
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_execute_api_call_unknown_tool(self, client):
        """Test API call with unknown tool."""
        with pytest.raises(ValueError) as exc_info:
            await client._execute_api_call("unknown_tool", {})
        
        assert "Unknown tool: unknown_tool" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnect method."""
        # Add some cache data
        client._cache["test_key"] = {"data": "test", "timestamp": datetime.now()}
        
        await client.disconnect()
        
        # Cache should be cleared
        assert len(client._cache) == 0
    
    def test_register_tools_to_registry(self, client):
        """Test tool registration."""
        registry = Mock(spec=ToolRegistry)
        
        client.register_tools_to_registry(registry)
        
        # Should register all 5 tools
        assert registry.register_tool.call_count == 5
        
        # Check registration details
        calls = registry.register_tool.call_args_list
        registered_names = [call[0][0]['name'] for call in calls]
        assert "current_weather" in registered_names
        assert "weather_forecast" in registered_names
        assert "weather_by_coords" in registered_names
        assert "air_pollution" in registered_names
        assert "uv_index" in registered_names
        
        # Check tool info structure
        first_call = calls[0][0][0]
        assert first_call['server_type'] == 'weather'
        assert first_call['endpoint'] == 'OpenWeather API'
        assert 'capabilities' in first_call
        assert 'input_schema' in first_call
    
    @pytest.mark.asyncio
    async def test_call_tool_exception_handling(self, client):
        """Test exception handling in call_tool."""
        with patch.object(client, '_execute_api_call', side_effect=Exception("Test error")):
            result = await client.call_tool("current_weather", {"location": "London"})
            
            assert result["success"] is False
            assert "Test error" in result["error"]
            assert "execution_time" in result


class TestMockWeatherMCPServer:
    """Unit tests for Mock Weather MCP Server."""
    
    @pytest.fixture
    def server(self):
        """Create mock server instance."""
        return MockWeatherMCPServer()
    
    def test_mock_server_initialization(self, server):
        """Test mock server initialization."""
        assert server.initialized is False
        assert len(server.tools) == 5
        assert len(server.weather_conditions) > 0
        assert len(server.city_coords) > 0
    
    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        """Test initialization request handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1
        }
        
        response = await server.handle_request(request)
        
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "1.0"
        assert server.initialized is True
    
    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        """Test tools list request handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        response = await server.handle_request(request)
        
        assert response["id"] == 2
        assert "result" in response
        assert len(response["result"]["tools"]) == 5
    
    @pytest.mark.asyncio
    async def test_mock_current_weather(self, server):
        """Test mock current weather generation."""
        args = {"location": "London,UK", "units": "metric"}
        
        result = await server._mock_current_weather(args)
        
        assert "coord" in result
        assert "weather" in result
        assert "main" in result
        assert "temp" in result["main"]
        assert result["name"] == "London"
        assert result["sys"]["country"] == "UK"
    
    @pytest.mark.asyncio
    async def test_mock_weather_forecast(self, server):
        """Test mock weather forecast generation."""
        args = {"location": "New York,US", "days": 2, "units": "imperial"}
        
        result = await server._mock_weather_forecast(args)
        
        assert "list" in result
        assert len(result["list"]) == 16  # 2 days * 8 intervals
        assert result["city"]["name"] == "New York"
        assert result["city"]["country"] == "US"
    
    def test_temperature_conversion(self, server):
        """Test temperature unit conversion."""
        celsius = 20.0
        
        # Test metric (Celsius)
        assert server._convert_temperature(celsius, "metric") == 20.0
        
        # Test imperial (Fahrenheit)
        fahrenheit = server._convert_temperature(celsius, "imperial")
        assert fahrenheit == 68.0
        
        # Test Kelvin
        kelvin = server._convert_temperature(celsius, "kelvin")
        assert kelvin == 293.15
    
    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        """Test handling unknown method."""
        request = {
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
            "id": 1
        }
        
        response = await server.handle_request(request)
        
        assert "error" in response
        assert "Unknown method" in response["error"]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])