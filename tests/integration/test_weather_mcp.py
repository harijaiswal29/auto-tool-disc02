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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.custom_wrappers.weather_mcp import WeatherMCPClient
from src.tools.custom_wrappers.mock_weather_mcp import MockWeatherMCPServer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestWeatherMCP:
    """Test Weather MCP functionality."""
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock Weather MCP server."""
        server = MockWeatherMCPServer()
        await server.start()
        yield server
        await server.stop()
    
    @pytest.fixture
    def weather_mcp(self):
        """Create a Weather MCP instance."""
        # Use a test API key or mock
        return WeatherMCPClient(api_key="test_api_key")
    
    @pytest.mark.asyncio
    async def test_weather_mcp_initialization(self, weather_mcp):
        """Test Weather MCP initialization."""
        assert weather_mcp.name == "weather_mcp"
        assert weather_mcp.description == "Weather information retrieval via MCP"
        assert weather_mcp.api_key == "test_api_key"
    
    @pytest.mark.asyncio
    async def test_get_current_weather_mock(self, weather_mcp):
        """Test getting current weather with mocked response."""
        # Mock the API call
        mock_response = {
            "location": {
                "name": "London",
                "country": "United Kingdom",
                "localtime": "2024-01-15 14:30"
            },
            "current": {
                "temp_c": 10.5,
                "temp_f": 50.9,
                "condition": {
                    "text": "Partly cloudy"
                },
                "humidity": 75,
                "wind_kph": 15.5,
                "wind_dir": "SW"
            }
        }
        
        with patch.object(weather_mcp, '_make_api_request', 
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await weather_mcp.execute({
                'action': 'current',
                'location': 'London'
            })
            
            assert result['success'] is True
            assert result['data']['location']['name'] == 'London'
            assert result['data']['current']['temp_c'] == 10.5
            assert result['data']['current']['condition']['text'] == 'Partly cloudy'
    
    @pytest.mark.asyncio
    async def test_get_forecast_mock(self, weather_mcp):
        """Test getting weather forecast with mocked response."""
        mock_response = {
            "location": {
                "name": "New York",
                "country": "USA"
            },
            "forecast": {
                "forecastday": [
                    {
                        "date": "2024-01-15",
                        "day": {
                            "maxtemp_c": 5.2,
                            "mintemp_c": -2.1,
                            "condition": {
                                "text": "Sunny"
                            },
                            "avghumidity": 60
                        }
                    },
                    {
                        "date": "2024-01-16",
                        "day": {
                            "maxtemp_c": 7.5,
                            "mintemp_c": 0.3,
                            "condition": {
                                "text": "Cloudy"
                            },
                            "avghumidity": 65
                        }
                    }
                ]
            }
        }
        
        with patch.object(weather_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await weather_mcp.execute({
                'action': 'forecast',
                'location': 'New York',
                'days': 2
            })
            
            assert result['success'] is True
            assert len(result['data']['forecast']['forecastday']) == 2
            assert result['data']['forecast']['forecastday'][0]['day']['condition']['text'] == 'Sunny'
    
    @pytest.mark.asyncio
    async def test_search_location_mock(self, weather_mcp):
        """Test searching for locations with mocked response."""
        mock_response = [
            {
                "id": 2801268,
                "name": "London",
                "region": "City of London, Greater London",
                "country": "United Kingdom",
                "lat": 51.52,
                "lon": -0.11
            },
            {
                "id": 2643123,
                "name": "London",
                "region": "Ontario",
                "country": "Canada",
                "lat": 42.98,
                "lon": -81.23
            }
        ]
        
        with patch.object(weather_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await weather_mcp.execute({
                'action': 'search',
                'query': 'London'
            })
            
            assert result['success'] is True
            assert len(result['data']) == 2
            assert result['data'][0]['country'] == 'United Kingdom'
            assert result['data'][1]['country'] == 'Canada'
    
    @pytest.mark.asyncio
    async def test_invalid_action(self, weather_mcp):
        """Test handling invalid action."""
        result = await weather_mcp.execute({
            'action': 'invalid_action',
            'location': 'London'
        })
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Unsupported action' in result['error']
    
    @pytest.mark.asyncio
    async def test_missing_location(self, weather_mcp):
        """Test handling missing location parameter."""
        result = await weather_mcp.execute({
            'action': 'current'
        })
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Location is required' in result['error']
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, weather_mcp):
        """Test handling API errors."""
        with patch.object(weather_mcp, '_make_api_request',
                         side_effect=Exception("API Error")):
            result = await weather_mcp.execute({
                'action': 'current',
                'location': 'London'
            })
            
            assert result['success'] is False
            assert 'error' in result
            assert 'API Error' in result['error']
    
    @pytest.mark.asyncio
    async def test_capabilities(self, weather_mcp):
        """Test getting tool capabilities."""
        capabilities = weather_mcp.get_capabilities()
        
        assert capabilities['name'] == 'weather_mcp'
        assert 'operations' in capabilities
        
        operations = capabilities['operations']
        assert any(op['name'] == 'current' for op in operations)
        assert any(op['name'] == 'forecast' for op in operations)
        assert any(op['name'] == 'search' for op in operations)
    
    @pytest.mark.asyncio
    async def test_mock_server_interaction(self, mock_server):
        """Test interaction with mock Weather MCP server."""
        # Get server info
        response = mock_server.handle_tool_list()
        
        assert response['result'] is not None
        assert 'tools' in response['result']
        
        tools = response['result']['tools']
        assert any(tool['name'] == 'get_weather' for tool in tools)
        assert any(tool['name'] == 'get_forecast' for tool in tools)
        
        # Get weather through mock
        weather_response = mock_server.handle_tool_call(
            'get_weather',
            {'location': 'London'}
        )
        
        assert weather_response['result'] is not None
        assert 'temperature' in weather_response['result']
        assert 'conditions' in weather_response['result']
    
    @pytest.mark.asyncio
    async def test_temperature_units(self, weather_mcp):
        """Test temperature unit conversion."""
        mock_response = {
            "location": {"name": "Paris"},
            "current": {
                "temp_c": 20.0,
                "temp_f": 68.0,
                "condition": {"text": "Clear"}
            }
        }
        
        with patch.object(weather_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            # Test Celsius (default)
            result_c = await weather_mcp.execute({
                'action': 'current',
                'location': 'Paris'
            })
            
            assert result_c['data']['current']['temp_c'] == 20.0
            
            # Test with explicit unit request
            result_f = await weather_mcp.execute({
                'action': 'current',
                'location': 'Paris',
                'units': 'fahrenheit'
            })
            
            assert result_f['data']['current']['temp_f'] == 68.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])