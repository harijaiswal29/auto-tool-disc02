"""
Weather MCP Client Wrapper

This module provides a custom MCP wrapper for OpenWeather API,
enabling weather data access through the Model Context Protocol.
"""

import asyncio
import json
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.tools.custom_wrappers.mock_weather_mcp import MockWeatherMCPServer

logger = get_logger(__name__)

class WeatherMCPClient:
    """
    Custom MCP wrapper for OpenWeather API.
    
    This client wraps OpenWeather API calls in MCP protocol format,
    enabling weather data access through standardized MCP tools.
    """
    
    # OpenWeather API endpoints
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    GEO_URL = "https://api.openweathermap.org/geo/1.0"
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Weather MCP client.
        
        Args:
            api_key: OpenWeather API key
            config: Additional configuration options
        """
        self.api_key = api_key
        self.config = config or {}
        self.server_name = "weather"
        
        # MCP protocol properties
        self.capabilities: Dict[str, Any] = {
            "tools": True,
            "resources": False
        }
        self.tools: List[Dict[str, Any]] = self._define_tools()
        self._message_id = 0
        self.mock_server: Optional[MockWeatherMCPServer] = None
        self.use_mock = False
        
        # Cache for API responses (simple in-memory cache)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 600  # 10 minutes
        
        logger.info(f"[INIT] Weather MCP Client initialized")
    
    def _next_message_id(self) -> int:
        """Generate next message ID for JSON-RPC."""
        self._message_id += 1
        return self._message_id
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available weather tools following MCP protocol."""
        return [
            {
                "name": "current_weather",
                "description": "Get current weather conditions for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, state code and country code (e.g., 'London,UK')"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial", "kelvin"],
                            "default": "metric",
                            "description": "Units of measurement"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "weather_forecast",
                "description": "Get weather forecast for up to 5 days",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, state code and country code"
                        },
                        "days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 3,
                            "description": "Number of days to forecast"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial", "kelvin"],
                            "default": "metric"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "weather_by_coords",
                "description": "Get weather by geographic coordinates",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude"
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial", "kelvin"],
                            "default": "metric"
                        }
                    },
                    "required": ["lat", "lon"]
                }
            },
            {
                "name": "air_pollution",
                "description": "Get current air pollution data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude"
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude"
                        }
                    },
                    "required": ["lat", "lon"]
                }
            },
            {
                "name": "uv_index",
                "description": "Get UV index data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude"
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude"
                        }
                    },
                    "required": ["lat", "lon"]
                }
            }
        ]
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Initialize connection (validate API key or setup mock).
        
        Args:
            use_mock: If True, use mock server instead of real API
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if use_mock:
                logger.info(f"[CONNECTING] Using mock Weather MCP server...")
                self.mock_server = MockWeatherMCPServer(self.config)
                self.use_mock = True
                logger.info(f"[SUCCESS] Connected to mock Weather MCP server")
                return True
            else:
                if not self.api_key:
                    logger.error("[ERROR] No API key provided for Weather MCP")
                    return False
                
                logger.info(f"[CONNECTING] Validating OpenWeather API key...")
                
                # Validate API key with a test request
                test_url = f"{self.BASE_URL}/weather?q=London&appid={self.api_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(test_url) as response:
                        if response.status == 200:
                            logger.info(f"[SUCCESS] Connected to OpenWeather API")
                            return True
                        else:
                            logger.error(f"[ERROR] Invalid API key or API error: {response.status}")
                            return False
                            
        except Exception as e:
            logger.error(f"[ERROR] Connection error: {str(e)}")
            return False
    
    async def get_current_weather(self, location: str, units: str = "metric") -> Dict[str, Any]:
        """
        Get current weather conditions.
        
        Args:
            location: City name, optionally with country code
            units: Temperature units (metric, imperial, kelvin)
            
        Returns:
            Weather data
        """
        logger.info(f"[CURRENT_WEATHER] Location: {location}, Units: {units}")
        
        return await self.call_tool("current_weather", {
            "location": location,
            "units": units
        })
    
    async def get_forecast(self, location: str, days: int = 3, units: str = "metric") -> Dict[str, Any]:
        """
        Get weather forecast.
        
        Args:
            location: City name, optionally with country code
            days: Number of days (1-5)
            units: Temperature units
            
        Returns:
            Forecast data
        """
        logger.info(f"[FORECAST] Location: {location}, Days: {days}, Units: {units}")
        
        return await self.call_tool("weather_forecast", {
            "location": location,
            "days": days,
            "units": units
        })
    
    async def get_weather_by_coords(self, lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
        """
        Get weather by coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            units: Temperature units
            
        Returns:
            Weather data
        """
        logger.info(f"[WEATHER_COORDS] Lat: {lat}, Lon: {lon}, Units: {units}")
        
        return await self.call_tool("weather_by_coords", {
            "lat": lat,
            "lon": lon,
            "units": units
        })
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a weather tool following MCP protocol.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution results
        """
        logger.debug(f"[TOOL] Calling {tool_name} with args: {arguments}")
        
        # Check if tool exists
        tool_exists = any(tool.get("name") == tool_name for tool in self.tools)
        if not tool_exists:
            logger.error(f"[ERROR] Tool not found: {tool_name}")
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        
        start_time = datetime.now()
        
        try:
            if self.use_mock:
                # Use mock server
                call_request = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": self._next_message_id()
                }
                response = await self.mock_server.handle_request(call_request)
                
                # Check if mock server returned an error
                if "error" in response:
                    error_msg = response["error"].get("message", "Unknown mock server error")
                    raise Exception(error_msg)
                
                result = response.get("result", {})
            else:
                # Make real API call
                result = await self._execute_api_call(tool_name, arguments)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"[ERROR] Tool execution failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time
            }
    
    async def _execute_api_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actual OpenWeather API call."""
        # Check cache first
        cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self._cache_ttl:
                logger.debug(f"[CACHE] Returning cached result for {tool_name}")
                return cached['data']
        
        async with aiohttp.ClientSession() as session:
            if tool_name == "current_weather":
                url = f"{self.BASE_URL}/weather"
                params = {
                    "q": arguments["location"],
                    "units": arguments.get("units", "metric"),
                    "appid": self.api_key
                }
                
            elif tool_name == "weather_forecast":
                url = f"{self.BASE_URL}/forecast"
                params = {
                    "q": arguments["location"],
                    "units": arguments.get("units", "metric"),
                    "cnt": arguments.get("days", 3) * 8,  # 8 forecasts per day (3-hour intervals)
                    "appid": self.api_key
                }
                
            elif tool_name == "weather_by_coords":
                url = f"{self.BASE_URL}/weather"
                params = {
                    "lat": arguments["lat"],
                    "lon": arguments["lon"],
                    "units": arguments.get("units", "metric"),
                    "appid": self.api_key
                }
                
            elif tool_name == "air_pollution":
                url = f"{self.BASE_URL}/air_pollution"
                params = {
                    "lat": arguments["lat"],
                    "lon": arguments["lon"],
                    "appid": self.api_key
                }
                
            elif tool_name == "uv_index":
                url = f"{self.BASE_URL}/uvi"
                params = {
                    "lat": arguments["lat"],
                    "lon": arguments["lon"],
                    "appid": self.api_key
                }
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the result
                    self._cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
    
    async def disconnect(self) -> None:
        """Cleanup resources."""
        logger.info("[DISCONNECTING] Closing Weather MCP connection...")
        self._cache.clear()
        logger.info("[SUCCESS] Disconnected from Weather MCP")
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register Weather tools with the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        for tool in self.tools:
            tool_info = {
                'id': f'weather.{tool["name"]}',
                'name': tool['name'],
                'server_type': 'weather',
                'endpoint': 'OpenWeather API',
                'description': tool.get('description', ''),
                'capabilities': self.capabilities,
                'input_schema': tool.get('inputSchema', {})
            }
            registry.register_tool(tool_info)
            logger.info(f"[REGISTERED] Weather tool: {tool_info['id']}")


async def test_weather_mcp():
    """Test the Weather MCP client implementation."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Weather MCP Client")
    logger.info("=" * 60)
    
    # Initialize client with mock API key
    client = WeatherMCPClient(api_key="test_api_key")
    
    try:
        # Try to connect to real API first, fall back to mock if it fails
        connected = await client.connect()
        if not connected:
            logger.warning("⚠️  Could not connect to OpenWeather API")
            logger.info("💡 Trying with mock server instead...")
            connected = await client.connect(use_mock=True)
            if not connected:
                logger.error("[ERROR] Could not connect to mock server either")
                return
        
        # Test current weather
        weather_result = await client.get_current_weather("London,UK", "metric")
        logger.info(f"[CURRENT_WEATHER] Result: {weather_result}")
        
        # Test weather forecast
        forecast_result = await client.get_forecast("New York,US", days=3, units="imperial")
        logger.info(f"[FORECAST] Result: {forecast_result}")
        
        # Test weather by coordinates
        coords_result = await client.get_weather_by_coords(51.5074, -0.1278, "metric")
        logger.info(f"[WEATHER_COORDS] Result: {coords_result}")
        
        # Test air pollution
        pollution_result = await client.call_tool("air_pollution", {"lat": 51.5074, "lon": -0.1278})
        logger.info(f"[AIR_POLLUTION] Result: {pollution_result}")
        
        # Test UV index
        uv_result = await client.call_tool("uv_index", {"lat": 40.7128, "lon": -74.0060})
        logger.info(f"[UV_INDEX] Result: {uv_result}")
        
        # Test with tool registry
        registry = ToolRegistry("data/test_weather_registry.db")
        client.register_tools_to_registry(registry)
        
        # List registered Weather tools
        weather_tools = registry.list_tools("weather")
        logger.info(f"[REGISTRY] Registered {len(weather_tools)} Weather tools")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()
    
    logger.info("[TEST] Weather MCP test complete!")


if __name__ == "__main__":
    asyncio.run(test_weather_mcp())