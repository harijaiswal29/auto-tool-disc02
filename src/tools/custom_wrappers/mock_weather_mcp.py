"""
Mock Weather MCP Server

A mock implementation of Weather MCP server for testing without requiring
actual OpenWeather API calls.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockWeatherMCPServer:
    """
    Mock Weather MCP server that simulates OpenWeather API responses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        self.tools = self._define_tools()
        
        # Mock weather conditions
        self.weather_conditions = [
            "Clear", "Clouds", "Rain", "Drizzle", "Thunderstorm", 
            "Snow", "Mist", "Fog", "Haze"
        ]
        
        # Mock city coordinates
        self.city_coords = {
            "London,UK": {"lat": 51.5074, "lon": -0.1278},
            "New York,US": {"lat": 40.7128, "lon": -74.0060},
            "Tokyo,JP": {"lat": 35.6762, "lon": 139.6503},
            "Paris,FR": {"lat": 48.8566, "lon": 2.3522},
            "Sydney,AU": {"lat": -33.8688, "lon": 151.2093}
        }
        
        logger.info(f"[MOCK] Mock Weather MCP Server initialized")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available weather tools (same as real client)."""
        return [
            {
                "name": "current_weather",
                "description": "Get current weather conditions for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, state code and country code"
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
                            "default": 3
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
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
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
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
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
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
                    },
                    "required": ["lat", "lon"]
                }
            }
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.debug(f"[MOCK] Handling request: {method}")
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return await self._handle_tool_call(request_id, params)
        else:
            return self._error_response(request_id, f"Unknown method: {method}")
    
    def _handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        self.initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "MockWeatherMCPServer",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True,
                    "resources": False
                }
            }
        }
    
    def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def _handle_tool_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "current_weather":
                result = await self._mock_current_weather(arguments)
            elif tool_name == "weather_forecast":
                result = await self._mock_weather_forecast(arguments)
            elif tool_name == "weather_by_coords":
                result = await self._mock_weather_by_coords(arguments)
            elif tool_name == "air_pollution":
                result = await self._mock_air_pollution(arguments)
            elif tool_name == "uv_index":
                result = await self._mock_uv_index(arguments)
            else:
                return self._error_response(request_id, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self._error_response(request_id, str(e))
    
    def _convert_temperature(self, temp_celsius: float, units: str) -> float:
        """Convert temperature to requested units."""
        if units == "imperial":
            return (temp_celsius * 9/5) + 32  # Fahrenheit
        elif units == "kelvin":
            return temp_celsius + 273.15
        return temp_celsius  # metric (Celsius)
    
    async def _mock_current_weather(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock current weather data."""
        location = args.get("location", "Unknown")
        units = args.get("units", "metric")
        
        # Get coordinates for known cities or generate random
        coords = self.city_coords.get(location, {
            "lat": random.uniform(-90, 90),
            "lon": random.uniform(-180, 180)
        })
        
        # Generate mock weather data
        base_temp = 20 + random.uniform(-15, 15)  # Base temperature in Celsius
        weather_condition = random.choice(self.weather_conditions)
        
        return {
            "coord": coords,
            "weather": [{
                "id": random.randint(200, 800),
                "main": weather_condition,
                "description": f"{weather_condition.lower()} sky",
                "icon": "01d"
            }],
            "base": "stations",
            "main": {
                "temp": self._convert_temperature(base_temp, units),
                "feels_like": self._convert_temperature(base_temp - random.uniform(0, 3), units),
                "temp_min": self._convert_temperature(base_temp - 3, units),
                "temp_max": self._convert_temperature(base_temp + 3, units),
                "pressure": random.randint(1000, 1020),
                "humidity": random.randint(30, 90)
            },
            "visibility": random.randint(5000, 10000),
            "wind": {
                "speed": random.uniform(0, 10),
                "deg": random.randint(0, 360)
            },
            "clouds": {
                "all": random.randint(0, 100)
            },
            "dt": int(datetime.now().timestamp()),
            "sys": {
                "type": 1,
                "id": random.randint(1000, 9999),
                "country": location.split(',')[-1] if ',' in location else "XX",
                "sunrise": int((datetime.now().replace(hour=6, minute=0)).timestamp()),
                "sunset": int((datetime.now().replace(hour=18, minute=0)).timestamp())
            },
            "timezone": 0,
            "id": random.randint(1000000, 9999999),
            "name": location.split(',')[0],
            "cod": 200
        }
    
    async def _mock_weather_forecast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock weather forecast data."""
        location = args.get("location", "Unknown")
        days = args.get("days", 3)
        units = args.get("units", "metric")
        
        # Get coordinates
        coords = self.city_coords.get(location, {
            "lat": random.uniform(-90, 90),
            "lon": random.uniform(-180, 180)
        })
        
        # Generate forecast list (8 entries per day for 3-hour intervals)
        forecast_list = []
        current_time = datetime.now()
        base_temp = 20 + random.uniform(-15, 15)
        
        for i in range(days * 8):
            forecast_time = current_time + timedelta(hours=i*3)
            temp_variation = random.uniform(-5, 5)
            weather_condition = random.choice(self.weather_conditions)
            
            forecast_list.append({
                "dt": int(forecast_time.timestamp()),
                "main": {
                    "temp": self._convert_temperature(base_temp + temp_variation, units),
                    "feels_like": self._convert_temperature(base_temp + temp_variation - 2, units),
                    "temp_min": self._convert_temperature(base_temp + temp_variation - 3, units),
                    "temp_max": self._convert_temperature(base_temp + temp_variation + 3, units),
                    "pressure": random.randint(1000, 1020),
                    "humidity": random.randint(30, 90)
                },
                "weather": [{
                    "id": random.randint(200, 800),
                    "main": weather_condition,
                    "description": f"{weather_condition.lower()}",
                    "icon": "01d"
                }],
                "clouds": {"all": random.randint(0, 100)},
                "wind": {
                    "speed": random.uniform(0, 10),
                    "deg": random.randint(0, 360)
                },
                "visibility": random.randint(5000, 10000),
                "pop": random.uniform(0, 1),  # Probability of precipitation
                "dt_txt": forecast_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "cod": "200",
            "message": 0,
            "cnt": len(forecast_list),
            "list": forecast_list,
            "city": {
                "id": random.randint(1000000, 9999999),
                "name": location.split(',')[0],
                "coord": coords,
                "country": location.split(',')[-1] if ',' in location else "XX",
                "population": random.randint(100000, 10000000),
                "timezone": 0,
                "sunrise": int((current_time.replace(hour=6, minute=0)).timestamp()),
                "sunset": int((current_time.replace(hour=18, minute=0)).timestamp())
            }
        }
    
    async def _mock_weather_by_coords(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock weather data by coordinates."""
        lat = args.get("lat", 0)
        lon = args.get("lon", 0)
        units = args.get("units", "metric")
        
        # Find nearest city or use coordinates
        nearest_city = "Unknown Location"
        for city, coords in self.city_coords.items():
            if abs(coords["lat"] - lat) < 1 and abs(coords["lon"] - lon) < 1:
                nearest_city = city
                break
        
        # Use the current weather mock with coordinates
        result = await self._mock_current_weather({
            "location": nearest_city,
            "units": units
        })
        
        # Update coordinates to match request
        result["coord"] = {"lat": lat, "lon": lon}
        
        return result
    
    async def _mock_air_pollution(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock air pollution data."""
        lat = args.get("lat", 0)
        lon = args.get("lon", 0)
        
        # Generate AQI (Air Quality Index) 1-5
        aqi = random.randint(1, 5)
        
        return {
            "coord": {"lat": lat, "lon": lon},
            "list": [{
                "main": {
                    "aqi": aqi
                },
                "components": {
                    "co": random.uniform(200, 1000),  # μg/m3
                    "no": random.uniform(0, 50),
                    "no2": random.uniform(0, 200),
                    "o3": random.uniform(0, 180),
                    "so2": random.uniform(0, 250),
                    "pm2_5": random.uniform(0, 75),
                    "pm10": random.uniform(0, 150),
                    "nh3": random.uniform(0, 200)
                },
                "dt": int(datetime.now().timestamp())
            }]
        }
    
    async def _mock_uv_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock UV index data."""
        lat = args.get("lat", 0)
        lon = args.get("lon", 0)
        
        # Generate UV index based on latitude (higher near equator)
        base_uv = 11 - (abs(lat) / 90 * 10)  # Max 11 at equator, lower at poles
        uv_value = max(0, base_uv + random.uniform(-2, 2))
        
        return {
            "lat": lat,
            "lon": lon,
            "date_iso": datetime.now().isoformat(),
            "date": int(datetime.now().timestamp()),
            "value": round(uv_value, 2)
        }
    
    def _error_response(self, request_id: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }


async def test_mock_server():
    """Test the mock Weather MCP server."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Mock Weather MCP Server")
    logger.info("=" * 60)
    
    # Create mock server
    server = MockWeatherMCPServer()
    
    # Test initialization
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0",
            "clientInfo": {"name": "TestClient", "version": "0.1.0"}
        },
        "id": 1
    }
    init_response = await server.handle_request(init_request)
    logger.info(f"[INIT] Response: {init_response}")
    
    # Test tools list
    list_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }
    list_response = await server.handle_request(list_request)
    logger.info(f"[TOOLS] Available tools: {len(list_response['result']['tools'])}")
    
    # Test current weather
    weather_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "current_weather",
            "arguments": {
                "location": "London,UK",
                "units": "metric"
            }
        },
        "id": 3
    }
    weather_response = await server.handle_request(weather_request)
    logger.info(f"[CURRENT_WEATHER] Temperature: {weather_response['result']['main']['temp']}°C")
    
    # Test forecast
    forecast_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "weather_forecast",
            "arguments": {
                "location": "New York,US",
                "days": 2,
                "units": "imperial"
            }
        },
        "id": 4
    }
    forecast_response = await server.handle_request(forecast_request)
    logger.info(f"[FORECAST] Entries: {len(forecast_response['result']['list'])}")
    
    # Test UV index
    uv_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "uv_index",
            "arguments": {
                "lat": 0,  # Equator
                "lon": 0
            }
        },
        "id": 5
    }
    uv_response = await server.handle_request(uv_request)
    logger.info(f"[UV_INDEX] UV Value: {uv_response['result']['value']}")
    
    logger.info("[TEST] Mock server test complete!")


if __name__ == "__main__":
    asyncio.run(test_mock_server())