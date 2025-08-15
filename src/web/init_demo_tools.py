#!/usr/bin/env python3
"""
Initialize demo tools for the web demonstration.
This ensures the tool registry has some tools available for demonstration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def init_demo_tools():
    """Initialize demo tools in the registry."""
    
    # Initialize registry
    registry_path = "data/registry/tools.db"
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    registry = ToolRegistry(registry_path)
    await registry.initialize()
    
    # DISABLED: Demo tools cause conflicts with actual MCP server tools
    # The system should use tools registered by MCP servers instead
    demo_tools = [
    ] if False else [  # Disabled to prevent conflicts
        {
            "id": "demo_search_tool",
            "name": "Demo Search Tool",
            "type": "search",
            "capabilities": {
                "operations": ["search", "query", "find"],
                "data_types": ["text", "documents"]
            },
            "performance_score": 0.85
        },
        {
            "id": "demo_web_search_tool",
            "name": "Demo Web Search Tool",
            "type": "search",
            "capabilities": {
                "operations": ["search", "query", "web"],
                "data_types": ["web", "urls"]
            },
            "performance_score": 0.82
        },
        {
            "id": "demo_filesystem_tool",
            "name": "Demo File System Tool",
            "type": "filesystem",
            "capabilities": {
                "operations": ["list", "read", "find"],
                "data_types": ["files", "directories"]
            },
            "performance_score": 0.90
        },
        {
            "id": "demo_database_tool",
            "name": "Demo Database Tool",
            "type": "database",
            "capabilities": {
                "operations": ["query", "list", "retrieve"],
                "data_types": ["tables", "records"]
            },
            "performance_score": 0.88
        },
        {
            "id": "demo_weather_tool",
            "name": "Demo Weather Tool",
            "type": "weather",
            "capabilities": {
                "operations": ["get", "fetch", "retrieve"],
                "data_types": ["weather", "forecast"]
            },
            "performance_score": 0.92
        },
        {
            "id": "demo_analytics_tool",
            "name": "Demo Analytics Tool",
            "type": "analytics",
            "capabilities": {
                "operations": ["analyze", "examine", "evaluate"],
                "data_types": ["data", "metrics"]
            },
            "performance_score": 0.80
        }
    ]
    
    # Skip demo tool registration - use only MCP server tools
    registered_count = 0
    for tool_data in []:
        try:
            # Check if tool already exists
            existing_tools = await registry.search_tools(tool_data["name"])
            if not any(t.get("id") == tool_data["id"] for t in existing_tools):
                # Prepare tool info in the format expected by register_tool
                tool_info = {
                    "id": tool_data["id"],
                    "name": tool_data["name"],
                    "server_type": tool_data["type"],
                    "endpoint": f"demo://{tool_data['id']}",  # Mock endpoint
                    "description": f"Demo {tool_data['type']} tool for testing",
                    "capabilities": tool_data["capabilities"],
                    "input_schema": {},  # Empty schema for demo
                    "performance_score": tool_data.get("performance_score", 0.8)
                }
                registry.register_tool(tool_info)
                registered_count += 1
                logger.info(f"Registered demo tool: {tool_data['name']}")
            else:
                logger.info(f"Demo tool already exists: {tool_data['name']}")
        except Exception as e:
            logger.error(f"Failed to register tool {tool_data['name']}: {e}")
    
    # Tool relationships - only add if demo tools exist
    relationships = [
        # Disabled since demo tools are not registered
        # ("demo_search_tool", "demo_database_tool", "complements"),
        # ("demo_filesystem_tool", "demo_analytics_tool", "complements"),
    ]
    
    for tool1, tool2, rel_type in relationships:
        try:
            await registry.add_tool_relationship(tool1, tool2, rel_type)
            logger.info(f"Added relationship: {tool1} {rel_type} {tool2}")
        except Exception as e:
            logger.debug(f"Relationship might already exist: {e}")
    
    logger.info(f"Demo tools initialization complete. Registered {registered_count} new tools.")
    
    # List all tools
    all_tools = registry.list_tools()
    logger.info(f"Total tools in registry: {len(all_tools)}")
    for tool in all_tools:
        logger.info(f"  - {tool.get('name')} ({tool.get('type')})")
    
    await registry.close()


if __name__ == "__main__":
    print("Initializing demo tools...")
    asyncio.run(init_demo_tools())
    print("Demo tools initialized successfully!")