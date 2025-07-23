#!/usr/bin/env python3
"""
Initialize Tool Registry Database

This script initializes the tool registry database with all MCP tools.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def initialize_tool_registry():
    """Initialize the tool registry with all MCP tools."""
    logger.info("Initializing Tool Registry Database...")
    
    # Create tool registry instance
    registry = ToolRegistry(db_path="data/registry/tools.db")
    
    # Define all 9 MCP tools
    tools = [
        {
            "id": "sqlite_mcp",
            "name": "SQLite MCP",
            "type": "mcp",
            "endpoint": "stdio://mcp-server-sqlite",
            "capabilities": {
                "operations": ["query", "execute", "schema"],
                "constraints": {"max_query_size": 10000},
                "semantic_tags": ["database", "sql", "storage", "query"]
            },
            "metadata": {
                "description": "SQLite database operations via MCP",
                "version": "1.0.0",
                "requires_config": {"database_path": "str"}
            }
        },
        {
            "id": "search_mcp",
            "name": "Search MCP",
            "type": "mcp",
            "endpoint": "stdio://mcp-server-brave-search",
            "capabilities": {
                "operations": ["search", "news", "images"],
                "constraints": {"max_results": 100},
                "semantic_tags": ["search", "web", "information", "query"]
            },
            "metadata": {
                "description": "Web search capabilities via Brave Search API",
                "version": "1.0.0",
                "requires_config": {"api_key": "str"}
            }
        },
        {
            "id": "weather_mcp",
            "name": "Weather MCP",
            "type": "api",
            "endpoint": "https://api.weatherapi.com/v1",
            "capabilities": {
                "operations": ["current", "forecast", "search"],
                "constraints": {"max_forecast_days": 14},
                "semantic_tags": ["weather", "forecast", "climate", "temperature"]
            },
            "metadata": {
                "description": "Weather information retrieval via MCP",
                "version": "1.0.0",
                "requires_config": {"api_key": "str"}
            }
        },
        {
            "id": "filesystem_mcp",
            "name": "Filesystem MCP",
            "type": "mcp",
            "endpoint": "stdio://mcp-server-filesystem",
            "capabilities": {
                "operations": ["read_file", "write_file", "list_directory", "delete_file"],
                "constraints": {"max_file_size_mb": 100},
                "semantic_tags": ["filesystem", "file", "directory", "storage", "io"]
            },
            "metadata": {
                "description": "Filesystem operations via MCP",
                "version": "1.0.0",
                "requires_config": {"base_path": "str"}
            }
        },
        {
            "id": "postgres_mcp",
            "name": "PostgreSQL MCP",
            "type": "mcp",
            "endpoint": "stdio://mcp-server-postgres",
            "capabilities": {
                "operations": ["query", "execute", "schema", "transaction"],
                "constraints": {"max_query_size": 50000},
                "semantic_tags": ["database", "postgresql", "sql", "relational"]
            },
            "metadata": {
                "description": "PostgreSQL database operations via MCP",
                "version": "1.0.0",
                "requires_config": {"connection_string": "str"}
            }
        },
        {
            "id": "github_mcp",
            "name": "GitHub MCP",
            "type": "mcp",
            "endpoint": "stdio://mcp-server-github",
            "capabilities": {
                "operations": ["search_repositories", "get_issues", "create_issue", "get_pull_requests"],
                "constraints": {"rate_limit": 5000},
                "semantic_tags": ["github", "git", "repository", "code", "version_control"]
            },
            "metadata": {
                "description": "GitHub API operations via MCP",
                "version": "1.0.0",
                "requires_config": {"token": "str"}
            }
        },
        {
            "id": "financial_datasets_mcp",
            "name": "Financial Datasets MCP",
            "type": "api",
            "endpoint": "https://api.financialdatasets.ai",
            "capabilities": {
                "operations": ["get_stock_data", "get_fundamentals", "search_symbols"],
                "constraints": {"max_timeframe_days": 3650},
                "semantic_tags": ["finance", "stocks", "market", "trading", "investment"]
            },
            "metadata": {
                "description": "Financial market data retrieval via MCP",
                "version": "1.0.0",
                "requires_config": {"api_key": "str"}
            }
        },
        {
            "id": "zerodha_mcp",
            "name": "Zerodha MCP",
            "type": "api",
            "endpoint": "https://api.kite.trade",
            "capabilities": {
                "operations": ["get_holdings", "place_order", "get_positions", "get_margins"],
                "constraints": {"max_orders_per_second": 10},
                "semantic_tags": ["trading", "stocks", "portfolio", "investment", "zerodha"]
            },
            "metadata": {
                "description": "Trading and portfolio management via Zerodha Kite API",
                "version": "1.0.0",
                "requires_config": {"api_key": "str", "api_secret": "str"}
            }
        },
        {
            "id": "notion_mcp",
            "name": "Notion MCP",
            "type": "api",
            "endpoint": "https://api.notion.com/v1",
            "capabilities": {
                "operations": ["search", "create_page", "update_page", "query_database"],
                "constraints": {"max_blocks_per_page": 1000},
                "semantic_tags": ["notion", "notes", "documentation", "workspace", "productivity"]
            },
            "metadata": {
                "description": "Notion workspace integration via MCP",
                "version": "1.0.0",
                "requires_config": {"api_key": "str"}
            }
        }
    ]
    
    # Define tool relationships
    relationships = [
        # Complementary relationships
        ("sqlite_mcp", "filesystem_mcp", "complements", 0.8),  # Often used together for data storage
        ("search_mcp", "weather_mcp", "complements", 0.6),  # Search for location, get weather
        ("github_mcp", "filesystem_mcp", "complements", 0.9),  # Code management and file operations
        ("financial_datasets_mcp", "zerodha_mcp", "complements", 0.9),  # Market data and trading
        ("notion_mcp", "github_mcp", "complements", 0.7),  # Documentation and code
        
        # Enhancement relationships
        ("postgres_mcp", "sqlite_mcp", "enhances", 0.7),  # PostgreSQL enhances SQLite capabilities
        ("zerodha_mcp", "financial_datasets_mcp", "enhances", 0.8),  # Trading enhances data analysis
        
        # Requires relationships
        ("zerodha_mcp", "financial_datasets_mcp", "requires", 0.6),  # Trading may need market data
    ]
    
    # Register all tools
    for tool in tools:
        tool_info = {
            "id": tool["id"],
            "name": tool["name"],
            "server_type": tool["type"],
            "endpoint": tool["endpoint"],
            "description": tool["metadata"]["description"],
            "capabilities": tool["capabilities"],
            "input_schema": tool["metadata"].get("requires_config", {})
        }
        
        success = registry.register_tool(tool_info)
        
        if success:
            logger.info(f"✅ Registered: {tool['name']}")
        else:
            logger.warning(f"⚠️  Failed to register: {tool['name']}")
    
    # Add relationships
    # Note: The current ToolRegistry doesn't have add_relationship method
    # Relationships can be added manually to the database if needed
    logger.info("\n📝 Note: Tool relationships defined but not added (method not available in current ToolRegistry)")
    
    # Verify registration
    all_tools = registry.get_all_tools()
    logger.info(f"\n📊 Total tools registered: {len(all_tools)}")
    
    # Display registered tools
    logger.info("\n🛠️  Registered Tools:")
    for tool in all_tools:
        logger.info(f"  - {tool['name']} ({tool['id']}): {tool['server_type']}")
    
    # Test search functionality
    logger.info("\n🔍 Testing search functionality:")
    
    # Search for database tools
    db_tools = await registry.search_tools("database")
    logger.info(f"  Database tools found: {len(db_tools)}")
    for tool in db_tools[:5]:  # Limit to first 5
        logger.info(f"    - {tool['name']} (score: {tool.get('score', 'N/A')})")
    
    # Search for financial tools
    finance_tools = await registry.search_tools("finance trading")
    logger.info(f"  Finance tools found: {len(finance_tools)}")
    for tool in finance_tools[:5]:  # Limit to first 5
        logger.info(f"    - {tool['name']} (score: {tool.get('score', 'N/A')})")
    
    logger.info("\n✅ Tool Registry initialization complete!")


if __name__ == "__main__":
    asyncio.run(initialize_tool_registry())