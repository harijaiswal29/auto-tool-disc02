#!/usr/bin/env python3
"""
Start Mock MCP Servers

This script starts mock MCP servers that simulate real MCP functionality
without requiring external APIs or npm packages.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tools.mock_mcp_servers import start_all_mock_servers
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Start all mock MCP servers."""
    logger.info("Starting Mock MCP Servers")
    logger.info("========================")
    logger.info("")
    logger.info("This will start mock implementations of all MCP tools:")
    logger.info("- Filesystem MCP (read/write files)")
    logger.info("- SQLite MCP (database operations)")
    logger.info("- Search MCP (web search simulation)")
    logger.info("- GitHub MCP (repository operations)")
    logger.info("- PostgreSQL MCP (database operations)")
    logger.info("- Weather MCP (weather data)")
    logger.info("- Financial MCP (market data)")
    logger.info("- Notion MCP (note operations)")
    logger.info("- Zerodha MCP (trading operations)")
    logger.info("")
    
    try:
        await start_all_mock_servers()
    except KeyboardInterrupt:
        logger.info("\nShutting down mock servers...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())