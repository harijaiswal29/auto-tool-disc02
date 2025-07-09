#!/usr/bin/env python3
"""
Test to discover and list all real GitHub MCP tools
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.tools.github_mcp import GitHubMCPClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def list_real_tools():
    """List all tools available in the real GitHub MCP server"""
    
    # Create client
    client = GitHubMCPClient()
    
    # Connect to real server
    logger.info("Connecting to real GitHub MCP server...")
    connected = await client.connect(use_mock=False)
    
    if connected and not client.use_mock:
        logger.info(f"✅ Connected to real server with {len(client.tools)} tools")
        logger.info("\nAvailable GitHub MCP Tools:")
        logger.info("=" * 60)
        
        for i, tool in enumerate(client.tools, 1):
            logger.info(f"\n{i}. {tool['name']}")
            logger.info(f"   Description: {tool.get('description', 'No description')}")
            if 'inputSchema' in tool:
                logger.info(f"   Parameters: {', '.join(tool['inputSchema'].get('properties', {}).keys())}")
    else:
        logger.warning("Failed to connect to real server or fell back to mock")
    
    await client.disconnect()

async def test_specific_tools():
    """Test specific tools with the real server"""
    
    client = GitHubMCPClient()
    connected = await client.connect(use_mock=False)
    
    if connected and not client.use_mock:
        logger.info("\n\nTesting specific tools:")
        logger.info("=" * 60)
        
        # Test search_repositories (note the different name)
        logger.info("\n1. Testing search_repositories...")
        try:
            result = await client.execute_tool("search_repositories", {
                "query": "language:python stars:>1000 topic:machine-learning"
            })
            logger.info(f"   Found {result.get('total_count', 0)} repositories")
            if 'items' in result and result['items']:
                logger.info(f"   Top result: {result['items'][0]['full_name']} ⭐ {result['items'][0]['stargazers_count']}")
        except Exception as e:
            logger.error(f"   Error: {e}")
    
    await client.disconnect()

async def main():
    """Run the tests"""
    await list_real_tools()
    await test_specific_tools()

if __name__ == "__main__":
    asyncio.run(main())