#!/usr/bin/env python3
"""
Demo script showing how to use GitHub MCP in the autonomous tool discovery system
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.core.tool_registry import ToolRegistry
from src.core.mcp_integration import MCPIntegration
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def demo_github_integration():
    """Demonstrate GitHub MCP integration with the autonomous tool discovery system."""
    
    # Initialize the system
    logger.info("Initializing Autonomous Tool Discovery System with GitHub MCP...")
    registry = ToolRegistry()
    integration = MCPIntegration(registry)
    
    # Add GitHub server (will use token from environment)
    logger.info("\nAdding GitHub MCP server...")
    success = await integration.add_github_server()
    
    if not success:
        logger.error("Failed to add GitHub server. Make sure GITHUB_TOKEN is set.")
        return
    
    logger.info("✅ GitHub MCP server successfully integrated!")
    
    # Show available GitHub tools
    logger.info("\n📋 Available GitHub Tools:")
    github_tools = registry.list_tools(server_type='github')
    for tool in github_tools:
        logger.info(f"  • {tool['name']}: {tool['description']}")
    
    # Example 1: List your repositories
    logger.info("\n🔍 Example 1: Listing your repositories...")
    result = await integration.execute_tool("github.list_repos", {"username": None})
    if result.get('success'):
        repos = result['result']
        logger.info(f"Found {len(repos)} repositories:")
        for repo in repos[:5]:  # Show first 5
            logger.info(f"  - {repo['full_name']} ⭐ {repo['stargazers_count']}")
    
    # Example 2: Search for repositories
    logger.info("\n🔍 Example 2: Searching for Python MCP repositories...")
    result = await integration.execute_tool("github.search_repos", {
        "q": "language:python mcp",
        "sort": "stars"
    })
    if result.get('success'):
        search_results = result['result']
        logger.info(f"Found {search_results.get('total_count', 0)} repositories")
        items = search_results.get('items', [])
        for repo in items[:3]:  # Show top 3
            logger.info(f"  - {repo['full_name']} ⭐ {repo['stargazers_count']}")
    
    # Example 3: Tool discovery simulation
    logger.info("\n🤖 Example 3: Autonomous Tool Discovery Simulation")
    logger.info("User Query: 'I need to find popular Python AI projects on GitHub'")
    
    # Simulate the system discovering that GitHub search tool is appropriate
    logger.info("System: Analyzing query intent...")
    logger.info("System: Discovered relevant tool: github.search_repos")
    
    # Execute the discovered tool
    result = await integration.execute_tool("github.search_repos", {
        "q": "language:python AI machine learning stars:>100",
        "sort": "stars"
    })
    
    if result.get('success'):
        search_results = result['result']
        logger.info(f"System: Found {search_results.get('total_count', 0)} relevant repositories")
        items = search_results.get('items', [])
        logger.info("\nTop AI/ML Python projects:")
        for repo in items[:5]:
            logger.info(f"  📦 {repo['full_name']}")
            logger.info(f"     ⭐ Stars: {repo['stargazers_count']}")
            logger.info(f"     📝 {repo.get('description', 'No description')[:80]}...")
            logger.info("")
    
    # Show performance metrics
    logger.info("\n📊 Tool Performance Metrics:")
    github_tools = registry.list_tools(server_type='github')
    for tool in github_tools:
        if tool['usage_count'] > 0:
            logger.info(f"  • {tool['name']}: {tool['usage_count']} uses, "
                       f"{tool['performance_score']:.2%} success rate")

async def main():
    """Run the demo."""
    logger.info("="*60)
    logger.info("GitHub MCP Integration Demo")
    logger.info("="*60)
    
    # Check for token
    if not os.environ.get('GITHUB_TOKEN'):
        logger.error("❌ GITHUB_TOKEN environment variable not set!")
        logger.info("Please set it with: export GITHUB_TOKEN=your_token_here")
        return
    
    await demo_github_integration()
    
    logger.info("\n✨ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())