#!/usr/bin/env python3
"""
Demo script showing the REAL GitHub MCP server integration
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.core.tool_registry import ToolRegistry
from src.core.mcp_integration import MCPIntegration
from src.tools.github_mcp import GitHubMCPClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def demo_real_github():
    """Demonstrate real GitHub MCP integration"""
    
    # Check token
    if not os.environ.get('GITHUB_TOKEN'):
        logger.error("❌ GITHUB_TOKEN not set!")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 Real GitHub MCP Server Demo")
    logger.info("=" * 60)
    
    # Create direct client
    client = GitHubMCPClient()
    
    logger.info("\n📡 Connecting to real GitHub MCP server...")
    connected = await client.connect(use_mock=False)
    
    if connected and not client.use_mock:
        logger.info(f"✅ Successfully connected to REAL GitHub MCP server!")
        logger.info(f"📊 Discovered {len(client.tools)} tools")
        
        # Demo 1: Search for popular Python repositories
        logger.info("\n🔍 Demo 1: Searching for popular Python ML repositories...")
        try:
            result = await client.execute_tool("search_repositories", {
                "query": "language:python stars:>5000 machine-learning",
                "perPage": 5
            })
            
            total = result.get('total_count', 0)
            logger.info(f"Found {total} repositories!")
            
            if 'items' in result:
                logger.info("\nTop 5 results:")
                for i, repo in enumerate(result['items'][:5], 1):
                    logger.info(f"{i}. {repo['full_name']} ⭐ {repo['stargazers_count']:,}")
                    logger.info(f"   {repo.get('description', 'No description')[:80]}...")
        except Exception as e:
            logger.error(f"Search failed: {e}")
        
        # Demo 2: Get file contents from a repository
        logger.info("\n📄 Demo 2: Reading README from a repository...")
        try:
            result = await client.execute_tool("get_file_contents", {
                "owner": "microsoft",
                "repo": "vscode",
                "path": "README.md"
            })
            
            if isinstance(result, dict) and 'content' in result:
                logger.info("✅ Successfully retrieved README.md")
                logger.info(f"   Size: {result.get('size', 0)} bytes")
                logger.info(f"   SHA: {result.get('sha', 'N/A')[:8]}...")
            else:
                logger.info("File content retrieved (structure varies)")
        except Exception as e:
            logger.error(f"File read failed: {e}")
        
        # Demo 3: Search for code
        logger.info("\n💻 Demo 3: Searching for Python async code...")
        try:
            result = await client.execute_tool("search_code", {
                "q": "async def language:python",
                "per_page": 3
            })
            
            total = result.get('total_count', 0)
            logger.info(f"Found {total} code matches!")
            
            if 'items' in result:
                logger.info("\nTop 3 results:")
                for i, item in enumerate(result['items'][:3], 1):
                    logger.info(f"{i}. {item['repository']['full_name']}: {item['path']}")
        except Exception as e:
            logger.error(f"Code search failed: {e}")
        
        # Demo 4: List tools available
        logger.info("\n🛠️  Available GitHub MCP Tools:")
        logger.info("-" * 40)
        tool_categories = {
            'Repository': ['create_repository', 'fork_repository', 'search_repositories'],
            'Files': ['create_or_update_file', 'get_file_contents', 'push_files'],
            'Issues': ['create_issue', 'list_issues', 'update_issue', 'add_issue_comment'],
            'Pull Requests': ['create_pull_request', 'list_pull_requests', 'merge_pull_request'],
            'Search': ['search_code', 'search_issues', 'search_users'],
            'Branches': ['create_branch', 'list_commits']
        }
        
        for category, tools in tool_categories.items():
            available = [t for t in tools if any(tool['name'] == t for tool in client.tools)]
            if available:
                logger.info(f"\n{category}:")
                for tool in available:
                    logger.info(f"  • {tool}")
        
    else:
        logger.warning("⚠️  Failed to connect to real server or fell back to mock")
    
    await client.disconnect()
    logger.info("\n" + "=" * 60)
    logger.info("✨ Demo completed!")

async def demo_with_integration():
    """Demo using the integration layer"""
    logger.info("\n" + "=" * 60)
    logger.info("🔧 Testing with Integration Layer")
    logger.info("=" * 60)
    
    # Initialize system
    registry = ToolRegistry()
    integration = MCPIntegration(registry)
    
    # Add GitHub server (will attempt real connection)
    logger.info("\nAdding GitHub server to integration...")
    success = await integration.add_github_server(use_mock=False)
    
    if success:
        logger.info("✅ GitHub server added successfully!")
        
        # List registered tools
        github_tools = registry.list_tools(server_type='github')
        logger.info(f"\n📋 Registered {len(github_tools)} GitHub tools in registry")
        
        # Execute a tool through integration
        logger.info("\n🔍 Executing search through integration layer...")
        result = await integration.execute_tool("github.search_repositories", {
            "query": "stars:>50000",
            "perPage": 3
        })
        
        if result.get('success'):
            search_result = result['result']
            logger.info(f"Found {search_result.get('total_count', 0)} repositories with >50k stars")
            if 'items' in search_result:
                logger.info("\nTop 3:")
                for repo in search_result['items'][:3]:
                    logger.info(f"  • {repo['full_name']} ⭐ {repo['stargazers_count']:,}")

async def main():
    """Run the demo"""
    await demo_real_github()
    await demo_with_integration()

if __name__ == "__main__":
    # Ensure token is set
    if not os.environ.get('GITHUB_TOKEN'):
        logger.error("Please set GITHUB_TOKEN environment variable")
        logger.info("export GITHUB_TOKEN=your_github_pat_here")
        sys.exit(1)
    
    asyncio.run(main())