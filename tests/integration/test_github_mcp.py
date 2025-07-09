#!/usr/bin/env python3
"""
Test script for GitHub MCP Integration

This script tests the GitHub MCP server integration including:
- PAT authentication
- Repository operations
- Issue management
- Pull request operations
- Mock server fallback
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path (go up 2 levels from tests/integration/)
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.core.mcp_integration import MCPIntegration
from src.tools.github_mcp import GitHubMCPClient

logger = get_logger(__name__)

async def test_direct_client():
    """Test GitHub MCP client directly."""
    logger.info("\n" + "="*60)
    logger.info("Testing GitHub MCP Client Directly")
    logger.info("="*60)
    
    # Test with mock server first
    logger.info("\n[TEST] Testing with mock server...")
    client = GitHubMCPClient()
    
    try:
        # Connect using mock
        connected = await client.connect(use_mock=True)
        logger.info(f"[MOCK] Connection status: {connected}")
        
        if connected:
            # Test list repos
            logger.info("\n[TEST] Testing list_repos...")
            repos = await client.execute_tool("list_repos", {"username": "test-user"})
            logger.info(f"[RESULT] Found {len(repos) if isinstance(repos, list) else 'N/A'} repositories")
            
            # Test search repos
            logger.info("\n[TEST] Testing search_repos...")
            search_results = await client.execute_tool("search_repos", {"q": "mcp"})
            logger.info(f"[RESULT] Search returned {search_results.get('total_count', 0)} results")
            
            # Test create issue
            logger.info("\n[TEST] Testing create_issue...")
            issue = await client.execute_tool("create_issue", {
                "owner": "test-user",
                "repo": "test-repo",
                "title": "Test Issue from MCP",
                "body": "This is a test issue created via GitHub MCP integration"
            })
            logger.info(f"[RESULT] Created issue #{issue.get('number', 'N/A')}")
            
            # Test list issues
            logger.info("\n[TEST] Testing list_issues...")
            issues = await client.execute_tool("list_issues", {
                "owner": "test-user",
                "repo": "test-repo",
                "state": "open"
            })
            logger.info(f"[RESULT] Found {len(issues) if isinstance(issues, list) else 'N/A'} open issues")
            
            # Test create pull request
            logger.info("\n[TEST] Testing create_pull...")
            pr = await client.execute_tool("create_pull", {
                "owner": "test-user",
                "repo": "test-repo",
                "title": "Test PR from MCP",
                "head": "feature-branch",
                "base": "main",
                "body": "This is a test PR created via GitHub MCP integration"
            })
            logger.info(f"[RESULT] Created PR #{pr.get('number', 'N/A')}")
            
            # Test code search
            logger.info("\n[TEST] Testing search_code...")
            code_results = await client.execute_tool("search_code", {
                "q": "mcp integration",
                "language": "py"
            })
            logger.info(f"[RESULT] Code search returned {code_results.get('total_count', 0)} results")
        
        await client.disconnect()
        
    except Exception as e:
        logger.error(f"[ERROR] Mock test failed: {e}")
    
    # Test with real server if token available
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        logger.info("\n[TEST] Testing with real GitHub server...")
        client = GitHubMCPClient(github_token)
        
        try:
            connected = await client.connect(use_mock=False)
            logger.info(f"[REAL] Connection status: {connected}")
            
            if connected:
                # Test with real API (read-only operations)
                logger.info("\n[TEST] Testing real API - list_repos...")
                repos = await client.execute_tool("list_repos", {})
                logger.info(f"[RESULT] Found {len(repos) if isinstance(repos, list) else 'N/A'} real repositories")
                
                logger.info("\n[TEST] Testing real API - search_repos...")
                search_results = await client.execute_tool("search_repos", {"q": "language:python stars:>1000"})
                logger.info(f"[RESULT] Search returned {search_results.get('total_count', 0)} popular Python repos")
            
            await client.disconnect()
            
        except Exception as e:
            logger.error(f"[ERROR] Real server test failed: {e}")
    else:
        logger.warning("[SKIP] No GITHUB_TOKEN found, skipping real server tests")

async def test_mcp_integration():
    """Test GitHub MCP through the integration layer."""
    logger.info("\n" + "="*60)
    logger.info("Testing GitHub MCP Integration Layer")
    logger.info("="*60)
    
    # Create registry and integration
    registry = ToolRegistry()
    integration = MCPIntegration(registry)
    
    # Test adding GitHub server (mock)
    logger.info("\n[TEST] Adding GitHub server to integration (mock mode)...")
    success = await integration.add_github_server(use_mock=True)
    logger.info(f"[RESULT] Server added: {success}")
    
    if success:
        # List all registered tools
        logger.info("\n[TEST] Listing all registered GitHub tools...")
        all_tools = registry.list_tools(server_type='github')
        github_tools = all_tools
        logger.info(f"[RESULT] Found {len(github_tools)} GitHub tools:")
        for tool in github_tools:
            logger.info(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        
        # Test tool execution through integration
        logger.info("\n[TEST] Executing tool through integration layer...")
        
        # Find a GitHub tool ID
        if github_tools:
            tool_id = github_tools[0]['id']
            logger.info(f"[TEST] Executing tool: {tool_id}")
            
            # Execute based on tool name
            if "list_repos" in tool_id:
                result = await integration.execute_tool(tool_id, {"username": "test-user"})
            elif "search_repos" in tool_id:
                result = await integration.execute_tool(tool_id, {"q": "test"})
            else:
                result = {"error": "Unknown tool type for test"}
            
            logger.info(f"[RESULT] Execution result: {result}")
    
    # Test with real server if token available
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        logger.info("\n[TEST] Adding GitHub server with real API...")
        success = await integration.add_github_server(
            github_token=github_token,
            server_id="github_real",
            use_mock=False
        )
        logger.info(f"[RESULT] Real server added: {success}")

async def test_error_handling():
    """Test error handling scenarios."""
    logger.info("\n" + "="*60)
    logger.info("Testing Error Handling")
    logger.info("="*60)
    
    client = GitHubMCPClient()
    
    # Test without connection
    logger.info("\n[TEST] Testing execution without connection...")
    try:
        await client.execute_tool("list_repos", {})
    except RuntimeError as e:
        logger.info(f"[EXPECTED] Got expected error: {e}")
    
    # Test with invalid tool
    await client.connect(use_mock=True)
    logger.info("\n[TEST] Testing invalid tool name...")
    try:
        result = await client.execute_tool("invalid_tool", {})
        logger.info(f"[RESULT] {result}")
    except Exception as e:
        logger.info(f"[EXPECTED] Got expected error: {e}")
    
    await client.disconnect()

async def main():
    """Run all tests."""
    logger.info("\n" + "#"*60)
    logger.info("GitHub MCP Integration Test Suite")
    logger.info("#"*60)
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Run tests
        await test_direct_client()
        await test_mcp_integration()
        await test_error_handling()
        
        logger.info("\n" + "#"*60)
        logger.info("All tests completed!")
        logger.info("#"*60)
        
    except Exception as e:
        logger.error(f"[FATAL] Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())