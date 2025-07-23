#!/usr/bin/env python3
"""
Comprehensive Integration Test for All MCP Tools

This script tests all 9 MCP tools in the system to ensure they are properly
integrated and functioning correctly.
"""

import asyncio
import os
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPToolIntegrationTester:
    """Test all MCP tools for basic functionality."""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.results = {}
    
    async def test_all_tools(self):
        """Run tests for all registered MCP tools."""
        logger.info("=" * 60)
        logger.info("MCP Tools Integration Test Suite")
        logger.info("=" * 60)
        
        # Get all registered tools
        all_tools = self.registry.get_all_tools()
        logger.info(f"\nFound {len(all_tools)} registered tools")
        
        # Test each tool
        for tool in all_tools:
            await self.test_tool(tool)
        
        # Print summary
        self.print_summary()
    
    async def test_tool(self, tool):
        """Test a single tool for basic functionality."""
        tool_id = tool['id']
        tool_name = tool['name']
        
        logger.info(f"\n[TEST] Testing {tool_name} ({tool_id})")
        logger.info("-" * 40)
        
        try:
            # Check tool registration
            logger.info(f"✓ Tool registered in registry")
            logger.info(f"  Type: {tool.get('server_type', 'Unknown')}")
            logger.info(f"  Endpoint: {tool.get('endpoint', 'Unknown')}")
            
            # Check capabilities
            capabilities = tool.get('capabilities', {})
            if capabilities:
                operations = capabilities.get('operations', [])
                logger.info(f"✓ Capabilities defined")
                logger.info(f"  Operations: {operations}")
                
                semantic_tags = capabilities.get('semantic_tags', [])
                if semantic_tags:
                    logger.info(f"  Tags: {semantic_tags}")
            
            # Tool-specific tests
            if tool_id == "filesystem_mcp":
                await self.test_filesystem_mcp()
            elif tool_id == "sqlite_mcp":
                await self.test_sqlite_mcp()
            elif tool_id == "search_mcp":
                await self.test_search_mcp()
            elif tool_id == "weather_mcp":
                await self.test_weather_mcp()
            elif tool_id == "postgres_mcp":
                await self.test_postgres_mcp()
            elif tool_id == "github_mcp":
                await self.test_github_mcp()
            elif tool_id == "financial_datasets_mcp":
                await self.test_financial_datasets_mcp()
            elif tool_id == "zerodha_mcp":
                await self.test_zerodha_mcp()
            elif tool_id == "notion_mcp":
                await self.test_notion_mcp()
            
            self.results[tool_id] = {"status": "PASSED", "message": "Basic checks passed"}
            logger.info(f"✅ {tool_name} tests PASSED")
            
        except Exception as e:
            self.results[tool_id] = {"status": "FAILED", "message": str(e)}
            logger.error(f"❌ {tool_name} tests FAILED: {e}")
    
    async def test_filesystem_mcp(self):
        """Test Filesystem MCP functionality."""
        from src.tools.filesystem_mcp import FilesystemMCPClient
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            test_file = f.name
            f.write("Test content")
        
        try:
            client = FilesystemMCPClient()
            await client.connect(use_mock=True)
            
            # Test read operation
            result = await client.read_file(test_file)
            assert "Test content" in str(result), "Failed to read file content"
            logger.info("  ✓ File read operation successful")
            
            await client.disconnect()
        finally:
            os.unlink(test_file)
    
    async def test_sqlite_mcp(self):
        """Test SQLite MCP functionality."""
        from src.tools.sqlite_mcp import SQLiteMCPClient
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            client = SQLiteMCPClient(db_path)
            await client.connect(use_mock=True)
            logger.info("  ✓ SQLite connection successful")
            await client.disconnect()
        finally:
            os.unlink(db_path)
    
    async def test_search_mcp(self):
        """Test Search MCP functionality."""
        from src.tools.search_mcp import SearchMCPClient
        
        client = SearchMCPClient(api_key="test_key")
        await client.connect(use_mock=True)
        
        # Test search operation
        result = await client.search("test query")
        assert result is not None, "Search returned no results"
        logger.info("  ✓ Search operation successful")
        
        await client.disconnect()
    
    async def test_weather_mcp(self):
        """Test Weather MCP functionality."""
        from src.tools.custom_wrappers.weather_mcp import WeatherMCPClient
        
        client = WeatherMCPClient(api_key="test_key")
        # Weather MCP doesn't have connect/disconnect
        logger.info("  ✓ Weather MCP client initialized")
    
    async def test_postgres_mcp(self):
        """Test PostgreSQL MCP functionality."""
        from src.tools.postgres_mcp import PostgreSQLMCPClient
        
        client = PostgreSQLMCPClient(connection_string="postgresql://test:test@localhost/test")
        # Just test initialization since we don't have a real postgres instance
        logger.info("  ✓ PostgreSQL MCP client initialized")
    
    async def test_github_mcp(self):
        """Test GitHub MCP functionality."""
        from src.tools.github_mcp import GitHubMCPClient
        
        client = GitHubMCPClient(token="test_token")
        await client.connect(use_mock=True)
        logger.info("  ✓ GitHub MCP mock connection successful")
        await client.disconnect()
    
    async def test_financial_datasets_mcp(self):
        """Test Financial Datasets MCP functionality."""
        from src.tools.financial_datasets_mcp import FinancialDatasetsMCPClient
        
        client = FinancialDatasetsMCPClient(api_key="test_key")
        logger.info("  ✓ Financial Datasets MCP client initialized")
    
    async def test_zerodha_mcp(self):
        """Test Zerodha MCP functionality."""
        from src.tools.zerodha_mcp import ZerodhaMCPClient
        
        client = ZerodhaMCPClient(api_key="test_key", api_secret="test_secret")
        logger.info("  ✓ Zerodha MCP client initialized")
    
    async def test_notion_mcp(self):
        """Test Notion MCP functionality."""
        from src.tools.notion_mcp import NotionMCPClient
        
        client = NotionMCPClient(api_key="test_key")
        logger.info("  ✓ Notion MCP client initialized")
    
    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAILED")
        
        logger.info(f"\nTotal Tools Tested: {len(self.results)}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        
        if failed > 0:
            logger.info("\nFailed Tools:")
            for tool_id, result in self.results.items():
                if result["status"] == "FAILED":
                    logger.error(f"  - {tool_id}: {result['message']}")
        
        logger.info("\nDetailed Results:")
        for tool_id, result in sorted(self.results.items()):
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            logger.info(f"  {status_icon} {tool_id}: {result['status']}")


async def main():
    """Run the integration test suite."""
    tester = MCPToolIntegrationTester()
    await tester.test_all_tools()


if __name__ == "__main__":
    asyncio.run(main())