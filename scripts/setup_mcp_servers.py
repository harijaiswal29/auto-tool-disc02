#!/usr/bin/env python3
"""
Setup MCP Servers Script

This script automatically configures and initializes MCP servers based on:
1. Available API keys in environment variables
2. Configuration files in config/
3. Fallback to mock servers when real ones are unavailable

Usage:
    python scripts/setup_mcp_servers.py [--use-mock] [--test]
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import MCPIntegration
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPServerSetup:
    """Setup and configure MCP servers."""
    
    def __init__(self, force_mock: bool = False, test_mode: bool = False):
        """
        Initialize the setup.
        
        Args:
            force_mock: If True, always use mock servers
            test_mode: If True, run in test mode with validation
        """
        self.force_mock = force_mock
        self.test_mode = test_mode
        self.mcp_integration = None
        
        # Load environment variables
        load_dotenv()
        
        # Check API keys
        self.api_keys = self._check_api_keys()
        
    def _check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available."""
        keys = {
            'BRAVE_API_KEY': os.getenv('BRAVE_API_KEY'),
            'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN'),
            'POSTGRES_CONNECTION_STRING': os.getenv('POSTGRES_CONNECTION_STRING'),
            'OPENWEATHER_API_KEY': os.getenv('OPENWEATHER_API_KEY'),
            'FINANCIAL_DATASETS_API_KEY': os.getenv('FINANCIAL_DATASETS_API_KEY'),
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_API_SECRET': os.getenv('ZERODHA_API_SECRET'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'NOTION_INTEGRATION_TOKEN': os.getenv('NOTION_INTEGRATION_TOKEN')
        }
        
        # Check if keys are valid (not placeholders)
        valid_keys = {}
        placeholders = ['placeholder', 'your_', 'xxx', 'TODO']
        
        for key, value in keys.items():
            if value:
                is_valid = not any(p in value.lower() for p in placeholders)
                valid_keys[key] = is_valid
            else:
                valid_keys[key] = False
        
        return valid_keys
    
    def print_api_key_status(self):
        """Print the status of API keys."""
        print("\n" + "="*60)
        print("API Key Status")
        print("="*60)
        
        for key, is_valid in self.api_keys.items():
            status = "✓ Valid" if is_valid else "✗ Missing/Invalid"
            print(f"{key:30} {status}")
        
        print("="*60)
    
    async def setup_servers(self) -> Dict[str, bool]:
        """
        Setup all MCP servers based on configuration.
        
        Returns:
            Dictionary of server_id -> success status
        """
        logger.info("="*60)
        logger.info("MCP Server Setup")
        logger.info("="*60)
        
        # Initialize MCP Integration
        self.mcp_integration = MCPIntegration()
        
        results = {}
        
        # 1. SQLite Server (always available)
        logger.info("\n[1/9] Setting up SQLite server...")
        success = await self.mcp_integration.add_sqlite_server(
            db_path="data/test.db",
            server_id="sqlite_main",
            use_mock=self.force_mock
        )
        results['sqlite_main'] = success
        
        # 2. Search Server (Brave API)
        logger.info("\n[2/9] Setting up Search server...")
        if self.api_keys['BRAVE_API_KEY'] and not self.force_mock:
            config = {"api_key": os.getenv('BRAVE_API_KEY')}
            success = await self.mcp_integration.add_search_server(
                config=config,
                server_id="search_main",
                use_mock=False
            )
        else:
            success = await self.mcp_integration.add_search_server(
                server_id="search_main",
                use_mock=True
            )
        results['search_main'] = success
        
        # 3. GitHub Server
        logger.info("\n[3/9] Setting up GitHub server...")
        if self.api_keys['GITHUB_TOKEN'] and not self.force_mock:
            success = await self.mcp_integration.add_github_server(
                github_token=os.getenv('GITHUB_TOKEN'),
                server_id="github_main",
                use_mock=False
            )
        else:
            success = await self.mcp_integration.add_github_server(
                server_id="github_main",
                use_mock=True
            )
        results['github_main'] = success
        
        # 4. PostgreSQL Server
        logger.info("\n[4/9] Setting up PostgreSQL server...")
        if self.api_keys['POSTGRES_CONNECTION_STRING'] and not self.force_mock:
            success = await self.mcp_integration.add_postgres_server(
                connection_string=os.getenv('POSTGRES_CONNECTION_STRING'),
                server_id="postgres_main",
                use_mock=False
            )
        else:
            # Skip PostgreSQL mock if no connection string
            logger.info("Skipping PostgreSQL (no connection string)")
            success = False
        results['postgres_main'] = success
        
        # 5. Filesystem Server
        logger.info("\n[5/9] Setting up Filesystem server...")
        success = await self.mcp_integration.add_filesystem_server(
            base_path=".",
            server_id="filesystem_main",
            use_mock=True  # Always use mock for security
        )
        results['filesystem_main'] = success
        
        # 6. Weather Server
        logger.info("\n[6/9] Setting up Weather server...")
        success = await self.mcp_integration.add_weather_server(
            server_id="weather_main",
            use_mock=True  # Always mock (no API key support yet)
        )
        results['weather_main'] = success
        
        # 7. Financial Datasets Server
        logger.info("\n[7/9] Setting up Financial Datasets server...")
        if self.api_keys['FINANCIAL_DATASETS_API_KEY'] and not self.force_mock:
            success = await self.mcp_integration.add_financial_datasets_server(
                api_key=os.getenv('FINANCIAL_DATASETS_API_KEY'),
                server_id="financial_main",
                use_mock=False
            )
        else:
            logger.info("Skipping Financial Datasets (no API key)")
            success = False
        results['financial_main'] = success
        
        # 8. Zerodha Server
        logger.info("\n[8/9] Setting up Zerodha server...")
        zerodha_ready = all([
            self.api_keys['ZERODHA_API_KEY'],
            self.api_keys['ZERODHA_API_SECRET'],
            self.api_keys['ZERODHA_ACCESS_TOKEN']
        ])
        
        if zerodha_ready and not self.force_mock:
            success = await self.mcp_integration.add_zerodha_server(
                api_key=os.getenv('ZERODHA_API_KEY'),
                api_secret=os.getenv('ZERODHA_API_SECRET'),
                access_token=os.getenv('ZERODHA_ACCESS_TOKEN'),
                server_id="zerodha_main",
                use_mock=False
            )
        else:
            logger.info("Skipping Zerodha (incomplete credentials)")
            success = False
        results['zerodha_main'] = success
        
        # 9. Notion Server
        logger.info("\n[9/9] Setting up Notion server...")
        if self.api_keys['NOTION_INTEGRATION_TOKEN'] and not self.force_mock:
            success = await self.mcp_integration.add_notion_server(
                api_key=os.getenv('NOTION_INTEGRATION_TOKEN'),
                server_id="notion_main",
                use_mock=False
            )
        else:
            logger.info("Skipping Notion (no integration token)")
            success = False
        results['notion_main'] = success
        
        return results
    
    async def test_servers(self):
        """Test the configured servers with sample queries."""
        if not self.mcp_integration:
            logger.error("MCP Integration not initialized")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Testing Configured Servers")
        logger.info("="*60)
        
        # Discover all available tools
        tools = await self.mcp_integration.discover_all_tools()
        logger.info(f"\nTotal tools discovered: {len(tools)}")
        
        # Group tools by server type
        tools_by_type = {}
        for tool in tools:
            server_type = tool.get('server_type', 'unknown')
            if server_type not in tools_by_type:
                tools_by_type[server_type] = []
            tools_by_type[server_type].append(tool)
        
        # Print tool summary
        print("\nTools by Server Type:")
        print("-" * 40)
        for server_type, type_tools in tools_by_type.items():
            print(f"{server_type:15} {len(type_tools):3} tools")
            if self.test_mode:
                # Show first 3 tools
                for tool in type_tools[:3]:
                    print(f"  - {tool['name']}")
        
        # Test a tool from each active server
        if self.test_mode:
            print("\n" + "="*60)
            print("Testing Sample Tools")
            print("="*60)
            
            test_cases = [
                ("sqlite.query", {"query": "SELECT 1"}),
                ("search.web", {"query": "Python programming"}),
                ("filesystem.list", {"path": "."}),
                ("weather.get_current", {"location": "London"})
            ]
            
            for tool_id, args in test_cases:
                # Check if tool exists
                tool_exists = any(t['id'] == tool_id for t in tools)
                if not tool_exists:
                    print(f"\n✗ {tool_id}: Tool not available")
                    continue
                
                print(f"\nTesting {tool_id}...")
                try:
                    result = await self.mcp_integration.execute_tool(tool_id, args)
                    if result.get('success'):
                        print(f"✓ {tool_id}: Success")
                    else:
                        print(f"✗ {tool_id}: Failed - {result.get('error')}")
                except Exception as e:
                    print(f"✗ {tool_id}: Error - {e}")
    
    def print_summary(self, results: Dict[str, bool]):
        """Print setup summary."""
        print("\n" + "="*60)
        print("Setup Summary")
        print("="*60)
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nServers configured: {successful}/{total}")
        print("-" * 40)
        
        for server_id, success in results.items():
            status = "✓ Active" if success else "✗ Inactive"
            mode = self._get_server_mode(server_id)
            print(f"{server_id:20} {status:12} ({mode})")
        
        print("="*60)
    
    def _get_server_mode(self, server_id: str) -> str:
        """Get the mode (real/mock) of a server."""
        if not self.mcp_integration or server_id not in self.mcp_integration.servers:
            return "not configured"
        
        server = self.mcp_integration.servers[server_id]
        return "mock" if server.get('is_mock') else "real"
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.mcp_integration:
            await self.mcp_integration.shutdown_all()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Setup MCP servers for the Autonomous Tool Discovery System')
    parser.add_argument('--use-mock', action='store_true', help='Force use of mock servers')
    parser.add_argument('--test', action='store_true', help='Test configured servers after setup')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Create setup instance
    setup = MCPServerSetup(force_mock=args.use_mock, test_mode=args.test)
    
    try:
        # Print API key status
        setup.print_api_key_status()
        
        # Setup servers
        results = await setup.setup_servers()
        
        # Print summary
        setup.print_summary(results)
        
        # Test if requested
        if args.test:
            await setup.test_servers()
        
        print("\n✓ Setup complete!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    
    finally:
        await setup.cleanup()


if __name__ == "__main__":
    asyncio.run(main())