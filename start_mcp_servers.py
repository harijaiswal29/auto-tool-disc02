#!/usr/bin/env python3
"""
Start MCP Servers for Testing

This script helps you run actual MCP servers for testing the Q-Learning system.
It can use either real MCP servers (if npm packages are installed) or mock servers.
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
import json
import signal
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MCPServerManager:
    """Manages MCP server processes."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.mcp_integration = None
        self.use_mock = False
        
        # Check environment variables
        self.check_environment()
        
    def check_environment(self):
        """Check if required environment variables are set."""
        required_vars = {
            'GITHUB_TOKEN': 'GitHub API access',
            'BRAVE_API_KEY': 'Brave Search API',
            'POSTGRES_CONNECTION_STRING': 'PostgreSQL database',
        }
        
        optional_vars = {
            'OPENWEATHER_API_KEY': 'Weather API',
            'FINANCIAL_DATASETS_API_KEY': 'Financial data API',
        }
        
        missing_required = []
        missing_optional = []
        
        for var, desc in required_vars.items():
            if not os.getenv(var):
                missing_required.append(f"{var} ({desc})")
                
        for var, desc in optional_vars.items():
            if not os.getenv(var):
                missing_optional.append(f"{var} ({desc})")
        
        if missing_required:
            logger.warning("Missing required environment variables:")
            for var in missing_required:
                logger.warning(f"  - {var}")
            logger.warning("Some MCP servers will use mock mode.")
            self.use_mock = True
        
        if missing_optional:
            logger.info("Missing optional environment variables:")
            for var in missing_optional:
                logger.info(f"  - {var}")
    
    def check_npm_servers(self) -> Dict[str, bool]:
        """Check which npm MCP servers are installed."""
        npm_servers = {
            'filesystem': '@modelcontextprotocol/server-filesystem',
            'github': '@modelcontextprotocol/server-github',
            'postgres': '@modelcontextprotocol/server-postgres',
            'search': 'brave-search-mcp',
        }
        
        installed = {}
        for name, package in npm_servers.items():
            try:
                # Check if package is installed
                result = subprocess.run(
                    ['npm', 'list', package],
                    capture_output=True,
                    text=True
                )
                installed[name] = result.returncode == 0
                if installed[name]:
                    logger.info(f"✓ {name} MCP server is installed ({package})")
                else:
                    logger.warning(f"✗ {name} MCP server not installed ({package})")
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
                installed[name] = False
                
        return installed
    
    def start_npm_server(self, name: str, command: List[str]) -> bool:
        """Start an npm-based MCP server."""
        try:
            logger.info(f"Starting {name} MCP server...")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            self.processes[name] = process
            logger.info(f"✓ {name} MCP server started (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start {name} server: {e}")
            return False
    
    async def start_mock_servers(self):
        """Start mock MCP servers."""
        logger.info("Starting mock MCP servers...")
        
        # Import mock server module
        from src.tools import mock_mcp_servers
        
        # Start mock server process
        try:
            process = subprocess.Popen(
                [sys.executable, '-m', 'src.tools.mock_mcp_servers'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['mock_servers'] = process
            logger.info(f"✓ Mock MCP servers started (PID: {process.pid})")
            
            # Give servers time to start
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to start mock servers: {e}")
    
    async def initialize_mcp_integration(self):
        """Initialize MCP integration and register tools."""
        # Load configuration
        config_path = Path(__file__).parent / "config" / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Create MCP integration
        self.mcp_integration = MCPIntegration(config)
        await self.mcp_integration.initialize()
        
        logger.info("MCP Integration initialized")
        
    async def register_servers(self, use_mock: bool = False):
        """Register MCP servers with the integration."""
        if use_mock:
            logger.info("Registering mock MCP servers...")
            # Mock servers are automatically available
        else:
            logger.info("Registering real MCP servers...")
            
            # Add SQLite server (always available)
            await self.mcp_integration.add_sqlite_server(
                db_path="data/test.db",
                server_id="sqlite_test",
                use_mock=False
            )
            
            # Add other servers based on availability
            # This is where you'd add real server connections
            
        # List registered tools
        tools = self.mcp_integration.registry.search_tools("")
        logger.info(f"Registered {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5
            logger.info(f"  - {tool['name']} ({tool['id']})")
        if len(tools) > 5:
            logger.info(f"  ... and {len(tools) - 5} more")
    
    async def start_all_servers(self):
        """Start all available MCP servers."""
        logger.info("=== Starting MCP Servers ===")
        
        # Check npm installations
        npm_installed = self.check_npm_servers()
        
        # Decide whether to use real or mock servers
        use_all_mock = self.use_mock or not any(npm_installed.values())
        
        if use_all_mock:
            logger.info("Using mock MCP servers for testing")
            await self.start_mock_servers()
        else:
            logger.info("Starting available real MCP servers...")
            
            # Start installed npm servers
            if npm_installed['filesystem']:
                self.start_npm_server('filesystem', [
                    'npx', '@modelcontextprotocol/server-filesystem',
                    '/tmp'  # Base directory for filesystem operations
                ])
            
            if npm_installed['github'] and os.getenv('GITHUB_TOKEN'):
                self.start_npm_server('github', [
                    'npx', '@modelcontextprotocol/server-github'
                ])
            
            if npm_installed['postgres'] and os.getenv('POSTGRES_CONNECTION_STRING'):
                self.start_npm_server('postgres', [
                    'npx', '@modelcontextprotocol/server-postgres',
                    os.getenv('POSTGRES_CONNECTION_STRING')
                ])
            
            if npm_installed['search'] and os.getenv('BRAVE_API_KEY'):
                self.start_npm_server('search', [
                    'npx', 'brave-search-mcp'
                ])
            
            # Start mock servers for any missing ones
            if not all(npm_installed.values()):
                logger.info("Starting mock servers for missing npm packages...")
                await self.start_mock_servers()
        
        # Initialize MCP integration
        await self.initialize_mcp_integration()
        
        # Register servers
        await self.register_servers(use_mock=use_all_mock)
        
        logger.info("\n=== MCP Servers Ready ===")
        logger.info("You can now run your Q-Learning experiments!")
        logger.info("Press Ctrl+C to stop all servers")
    
    def stop_all_servers(self):
        """Stop all running MCP servers."""
        logger.info("\nStopping MCP servers...")
        
        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                logger.info(f"Stopping {name} server (PID: {process.pid})")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} server")
                    process.kill()
        
        self.processes.clear()
        logger.info("All servers stopped")
    
    async def run(self):
        """Run the server manager."""
        # Set up signal handler
        def signal_handler(sig, frame):
            logger.info("\nReceived interrupt signal")
            self.stop_all_servers()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Start all servers
            await self.start_all_servers()
            
            # Keep running
            while True:
                # Check if any servers have crashed
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"{name} server crashed! Exit code: {process.returncode}")
                        del self.processes[name]
                
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("\nShutdown requested")
        finally:
            self.stop_all_servers()
            if self.mcp_integration:
                await self.mcp_integration.shutdown()


async def main():
    """Main entry point."""
    manager = MCPServerManager()
    await manager.run()


if __name__ == "__main__":
    print("MCP Server Manager")
    print("==================")
    print()
    print("This script helps you run MCP servers for testing.")
    print("It will check for installed npm packages and environment variables,")
    print("then start either real or mock servers as appropriate.")
    print()
    
    asyncio.run(main())