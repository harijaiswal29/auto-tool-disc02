"""
Main entry point for the Autonomous Tool Discovery System.

This application demonstrates the complete pipeline from natural language
query to tool discovery, selection, and execution.
"""

import asyncio
import sys
import os
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import OrchestratorAgent
from src.core.mcp_integration import MCPIntegration
from src.utils.logger import get_logger
from src.utils.model_manager import preload_models


class AutonomousToolDiscoveryApp:
    """Main application class for the Autonomous Tool Discovery System."""
    
    def __init__(self):
        """Initialize the application."""
        self.logger = get_logger(__name__)
        self.orchestrator = None
        self.mcp_integration = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("="*60)
        self.logger.info("Autonomous Tool Discovery System")
        self.logger.info("="*60)
        self.logger.info("Initializing components...")
        
        try:
            # Preload ML models for better performance
            self.logger.info("Preloading ML models...")
            preload_models({
                'sentence_transformer': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'device': 'cpu'
                }
            })
            self.logger.info("Models preloaded successfully")
            
            # Initialize Orchestrator Agent
            self.orchestrator = OrchestratorAgent()
            await self.orchestrator.initialize()
            
            # Get MCP Integration from orchestrator
            self.mcp_integration = self.orchestrator.mcp_integration
            
            # Add some default MCP servers
            await self._setup_default_servers()
            
            self.is_initialized = True
            self.logger.info("System initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _setup_default_servers(self):
        """Setup default MCP servers."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.logger.info("Setting up default MCP servers...")
        
        # Check for API keys
        brave_api = os.getenv('BRAVE_API_KEY')
        github_token = os.getenv('GITHUB_TOKEN')
        postgres_conn = os.getenv('POSTGRES_CONNECTION_STRING')
        
        # Try to add SQLite server
        db_path = "data/test.db"
        success = await self.mcp_integration.add_sqlite_server(
            db_path=db_path,
            server_id="sqlite_main",
            use_mock=True  # Always mock for SQLite
        )
        if success:
            self.logger.info("✓ SQLite server added")
        
        # Try to add Search server with real API if available
        if brave_api and brave_api not in ['placeholder_brave_api_key_here', 'your_brave_api_key_here']:
            self.logger.info(f"Using real Brave Search API (key: {brave_api[:10]}...)")
            config = {"api_key": brave_api}
            success = await self.mcp_integration.add_search_server(
                config=config,
                server_id="search_main",
                use_mock=False  # Use real server
            )
            if success:
                self.logger.info("✓ Search server added (REAL Brave API)")
            else:
                # Fallback to mock if real server fails
                success = await self.mcp_integration.add_search_server(
                    server_id="search_main",
                    use_mock=True
                )
                if success:
                    self.logger.info("✓ Search server added (mock - real server failed)")
        else:
            success = await self.mcp_integration.add_search_server(
                server_id="search_main",
                use_mock=True
            )
            if success:
                self.logger.info("✓ Search server added (mock)")
        
        # Try to add GitHub server with real API if available
        if github_token and github_token not in ['ghp_placeholder_token_here', 'your_github_token_here']:
            self.logger.info(f"Using real GitHub API (token: {github_token[:10]}...)")
            success = await self.mcp_integration.add_github_server(
                github_token=github_token,
                server_id="github_main",
                use_mock=False  # Use real server
            )
            if success:
                self.logger.info("✓ GitHub server added (REAL API)")
            else:
                # Fallback to mock if real server fails
                success = await self.mcp_integration.add_github_server(
                    server_id="github_main",
                    use_mock=True
                )
                if success:
                    self.logger.info("✓ GitHub server added (mock - real server failed)")
        else:
            success = await self.mcp_integration.add_github_server(
                server_id="github_main",
                use_mock=True
            )
            if success:
                self.logger.info("✓ GitHub server added (mock)")
        
        # Try to add PostgreSQL server if connection string available
        if postgres_conn and postgres_conn != "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc":
            self.logger.info("Attempting PostgreSQL connection...")
            try:
                success = await self.mcp_integration.add_postgres_server(
                    connection_string=postgres_conn,
                    server_id="postgres_main",
                    use_mock=False  # Try real connection
                )
                if success:
                    self.logger.info("✓ PostgreSQL server added (REAL)")
            except Exception as e:
                self.logger.warning(f"PostgreSQL connection failed: {e}")
                success = await self.mcp_integration.add_postgres_server(
                    connection_string=postgres_conn,
                    server_id="postgres_main",
                    use_mock=True
                )
                if success:
                    self.logger.info("✓ PostgreSQL server added (mock - real connection failed)")
        
        # Try to add Weather server
        success = await self.mcp_integration.add_weather_server(
            server_id="weather_main",
            use_mock=True  # Always mock for weather (no API key provided)
        )
        if success:
            self.logger.info("✓ Weather server added")
        
        # Try to add Filesystem server
        success = await self.mcp_integration.add_filesystem_server(
            base_path=".",
            server_id="filesystem_main",
            use_mock=True  # Always mock for filesystem
        )
        if success:
            self.logger.info("✓ Filesystem server added")
    
    async def process_query(self, query: str) -> None:
        """
        Process a user query through the entire pipeline.
        
        Args:
            query: Natural language query from user
        """
        if not self.is_initialized:
            self.logger.error("System not initialized!")
            return
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            # Process through orchestrator
            result = await self.orchestrator.process_user_query(query)
            
            # Display results
            if result.intent and hasattr(result.intent, 'primary_intent'):
                print(f"\nIntent: {result.intent.primary_intent.type} "
                      f"(confidence: {result.intent.primary_intent.confidence:.2f})")
            else:
                print(f"\nIntent: Unable to recognize intent")
            
            print(f"\nDiscovered {len(result.discovered_tools)} tools:")
            for tool in result.discovered_tools[:3]:  # Show top 3
                print(f"  - {tool['name']} ({tool.get('relevance_score', 0):.2f})")
            
            if result.selected_tools:
                print(f"\nSelected {len(result.selected_tools)} tools for execution")
            
            if result.execution_results:
                print(f"\nExecution Results:")
                for exec_result in result.execution_results:
                    status = "✓" if exec_result.success else "✗"
                    print(f"  {status} {exec_result.tool_name}: ", end="")
                    if exec_result.success:
                        print(f"{self._summarize_result(exec_result.result)}")
                    else:
                        print(f"Error - {exec_result.error}")
            
            print(f"\nTotal execution time: {result.total_time_ms:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            print(f"\nError: {e}")
    
    def _summarize_result(self, result) -> str:
        """Create a brief summary of a result."""
        if result is None:
            return "No result"
        elif isinstance(result, dict):
            if 'error' in result:
                return f"Error: {result['error']}"
            return f"{len(result)} items"
        elif isinstance(result, list):
            return f"{len(result)} results"
        elif isinstance(result, str):
            return result[:50] + "..." if len(result) > 50 else result
        else:
            return str(result)[:50]
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("\nWelcome to the Autonomous Tool Discovery System!")
        print("Type 'help' for commands, 'exit' to quit.")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                elif query.lower() == 'help':
                    self._show_help()
                
                elif query.lower() == 'status':
                    self._show_status()
                
                elif query.lower() == 'tools':
                    await self._show_tools()
                
                else:
                    await self.process_query(query)
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  status  - Show system status")
        print("  tools   - List available tools")
        print("  exit    - Exit the application")
        print("\nExample queries:")
        print("  - Find all Python files in the project")
        print("  - Search for information about machine learning")
        print("  - Create a new configuration file")
        print("  - Check the weather in London")
        print("  - Query the database for user information")
    
    def _show_status(self):
        """Show system status."""
        if not self.is_initialized:
            print("System not initialized")
            return
        
        print("\nSystem Status:")
        server_status = self.mcp_integration.get_server_status()
        
        for server_id, status in server_status.items():
            print(f"  {server_id}: {status['status']} ({status['tools_count']} tools)")
    
    async def _show_tools(self):
        """Show available tools."""
        if not self.is_initialized:
            print("System not initialized")
            return
        
        print("\nAvailable Tools:")
        tools = self.mcp_integration.registry.list_tools()
        
        # Group by type
        tools_by_type = {}
        for tool in tools:
            tool_type = tool.get('type', 'unknown')
            if tool_type not in tools_by_type:
                tools_by_type[tool_type] = []
            tools_by_type[tool_type].append(tool)
        
        for tool_type, type_tools in tools_by_type.items():
            print(f"\n  {tool_type.upper()}:")
            for tool in type_tools:
                print(f"    - {tool['name']} ({tool['id']})")
    
    async def shutdown(self):
        """Shutdown all components."""
        self.logger.info("Shutting down...")
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        self.logger.info("Shutdown complete")
    
    async def run(self):
        """Main application run method."""
        try:
            # Initialize
            await self.initialize()
            
            # Run interactive mode
            await self.interactive_mode()
            
        finally:
            # Cleanup
            await self.shutdown()


async def main():
    """Main entry point."""
    app = AutonomousToolDiscoveryApp()
    await app.run()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())