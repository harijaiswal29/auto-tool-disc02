"""
MCP Integration Module

Central module for managing all MCP server integrations including SQLite.
Provides a unified interface for discovering and using MCP tools.
"""

import asyncio
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.utils.retry import (
    RetryManager, ExponentialBackoffRetry, CircuitBreaker,
    retry_async, RetryableError, NonRetryableError
)
from src.core.tool_registry import ToolRegistry
from src.core.connection_pool import ConnectionPool
from src.tools.sqlite_mcp import SQLiteMCPClient
from src.tools.search_mcp import SearchMCPClient
from src.tools.custom_wrappers.weather_mcp import WeatherMCPClient
from src.tools.filesystem_mcp import FileSystemMCPClient
from src.tools.postgres_mcp import PostgresMCPClient
from src.tools.github_mcp import GitHubMCPClient
from src.tools.financial_datasets_mcp import FinancialDatasetsMCPClient
from src.tools.zerodha_mcp import ZerodhaMCPClient

logger = get_logger(__name__)

class MCPIntegration:
    """
    Central integration point for all MCP servers.
    
    This class manages:
    - Server lifecycle (start/stop)
    - Tool discovery and registration
    - Unified interface for tool execution
    - Performance tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, registry: Optional[ToolRegistry] = None):
        """
        Initialize MCP integration.
        
        Args:
            config: Configuration dictionary
            registry: Tool registry for storing discovered tools (optional, will create if not provided)
        """
        # Load configuration
        if config is None:
            config = self._load_default_config()
        self.config = config
        
        # Initialize or use provided registry
        if registry is None:
            registry_path = config.get('database', {}).get('tool_registry', 'data/registry/tools.db')
            self.registry = ToolRegistry(registry_path)
        else:
            self.registry = registry
            
        self.servers: Dict[str, Any] = {}
        self.active_connections: Dict[str, Any] = {}
        
        # Load tool-to-server mappings
        self.tool_mappings = self._load_tool_mappings()
        self.server_configs = self._load_server_configs()
        
        # Initialize retry manager with configuration
        retry_config = config.get('retry_policies', {})
        circuit_breaker_config = config.get('circuit_breaker', {})
        
        self.retry_manager = RetryManager({
            'retry_policy': retry_config.get('default', {
                'type': 'exponential_backoff',
                'max_attempts': 5,
                'base_delay': 1.0,
                'max_delay': 16.0,
                'jitter_factor': 0.2
            }),
            'circuit_breaker': circuit_breaker_config.get('default', {
                'failure_threshold': 5,
                'recovery_timeout': 30.0,
                'half_open_test_requests': 3
            })
        })
        
        # Store server-specific retry configurations
        self.server_retry_configs = retry_config.get('mcp_servers', {})
        self.server_circuit_configs = circuit_breaker_config.get('mcp_servers', {})
        
        # Initialize connection pool
        pool_config = config.get('connection_pool', {})
        self.connection_pool = ConnectionPool(pool_config)
        
        logger.info("[INIT] MCP Integration initialized with retry support and connection pooling")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from config file."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Fallback configuration
        return {
            'database': {
                'tool_registry': 'data/registry/tools.db'
            }
        }
    
    def _load_tool_mappings(self) -> Dict[str, Any]:
        """Load tool-to-server mapping configuration."""
        mapping_path = Path(__file__).parent.parent.parent / 'config' / 'tool_server_mapping.json'
        
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                mappings = json.load(f)
                logger.info(f"[MAPPING] Loaded tool mappings from {mapping_path}")
                return mappings
        
        logger.warning("[MAPPING] Tool mapping file not found, using empty mappings")
        return {
            'tool_mappings': {},
            'server_types': {},
            'fallback_strategy': {
                'use_mock_on_failure': True,
                'retry_real_server': True,
                'max_retries': 3
            }
        }
    
    def _load_server_configs(self) -> Dict[str, Any]:
        """Load MCP server configurations."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'mcp_servers_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs = json.load(f)
                logger.info(f"[CONFIG] Loaded server configs from {config_path}")
                return configs
        
        logger.warning("[CONFIG] Server config file not found, using empty configs")
        return {'servers': {}}
    
    def get_server_for_tool(self, tool_id: str) -> Optional[str]:
        """
        Get the server ID for a given tool ID using mapping configuration.
        
        Args:
            tool_id: Tool identifier
            
        Returns:
            Server ID or None if not found
        """
        # Check direct mapping first
        tool_mappings = self.tool_mappings.get('tool_mappings', {})
        if tool_id in tool_mappings:
            return tool_mappings[tool_id]
        
        # Check pattern-based mapping
        server_types = self.tool_mappings.get('server_types', {})
        for server_type, config in server_types.items():
            patterns = config.get('patterns', [])
            for pattern in patterns:
                # Simple wildcard matching
                if pattern.endswith('*'):
                    prefix = pattern[:-1]
                    if tool_id.startswith(prefix):
                        return config.get('default_server')
                elif pattern == tool_id:
                    return config.get('default_server')
        
        # Try to find by server type from registry
        tool_info = self.registry.get_tool(tool_id)
        if tool_info:
            server_type = tool_info.get('server_type')
            if server_type in server_types:
                return server_types[server_type].get('default_server')
        
        logger.warning(f"[MAPPING] No server mapping found for tool: {tool_id}")
        return None
    
    async def initialize_from_config(self) -> Dict[str, bool]:
        """
        Initialize all servers based on configuration.
        
        Returns:
            Dictionary of server_id -> success status
        """
        results = {}
        server_configs = self.server_configs.get('servers', {})
        init_order = self.server_configs.get('initialization_order', list(server_configs.keys()))
        
        for server_id in init_order:
            if server_id not in server_configs:
                logger.warning(f"[INIT] Server {server_id} in init order but not in configs")
                continue
            
            config = server_configs[server_id]
            if not config.get('auto_initialize', True):
                logger.info(f"[INIT] Skipping {server_id} (auto_initialize=false)")
                continue
            
            # Check for required environment variables
            import os
            required_env = config.get('real_server', {}).get('required_env', [])
            env_available = all(os.getenv(var) for var in required_env)
            
            # Determine whether to use real or mock
            use_mock = not env_available or self.server_configs.get('mock_settings', {}).get('always_use_mock', False)
            
            # Initialize based on server type
            server_type = config.get('type')
            success = False
            
            try:
                if server_type == 'search':
                    api_key = os.getenv('BRAVE_API_KEY') if not use_mock else None
                    success = await self.add_search_server(
                        config={'api_key': api_key} if api_key else None,
                        server_id=server_id,
                        use_mock=use_mock
                    )
                elif server_type == 'github':
                    token = os.getenv('GITHUB_TOKEN') if not use_mock else None
                    success = await self.add_github_server(
                        github_token=token,
                        server_id=server_id,
                        use_mock=use_mock
                    )
                elif server_type == 'sqlite':
                    db_path = config.get('config', {}).get('default_db_path', 'data/test.db')
                    success = await self.add_sqlite_server(
                        db_path=db_path,
                        server_id=server_id,
                        use_mock=use_mock
                    )
                elif server_type == 'postgres':
                    conn_str = os.getenv('POSTGRES_CONNECTION_STRING') if not use_mock else None
                    if conn_str:
                        success = await self.add_postgres_server(
                            connection_string=conn_str,
                            server_id=server_id,
                            use_mock=use_mock
                        )
                elif server_type == 'filesystem':
                    base_path = config.get('config', {}).get('default_base_path', '.')
                    success = await self.add_filesystem_server(
                        base_path=base_path,
                        server_id=server_id,
                        use_mock=use_mock
                    )
                elif server_type == 'weather':
                    success = await self.add_weather_server(
                        server_id=server_id,
                        use_mock=use_mock
                    )
                else:
                    logger.warning(f"[INIT] Unknown server type: {server_type}")
                
                results[server_id] = success
                logger.info(f"[INIT] {server_id}: {'SUCCESS' if success else 'FAILED'} (mock={use_mock})")
                
            except Exception as e:
                logger.error(f"[INIT] Error initializing {server_id}: {e}")
                results[server_id] = False
        
        return results
    
    async def add_sqlite_server(self, db_path: str, server_id: str = "sqlite_default", use_mock: bool = False) -> bool:
        """
        Add a SQLite MCP server.
        
        Args:
            db_path: Path to SQLite database
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding SQLite server: {server_id}")
            
            # Create SQLite client
            client = SQLiteMCPClient(db_path)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'sqlite',
                    'db_path': db_path,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real"
                logger.info(f"[SUCCESS] SQLite server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to SQLite server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add SQLite server: {e}")
            return False
    
    async def add_search_server(self, config: Optional[Dict[str, Any]] = None, 
                               server_id: str = "search_default", use_mock: bool = False) -> bool:
        """
        Add a Search MCP server.
        
        Args:
            config: Configuration for search server (API keys, etc.)
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding Search server: {server_id}")
            
            # Create Search client
            client = SearchMCPClient(config)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'search',
                    'config': config,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real"
                logger.info(f"[SUCCESS] Search server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to Search server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add Search server: {e}")
            return False
    
    async def add_weather_server(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                                server_id: str = "weather_default", use_mock: bool = False) -> bool:
        """
        Add a Weather MCP server (custom wrapper for OpenWeather API).
        
        Args:
            api_key: OpenWeather API key
            config: Additional configuration
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding Weather server: {server_id}")
            
            # Create Weather client
            client = WeatherMCPClient(api_key=api_key, config=config)
            
            # Connect to server (validate API key or use mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'weather',
                    'api_key': api_key if not use_mock else None,
                    'config': config,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real API"
                logger.info(f"[SUCCESS] Weather server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to Weather server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add Weather server: {e}")
            return False
    
    async def add_filesystem_server(self, base_path: str = "/tmp", server_id: str = "filesystem_default", 
                                   use_mock: bool = False) -> bool:
        """
        Add a Filesystem MCP server.
        
        Args:
            base_path: Base directory for filesystem operations
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding Filesystem server: {server_id}")
            
            # Create Filesystem client
            client = FileSystemMCPClient(base_path)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'filesystem',
                    'base_path': base_path,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real"
                logger.info(f"[SUCCESS] Filesystem server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to Filesystem server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add Filesystem server: {e}")
            return False
    
    async def add_postgres_server(self, connection_string: str, server_id: str = "postgres_default", 
                                  use_mock: bool = False) -> bool:
        """
        Add a PostgreSQL MCP server.
        
        Args:
            connection_string: PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding PostgreSQL server: {server_id}")
            
            # Create PostgreSQL client
            client = PostgresMCPClient(connection_string)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'postgres',
                    'connection_string': connection_string,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real"
                logger.info(f"[SUCCESS] PostgreSQL server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to PostgreSQL server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add PostgreSQL server: {e}")
            return False
    
    async def add_github_server(self, github_token: Optional[str] = None, server_id: str = "github_default", 
                               use_mock: bool = False) -> bool:
        """
        Add a GitHub MCP server.
        
        Args:
            github_token: GitHub Personal Access Token (if not provided, uses GITHUB_TOKEN env var)
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding GitHub server: {server_id}")
            
            # Create GitHub client
            client = GitHubMCPClient(github_token)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'github',
                    'github_token': github_token if not use_mock else None,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real"
                logger.info(f"[SUCCESS] GitHub server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to GitHub server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add GitHub server: {e}")
            return False
    
    async def add_financial_datasets_server(self, api_key: Optional[str] = None, endpoint: Optional[str] = None,
                                          server_id: str = "financial_datasets_default", use_mock: bool = False) -> bool:
        """
        Add a Financial Datasets MCP server (remote).
        
        Args:
            api_key: Financial Datasets API key (if not provided, uses FINANCIAL_DATASETS_API_KEY env var)
            endpoint: Remote MCP server endpoint (optional, uses default if not provided)
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding Financial Datasets server: {server_id}")
            
            # Create Financial Datasets client
            client = FinancialDatasetsMCPClient(api_key, endpoint)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'financial_datasets',
                    'api_key': api_key if not use_mock else None,
                    'endpoint': endpoint,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "remote"
                logger.info(f"[SUCCESS] Financial Datasets server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to Financial Datasets server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add Financial Datasets server: {e}")
            return False
    
    async def add_zerodha_server(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                                 access_token: Optional[str] = None, endpoint: Optional[str] = None,
                                 server_id: str = "zerodha_default", use_mock: bool = False) -> bool:
        """
        Add a Zerodha MCP server for trading operations.
        
        Args:
            api_key: Zerodha API key (if not provided, uses ZERODHA_API_KEY env var)
            api_secret: Zerodha API secret (if not provided, uses ZERODHA_API_SECRET env var)
            access_token: User's access token (if not provided, uses ZERODHA_ACCESS_TOKEN env var)
            endpoint: Remote MCP server endpoint (optional, uses default if not provided)
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding Zerodha server: {server_id}")
            
            # Create Zerodha client
            client = ZerodhaMCPClient(api_key, api_secret, access_token, endpoint)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'zerodha',
                    'api_key': api_key if not use_mock else None,
                    'endpoint': endpoint,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "remote"
                logger.info(f"[SUCCESS] Zerodha server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to Zerodha server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add Zerodha server: {e}")
            return False
    
    async def add_notion_server(self, api_key: Optional[str] = None, endpoint: Optional[str] = None,
                                server_id: str = "notion_default", use_mock: bool = False) -> bool:
        """
        Add a Notion MCP server for workspace operations.
        
        Args:
            api_key: Notion Integration API key (if not provided, uses NOTION_API_KEY env var)
            endpoint: Remote MCP server endpoint (optional, uses default if not provided)
            server_id: Unique identifier for this server instance
            use_mock: If True, use mock server implementation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[ADD] Adding Notion server: {server_id}")
            
            # Import Notion MCP client
            from src.tools.notion_mcp import NotionMCPClient
            
            # Create Notion client
            client = NotionMCPClient(api_key, endpoint)
            
            # Connect to server (try real first, then mock)
            if await client.connect(use_mock=use_mock):
                # Store connection
                self.active_connections[server_id] = client
                
                # Register tools
                client.register_tools_to_registry(self.registry)
                
                # Store server info
                self.servers[server_id] = {
                    'type': 'notion',
                    'api_key': api_key if not use_mock else None,
                    'endpoint': endpoint,
                    'client': client,
                    'status': 'active',
                    'is_mock': client.use_mock
                }
                
                mode = "mock" if client.use_mock else "real"
                logger.info(f"[SUCCESS] Notion server {server_id} added successfully ({mode} mode)")
                return True
            else:
                logger.error(f"[FAILED] Could not connect to Notion server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add Notion server: {e}")
            return False
    
    async def execute_tool(self, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by its ID with retry logic and circuit breaker.
        
        Args:
            tool_id: Tool identifier (e.g., "sqlite.query")
            arguments: Tool arguments
            
        Returns:
            Execution results
        """
        logger.info(f"[EXECUTE] Tool: {tool_id}")
        
        # Get tool info from registry
        tool_info = self.registry.get_tool(tool_id)
        if not tool_info:
            logger.error(f"[ERROR] Tool not found in registry: {tool_id}")
            return {"error": f"Tool not found: {tool_id}", "success": False}
        
        # Find the appropriate server using mapping configuration
        server_id = self.get_server_for_tool(tool_id)
        
        # Fallback to old method if mapping not found
        if not server_id:
            server_type = tool_info['server_type']
            for sid, server in self.servers.items():
                if server['type'] == server_type and server['status'] == 'active':
                    server_id = sid
                    break
        
        if not server_id:
            server_type = tool_info.get('server_type', 'unknown')
            logger.error(f"[ERROR] No active server for tool: {tool_id} (type: {server_type})")
            return {"error": f"No active server for type: {server_type}"}
        
        # Execute through the appropriate client
        client = self.active_connections.get(server_id)
        if not client:
            logger.error(f"[ERROR] No active connection for server: {server_id}")
            return {"error": f"No active connection for server: {server_id}"}
        
        # Get server type for retry policy
        server_type = self.servers.get(server_id, {}).get('type', 'unknown')
        
        # Extract tool name (remove prefix)
        if '.' in tool_id:
            tool_name = tool_id.split('.', 1)[1]
        elif tool_id.startswith('financial_datasets_'):
            tool_name = tool_id.replace('financial_datasets_', '')
        elif tool_id.startswith('zerodha_'):
            tool_name = tool_id.replace('zerodha_', '')
        else:
            tool_name = tool_id
        
        # Validate that the tool exists on the server
        if hasattr(client, 'tools'):
            # Check if the tool exists in the client's tool list
            tool_exists = any(t.get('name') == tool_name for t in client.tools)
            if not tool_exists:
                logger.error(f"[ERROR] Tool '{tool_name}' not available on server '{server_id}'")
                # Also remove from registry if it's there
                try:
                    self.registry.remove_tool(tool_id)
                    logger.warning(f"[CLEANUP] Removed invalid tool '{tool_id}' from registry")
                except Exception:
                    pass  # Tool might not be in registry
                return {"error": f"Tool '{tool_name}' not available on server", "success": False}
        
        # Get retry policy and circuit breaker for this server
        server_type_lower = server_type.lower()
        retry_policy = self._get_retry_policy_for_server(server_type_lower)
        circuit_breaker = self.retry_manager.get_circuit_breaker(server_id)
        
        # Define the async function to execute with retry
        @retry_async(
            retry_policy=retry_policy,
            circuit_breaker=circuit_breaker,
            on_retry=lambda e, attempt: logger.warning(
                f"[RETRY] Tool {tool_id} execution retry {attempt + 1}, error: {e}"
            )
        )
        async def execute_with_retry():
            start_time = asyncio.get_event_loop().time()
            try:
                result = await client.call_tool(tool_name, arguments)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Check if the result indicates a retryable error
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown error")
                    
                    # Check if this is a non-retryable error
                    if self._is_non_retryable_error(error_msg):
                        raise NonRetryableError(error_msg)
                    
                    # Otherwise, raise a retryable error
                    raise RetryableError(error_msg)
                
                # Record successful usage
                self.registry.record_usage(
                    tool_id, 
                    True, 
                    execution_time,
                    error_message=None
                )
                
                return result
                
            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                error_msg = str(e)
                
                # Record failure
                self.registry.record_usage(tool_id, False, execution_time, error_message=error_msg)
                
                # Re-raise the exception for retry logic
                raise
        
        # Execute with retry logic
        try:
            return await execute_with_retry()
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERROR] Tool execution failed after retries: {error_msg}")
            return {"error": error_msg, "success": False}
    
    def _get_retry_policy_for_server(self, server_type: str):
        """Get retry policy for a specific server type."""
        # Check if there's a server-specific configuration
        if server_type in self.server_retry_configs:
            config = self.server_retry_configs[server_type]
            return self.retry_manager.create_retry_policy(config)
        
        # Use default retry policy
        return self.retry_manager.get_retry_policy(server_type)
    
    def _is_non_retryable_error(self, error_msg: str) -> bool:
        """Check if an error is non-retryable based on configuration."""
        non_retryable_errors = self.config.get('retry_policies', {}).get('no_retry_errors', [
            'NonRetryableError',
            'AuthenticationError',
            'InvalidArgumentError'
        ])
        
        # Check if error message contains any non-retryable error patterns
        error_lower = error_msg.lower()
        for pattern in non_retryable_errors:
            if pattern.lower() in error_lower:
                return True
        
        return False
    
    async def discover_all_tools(self) -> List[Dict[str, Any]]:
        """
        Discover tools from all connected servers.
        
        Returns:
            List of all available tools
        """
        all_tools = []
        
        for server_id, server in self.servers.items():
            if server['status'] == 'active':
                server_tools = self.registry.list_tools(server['type'])
                all_tools.extend(server_tools)
                logger.info(f"[DISCOVER] Found {len(server_tools)} tools from {server_id}")
        
        return all_tools
    
    async def shutdown_server(self, server_id: str) -> bool:
        """
        Shutdown a specific MCP server.
        
        Args:
            server_id: Server identifier
            
        Returns:
            True if successful
        """
        if server_id in self.active_connections:
            client = self.active_connections[server_id]
            await client.disconnect()
            
            del self.active_connections[server_id]
            if server_id in self.servers:
                self.servers[server_id]['status'] = 'inactive'
            
            logger.info(f"[SHUTDOWN] Server {server_id} shut down")
            return True
        
        return False
    
    async def shutdown_all(self) -> None:
        """Shutdown all MCP servers."""
        logger.info("[SHUTDOWN] Shutting down all MCP servers...")
        
        for server_id in list(self.active_connections.keys()):
            await self.shutdown_server(server_id)
        
        logger.info("[SHUTDOWN] All servers shut down")
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers including circuit breaker state."""
        status = {}
        
        for server_id, server in self.servers.items():
            # Get circuit breaker state if available
            circuit_breaker = self.retry_manager.circuit_breakers.get(server_id)
            cb_state = None
            cb_stats = None
            
            if circuit_breaker:
                cb_state = circuit_breaker.state.value
                cb_stats = circuit_breaker.statistics
            
            status[server_id] = {
                'type': server['type'],
                'status': server['status'],
                'tools_count': len(self.registry.list_tools(server['type'])),
                'circuit_breaker_state': cb_state,
                'circuit_breaker_stats': cb_stats
            }
        
        return status
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get retry and circuit breaker statistics for all servers."""
        return self.retry_manager.get_statistics()
    
    async def find_tools_by_intent(self, intent_type: str) -> List[Dict[str, Any]]:
        """
        Find tools that match a specific intent type.
        
        Args:
            intent_type: Type of intent (e.g., "query.search", "action.create")
            
        Returns:
            List of tools that match the intent
        """
        # Intent to capability mapping
        intent_capability_map = {
            'query.search': ['search', 'find', 'query', 'list', 'discover'],
            'query.retrieve': ['read', 'get', 'fetch', 'retrieve', 'load'],
            'query.analyze': ['analyze', 'examine', 'inspect', 'evaluate'],
            'action.create': ['create', 'write', 'generate', 'make', 'build'],
            'action.modify': ['update', 'edit', 'modify', 'change', 'alter'],
            'action.delete': ['delete', 'remove', 'clear', 'drop', 'erase'],
            'system.configure': ['configure', 'setup', 'initialize', 'install'],
            'system.monitor': ['monitor', 'track', 'watch', 'observe', 'log']
        }
        
        # Get capabilities for the intent
        required_capabilities = intent_capability_map.get(intent_type, [])
        
        if not required_capabilities:
            logger.warning(f"[INTENT] Unknown intent type: {intent_type}")
            return []
        
        # Find tools with matching capabilities
        matching_tools = []
        all_tools = self.registry.list_tools()
        
        for tool in all_tools:
            tool_caps = tool.get('capabilities', {})
            
            # Parse capabilities if they're stored as JSON string
            if isinstance(tool_caps, str):
                try:
                    tool_caps = json.loads(tool_caps)
                except:
                    tool_caps = {}
            
            # Check if tool has any of the required capabilities
            if isinstance(tool_caps, dict):
                operations = tool_caps.get('operations', [])
                for op in operations:
                    op_name = op.get('name', '') if isinstance(op, dict) else str(op)
                    if any(cap in op_name.lower() for cap in required_capabilities):
                        matching_tools.append(tool)
                        break
        
        logger.info(f"[INTENT] Found {len(matching_tools)} tools for intent {intent_type}")
        return matching_tools
    
    async def get_tools_by_capabilities(self, capabilities: List[str]) -> List[Dict[str, Any]]:
        """
        Get tools that have specific capabilities.
        
        Args:
            capabilities: List of required capabilities
            
        Returns:
            List of tools with the specified capabilities
        """
        matching_tools = []
        all_tools = self.registry.list_tools()
        
        for tool in all_tools:
            tool_caps = tool.get('capabilities', {})
            
            # Parse capabilities if they're stored as JSON string
            if isinstance(tool_caps, str):
                try:
                    tool_caps = json.loads(tool_caps)
                except:
                    tool_caps = {}
            
            # Check if tool has all required capabilities
            if isinstance(tool_caps, dict):
                operations = tool_caps.get('operations', [])
                tool_cap_names = []
                
                for op in operations:
                    if isinstance(op, dict):
                        tool_cap_names.append(op.get('name', '').lower())
                    else:
                        tool_cap_names.append(str(op).lower())
                
                # Check if all required capabilities are present
                if all(any(req_cap in cap_name for cap_name in tool_cap_names) 
                       for req_cap in capabilities):
                    matching_tools.append(tool)
        
        logger.info(f"[CAPABILITY] Found {len(matching_tools)} tools with capabilities {capabilities}")
        return matching_tools
    
    async def execute_tool_by_intent(self, intent_type: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the best tool for a given intent.
        
        Args:
            intent_type: Type of intent
            arguments: Tool arguments
            
        Returns:
            Execution results
        """
        # Find tools matching the intent
        matching_tools = await self.find_tools_by_intent(intent_type)
        
        if not matching_tools:
            logger.error(f"[INTENT] No tools found for intent {intent_type}")
            return {
                'error': f'No tools available for intent: {intent_type}',
                'intent': intent_type
            }
        
        # Select the best tool (highest performance score)
        best_tool = max(matching_tools, key=lambda t: t.get('performance_score', 0.5))
        
        logger.info(f"[INTENT] Selected tool {best_tool['id']} for intent {intent_type}")
        
        # Execute the tool
        return await self.execute_tool(best_tool['id'], arguments)
    
    async def validate_registry(self) -> Dict[str, Any]:
        """Validate all registered tools against their servers.
        
        Returns:
            Validation report with valid/invalid tools
        """
        logger.info("[VALIDATE] Starting registry validation...")
        
        all_tools = self.registry.list_tools()
        valid_tools = []
        invalid_tools = []
        
        for tool in all_tools:
            tool_id = tool['id']
            server_type = tool.get('server_type', '')
            
            # Parse tool ID to get server and tool name
            parts = tool_id.split('.', 1)
            if len(parts) != 2:
                invalid_tools.append({'tool_id': tool_id, 'reason': 'Invalid tool ID format'})
                continue
            
            server_prefix, tool_name = parts
            
            # Find the active server for this tool
            server_found = False
            for server_id, server in self.servers.items():
                if server['type'] == server_type and server['status'] == 'active':
                    client = self.active_connections.get(server_id)
                    if client and hasattr(client, 'tools'):
                        # Check if tool exists on server
                        if any(t.get('name') == tool_name for t in client.tools):
                            valid_tools.append(tool_id)
                            server_found = True
                            break
            
            if not server_found:
                invalid_tools.append({'tool_id': tool_id, 'reason': 'Tool not found on any active server'})
        
        # Remove invalid tools from registry
        for invalid in invalid_tools:
            try:
                self.registry.remove_tool(invalid['tool_id'])
                logger.warning(f"[CLEANUP] Removed invalid tool: {invalid['tool_id']} - {invalid['reason']}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to remove tool {invalid['tool_id']}: {e}")
        
        report = {
            'total_tools': len(all_tools),
            'valid_tools': len(valid_tools),
            'invalid_tools': len(invalid_tools),
            'invalid_details': invalid_tools,
            'cleanup_performed': len(invalid_tools) > 0
        }
        
        logger.info(f"[VALIDATE] Validation complete: {report['valid_tools']} valid, {report['invalid_tools']} invalid")
        return report
    
    async def initialize(self):
        """Initialize the MCP Integration and registry."""
        logger.info("[INIT] Initializing MCP Integration components...")
        
        # Initialize the tool registry
        await self.registry.initialize()
        
        # Start connection pool
        await self.connection_pool.start()
        
        logger.info("[INIT] MCP Integration initialization complete")
    
    async def shutdown(self):
        """Shutdown all components cleanly."""
        logger.info("[SHUTDOWN] Shutting down MCP Integration...")
        
        # Stop connection pool
        await self.connection_pool.stop()
        
        # Shutdown all servers
        await self.shutdown_all()
        
        # Close the registry
        await self.registry.close()
        
        logger.info("[SHUTDOWN] MCP Integration shutdown complete")
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self.connection_pool.get_statistics()


async def test_mcp_integration():
    """Test the MCP integration with SQLite and Search."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing MCP Integration")
    logger.info("=" * 60)
    
    # Create registry and integration
    registry = ToolRegistry("data/test_integration_registry.db")
    integration = MCPIntegration(registry)
    
    try:
        # Add SQLite server (try real first, then mock)
        test_db = "data/test_integration.db"
        success = await integration.add_sqlite_server(test_db, "test_sqlite")
        
        if not success:
            logger.warning("⚠️  Could not add SQLite server, trying with mock...")
            success = await integration.add_sqlite_server(test_db, "test_sqlite", use_mock=True)
            if not success:
                logger.error("[ERROR] Could not add SQLite server even with mock")
                return
        
        # Add Search server (try real first, then mock)
        search_config = {"api_key": "test_key", "max_results": 10}
        search_success = await integration.add_search_server(search_config, "test_search")
        
        if not search_success:
            logger.warning("⚠️  Could not add Search server, trying with mock...")
            search_success = await integration.add_search_server(search_config, "test_search", use_mock=True)
            if not search_success:
                logger.error("[ERROR] Could not add Search server even with mock")
                return
        
        # Add Weather server (custom MCP wrapper)
        # Note: Replace with real API key for actual testing
        weather_api_key = "test_weather_api_key"
        weather_success = await integration.add_weather_server(
            api_key=weather_api_key, 
            server_id="test_weather"
        )
        
        if not weather_success:
            logger.warning("⚠️  Could not add Weather server with API key, trying with mock...")
            weather_success = await integration.add_weather_server(
                server_id="test_weather", 
                use_mock=True
            )
            if not weather_success:
                logger.error("[ERROR] Could not add Weather server even with mock")
                return
        
        # Get server status
        status = integration.get_server_status()
        logger.info(f"[STATUS] Server status: {status}")
        
        # Discover tools
        tools = await integration.discover_all_tools()
        logger.info(f"[TOOLS] Discovered {len(tools)} tools total")
        
        # Execute a tool - create table
        create_result = await integration.execute_tool(
            "sqlite.query",
            {
                "sql": """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    completed BOOLEAN DEFAULT 0
                )
                """
            }
        )
        logger.info(f"[CREATE] Result: {create_result}")
        
        # Execute a tool - insert data
        insert_result = await integration.execute_tool(
            "sqlite.query",
            {
                "sql": "INSERT INTO tasks (title) VALUES (?)",
                "params": ["Test task from MCP integration"]
            }
        )
        logger.info(f"[INSERT] Result: {insert_result}")
        
        # Execute a tool - select data
        select_result = await integration.execute_tool(
            "sqlite.query",
            {
                "sql": "SELECT * FROM tasks"
            }
        )
        logger.info(f"[SELECT] Result: {select_result}")
        
        # Check tool performance
        if tools:
            tool_id = tools[0]['id']
            perf = registry.get_tool_performance(tool_id)
            logger.info(f"[PERFORMANCE] Tool {tool_id}: {perf}")
        
        # Test Search tool execution
        logger.info("\n[TEST] Testing Search tools...")
        
        # Test web search
        web_search_result = await integration.execute_tool(
            "search.web_search",
            {
                "query": "Model Context Protocol",
                "num_results": 3
            }
        )
        logger.info(f"[WEB_SEARCH] Result: {web_search_result}")
        
        # Test code search
        code_search_result = await integration.execute_tool(
            "search.code_search",
            {
                "query": "async def",
                "language": "python"
            }
        )
        logger.info(f"[CODE_SEARCH] Result: {code_search_result}")
        
        # Check Search tool performance
        search_tools = registry.list_tools("search")
        if search_tools:
            search_tool_id = search_tools[0]['id']
            search_perf = registry.get_tool_performance(search_tool_id)
            logger.info(f"[PERFORMANCE] Search tool {search_tool_id}: {search_perf}")
        
        # Test Weather tool execution
        logger.info("\n[TEST] Testing Weather tools...")
        
        # Test current weather
        weather_result = await integration.execute_tool(
            "weather.current_weather",
            {
                "location": "London,UK",
                "units": "metric"
            }
        )
        logger.info(f"[CURRENT_WEATHER] Result: {weather_result}")
        
        # Test weather forecast
        forecast_result = await integration.execute_tool(
            "weather.weather_forecast",
            {
                "location": "New York,US",
                "days": 2,
                "units": "imperial"
            }
        )
        logger.info(f"[WEATHER_FORECAST] Result: {forecast_result}")
        
        # Check Weather tool performance
        weather_tools = registry.list_tools("weather")
        if weather_tools:
            weather_tool_id = weather_tools[0]['id']
            weather_perf = registry.get_tool_performance(weather_tool_id)
            logger.info(f"[PERFORMANCE] Weather tool {weather_tool_id}: {weather_perf}")
        
        # Test Filesystem tool execution
        logger.info("\n[TEST] Testing Filesystem tools...")
        
        # Add Filesystem server
        test_fs_dir = "data/test_mcp_fs"
        fs_success = await integration.add_filesystem_server(
            base_path=test_fs_dir,
            server_id="test_filesystem"
        )
        
        if not fs_success:
            logger.warning("⚠️  Could not add Filesystem server, trying with mock...")
            fs_success = await integration.add_filesystem_server(
                base_path=test_fs_dir,
                server_id="test_filesystem",
                use_mock=True
            )
            if not fs_success:
                logger.error("[ERROR] Could not add Filesystem server even with mock")
                return
        
        # Test write file
        write_result = await integration.execute_tool(
            "filesystem.write_file",
            {
                "path": "test_integration.txt",
                "content": "Hello from MCP Integration!\nTesting filesystem operations."
            }
        )
        logger.info(f"[WRITE_FILE] Result: {write_result}")
        
        # Test read file
        read_result = await integration.execute_tool(
            "filesystem.read_file",
            {
                "path": "test_integration.txt"
            }
        )
        logger.info(f"[READ_FILE] Result: {read_result}")
        
        # Test list directory
        list_result = await integration.execute_tool(
            "filesystem.list_directory",
            {
                "path": "."
            }
        )
        logger.info(f"[LIST_DIR] Result: {list_result}")
        
        # Check Filesystem tool performance
        fs_tools = registry.list_tools("filesystem")
        if fs_tools:
            fs_tool_id = fs_tools[0]['id']
            fs_perf = registry.get_tool_performance(fs_tool_id)
            logger.info(f"[PERFORMANCE] Filesystem tool {fs_tool_id}: {fs_perf}")
        
        # Test PostgreSQL tool execution
        logger.info("\n[TEST] Testing PostgreSQL tools...")
        
        # Add PostgreSQL server
        test_postgres_connection = "postgresql://testuser:testpass@localhost:5432/auto_tool_disc"
        pg_success = await integration.add_postgres_server(
            connection_string=test_postgres_connection,
            server_id="test_postgres"
        )
        
        if not pg_success:
            logger.warning("⚠️  Could not add PostgreSQL server, trying with mock...")
            pg_success = await integration.add_postgres_server(
                connection_string=test_postgres_connection,
                server_id="test_postgres",
                use_mock=True
            )
            if not pg_success:
                logger.error("[ERROR] Could not add PostgreSQL server even with mock")
                return
        
        # Test PostgreSQL version query
        version_result = await integration.execute_tool(
            "postgres.query",
            {
                "sql": "SELECT version()"
            }
        )
        logger.info(f"[PG_VERSION] Result: {version_result}")
        
        # Test PostgreSQL table listing
        tables_result = await integration.execute_tool(
            "postgres.list_tables",
            {}
        )
        logger.info(f"[PG_TABLES] Result: {tables_result}")
        
        # Test PostgreSQL schema query
        schema_result = await integration.execute_tool(
            "postgres.query",
            {
                "sql": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            }
        )
        logger.info(f"[PG_SCHEMA] Result: {schema_result}")
        
        # Check PostgreSQL tool performance
        pg_tools = registry.list_tools("postgres")
        if pg_tools:
            pg_tool_id = pg_tools[0]['id']
            pg_perf = registry.get_tool_performance(pg_tool_id)
            logger.info(f"[PERFORMANCE] PostgreSQL tool {pg_tool_id}: {pg_perf}")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown all servers
        await integration.shutdown_all()
    
    logger.info("[TEST] MCP Integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_mcp_integration())