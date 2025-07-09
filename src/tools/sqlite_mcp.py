"""
SQLite MCP Client Wrapper

This module provides a client wrapper for the SQLite MCP server,
enabling database operations through the Model Context Protocol.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.tools.mock_sqlite_mcp import MockSQLiteMCPServer

logger = get_logger(__name__)

class SQLiteMCPClient:
    """
    Client wrapper for SQLite MCP server.
    
    This client enables database operations like:
    - Query execution (SELECT, INSERT, UPDATE, DELETE)
    - Schema inspection
    - Transaction management
    - Database metadata retrieval
    """
    
    def __init__(self, db_path: str, server_command: Optional[List[str]] = None):
        """
        Initialize SQLite MCP client.
        
        Args:
            db_path: Path to the SQLite database file
            server_command: Command to start the MCP server (if not provided, uses default)
        """
        self.db_path = Path(db_path).resolve()
        self.server_name = "sqlite"
        
        # Default to using npx to run the SQLite MCP server
        self.server_command = server_command or [
            "npx", "@modelcontextprotocol/server-sqlite", 
            str(self.db_path)
        ]
        
        self.process: Optional[subprocess.Popen] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.mock_server: Optional[MockSQLiteMCPServer] = None
        self.use_mock = False
        
        logger.info(f"[INIT] SQLite MCP Client for database: {self.db_path}")
    
    def _next_message_id(self) -> int:
        """Generate next message ID for JSON-RPC."""
        self._message_id += 1
        return self._message_id
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Connect to the SQLite MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if use_mock:
                logger.info(f"[CONNECTING] Using mock SQLite MCP server...")
                self.mock_server = MockSQLiteMCPServer(str(self.db_path))
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "SQLiteMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to mock SQLite MCP server")
                    
                    # Get available tools
                    await self._discover_tools_mock()
                    self.use_mock = True
                    return True
                else:
                    logger.error(f"[FAILED] Failed to initialize mock: {response}")
                    return False
            else:
                logger.info(f"[CONNECTING] Starting SQLite MCP server...")
                
                # Start the MCP server process
                self.process = subprocess.Popen(
                    self.server_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Send initialization request
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "SQLiteMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                await self._send_message(init_request)
                response = await self._receive_message()
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to SQLite MCP server")
                    logger.debug(f"Capabilities: {self.capabilities}")
                    
                    # Get available tools
                    await self._discover_tools()
                    return True
                else:
                    logger.error(f"[FAILED] Failed to initialize: {response}")
                    return False
                
        except Exception as e:
            logger.error(f"[ERROR] Connection error: {str(e)}")
            return False
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the SQLite MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        await self._send_message(list_request)
        response = await self._receive_message()
        
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            logger.info(f"[TOOLS] Discovered {len(self.tools)} SQLite tools:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def _discover_tools_mock(self) -> None:
        """Discover available tools from the mock SQLite MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        response = await self.mock_server.handle_request(list_request)
        
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            logger.info(f"[TOOLS] Discovered {len(self.tools)} SQLite tools from mock:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters for prepared statements
            
        Returns:
            Query results or error information
        """
        logger.info(f"[QUERY] Executing: {query[:100]}...")
        
        # Find the appropriate tool (usually "query" or "execute")
        query_tool = None
        for tool in self.tools:
            if tool.get("name") in ["query", "execute", "run_query"]:
                query_tool = tool
                break
        
        if not query_tool:
            logger.error("[ERROR] No query execution tool found")
            return {"error": "No query execution tool available"}
        
        # Prepare arguments based on tool schema
        arguments = {"sql": query}
        if params:
            arguments["params"] = params
        
        return await self.call_tool(query_tool["name"], arguments)
    
    async def get_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            table_name: Specific table to get schema for (optional)
            
        Returns:
            Schema information
        """
        # Look for schema tool
        schema_tool = None
        for tool in self.tools:
            if "schema" in tool.get("name", "").lower():
                schema_tool = tool
                break
        
        if not schema_tool:
            # Fallback to query
            if table_name:
                query = f"PRAGMA table_info({table_name})"
            else:
                query = "SELECT name FROM sqlite_master WHERE type='table'"
            return await self.execute_query(query)
        
        arguments = {}
        if table_name:
            arguments["table"] = table_name
            
        return await self.call_tool(schema_tool["name"], arguments)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific SQLite MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution results
        """
        logger.debug(f"[TOOL] Calling {tool_name} with args: {arguments}")
        
        call_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_message_id()
        }
        
        start_time = datetime.now()
        
        if self.use_mock:
            response = await self.mock_server.handle_request(call_request)
        else:
            await self._send_message(call_request)
            response = await self._receive_message()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if response and "result" in response:
            logger.info(f"[SUCCESS] Tool {tool_name} executed in {execution_time:.2f}s")
            return {
                "success": True,
                "result": response["result"],
                "execution_time": execution_time
            }
        else:
            error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
            logger.error(f"[ERROR] Tool execution failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time
            }
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message to the MCP server."""
        if self.process and self.process.stdin:
            json_message = json.dumps(message) + "\n"
            self.process.stdin.write(json_message)
            self.process.stdin.flush()
            logger.debug(f"→ Sent: {message.get('method', 'response')} (id: {message.get('id')})")
    
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive a JSON-RPC message from the MCP server."""
        if self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    response = json.loads(line.strip())
                    logger.debug(f"← Received: {response.get('id', 'notification')}")
                    return response
            except json.JSONDecodeError as e:
                logger.error(f"[ERROR] JSON decode error: {e}")
                logger.debug(f"Raw line: {line}")
        return None
    
    async def disconnect(self) -> None:
        """Disconnect from the SQLite MCP server."""
        logger.info("[DISCONNECTING] Closing SQLite MCP connection...")
        
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            
            if self.process.poll() is None:
                self.process.kill()
            
            logger.info("[SUCCESS] Disconnected from SQLite MCP server")
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register SQLite tools with the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        for tool in self.tools:
            tool_info = {
                'id': f'sqlite.{tool["name"]}',
                'name': tool['name'],
                'server_type': 'sqlite',
                'endpoint': str(self.db_path),
                'description': tool.get('description', ''),
                'capabilities': self.capabilities,
                'input_schema': tool.get('inputSchema', {})
            }
            registry.register_tool(tool_info)
            logger.info(f"[REGISTERED] SQLite tool: {tool_info['id']}")


async def test_sqlite_mcp():
    """Test the SQLite MCP client implementation."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing SQLite MCP Client")
    logger.info("=" * 60)
    
    # Create a test database
    test_db = Path("data/test_sqlite.db")
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = SQLiteMCPClient(str(test_db))
    
    try:
        # Try to connect to real server first, fall back to mock if it fails
        connected = await client.connect()
        if not connected:
            logger.warning("⚠️  Could not connect to SQLite MCP server")
            logger.info("💡 Trying with mock server instead...")
            connected = await client.connect(use_mock=True)
            if not connected:
                logger.error("[ERROR] Could not connect to mock server either")
                return
        
        # Create a test table
        create_result = await client.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info(f"[CREATE TABLE] Result: {create_result}")
        
        # Insert test data
        insert_result = await client.execute_query(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            ["Test User", "test@example.com"]
        )
        logger.info(f"[INSERT] Result: {insert_result}")
        
        # Query data
        select_result = await client.execute_query("SELECT * FROM users")
        logger.info(f"[SELECT] Result: {select_result}")
        
        # Get schema
        schema_result = await client.get_schema("users")
        logger.info(f"[SCHEMA] Result: {schema_result}")
        
        # Test with tool registry
        registry = ToolRegistry("data/test_registry.db")
        client.register_tools_to_registry(registry)
        
        # List registered SQLite tools
        sqlite_tools = registry.list_tools("sqlite")
        logger.info(f"[REGISTRY] Registered {len(sqlite_tools)} SQLite tools")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()
    
    logger.info("[TEST] SQLite MCP test complete!")


if __name__ == "__main__":
    asyncio.run(test_sqlite_mcp())