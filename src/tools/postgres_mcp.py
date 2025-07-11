"""
PostgreSQL MCP Client Wrapper

This module provides a client wrapper for the PostgreSQL MCP server,
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
from src.tools.mock_postgres_mcp import MockPostgresMCPServer

logger = get_logger(__name__)

class PostgresMCPClient:
    """
    Client wrapper for PostgreSQL MCP server.
    
    This client enables database operations like:
    - Read-only query execution (SELECT)
    - Schema inspection
    - Table metadata retrieval
    - Database introspection
    """
    
    def __init__(self, connection_string: str, server_command: Optional[List[str]] = None):
        """
        Initialize PostgreSQL MCP client.
        
        Args:
            connection_string: PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)
            server_command: Command to start the MCP server (if not provided, uses default)
        """
        self.connection_string = connection_string
        self.server_name = "postgres"
        
        # Default to using local binary to run the PostgreSQL MCP server
        self.server_command = server_command or [
            "./node_modules/.bin/mcp-server-postgres", 
            connection_string
        ]
        
        self.process: Optional[subprocess.Popen] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.mock_server: Optional[MockPostgresMCPServer] = None
        self.use_mock = False
        
        logger.info(f"[INIT] PostgreSQL MCP Client for connection: {self._safe_connection_string()}")
    
    def _safe_connection_string(self) -> str:
        """Return connection string with password masked for logging."""
        if "@" in self.connection_string:
            parts = self.connection_string.split("@")
            if ":" in parts[0]:
                user_part = parts[0].split(":")
                if len(user_part) >= 2:
                    masked = ":".join(user_part[:-1]) + ":***@"
                    return masked + "@".join(parts[1:])
        return self.connection_string
    
    def _next_message_id(self) -> int:
        """Generate next message ID for JSON-RPC."""
        self._message_id += 1
        return self._message_id
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Connect to the PostgreSQL MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if use_mock:
                logger.info(f"[CONNECTING] Using mock PostgreSQL MCP server...")
                self.mock_server = MockPostgresMCPServer(self.connection_string)
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "PostgresMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to mock PostgreSQL MCP server")
                    
                    # Get available tools
                    await self._discover_tools_mock()
                    self.use_mock = True
                    return True
                else:
                    logger.error(f"[FAILED] Failed to initialize mock: {response}")
                    return False
            else:
                logger.info(f"[CONNECTING] Starting PostgreSQL MCP server...")
                
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
                            "name": "PostgresMCPClient",
                            "version": "0.1.0"
                        },
                        "capabilities": {
                            "tools": True,
                            "resources": True
                        }
                    },
                    "id": self._next_message_id()
                }
                
                await self._send_message(init_request)
                response = await self._receive_message()
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to PostgreSQL MCP server")
                    logger.debug(f"Capabilities: {self.capabilities}")
                    
                    # Get available tools
                    await self._discover_tools()
                    return True
                else:
                    error_detail = response.get("error", {}) if response else "No response received"
                    logger.error(f"[FAILED] Failed to initialize PostgreSQL MCP server")
                    logger.error(f"[ERROR] Server response: {response}")
                    logger.error(f"[ERROR] Error details: {error_detail}")
                    
                    # Check stderr for server errors
                    if self.process and self.process.stderr:
                        stderr_output = self.process.stderr.read()
                        if stderr_output:
                            logger.error(f"[ERROR] Server stderr: {stderr_output}")
                    return False
                
        except Exception as e:
            logger.error(f"[ERROR] Connection error: {str(e)}")
            return False
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the PostgreSQL MCP server."""
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
            logger.info(f"[TOOLS] Discovered {len(self.tools)} PostgreSQL tools:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def _discover_tools_mock(self) -> None:
        """Discover available tools from the mock PostgreSQL MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        response = await self.mock_server.handle_request(list_request)
        
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            logger.info(f"[TOOLS] Discovered {len(self.tools)} PostgreSQL tools from mock:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Execute a read-only SQL query.
        
        Args:
            query: SQL query to execute (must be read-only)
            params: Query parameters for prepared statements
            
        Returns:
            Query results or error information
        """
        logger.info(f"[QUERY] Executing: {query[:100]}...")
        
        # Find the appropriate tool (usually "query")
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
        Get database schema information using SQL queries.
        Note: The official PostgreSQL MCP server only provides a 'query' tool,
        so we implement schema retrieval using SQL queries.
        
        Args:
            table_name: Specific table to get schema for (optional)
            
        Returns:
            Schema information
        """
        if table_name:
            # Get specific table schema
            query = """
                SELECT column_name, data_type, is_nullable, column_default,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns 
                WHERE table_name = $1 AND table_schema = 'public'
                ORDER BY ordinal_position
            """
            result = await self.execute_query(query, [table_name])
            
            # Format the result for consistency with mock server
            if result.get("success") and result.get("result"):
                rows = result["result"].get("rows", [])
                return {
                    "success": True,
                    "table": table_name,
                    "columns": rows,
                    "execution_time": result.get("execution_time")
                }
            return result
        else:
            # Get all tables
            query = """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            return await self.execute_query(query)
    
    async def list_tables(self) -> Dict[str, Any]:
        """
        List all tables in the database using SQL queries.
        Note: The official PostgreSQL MCP server only provides a 'query' tool,
        so we implement table listing using SQL queries.
        
        Returns:
            List of table names and types
        """
        query = """
            SELECT 
                table_name,
                table_type,
                table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """
        result = await self.execute_query(query)
        
        # Format the result for consistency with mock server
        if result.get("success") and result.get("result"):
            rows = result["result"].get("rows", [])
            return {
                "success": True,
                "tables": rows,
                "execution_time": result.get("execution_time")
            }
        return result
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table information including columns, types, constraints
        """
        query = """
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                tc.constraint_type
            FROM information_schema.columns c
            LEFT JOIN information_schema.constraint_column_usage ccu 
                ON c.table_name = ccu.table_name 
                AND c.column_name = ccu.column_name
                AND c.table_schema = ccu.table_schema
            LEFT JOIN information_schema.table_constraints tc 
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE c.table_name = $1 AND c.table_schema = 'public'
            ORDER BY c.ordinal_position
        """
        return await self.execute_query(query, [table_name])
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific PostgreSQL MCP tool.
        
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
        """Disconnect from the PostgreSQL MCP server."""
        logger.info("[DISCONNECTING] Closing PostgreSQL MCP connection...")
        
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            
            if self.process.poll() is None:
                self.process.kill()
            
            logger.info("[SUCCESS] Disconnected from PostgreSQL MCP server")
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register PostgreSQL tools with the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        for tool in self.tools:
            tool_info = {
                'id': f'postgres.{tool["name"]}',
                'name': tool['name'],
                'server_type': 'postgres',
                'endpoint': self._safe_connection_string(),
                'description': tool.get('description', ''),
                'capabilities': self.capabilities,
                'input_schema': tool.get('inputSchema', {})
            }
            registry.register_tool(tool_info)
            logger.info(f"[REGISTERED] PostgreSQL tool: {tool_info['id']}")


async def test_postgres_mcp():
    """Test the PostgreSQL MCP client implementation."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing PostgreSQL MCP Client")
    logger.info("=" * 60)
    
    # Use a test connection string
    test_connection = "postgresql://testuser:testpass@localhost:5432/auto_tool_disc"
    
    # Initialize client
    client = PostgresMCPClient(test_connection)
    
    try:
        # Try to connect to real server first, fall back to mock if it fails
        connected = await client.connect()
        if not connected:
            logger.warning("⚠️  Could not connect to PostgreSQL MCP server")
            logger.info("💡 Trying with mock server instead...")
            connected = await client.connect(use_mock=True)
            if not connected:
                logger.error("[ERROR] Could not connect to mock server either")
                return
        
        # Test schema discovery
        schema_result = await client.get_schema()
        logger.info(f"[SCHEMA] Result: {schema_result}")
        
        # Test table listing
        tables_result = await client.list_tables()
        logger.info(f"[TABLES] Result: {tables_result}")
        
        # Test a simple query
        query_result = await client.execute_query("SELECT version()")
        logger.info(f"[VERSION] Result: {query_result}")
        
        # Test with tool registry
        registry = ToolRegistry("data/test_registry.db")
        client.register_tools_to_registry(registry)
        
        # List registered PostgreSQL tools
        postgres_tools = registry.list_tools("postgres")
        logger.info(f"[REGISTRY] Registered {len(postgres_tools)} PostgreSQL tools")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()
    
    logger.info("[TEST] PostgreSQL MCP test complete!")


if __name__ == "__main__":
    asyncio.run(test_postgres_mcp())