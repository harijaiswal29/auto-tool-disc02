"""
Mock PostgreSQL MCP Server

A mock implementation of PostgreSQL MCP server for testing without requiring
the actual PostgreSQL database or MCP server to be installed.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockPostgresMCPServer:
    """
    Mock PostgreSQL MCP server that simulates the protocol without external dependencies.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.initialized = False
        self.tools = self._define_tools()
        
        # Mock database with sample data
        self.mock_data = self._initialize_mock_data()
        
        logger.info(f"[MOCK] Mock PostgreSQL MCP Server initialized for: {self._safe_connection_string()}")
    
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
    
    def _initialize_mock_data(self) -> Dict[str, Any]:
        """Initialize mock database with sample data."""
        return {
            "tables": {
                "users": {
                    "columns": [
                        {"column_name": "id", "data_type": "integer", "is_nullable": "NO", "column_default": "nextval('users_id_seq'::regclass)"},
                        {"column_name": "username", "data_type": "character varying", "is_nullable": "NO", "column_default": None},
                        {"column_name": "email", "data_type": "character varying", "is_nullable": "YES", "column_default": None},
                        {"column_name": "created_at", "data_type": "timestamp with time zone", "is_nullable": "NO", "column_default": "CURRENT_TIMESTAMP"}
                    ],
                    "data": [
                        {"id": 1, "username": "admin", "email": "admin@example.com", "created_at": "2024-01-01T00:00:00Z"},
                        {"id": 2, "username": "user1", "email": "user1@example.com", "created_at": "2024-01-02T00:00:00Z"},
                        {"id": 3, "username": "user2", "email": "user2@example.com", "created_at": "2024-01-03T00:00:00Z"}
                    ]
                },
                "tools": {
                    "columns": [
                        {"column_name": "id", "data_type": "integer", "is_nullable": "NO", "column_default": "nextval('tools_id_seq'::regclass)"},
                        {"column_name": "name", "data_type": "character varying", "is_nullable": "NO", "column_default": None},
                        {"column_name": "description", "data_type": "text", "is_nullable": "YES", "column_default": None},
                        {"column_name": "type", "data_type": "character varying", "is_nullable": "NO", "column_default": None},
                        {"column_name": "capabilities", "data_type": "jsonb", "is_nullable": "YES", "column_default": None}
                    ],
                    "data": [
                        {"id": 1, "name": "filesystem_mcp", "description": "File system operations", "type": "mcp", "capabilities": {"read": True, "write": True}},
                        {"id": 2, "name": "search_mcp", "description": "Web search capabilities", "type": "mcp", "capabilities": {"web_search": True}},
                        {"id": 3, "name": "weather_mcp", "description": "Weather information", "type": "mcp", "capabilities": {"current_weather": True, "forecast": True}}
                    ]
                },
                "execution_history": {
                    "columns": [
                        {"column_name": "id", "data_type": "uuid", "is_nullable": "NO", "column_default": "gen_random_uuid()"},
                        {"column_name": "user_id", "data_type": "integer", "is_nullable": "YES", "column_default": None},
                        {"column_name": "query", "data_type": "text", "is_nullable": "NO", "column_default": None},
                        {"column_name": "tools_used", "data_type": "jsonb", "is_nullable": "YES", "column_default": None},
                        {"column_name": "success", "data_type": "boolean", "is_nullable": "NO", "column_default": "true"},
                        {"column_name": "execution_time_ms", "data_type": "integer", "is_nullable": "YES", "column_default": None},
                        {"column_name": "created_at", "data_type": "timestamp with time zone", "is_nullable": "NO", "column_default": "CURRENT_TIMESTAMP"}
                    ],
                    "data": [
                        {"id": "550e8400-e29b-41d4-a716-446655440001", "user_id": 1, "query": "Find Python files", "tools_used": ["filesystem_mcp"], "success": True, "execution_time_ms": 250, "created_at": "2024-01-15T10:00:00Z"},
                        {"id": "550e8400-e29b-41d4-a716-446655440002", "user_id": 2, "query": "Search for weather", "tools_used": ["search_mcp", "weather_mcp"], "success": True, "execution_time_ms": 1200, "created_at": "2024-01-15T11:00:00Z"}
                    ]
                }
            }
        }
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available PostgreSQL tools."""
        return [
            {
                "name": "query",
                "description": "Execute a read-only SQL query on the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL query to execute (must be read-only)"
                        },
                        "params": {
                            "type": "array",
                            "description": "Query parameters for prepared statements",
                            "items": {}
                        }
                    },
                    "required": ["sql"]
                }
            },
            {
                "name": "get_schema",
                "description": "Get database schema information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Specific table name (optional)"
                        }
                    }
                }
            },
            {
                "name": "list_tables",
                "description": "List all tables in the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.debug(f"[MOCK] Handling request: {method}")
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return await self._handle_tool_call(request_id, params)
        else:
            return self._error_response(request_id, f"Unknown method: {method}")
    
    def _handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        self.initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "MockPostgresMCPServer",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True,
                    "resources": True
                }
            }
        }
    
    def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def _handle_tool_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "query":
                result = await self._execute_query(arguments)
            elif tool_name == "get_schema":
                result = await self._get_schema(arguments)
            elif tool_name == "list_tables":
                result = await self._list_tables()
            else:
                return self._error_response(request_id, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self._error_response(request_id, str(e))
    
    async def _execute_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query on mock data."""
        sql = args.get("sql", "").strip()
        params = args.get("params", [])
        
        # Simulate read-only constraint
        sql_upper = sql.upper()
        if any(keyword in sql_upper for keyword in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]):
            raise Exception("Only read-only queries are allowed")
        
        # Handle special queries
        if "SELECT version()" in sql:
            return {
                "rows": [{"version": "PostgreSQL 14.10 (Mock Database)"}],
                "columns": ["version"]
            }
        
        if "information_schema.tables" in sql.lower():
            tables_data = []
            for table_name in self.mock_data["tables"].keys():
                tables_data.append({
                    "table_name": table_name,
                    "table_type": "BASE TABLE",
                    "table_schema": "public"
                })
            return {
                "rows": tables_data,
                "columns": ["table_name", "table_type", "table_schema"]
            }
        
        if "information_schema.columns" in sql.lower():
            # Extract table name from query if present
            table_name = None
            if "table_name = $1" in sql or "table_name = %s" in sql:
                if params:
                    table_name = params[0]
            
            if table_name and table_name in self.mock_data["tables"]:
                return {
                    "rows": self.mock_data["tables"][table_name]["columns"],
                    "columns": ["column_name", "data_type", "is_nullable", "column_default"]
                }
            else:
                # Return all columns from all tables
                all_columns = []
                for table, data in self.mock_data["tables"].items():
                    for col in data["columns"]:
                        col_with_table = col.copy()
                        col_with_table["table_name"] = table
                        all_columns.append(col_with_table)
                return {
                    "rows": all_columns,
                    "columns": ["table_name", "column_name", "data_type", "is_nullable", "column_default"]
                }
        
        # Handle simple table queries
        for table_name in self.mock_data["tables"].keys():
            if f"from {table_name}" in sql.lower() or f"FROM {table_name}" in sql:
                table_data = self.mock_data["tables"][table_name]["data"]
                if table_data:
                    return {
                        "rows": table_data,
                        "columns": list(table_data[0].keys()) if table_data else []
                    }
                else:
                    return {
                        "rows": [],
                        "columns": []
                    }
        
        # Default response for unhandled queries
        return {
            "rows": [{"result": "Mock query executed successfully", "query": sql}],
            "columns": ["result", "query"]
        }
    
    async def _get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get database schema."""
        table_name = args.get("table")
        
        if table_name:
            if table_name in self.mock_data["tables"]:
                return {
                    "table": table_name,
                    "columns": self.mock_data["tables"][table_name]["columns"]
                }
            else:
                raise Exception(f"Table '{table_name}' not found")
        else:
            # Return all table schemas
            schemas = {}
            for table, data in self.mock_data["tables"].items():
                schemas[table] = data["columns"]
            return {"schemas": schemas}
    
    async def _list_tables(self) -> Dict[str, Any]:
        """List all tables."""
        tables = []
        for table_name in self.mock_data["tables"].keys():
            tables.append({
                "table_name": table_name,
                "table_type": "BASE TABLE",
                "table_schema": "public"
            })
        
        return {
            "tables": tables
        }
    
    def _error_response(self, request_id: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }


async def test_mock_server():
    """Test the mock PostgreSQL MCP server."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Mock PostgreSQL MCP Server")
    logger.info("=" * 60)
    
    # Create mock server
    connection_string = "postgresql://testuser:testpass@localhost:5432/auto_tool_disc"
    server = MockPostgresMCPServer(connection_string)
    
    # Test initialization
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0",
            "clientInfo": {"name": "TestClient", "version": "0.1.0"}
        },
        "id": 1
    }
    init_response = await server.handle_request(init_request)
    logger.info(f"[INIT] Response: {init_response}")
    
    # Test tools list
    list_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }
    list_response = await server.handle_request(list_request)
    logger.info(f"[TOOLS] Available tools: {len(list_response['result']['tools'])}")
    
    # Test version query
    version_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "sql": "SELECT version()"
            }
        },
        "id": 3
    }
    version_response = await server.handle_request(version_request)
    logger.info(f"[VERSION] Response: {version_response}")
    
    # Test table listing
    tables_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "list_tables",
            "arguments": {}
        },
        "id": 4
    }
    tables_response = await server.handle_request(tables_request)
    logger.info(f"[TABLES] Response: {tables_response}")
    
    # Test schema query
    schema_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_schema",
            "arguments": {
                "table": "users"
            }
        },
        "id": 5
    }
    schema_response = await server.handle_request(schema_request)
    logger.info(f"[SCHEMA] Response: {schema_response}")
    
    # Test data query
    data_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "sql": "SELECT * FROM users"
            }
        },
        "id": 6
    }
    data_response = await server.handle_request(data_request)
    logger.info(f"[DATA] Response: {data_response}")
    
    # Test information_schema query
    info_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "sql": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            }
        },
        "id": 7
    }
    info_response = await server.handle_request(info_request)
    logger.info(f"[INFO_SCHEMA] Response: {info_response}")
    
    # Test read-only constraint
    try:
        insert_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {
                    "sql": "INSERT INTO users (username) VALUES ('test')"
                }
            },
            "id": 8
        }
        insert_response = await server.handle_request(insert_request)
        logger.info(f"[INSERT] Should fail - Response: {insert_response}")
    except Exception as e:
        logger.info(f"[INSERT] Correctly blocked: {e}")
    
    logger.info("[TEST] Mock PostgreSQL server test complete!")


if __name__ == "__main__":
    asyncio.run(test_mock_server())