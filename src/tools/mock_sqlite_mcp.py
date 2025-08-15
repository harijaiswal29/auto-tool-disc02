"""
Mock SQLite MCP Server

A mock implementation of SQLite MCP server for testing without requiring
the actual MCP server to be installed.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockSQLiteMCPServer:
    """
    Mock SQLite MCP server that simulates the protocol without external dependencies.
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialized = False
        self.tools = self._define_tools()
        
        logger.info(f"[MOCK] Mock SQLite MCP Server initialized for: {db_path}")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available SQLite tools - matching standard SQLite MCP servers."""
        return [
            {
                "name": "read_query",
                "description": "Execute a read-only SQL query (SELECT only)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query to execute"
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
                "name": "write_query",
                "description": "Execute a write SQL query (INSERT, UPDATE, DELETE)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL write query to execute"
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
                "name": "list_tables",
                "description": "List all tables in the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "describe_table",
                "description": "Get detailed schema information for a specific table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to describe"
                        }
                    },
                    "required": ["table_name"]
                }
            },
            {
                "name": "create_table",
                "description": "Create a new table (CREATE TABLE statements only)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "CREATE TABLE SQL statement"
                        }
                    },
                    "required": ["sql"]
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
                    "name": "MockSQLiteMCPServer",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True,
                    "resources": False
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
            if tool_name == "read_query":
                result = await self._execute_read_query(arguments)
            elif tool_name == "write_query":
                result = await self._execute_write_query(arguments)
            elif tool_name == "list_tables":
                result = await self._list_tables()
            elif tool_name == "describe_table":
                result = await self._describe_table(arguments)
            elif tool_name == "create_table":
                result = await self._create_table(arguments)
            else:
                return self._error_response(request_id, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self._error_response(request_id, str(e))
    
    async def _execute_read_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute read-only SQL query (SELECT only)."""
        sql = args.get("sql", "")
        params = args.get("params", [])
        
        # Ensure it's a SELECT query
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed in read_query")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # It's definitely a SELECT query
            rows = cursor.fetchall()
            return {
                "rows": [dict(row) for row in rows],
                "columns": list(rows[0].keys()) if rows else []
            }
    
    async def _execute_write_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute write SQL query (INSERT, UPDATE, DELETE)."""
        sql = args.get("sql", "")
        params = args.get("params", [])
        
        # Ensure it's NOT a SELECT query
        if sql.strip().upper().startswith("SELECT"):
            raise ValueError("SELECT queries should use read_query")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            conn.commit()
            return {
                "changes": cursor.rowcount,
                "lastrowid": cursor.lastrowid
            }
    
    async def _create_table(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new table (CREATE TABLE only)."""
        sql = args.get("sql", "")
        
        # Ensure it's a CREATE TABLE statement
        if not sql.strip().upper().startswith("CREATE TABLE"):
            raise ValueError("Only CREATE TABLE statements are allowed")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            return {
                "success": True,
                "message": "Table created successfully"
            }
    
    async def _describe_table(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed schema information for a specific table."""
        table_name = args.get("table_name")
        
        if not table_name:
            raise ValueError("table_name is required")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            
            return {
                "table": table_name,
                "columns": [
                    {
                        "cid": col[0],
                        "name": col[1],
                        "type": col[2],
                        "notnull": bool(col[3]),
                        "default": col[4],
                        "pk": bool(col[5])
                    }
                    for col in columns
                ],
                "indexes": [
                    {
                        "seq": idx[0],
                        "name": idx[1],
                        "unique": bool(idx[2]),
                        "origin": idx[3],
                        "partial": bool(idx[4])
                    }
                    for idx in indexes
                ],
                "foreign_keys": [
                    {
                        "id": fk[0],
                        "seq": fk[1],
                        "table": fk[2],
                        "from": fk[3],
                        "to": fk[4],
                        "on_update": fk[5],
                        "on_delete": fk[6],
                        "match": fk[7]
                    }
                    for fk in foreign_keys
                ]
            }
    
    # Note: _get_schema removed in favor of describe_table for consistency with standard SQLite MCP
    
    async def _list_tables(self) -> Dict[str, Any]:
        """List all tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            return {
                "tables": [table[0] for table in tables]
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
    """Test the mock SQLite MCP server."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Mock SQLite MCP Server")
    logger.info("=" * 60)
    
    # Create mock server
    server = MockSQLiteMCPServer("data/mock_test.db")
    
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
    
    # Test query execution - create table
    create_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "sql": """
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value REAL
                )
                """
            }
        },
        "id": 3
    }
    create_response = await server.handle_request(create_request)
    logger.info(f"[CREATE] Response: {create_response}")
    
    # Test insert
    insert_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "sql": "INSERT INTO test_table (name, value) VALUES (?, ?)",
                "params": ["test_item", 42.5]
            }
        },
        "id": 4
    }
    insert_response = await server.handle_request(insert_request)
    logger.info(f"[INSERT] Response: {insert_response}")
    
    # Test select
    select_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "sql": "SELECT * FROM test_table"
            }
        },
        "id": 5
    }
    select_response = await server.handle_request(select_request)
    logger.info(f"[SELECT] Response: {select_response}")
    
    # Test schema
    schema_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_schema",
            "arguments": {
                "table": "test_table"
            }
        },
        "id": 6
    }
    schema_response = await server.handle_request(schema_request)
    logger.info(f"[SCHEMA] Response: {schema_response}")
    
    logger.info("[TEST] Mock server test complete!")


if __name__ == "__main__":
    asyncio.run(test_mock_server())