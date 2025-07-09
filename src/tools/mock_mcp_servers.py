"""
Mock MCP Servers for Development
Since not all official MCP servers are available, we'll create mock versions.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockMCPServer:
    """Base class for mock MCP servers."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools = []
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC requests."""
        method = request.get("method", "")
        request_id = request.get("id", 0)
        
        if method == "initialize":
            return self.handle_initialize(request_id)
        elif method == "tools/list":
            return self.handle_tools_list(request_id)
        elif method == "tools/call":
            return await self.handle_tool_call(request, request_id)
        else:
            return self.error_response(request_id, -32601, "Method not found")
    
    def handle_initialize(self, request_id: int) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": self.name,
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True
                }
            }
        }
    
    def handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """List available tools."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def handle_tool_call(self, request: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle tool execution - to be overridden by subclasses."""
        return self.error_response(request_id, -32601, "Tool not implemented")
    
    def error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

class FileSystemMCPServer(MockMCPServer):
    """Mock filesystem MCP server."""
    
    def __init__(self, base_path: str = "/tmp"):
        super().__init__("filesystem-mock", "Mock filesystem operations")
        self.base_path = Path(base_path)
        self.tools = [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write contents to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List contents of a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            }
        ]
    
    async def handle_tool_call(self, request: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Execute filesystem operations."""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "read_file":
                path = self.base_path / arguments["path"]
                content = path.read_text()
                result = {"content": content}
            
            elif tool_name == "write_file":
                path = self.base_path / arguments["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(arguments["content"])
                result = {"success": True, "path": str(path)}
            
            elif tool_name == "list_directory":
                path = self.base_path / arguments["path"]
                items = [{"name": p.name, "type": "file" if p.is_file() else "directory"} 
                        for p in path.iterdir()]
                result = {"items": items}
            
            else:
                return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self.error_response(request_id, -32603, str(e))

class TimeMCPServer(MockMCPServer):
    """Mock time MCP server."""
    
    def __init__(self):
        super().__init__("time-mock", "Mock time operations")
        self.tools = [
            {
                "name": "get_current_time",
                "description": "Get current time in specified timezone",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone (e.g., 'UTC', 'EST')"}
                    }
                }
            },
            {
                "name": "format_time",
                "description": "Format timestamp",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "description": "Time format string"}
                    }
                }
            }
        ]
    
    async def handle_tool_call(self, request: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Execute time operations."""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "get_current_time":
                # Simple implementation - just return UTC time
                current_time = datetime.utcnow().isoformat()
                result = {"time": current_time, "timezone": arguments.get("timezone", "UTC")}
            
            elif tool_name == "format_time":
                format_str = arguments.get("format", "%Y-%m-%d %H:%M:%S")
                formatted = datetime.now().strftime(format_str)
                result = {"formatted": formatted}
            
            else:
                return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self.error_response(request_id, -32603, str(e))

class SQLiteMCPServer(MockMCPServer):
    """Mock SQLite MCP server."""
    
    def __init__(self, db_path: str = "data/test.db"):
        super().__init__("sqlite-mock", "Mock SQLite operations")
        self.db_path = db_path
        self.tools = [
            {
                "name": "execute_query",
                "description": "Execute SQL query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query"},
                        "params": {"type": "array", "description": "Query parameters"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "list_tables",
                "description": "List all tables in database",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    async def handle_tool_call(self, request: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Execute SQLite operations."""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if tool_name == "execute_query":
                query = arguments["query"]
                query_params = arguments.get("params", [])
                cursor.execute(query, query_params)
                
                if query.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    result = {"rows": rows, "columns": columns}
                else:
                    conn.commit()
                    result = {"affected_rows": cursor.rowcount}
            
            elif tool_name == "list_tables":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                result = {"tables": tables}
            
            else:
                conn.close()
                return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
            
            conn.close()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self.error_response(request_id, -32603, str(e))

# Test script
async def test_mock_servers():
    """Test our mock MCP servers."""
    logger.info("[TEST] Starting mock MCP server tests")
    
    # Test filesystem server
    fs_server = FileSystemMCPServer("data/test_fs")
    
    # Initialize
    init_response = await fs_server.handle_request({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1
    })
    logger.info(f"[FS] Initialize response: {init_response}")
    
    # List tools
    tools_response = await fs_server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    })
    logger.info(f"[FS] Available tools: {len(tools_response['result']['tools'])} tools")
    
    # Test time server
    time_server = TimeMCPServer()
    time_response = await time_server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_current_time",
            "arguments": {"timezone": "UTC"}
        },
        "id": 3
    })
    logger.info(f"[TIME] Current time: {time_response}")
    
    logger.info("[TEST] Mock server tests complete!")

if __name__ == "__main__":
    asyncio.run(test_mock_servers())