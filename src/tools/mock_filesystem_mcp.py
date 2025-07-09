"""
Mock Filesystem MCP Server

This module provides a mock implementation of the Filesystem MCP server
for development and testing when the real server is not available.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockFileSystemMCPServer:
    """Mock filesystem MCP server for testing."""
    
    def __init__(self, base_path: str = "/tmp"):
        self.name = "filesystem-mock"
        self.description = "Mock filesystem operations"
        self.base_path = Path(base_path)
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.tools = [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to base directory"
                        }
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
                        "path": {
                            "type": "string",
                            "description": "File path relative to base directory"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
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
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to base directory"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "create_directory",
                "description": "Create a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to create"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "delete_file",
                "description": "Delete a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to delete"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "move_file",
                "description": "Move or rename a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source file path"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination file path"
                        }
                    },
                    "required": ["source", "destination"]
                }
            }
        ]
        
        logger.info(f"[INIT] Mock Filesystem MCP server initialized with base path: {self.base_path}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC requests."""
        method = request.get("method", "")
        request_id = request.get("id", 0)
        
        logger.debug(f"[REQUEST] Method: {method}, ID: {request_id}")
        
        if method == "initialize":
            return self.handle_initialize(request_id)
        elif method == "tools/list":
            return self.handle_tools_list(request_id)
        elif method == "tools/call":
            return await self.handle_tool_call(request, request_id)
        else:
            return self.error_response(request_id, -32601, f"Method not found: {method}")
    
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
                    "tools": True,
                    "filesystem": {
                        "read": True,
                        "write": True,
                        "list": True,
                        "create": True,
                        "delete": True,
                        "move": True
                    }
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
        """Execute filesystem operations."""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        logger.info(f"[TOOL_CALL] Tool: {tool_name}, Args: {arguments}")
        
        try:
            if tool_name == "read_file":
                return await self.read_file(arguments, request_id)
            
            elif tool_name == "write_file":
                return await self.write_file(arguments, request_id)
            
            elif tool_name == "list_directory":
                return await self.list_directory(arguments, request_id)
            
            elif tool_name == "create_directory":
                return await self.create_directory(arguments, request_id)
            
            elif tool_name == "delete_file":
                return await self.delete_file(arguments, request_id)
            
            elif tool_name == "move_file":
                return await self.move_file(arguments, request_id)
            
            else:
                return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Tool execution failed: {str(e)}")
            return self.error_response(request_id, -32603, str(e))
    
    async def read_file(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Read file contents."""
        file_path = arguments.get("path", "")
        
        # Resolve path relative to base
        full_path = self.base_path / file_path
        
        # Security check - ensure path is within base directory
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        if not full_path.exists():
            return self.error_response(request_id, -32602, f"File not found: {file_path}")
        
        if not full_path.is_file():
            return self.error_response(request_id, -32602, f"Path is not a file: {file_path}")
        
        try:
            content = full_path.read_text(encoding='utf-8')
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": content,
                    "path": str(file_path),
                    "size": len(content),
                    "encoding": "utf-8"
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error reading file: {str(e)}")
    
    async def write_file(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Write content to file."""
        file_path = arguments.get("path", "")
        content = arguments.get("content", "")
        
        # Resolve path relative to base
        full_path = self.base_path / file_path
        
        # Security check
        try:
            full_path = full_path.resolve()
            # Allow creating new files within base directory
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        try:
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            full_path.write_text(content, encoding='utf-8')
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "success": True,
                    "path": str(file_path),
                    "bytes_written": len(content.encode('utf-8'))
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error writing file: {str(e)}")
    
    async def list_directory(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """List directory contents."""
        dir_path = arguments.get("path", ".")
        
        # Resolve path relative to base
        full_path = self.base_path / dir_path
        
        # Security check
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        if not full_path.exists():
            return self.error_response(request_id, -32602, f"Directory not found: {dir_path}")
        
        if not full_path.is_dir():
            return self.error_response(request_id, -32602, f"Path is not a directory: {dir_path}")
        
        try:
            items = []
            for item in sorted(full_path.iterdir()):
                item_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": item.stat().st_mtime
                }
                items.append(item_info)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "path": str(dir_path),
                    "items": items,
                    "count": len(items)
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error listing directory: {str(e)}")
    
    async def create_directory(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Create a directory."""
        dir_path = arguments.get("path", "")
        
        # Resolve path relative to base
        full_path = self.base_path / dir_path
        
        # Security check
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "success": True,
                    "path": str(dir_path),
                    "created": True
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error creating directory: {str(e)}")
    
    async def delete_file(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Delete a file."""
        file_path = arguments.get("path", "")
        
        # Resolve path relative to base
        full_path = self.base_path / file_path
        
        # Security check
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        if not full_path.exists():
            return self.error_response(request_id, -32602, f"File not found: {file_path}")
        
        if not full_path.is_file():
            return self.error_response(request_id, -32602, f"Path is not a file: {file_path}")
        
        try:
            full_path.unlink()
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "success": True,
                    "path": str(file_path),
                    "deleted": True
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error deleting file: {str(e)}")
    
    async def move_file(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Move or rename a file."""
        source = arguments.get("source", "")
        destination = arguments.get("destination", "")
        
        # Resolve paths relative to base
        source_path = self.base_path / source
        dest_path = self.base_path / destination
        
        # Security check
        try:
            source_path = source_path.resolve()
            dest_path = dest_path.resolve()
            if not str(source_path).startswith(str(self.base_path)) or \
               not str(dest_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        if not source_path.exists():
            return self.error_response(request_id, -32602, f"Source file not found: {source}")
        
        try:
            # Create destination directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move/rename file
            source_path.rename(dest_path)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "success": True,
                    "source": str(source),
                    "destination": str(destination),
                    "moved": True
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error moving file: {str(e)}")
    
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


# Test script
async def test_mock_filesystem_server():
    """Test the mock filesystem MCP server."""
    logger.info("[TEST] Starting mock Filesystem MCP server tests")
    
    # Create test directory
    test_dir = Path("data/test_mock_fs")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize mock server
    server = MockFileSystemMCPServer(str(test_dir))
    
    # Test initialize
    init_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1
    })
    logger.info(f"[INIT] Response: {init_response['result']['serverInfo']['name']}")
    
    # Test tools list
    tools_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    })
    logger.info(f"[TOOLS] Available tools: {len(tools_response['result']['tools'])}")
    
    # Test write file
    write_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "write_file",
            "arguments": {
                "path": "test.txt",
                "content": "Hello from mock filesystem!"
            }
        },
        "id": 3
    })
    logger.info(f"[WRITE] Response: {write_response}")
    
    # Test read file
    read_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {
                "path": "test.txt"
            }
        },
        "id": 4
    })
    logger.info(f"[READ] Content: {read_response['result']['content']}")
    
    # Test list directory
    list_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "list_directory",
            "arguments": {
                "path": "."
            }
        },
        "id": 5
    })
    logger.info(f"[LIST] Files: {[item['name'] for item in list_response['result']['items']]}")
    
    logger.info("[TEST] Mock filesystem server tests complete!")


if __name__ == "__main__":
    asyncio.run(test_mock_filesystem_server())