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
        
        # EXACT tools from official @modelcontextprotocol/server-filesystem
        self.tools = [
            {
                "name": "read_text_file",
                "description": "Read text contents of a file",
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
                "name": "read_media_file",
                "description": "Read media file as base64",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Media file path relative to base directory"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "read_multiple_files",
                "description": "Read multiple files at once",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of file paths to read"
                        }
                    },
                    "required": ["paths"]
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
                "name": "edit_file",
                "description": "Edit a file with search and replace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to edit"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "String to search for"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "String to replace with"
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
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
            },
            {
                "name": "search_files",
                "description": "Search for files by pattern",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (glob or regex)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "get_file_info",
                "description": "Get file metadata and information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to get info for"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_allowed_directories",
                "description": "List directories that are allowed for access",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
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
            # Map new tool names to handler methods
            if tool_name == "read_text_file":
                return await self.read_file(arguments, request_id)
            
            elif tool_name == "read_media_file":
                return await self.read_media_file(arguments, request_id)
            
            elif tool_name == "read_multiple_files":
                return await self.read_multiple_files(arguments, request_id)
            
            elif tool_name == "write_file":
                return await self.write_file(arguments, request_id)
            
            elif tool_name == "edit_file":
                return await self.edit_file(arguments, request_id)
            
            elif tool_name == "create_directory":
                return await self.create_directory(arguments, request_id)
            
            elif tool_name == "list_directory":
                return await self.list_directory(arguments, request_id)
            
            elif tool_name == "move_file":
                return await self.move_file(arguments, request_id)
            
            elif tool_name == "search_files":
                return await self.search_files(arguments, request_id)
            
            elif tool_name == "get_file_info":
                return await self.get_file_info(arguments, request_id)
            
            elif tool_name == "list_allowed_directories":
                return await self.list_allowed_directories(arguments, request_id)
            
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
    
    # Note: delete_file removed as it's not in official @modelcontextprotocol/server-filesystem
    
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
    
    async def read_media_file(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Read media file as base64."""
        import base64
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
        
        try:
            with open(full_path, "rb") as f:
                content = f.read()
                encoded = base64.b64encode(content).decode('utf-8')
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": encoded,
                    "encoding": "base64",
                    "path": str(file_path)
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error reading media file: {str(e)}")
    
    async def read_multiple_files(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Read multiple files at once."""
        paths = arguments.get("paths", [])
        results = []
        
        for path in paths:
            result = await self.read_file({"path": path}, request_id)
            if "result" in result:
                results.append({
                    "path": path,
                    "content": result["result"]["content"],
                    "success": True
                })
            else:
                results.append({
                    "path": path,
                    "error": result.get("error", {}).get("message", "Unknown error"),
                    "success": False
                })
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "files": results,
                "total": len(paths),
                "successful": sum(1 for r in results if r["success"])
            }
        }
    
    async def edit_file(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Edit file with search and replace."""
        file_path = arguments.get("path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        
        # First read the file
        read_result = await self.read_file({"path": file_path}, request_id)
        if "error" in read_result:
            return read_result
        
        # Replace content
        content = read_result["result"]["content"]
        new_content = content.replace(old_string, new_string)
        
        # Write back
        write_result = await self.write_file({
            "path": file_path,
            "content": new_content
        }, request_id)
        
        if "error" in write_result:
            return write_result
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "success": True,
                "path": file_path,
                "replacements": content.count(old_string)
            }
        }
    
    async def search_files(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Search for files by pattern."""
        import glob
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", ".")
        
        # Resolve search path
        full_path = self.base_path / search_path
        
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        try:
            # Use glob to find matching files
            matches = []
            for match in glob.glob(str(full_path / pattern), recursive=True):
                match_path = Path(match)
                if match_path.is_file():
                    rel_path = match_path.relative_to(self.base_path)
                    matches.append(str(rel_path))
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "matches": matches,
                    "count": len(matches),
                    "pattern": pattern
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error searching files: {str(e)}")
    
    async def get_file_info(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Get file metadata."""
        import os
        file_path = arguments.get("path", "")
        
        # Resolve path
        full_path = self.base_path / file_path
        
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.base_path)):
                raise ValueError("Path traversal attempt detected")
        except Exception as e:
            return self.error_response(request_id, -32602, f"Invalid path: {str(e)}")
        
        try:
            stat = full_path.stat()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "path": str(file_path),
                    "exists": full_path.exists(),
                    "is_file": full_path.is_file(),
                    "is_directory": full_path.is_dir(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "created": stat.st_ctime,
                    "permissions": oct(stat.st_mode)[-3:]
                }
            }
        except Exception as e:
            return self.error_response(request_id, -32603, f"Error getting file info: {str(e)}")
    
    async def list_allowed_directories(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """List allowed directories for access."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "directories": [str(self.base_path)],
                "base_path": str(self.base_path)
            }
        }
    
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
    
    # Test read file (using new tool name)
    read_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "read_text_file",
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