"""
Filesystem MCP Client Wrapper

This module provides a client wrapper for the Filesystem MCP server,
enabling file operations through the Model Context Protocol.
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
from src.tools.mock_filesystem_mcp import MockFileSystemMCPServer

logger = get_logger(__name__)

class FileSystemMCPClient:
    """
    Client wrapper for Filesystem MCP server.
    
    This client enables file operations like:
    - Reading file contents
    - Writing to files
    - Listing directory contents
    - Creating directories
    - Checking file/directory existence
    """
    
    def __init__(self, base_path: str = "/tmp", server_command: Optional[List[str]] = None):
        """
        Initialize Filesystem MCP client.
        
        Args:
            base_path: Base directory for file operations
            server_command: Command to start the MCP server (if not provided, uses default)
        """
        self.base_path = Path(base_path).resolve()
        self.server_name = "filesystem"
        
        # Default to using npx to run the Filesystem MCP server
        self.server_command = server_command or [
            "npx", "@modelcontextprotocol/server-filesystem", 
            str(self.base_path)
        ]
        
        self.process: Optional[subprocess.Popen] = None
        self.capabilities: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self.mock_server: Optional[MockFileSystemMCPServer] = None
        self.use_mock = False
        
        logger.info(f"[INIT] Filesystem MCP Client for path: {self.base_path}")
    
    def _next_message_id(self) -> int:
        """Generate next message ID for JSON-RPC."""
        self._message_id += 1
        return self._message_id
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Connect to the Filesystem MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if use_mock:
                logger.info(f"[CONNECTING] Using mock Filesystem MCP server...")
                self.mock_server = MockFileSystemMCPServer(str(self.base_path))
                
                # Simulate initialization
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "FileSystemMCPClient",
                            "version": "0.1.0"
                        }
                    },
                    "id": self._next_message_id()
                }
                
                response = await self.mock_server.handle_request(init_request)
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to mock Filesystem MCP server")
                    
                    # Get available tools
                    await self._discover_tools_mock()
                    self.use_mock = True
                    return True
                else:
                    logger.error(f"[FAILED] Failed to initialize mock: {response}")
                    return False
            else:
                logger.info(f"[CONNECTING] Starting Filesystem MCP server...")
                
                # Start the MCP server process
                self.process = subprocess.Popen(
                    self.server_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Send initialization request with proper capabilities
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "FileSystemMCPClient",
                            "version": "0.1.0"
                        },
                        "capabilities": {}  # Add empty capabilities as required
                    },
                    "id": self._next_message_id()
                }
                
                await self._send_message(init_request)
                response = await self._receive_message()
                
                if response and "result" in response:
                    self.capabilities = response["result"].get("capabilities", {})
                    logger.info(f"[SUCCESS] Connected to Filesystem MCP server")
                    logger.debug(f"Capabilities: {self.capabilities}")
                    
                    # Send initialized notification
                    await self._send_message({
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    })
                    
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
        """Discover available tools from the Filesystem MCP server."""
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
            logger.info(f"[TOOLS] Discovered {len(self.tools)} Filesystem tools:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def _discover_tools_mock(self) -> None:
        """Discover available tools from the mock Filesystem MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_message_id()
        }
        
        response = await self.mock_server.handle_request(list_request)
        
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            logger.info(f"[TOOLS] Discovered {len(self.tools)} Filesystem tools from mock:")
            for tool in self.tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
    
    async def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read contents of a file.
        
        Args:
            file_path: Path to the file to read (relative to base path or absolute)
            
        Returns:
            File contents or error information
        """
        # Convert to absolute path within base directory if relative
        if not Path(file_path).is_absolute():
            file_path = str(self.base_path / file_path)
        
        logger.info(f"[READ] Reading file: {file_path}")
        
        return await self.call_tool("read_file", {"path": file_path})
    
    async def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file to write (relative to base path or absolute)
            content: Content to write
            
        Returns:
            Success status or error information
        """
        # Convert to absolute path within base directory if relative
        if not Path(file_path).is_absolute():
            file_path = str(self.base_path / file_path)
            
        logger.info(f"[WRITE] Writing to file: {file_path}")
        
        return await self.call_tool("write_file", {"path": file_path, "content": content})
    
    async def list_directory(self, dir_path: str = ".") -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            dir_path: Path to the directory (relative to base path or absolute, default: base directory)
            
        Returns:
            Directory contents or error information
        """
        # Convert to absolute path within base directory if relative
        if dir_path == "" or dir_path == ".":
            dir_path = str(self.base_path)
        elif not Path(dir_path).is_absolute():
            dir_path = str(self.base_path / dir_path)
            
        logger.info(f"[LIST] Listing directory: {dir_path}")
        
        return await self.call_tool("list_directory", {"path": dir_path})
    
    async def create_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Create a directory.
        
        Args:
            dir_path: Path to the directory to create (relative to base path or absolute)
            
        Returns:
            Success status or error information
        """
        # Convert to absolute path within base directory if relative
        if not Path(dir_path).is_absolute():
            dir_path = str(self.base_path / dir_path)
            
        logger.info(f"[CREATE] Creating directory: {dir_path}")
        
        # Some MCP servers might have this as a separate tool
        # If not, we can create via write_file with a marker
        for tool in self.tools:
            if "create_directory" in tool.get("name", "").lower():
                return await self.call_tool(tool["name"], {"path": dir_path})
        
        # Fallback - create a marker file
        marker_path = f"{dir_path}/.mcp_directory"
        return await self.write_file(marker_path, "")
    
    async def file_exists(self, file_path: str) -> Dict[str, Any]:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            Existence status or error information
        """
        logger.info(f"[EXISTS] Checking file existence: {file_path}")
        
        # Try to read the file - if it exists, we'll get content
        # If it doesn't exist, we'll get an error
        result = await self.read_file(file_path)
        if result.get("success"):
            return {"success": True, "exists": True, "path": file_path}
        else:
            # Check if error is "file not found" type
            error = result.get("error", "")
            if "not found" in error.lower() or "no such file" in error.lower():
                return {"success": True, "exists": False, "path": file_path}
            else:
                return result  # Other error
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific Filesystem MCP tool.
        
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
                line = await asyncio.wait_for(
                    asyncio.to_thread(self.process.stdout.readline),
                    timeout=5.0
                )
                if line:
                    response = json.loads(line.strip())
                    logger.debug(f"← Received: {response.get('id', 'notification')}")
                    return response
            except asyncio.TimeoutError:
                logger.warning("[TIMEOUT] No response received within timeout")
            except json.JSONDecodeError as e:
                logger.error(f"[ERROR] JSON decode error: {e}")
                logger.debug(f"Raw line: {line}")
        return None
    
    async def disconnect(self) -> None:
        """Disconnect from the Filesystem MCP server."""
        logger.info("[DISCONNECTING] Closing Filesystem MCP connection...")
        
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            
            if self.process.poll() is None:
                self.process.kill()
            
            logger.info("[SUCCESS] Disconnected from Filesystem MCP server")
    
    def register_tools_to_registry(self, registry: ToolRegistry) -> None:
        """
        Register Filesystem tools with the tool registry.
        
        Args:
            registry: Tool registry instance
        """
        for tool in self.tools:
            tool_info = {
                'id': f'filesystem.{tool["name"]}',
                'name': tool['name'],
                'server_type': 'filesystem',
                'endpoint': str(self.base_path),
                'description': tool.get('description', ''),
                'capabilities': self.capabilities,
                'input_schema': tool.get('inputSchema', {})
            }
            registry.register_tool(tool_info)
            logger.info(f"[REGISTERED] Filesystem tool: {tool_info['id']}")


async def test_filesystem_mcp():
    """Test the Filesystem MCP client implementation."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Filesystem MCP Client")
    logger.info("=" * 60)
    
    # Create a test directory
    test_dir = Path("data/test_filesystem")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = FileSystemMCPClient(str(test_dir))
    
    try:
        # Try to connect to real server first, fall back to mock if it fails
        connected = await client.connect()
        if not connected:
            logger.warning("⚠️  Could not connect to Filesystem MCP server")
            logger.info("💡 Trying with mock server instead...")
            connected = await client.connect(use_mock=True)
            if not connected:
                logger.error("[ERROR] Could not connect to mock server either")
                return
        
        # Test write file
        write_result = await client.write_file(
            "test.txt", 
            "Hello from Filesystem MCP!\nThis is a test file."
        )
        logger.info(f"[WRITE] Result: {write_result}")
        
        # Test read file
        read_result = await client.read_file("test.txt")
        logger.info(f"[READ] Result: {read_result}")
        
        # Test list directory
        list_result = await client.list_directory(".")
        logger.info(f"[LIST] Result: {list_result}")
        
        # Test file exists
        exists_result = await client.file_exists("test.txt")
        logger.info(f"[EXISTS] test.txt exists: {exists_result}")
        
        not_exists_result = await client.file_exists("nonexistent.txt")
        logger.info(f"[EXISTS] nonexistent.txt exists: {not_exists_result}")
        
        # Test with tool registry
        registry = ToolRegistry("data/test_fs_registry.db")
        client.register_tools_to_registry(registry)
        
        # List registered filesystem tools
        fs_tools = registry.list_tools("filesystem")
        logger.info(f"[REGISTRY] Registered {len(fs_tools)} Filesystem tools")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()
    
    logger.info("[TEST] Filesystem MCP test complete!")


if __name__ == "__main__":
    asyncio.run(test_filesystem_mcp())