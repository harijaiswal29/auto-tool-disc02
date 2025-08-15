"""
Fixed GitHub MCP Client Wrapper

This module provides a corrected client wrapper for the GitHub MCP server.
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import threading
import queue

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.tools.mock_github_mcp import MockGitHubMCPServer

logger = get_logger(__name__)

class GitHubMCPClient:
    """
    Fixed client wrapper for GitHub MCP server.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub MCP client.
        
        Args:
            github_token: GitHub Personal Access Token
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.server_name = "github"
        self.process = None
        self.tools = []
        self.connected = False
        self.use_mock = False
        self.mock_server = None
        self.reader_thread = None
        self.response_queue = queue.Queue()
        self.message_id = 0
        
        logger.info(f"[INIT] GitHub MCP client initialized")
    
    def _read_stdout(self):
        """Background thread to read stdout from the process."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            # Try to parse as JSON
                            response = json.loads(line)
                            self.response_queue.put(response)
                            logger.debug(f"[READER] Received: {line}")
                        except json.JSONDecodeError:
                            # Not JSON, might be a server message
                            logger.debug(f"[READER] Non-JSON output: {line}")
                            if line.startswith("Server started"):
                                # Server ready message
                                self.response_queue.put({"server_ready": True})
            except Exception as e:
                logger.error(f"[READER] Error reading stdout: {e}")
                break
    
    def _read_stderr(self):
        """Background thread to read stderr from the process."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stderr.readline()
                if line:
                    logger.debug(f"[STDERR] {line.strip()}")
            except:
                break
    
    async def connect(self, use_mock: bool = False) -> bool:
        """
        Connect to the GitHub MCP server.
        
        Args:
            use_mock: If True, use mock server instead of real one
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try real server first if not explicitly using mock
            if not use_mock and self.github_token:
                logger.info("[CONNECT] Attempting to connect to real GitHub MCP server")
                
                # Set up environment with GitHub token
                env = os.environ.copy()
                env['GITHUB_TOKEN'] = self.github_token
                
                # Start the MCP server process
                self.process = subprocess.Popen(
                    ['node', 'node_modules/.bin/mcp-server-github'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    env=env
                )
                
                # Start reader threads
                self.reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
                self.reader_thread.start()
                
                stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
                stderr_thread.start()
                
                # Wait a moment for the server to start
                await asyncio.sleep(1)
                
                # Check if process is still running
                if self.process.poll() is not None:
                    logger.error("[ERROR] Server process exited immediately")
                    stdout, stderr = self.process.communicate()
                    if stderr:
                        logger.error(f"[STDERR] {stderr}")
                    return False
                
                # Send initialization request
                self.message_id += 1
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "github-mcp-client",
                            "version": "0.1.0"
                        },
                        "capabilities": {}
                    },
                    "id": self.message_id
                }
                
                # Send request
                logger.debug(f"[SEND] {json.dumps(init_request)}")
                self.process.stdin.write(json.dumps(init_request) + '\n')
                self.process.stdin.flush()
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(self.response_queue.get),
                        timeout=10.0
                    )
                    
                    # Check for server ready or init response
                    if response.get("server_ready") or "result" in response:
                        logger.info("[SUCCESS] Connected to real GitHub MCP server")
                        
                        # If we got server_ready, wait for actual init response
                        if response.get("server_ready"):
                            response = await asyncio.wait_for(
                                asyncio.to_thread(self.response_queue.get),
                                timeout=5.0
                            )
                        
                        # Now discover tools
                        await self._discover_tools()
                        
                        self.connected = True
                        self.use_mock = False
                        return True
                    
                except asyncio.TimeoutError:
                    logger.warning("[TIMEOUT] Real GitHub MCP server connection timed out")
                except Exception as e:
                    logger.warning(f"[ERROR] Failed to connect to real server: {e}")
                
                # Clean up failed connection
                if self.process:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except:
                        self.process.kill()
                    self.process = None
            
            # Fall back to mock server
            logger.info("[FALLBACK] Using mock GitHub MCP server")
            self.mock_server = MockGitHubMCPServer()
            
            # Initialize mock server
            init_response = await self.mock_server.handle_request({
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1
            })
            
            if "result" in init_response:
                # Get tools from mock server
                tools_response = await self.mock_server.handle_request({
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 2
                })
                
                self.tools = tools_response.get("result", {}).get("tools", [])
                self.connected = True
                self.use_mock = True
                logger.info(f"[MOCK] Connected successfully with {len(self.tools)} tools")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] Connection failed: {e}", exc_info=True)
            return False
    
    async def _discover_tools(self):
        """Discover available tools from the MCP server."""
        if self.use_mock:
            return
        
        try:
            # Request tool list
            self.message_id += 1
            tools_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": self.message_id
            }
            
            logger.debug(f"[SEND] {json.dumps(tools_request)}")
            self.process.stdin.write(json.dumps(tools_request) + '\n')
            self.process.stdin.flush()
            
            # Read response
            response = await asyncio.wait_for(
                asyncio.to_thread(self.response_queue.get),
                timeout=5.0
            )
            
            if "result" in response:
                self.tools = response["result"].get("tools", [])
                logger.info(f"[TOOLS] Discovered {len(self.tools)} GitHub tools")
                for tool in self.tools:
                    logger.debug(f"  - {tool['name']}: {tool.get('description', '')}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to discover tools: {e}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a GitHub tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if not self.connected:
            raise RuntimeError("Not connected to GitHub MCP server")
        
        try:
            if self.use_mock:
                # Use mock server
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": 3
                }
                response = await self.mock_server.handle_request(request)
            else:
                # Use real server
                self.message_id += 1
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": self.message_id
                }
                
                logger.debug(f"[SEND] {json.dumps(request)}")
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.response_queue.get),
                    timeout=30.0
                )
            
            if "result" in response:
                return response["result"]
            elif "error" in response:
                raise RuntimeError(f"Tool execution error: {response['error']}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to execute tool {tool_name}: {e}")
            raise
    
    def register_tools_to_registry(self, registry: ToolRegistry):
        """
        Register all GitHub tools to the tool registry.
        Only registers tools that were discovered from the actual server.
        
        Args:
            registry: Tool registry instance
        """
        if not self.tools:
            logger.warning("[REGISTRY] No tools discovered from GitHub server, skipping registration")
            return
        
        # Only register tools that we actually discovered from the server
        registered_count = 0
        for tool in self.tools:
            # Validate tool has required fields
            if 'name' not in tool:
                logger.warning(f"[REGISTRY] Skipping tool without name: {tool}")
                continue
            
            tool_info = {
                'id': f"github.{tool['name']}",
                'name': tool['name'],
                'server_type': 'github',
                'endpoint': 'mock://github' if self.use_mock else 'node node_modules/.bin/mcp-server-github',
                'description': tool.get('description', ''),
                'capabilities': tool,
                'input_schema': tool.get('inputSchema', {}),
                'is_mock': self.use_mock  # Track if this is from mock server
            }
            
            try:
                registry.register_tool(tool_info)
                registered_count += 1
                logger.debug(f"[REGISTRY] Registered tool: {tool['name']}")
            except Exception as e:
                logger.error(f"[REGISTRY] Failed to register tool {tool['name']}: {e}")
        
        mode = "mock" if self.use_mock else "real"
        logger.info(f"[REGISTRY] Registered {registered_count}/{len(self.tools)} GitHub tools from {mode} server")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool (alias for execute_tool for compatibility).
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result with success flag
        """
        try:
            result = await self.execute_tool(tool_name, arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def disconnect(self):
        """Disconnect from the GitHub MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None
        
        self.connected = False
        self.tools = []
        logger.info("[DISCONNECT] Disconnected from GitHub MCP server")