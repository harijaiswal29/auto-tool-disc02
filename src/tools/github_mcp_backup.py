"""
GitHub MCP Client Wrapper

This module provides a client wrapper for the GitHub MCP server,
enabling GitHub operations through the Model Context Protocol.
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.core.tool_registry import ToolRegistry
from src.tools.mock_github_mcp import MockGitHubMCPServer

logger = get_logger(__name__)

class GitHubMCPClient:
    """
    Client wrapper for GitHub MCP server.
    
    This client enables GitHub operations like:
    - Repository management (list, create, search)
    - Issue management (create, update, list, search)
    - Pull request operations (create, update, merge)
    - Code search across repositories
    - User and organization management
    - GitHub Actions workflow management
    """
    
    def __init__(self, github_token: Optional[str] = None, server_command: Optional[List[str]] = None):
        """
        Initialize GitHub MCP client.
        
        Args:
            github_token: GitHub Personal Access Token (if not provided, uses GITHUB_TOKEN env var)
            server_command: Command to start the MCP server (if not provided, uses default)
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.server_name = "github"
        
        # Default to using npx to run the GitHub MCP server
        self.server_command = server_command or [
            "npx", "@modelcontextprotocol/server-github"
        ]
        
        self.process = None
        self.tools = []
        self.connected = False
        self.use_mock = False
        self.mock_server = None
        
        logger.info(f"[INIT] GitHub MCP client initialized")
    
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
                    self.server_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                    env=env
                )
                
                # Send initialization request
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0",
                        "clientInfo": {
                            "name": "github-mcp-client",
                            "version": "0.1.0"
                        }
                    },
                    "id": 1
                }
                
                # Send request
                self.process.stdin.write(json.dumps(init_request) + '\n')
                self.process.stdin.flush()
                
                # Try to read response with timeout
                try:
                    # Skip the initial server message
                    initial_msg = await asyncio.wait_for(
                        asyncio.to_thread(self.process.stdout.readline),
                        timeout=2.0
                    )
                    logger.debug(f"[SERVER] Initial message: {initial_msg.strip()}")
                    
                    # Now read the actual response
                    response_line = await asyncio.wait_for(
                        asyncio.to_thread(self.process.stdout.readline),
                        timeout=5.0
                    )
                    
                    if response_line:
                        response = json.loads(response_line.strip())
                        if "result" in response:
                            logger.info("[SUCCESS] Connected to real GitHub MCP server")
                            
                            # Discover available tools
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
            logger.error(f"[ERROR] Connection failed: {e}")
            return False
    
    async def _discover_tools(self):
        """Discover available tools from the MCP server."""
        if self.use_mock:
            return
        
        try:
            # Request tool list
            tools_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
            
            self.process.stdin.write(json.dumps(tools_request) + '\n')
            self.process.stdin.flush()
            
            # Read response
            response_line = await asyncio.wait_for(
                asyncio.to_thread(self.process.stdout.readline),
                timeout=5.0
            )
            
            if response_line:
                response = json.loads(response_line.strip())
                if "result" in response:
                    self.tools = response["result"].get("tools", [])
                    logger.info(f"[TOOLS] Discovered {len(self.tools)} GitHub tools")
            
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
                request = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": 3
                }
                
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                
                response_line = await asyncio.wait_for(
                    asyncio.to_thread(self.process.stdout.readline),
                    timeout=30.0
                )
                
                response = json.loads(response_line.strip())
            
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
        
        Args:
            registry: Tool registry instance
        """
        for tool in self.tools:
            tool_info = {
                'id': f"github.{tool['name']}",
                'name': tool['name'],
                'server_type': 'github',
                'endpoint': ' '.join(self.server_command),
                'description': tool.get('description', ''),
                'capabilities': tool,
                'input_schema': tool.get('inputSchema', {})
            }
            registry.register_tool(tool_info)
        
        logger.info(f"[REGISTRY] Registered {len(self.tools)} GitHub tools")
    
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
            self.process = None
        
        self.connected = False
        self.tools = []
        logger.info("[DISCONNECT] Disconnected from GitHub MCP server")

# Example usage functions for common GitHub operations
    
async def list_user_repos(client: GitHubMCPClient, username: str = None) -> List[Dict]:
    """List repositories for a user."""
    return await client.execute_tool("list_repos", {"username": username})

async def search_repos(client: GitHubMCPClient, query: str, **kwargs) -> List[Dict]:
    """Search for repositories."""
    args = {"q": query}
    args.update(kwargs)
    return await client.execute_tool("search_repos", args)

async def create_issue(client: GitHubMCPClient, owner: str, repo: str, title: str, body: str) -> Dict:
    """Create a new issue."""
    return await client.execute_tool("create_issue", {
        "owner": owner,
        "repo": repo,
        "title": title,
        "body": body
    })

async def list_issues(client: GitHubMCPClient, owner: str, repo: str, **kwargs) -> List[Dict]:
    """List issues for a repository."""
    args = {"owner": owner, "repo": repo}
    args.update(kwargs)
    return await client.execute_tool("list_issues", args)

async def create_pull_request(client: GitHubMCPClient, owner: str, repo: str, title: str, 
                            head: str, base: str, body: str = "") -> Dict:
    """Create a pull request."""
    return await client.execute_tool("create_pull", {
        "owner": owner,
        "repo": repo,
        "title": title,
        "head": head,
        "base": base,
        "body": body
    })