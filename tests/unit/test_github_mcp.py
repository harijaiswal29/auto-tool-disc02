"""
Unit tests for GitHub MCP Client

This module tests the GitHubMCPClient class functionality with proper mocking
of external dependencies.
"""

import pytest
import asyncio
import json
import queue
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import subprocess
import threading

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.github_mcp import GitHubMCPClient
from src.tools.mock_github_mcp import MockGitHubMCPServer


class TestGitHubMCPClient:
    """Test cases for GitHubMCPClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = GitHubMCPClient()
        
    def test_initialization(self):
        """Test client initialization."""
        # Test with no token
        client = GitHubMCPClient()
        assert client.github_token is None
        assert client.server_name == "github"
        assert client.process is None
        assert client.connected is False
        assert client.use_mock is False
        
        # Test with token
        client = GitHubMCPClient("test_token")
        assert client.github_token == "test_token"
        
    @patch.dict('os.environ', {'GITHUB_TOKEN': 'env_token'})
    def test_initialization_with_env_token(self):
        """Test client initialization with environment token."""
        client = GitHubMCPClient()
        assert client.github_token == "env_token"
        
    @pytest.mark.asyncio
    async def test_connect_mock_server(self):
        """Test connection with mock server."""
        client = GitHubMCPClient()
        
        # Connect with mock
        result = await client.connect(use_mock=True)
        assert result is True
        assert client.connected is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert len(client.tools) > 0
        
        await client.disconnect()
        
    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    async def test_connect_real_server_failure(self, mock_popen):
        """Test connection failure with real server."""
        # Mock process that exits immediately
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited
        mock_process.communicate.return_value = ("", "Error")
        mock_popen.return_value = mock_process
        
        client = GitHubMCPClient("test_token")
        result = await client.connect(use_mock=False)
        
        # Should fall back to mock
        assert result is True
        assert client.use_mock is True
        assert client.connected is True
        
    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    @patch('threading.Thread')
    async def test_connect_real_server_success(self, mock_thread, mock_popen):
        """Test successful connection with real server."""
        # Mock process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process running
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process
        
        # Mock threads
        mock_thread.return_value.start = Mock()
        
        client = GitHubMCPClient("test_token")
        client.response_queue = queue.Queue()
        
        # Simulate server response
        client.response_queue.put({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {"name": "github-mcp", "version": "1.0"}
            }
        })
        
        # Mock discover tools
        with patch.object(client, '_discover_tools', new_callable=AsyncMock):
            result = await client.connect(use_mock=False)
            
        assert result is True
        assert client.connected is True
        assert client.use_mock is False
        
    @pytest.mark.asyncio
    async def test_execute_tool_not_connected(self):
        """Test tool execution when not connected."""
        client = GitHubMCPClient()
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.execute_tool("list_repos", {})
            
    @pytest.mark.asyncio
    async def test_execute_tool_with_mock(self):
        """Test tool execution with mock server."""
        client = GitHubMCPClient()
        await client.connect(use_mock=True)
        
        # Execute a tool with old name (should work due to backward compatibility)
        result = await client.execute_tool("list_repos", {"username": "test"})
        
        # Should return mock data
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Also test with new name
        result2 = await client.execute_tool("list_repositories", {"username": "test"})
        assert isinstance(result2, list)
        assert len(result2) > 0
        
        await client.disconnect()
        
    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    async def test_execute_tool_with_real_server(self, mock_popen):
        """Test tool execution with real server."""
        # Setup mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process
        
        client = GitHubMCPClient("test_token")
        client.connected = True
        client.use_mock = False
        client.process = mock_process
        client.response_queue = queue.Queue()
        
        # Simulate tool response
        client.response_queue.put({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"repositories": ["repo1", "repo2"]}
        })
        
        result = await client.execute_tool("list_repos", {"username": "test"})
        
        assert result == {"repositories": ["repo1", "repo2"]}
        mock_process.stdin.write.assert_called()
        
    @pytest.mark.asyncio
    async def test_execute_tool_error_response(self):
        """Test tool execution with error response."""
        client = GitHubMCPClient()
        await client.connect(use_mock=True)
        
        # Mock error response
        client.mock_server.handle_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32602, "message": "Invalid params"}
        })
        
        with pytest.raises(RuntimeError, match="Tool execution error"):
            await client.execute_tool("invalid_tool", {})
            
    def test_register_tools_to_registry(self):
        """Test registering tools to registry."""
        from src.core.tool_registry import ToolRegistry
        
        client = GitHubMCPClient()
        client.tools = [
            {"name": "list_repositories", "description": "List repositories"},
            {"name": "create_issue", "description": "Create issue"},
            {"name": "get_file_contents", "description": "Get file contents"}
        ]
        
        registry = Mock(spec=ToolRegistry)
        client.register_tools_to_registry(registry)
        
        # Should register 3 tools
        assert registry.register_tool.call_count == 3
        
        # Check registered tool format
        first_call = registry.register_tool.call_args_list[0][0][0]
        assert first_call['id'] == "github.list_repositories"
        assert first_call['name'] == "list_repositories"
        assert first_call['server_type'] == "github"
        
    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test call_tool method success."""
        client = GitHubMCPClient()
        await client.connect(use_mock=True)
        
        result = await client.call_tool("list_repos", {"username": "test"})
        
        assert result["success"] is True
        assert "result" in result
        
    @pytest.mark.asyncio
    async def test_call_tool_failure(self):
        """Test call_tool method failure."""
        client = GitHubMCPClient()
        # Not connected
        
        result = await client.call_tool("list_repos", {})
        
        assert result["success"] is False
        assert "error" in result
        
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        client = GitHubMCPClient()
        await client.connect(use_mock=True)
        
        assert client.connected is True
        
        await client.disconnect()
        
        assert client.connected is False
        assert client.tools == []
        
    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    async def test_disconnect_with_process(self, mock_popen):
        """Test disconnection with running process."""
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        
        client = GitHubMCPClient()
        client.process = mock_process
        client.connected = True
        
        await client.disconnect()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert client.process is None
        
    def test_read_stdout_thread(self):
        """Test stdout reading thread."""
        client = GitHubMCPClient()
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 1]  # Running, then exits
        mock_process.stdout.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "result": {}}',
            'Server started',
            ''
        ]
        
        client.process = mock_process
        client._read_stdout()
        
        # Should have queued responses
        assert client.response_queue.qsize() == 2
        
    @pytest.mark.asyncio
    async def test_discover_tools(self):
        """Test tool discovery."""
        client = GitHubMCPClient()
        client.use_mock = False
        client.process = Mock()
        client.process.stdin = Mock()
        client.response_queue = queue.Queue()
        
        # Simulate tools response
        client.response_queue.put({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {"name": "list_repos", "description": "List repos"},
                    {"name": "create_issue", "description": "Create issue"}
                ]
            }
        })
        
        await client._discover_tools()
        
        assert len(client.tools) == 2
        assert client.tools[0]["name"] == "list_repos"
        
    @pytest.mark.asyncio
    async def test_discover_tools_error(self):
        """Test tool discovery error handling."""
        client = GitHubMCPClient()
        client.use_mock = False
        client.process = Mock()
        client.process.stdin = Mock()
        client.response_queue = queue.Queue()
        
        # No response (timeout)
        await client._discover_tools()
        
        # Should handle gracefully
        assert client.tools == []


class TestMockGitHubMCPServer:
    """Test cases for MockGitHubMCPServer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.server = MockGitHubMCPServer()
        
    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        """Test initialization request handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1
        }
        
        response = await self.server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "1.0"
        
    @pytest.mark.asyncio
    async def test_handle_tools_list(self):
        """Test tools list request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2
        }
        
        response = await self.server.handle_request(request)
        
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0
        
    @pytest.mark.asyncio
    async def test_handle_tool_call_list_repos(self):
        """Test list_repositories tool call with both old and new names."""
        # Test with old name (backward compatibility)
        request_old = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_repos",
                "arguments": {"username": "test"}
            },
            "id": 3
        }
        
        response_old = await self.server.handle_request(request_old)
        
        assert "result" in response_old
        assert isinstance(response_old["result"], list)
        
        # Test with new name
        request_new = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_repositories",
                "arguments": {"username": "test"}
            },
            "id": 4
        }
        
        response_new = await self.server.handle_request(request_new)
        
        assert "result" in response_new
        assert isinstance(response_new["result"], list)
        
    @pytest.mark.asyncio
    async def test_handle_tool_call_create_issue(self):
        """Test create_issue tool call."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "create_issue",
                "arguments": {
                    "owner": "test",
                    "repo": "test-repo",
                    "title": "Test Issue"
                }
            },
            "id": 4
        }
        
        response = await self.server.handle_request(request)
        
        assert "result" in response
        assert "number" in response["result"]
        assert response["result"]["title"] == "Test Issue"
        
    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        """Test unknown method handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "unknown_method",
            "id": 5
        }
        
        response = await self.server.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]
        
    @pytest.mark.asyncio
    async def test_handle_new_tools(self):
        """Test newly added tools."""
        # Test get_file_contents
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_file_contents",
                "arguments": {
                    "owner": "test",
                    "repo": "test-repo",
                    "path": "README.md"
                }
            },
            "id": 6
        }
        
        response = await self.server.handle_request(request)
        
        assert "result" in response
        assert response["result"]["type"] == "file"
        assert "content" in response["result"]
        assert response["result"]["encoding"] == "base64"
        
        # Test get_user
        request_user = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_user",
                "arguments": {"username": "testuser"}
            },
            "id": 7
        }
        
        response_user = await self.server.handle_request(request_user)
        
        assert "result" in response_user
        assert response_user["result"]["login"] == "testuser"
        assert "public_repos" in response_user["result"]
    
    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self):
        """Test unknown tool handling."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {}
            },
            "id": 8
        }
        
        response = await self.server.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == -32602
        
    @pytest.mark.asyncio
    async def test_handle_unimplemented_tool(self):
        """Test unimplemented real server tool."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "merge_pull_request",
                "arguments": {}
            },
            "id": 9
        }
        
        response = await self.server.handle_request(request)
        
        assert "error" in response
        assert "not implemented in mock server" in response["error"]["message"]