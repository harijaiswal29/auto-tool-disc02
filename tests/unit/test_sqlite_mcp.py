#!/usr/bin/env python3
"""
Unit tests for SQLite MCP Client.

Tests individual components and methods of the SQLite MCP client
with extensive mocking to ensure isolation from external dependencies.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime
import sqlite3
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.sqlite_mcp import SQLiteMCPClient
from src.tools.mock_sqlite_mcp import MockSQLiteMCPServer
from src.utils.retry import RetryableError, NonRetryableError, ExponentialBackoffRetry
from src.core.tool_registry import ToolRegistry


class TestSQLiteMCPClient:
    """Unit tests for SQLite MCP Client."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def client(self, temp_db):
        """Create SQLite MCP client instance."""
        return SQLiteMCPClient(temp_db)
    
    @pytest.fixture
    def mock_process(self):
        """Create mock subprocess."""
        process = Mock()
        process.stdin = Mock()
        process.stdout = Mock()
        process.stderr = Mock()
        process.poll = Mock(return_value=None)
        return process
    
    def test_initialization(self, temp_db):
        """Test client initialization."""
        # Test with default parameters
        client = SQLiteMCPClient(temp_db)
        assert client.db_path == Path(temp_db).resolve()
        assert client.server_name == "sqlite"
        assert client.server_command == ["npx", "@modelcontextprotocol/server-sqlite", str(Path(temp_db).resolve())]
        assert isinstance(client.retry_policy, ExponentialBackoffRetry)
        
        # Test with custom parameters
        custom_command = ["custom", "command"]
        custom_retry = ExponentialBackoffRetry(max_attempts=5)
        client2 = SQLiteMCPClient(temp_db, server_command=custom_command, retry_policy=custom_retry)
        assert client2.server_command == custom_command
        assert client2.retry_policy == custom_retry
    
    def test_message_id_generation(self, client):
        """Test message ID generation."""
        id1 = client._next_message_id()
        id2 = client._next_message_id()
        id3 = client._next_message_id()
        
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
    
    @pytest.mark.asyncio
    async def test_connect_with_mock_server(self, client):
        """Test connection with mock server."""
        # Connect with mock server
        result = await client.connect(use_mock=True)
        
        assert result is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert isinstance(client.mock_server, MockSQLiteMCPServer)
        assert len(client.tools) > 0  # Should discover tools
    
    @pytest.mark.asyncio
    async def test_connect_with_real_server_failure(self, client, mock_process):
        """Test connection failure with real server."""
        with patch('subprocess.Popen', return_value=mock_process):
            # Mock the receive_message to return None (connection failure)
            with patch.object(client, '_receive_message', return_value=None):
                result = await client.connect(use_mock=False)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_connect_with_real_server_success(self, client, mock_process):
        """Test successful connection with real server."""
        # Mock subprocess
        with patch('subprocess.Popen', return_value=mock_process):
            # Mock successful initialization response
            init_response = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "1.0",
                    "serverInfo": {"name": "SQLiteMCP", "version": "1.0"},
                    "capabilities": {"tools": True}
                }
            }
            
            # Mock tool list response
            tools_response = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "query", "description": "Execute SQL query"},
                        {"name": "get_schema", "description": "Get schema"}
                    ]
                }
            }
            
            responses = [init_response, tools_response]
            with patch.object(client, '_receive_message', side_effect=responses):
                with patch.object(client, '_send_message', return_value=None):
                    result = await client.connect(use_mock=False)
                    
                    assert result is True
                    assert client.use_mock is False
                    assert len(client.tools) == 2
                    assert client.capabilities == {"tools": True}
    
    def test_is_non_retryable_error(self, client):
        """Test non-retryable error detection."""
        # Non-retryable errors
        assert client._is_non_retryable_error("Syntax error in SQL") is True
        assert client._is_non_retryable_error("no such table: users") is True
        assert client._is_non_retryable_error("no such column: id") is True
        assert client._is_non_retryable_error("Permission denied") is True
        assert client._is_non_retryable_error("Authentication failed") is True
        assert client._is_non_retryable_error("Invalid argument provided") is True
        
        # Retryable errors
        assert client._is_non_retryable_error("Connection timeout") is False
        assert client._is_non_retryable_error("Network error") is False
        assert client._is_non_retryable_error("Server busy") is False
    
    @pytest.mark.asyncio
    async def test_execute_query_with_mock(self, client):
        """Test query execution with mock server."""
        await client.connect(use_mock=True)
        
        # Execute a simple query
        result = await client.execute_query("SELECT * FROM users")
        
        assert result is not None
        assert 'success' in result or 'result' in result
    
    @pytest.mark.asyncio
    async def test_execute_query_with_params(self, client):
        """Test query execution with parameters."""
        await client.connect(use_mock=True)
        
        # Execute query with parameters
        result = await client.execute_query(
            "SELECT * FROM users WHERE id = ?",
            [1]
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_execute_query_no_tools(self, client):
        """Test query execution when no query tool is available."""
        await client.connect(use_mock=True)
        client.tools = []  # Remove all tools
        
        result = await client.execute_query("SELECT * FROM users")
        
        assert result["error"] == "No query execution tool available"
    
    @pytest.mark.asyncio
    async def test_get_schema(self, client):
        """Test schema retrieval."""
        await client.connect(use_mock=True)
        
        # Get schema for specific table
        result = await client.get_schema("users")
        assert result is not None
        
        # Get all tables schema
        result_all = await client.get_schema()
        assert result_all is not None
    
    @pytest.mark.asyncio
    async def test_call_tool_with_retry(self, client):
        """Test tool execution with retry logic."""
        await client.connect(use_mock=True)
        
        # Test successful execution
        result = await client.call_tool("query", {"sql": "SELECT 1"})
        assert result["success"] is True
        
        # Test with retryable error
        with patch.object(client.mock_server, 'handle_request', 
                         side_effect=[RetryableError("Temporary error"), 
                                    {"jsonrpc": "2.0", "id": 1, "result": {"rows": []}}]):
            result = await client.call_tool("query", {"sql": "SELECT 1"})
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_call_tool_non_retryable_error(self, client):
        """Test tool execution with non-retryable error."""
        await client.connect(use_mock=True)
        
        # Mock non-retryable error response
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"message": "no such table: nonexistent"}
        }
        
        with patch.object(client.mock_server, 'handle_request', return_value=error_response):
            result = await client.call_tool("query", {"sql": "SELECT * FROM nonexistent"})
            assert result["success"] is False
            assert "no such table" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_receive_messages(self, client, mock_process):
        """Test message sending and receiving."""
        client.process = mock_process
        
        # Test send message
        message = {"jsonrpc": "2.0", "method": "test", "id": 1}
        await client._send_message(message)
        
        expected_call = json.dumps(message) + "\n"
        mock_process.stdin.write.assert_called_once_with(expected_call)
        mock_process.stdin.flush.assert_called_once()
        
        # Test receive message
        response = {"jsonrpc": "2.0", "result": "success", "id": 1}
        mock_process.stdout.readline.return_value = json.dumps(response) + "\n"
        
        received = await client._receive_message()
        assert received == response
        
        # Test receive with invalid JSON
        mock_process.stdout.readline.return_value = "invalid json\n"
        received = await client._receive_message()
        assert received is None
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_process):
        """Test disconnection."""
        client.process = mock_process
        
        await client.disconnect()
        
        mock_process.terminate.assert_called_once()
        # Should check if process is still running after terminate
        mock_process.poll.assert_called()
    
    def test_register_tools_to_registry(self, client):
        """Test tool registration to registry."""
        # Set up mock tools
        client.tools = [
            {"name": "query", "description": "Execute SQL", "inputSchema": {}},
            {"name": "get_schema", "description": "Get schema", "inputSchema": {}}
        ]
        client.capabilities = {"tools": True}
        
        # Mock registry
        mock_registry = Mock(spec=ToolRegistry)
        
        # Register tools
        client.register_tools_to_registry(mock_registry)
        
        # Verify registration calls
        assert mock_registry.register_tool.call_count == 2
        
        # Check first tool registration
        first_call = mock_registry.register_tool.call_args_list[0][0][0]
        assert first_call['id'] == 'sqlite.query'
        assert first_call['name'] == 'query'
        assert first_call['server_type'] == 'sqlite'
        assert first_call['description'] == 'Execute SQL'
        
        # Check second tool registration
        second_call = mock_registry.register_tool.call_args_list[1][0][0]
        assert second_call['id'] == 'sqlite.get_schema'
        assert second_call['name'] == 'get_schema'
    
    @pytest.mark.asyncio
    async def test_retry_policy_application(self, client):
        """Test that retry policy is properly applied."""
        # Create a client with custom retry policy
        custom_retry = ExponentialBackoffRetry(
            max_attempts=2,
            base_delay=0.1,
            max_delay=0.5
        )
        client = SQLiteMCPClient("test.db", retry_policy=custom_retry)
        
        await client.connect(use_mock=True)
        
        # Mock to fail twice then succeed
        call_count = 0
        def mock_handler(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Temporary failure")
            return {"jsonrpc": "2.0", "id": 1, "result": {"success": True}}
        
        with patch.object(client.mock_server, 'handle_request', side_effect=mock_handler):
            result = await client.call_tool("query", {"sql": "SELECT 1"})
            assert result["success"] is True
            assert call_count == 2  # Should retry once
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, client):
        """Test various connection error scenarios."""
        # Connect first
        await client.connect(use_mock=True)
        
        # Test timeout error
        with patch.object(client.mock_server, 'handle_request', 
                         side_effect=asyncio.TimeoutError("Connection timeout")):
            result = await client.call_tool("query", {"sql": "SELECT 1"})
            assert result["success"] is False
            assert "Connection error" in result["error"]
        
        # Test OS error
        with patch.object(client.mock_server, 'handle_request', 
                         side_effect=OSError("Socket error")):
            result = await client.call_tool("query", {"sql": "SELECT 1"})
            assert result["success"] is False
            assert "Connection error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_mock_server_functionality(self, temp_db):
        """Test mock server implementation."""
        server = MockSQLiteMCPServer(temp_db)
        
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"protocolVersion": "1.0"},
            "id": 1
        }
        response = await server.handle_request(init_request)
        assert response["result"]["protocolVersion"] == "1.0"
        
        # Test tool listing
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2
        }
        response = await server.handle_request(list_request)
        assert len(response["result"]["tools"]) > 0
        
        # Test query execution
        query_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {"sql": "CREATE TABLE test (id INTEGER)"}
            },
            "id": 3
        }
        response = await server.handle_request(query_request)
        assert "result" in response
        
        # Test error handling
        error_request = {
            "jsonrpc": "2.0",
            "method": "unknown_method",
            "id": 4
        }
        response = await server.handle_request(error_request)
        assert "error" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])