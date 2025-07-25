"""
Unit Tests for PostgreSQL MCP Real Server Mode

These unit tests focus on testing the real server code paths
of the PostgreSQL MCP client, but still use mocks as proper unit tests should.
They test behavior specific to real server connections without actually connecting.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from pathlib import Path
import sys
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.postgres_mcp import PostgresMCPClient
from src.core.tool_registry import ToolRegistry


class TestPostgresMCPRealServerUnit:
    """Unit tests for PostgreSQL MCP client real server mode."""

    @pytest.fixture
    def connection_string(self):
        """Test connection string."""
        return "postgresql://testuser:testpass@localhost:5432/testdb"

    @pytest.fixture
    def client(self, connection_string):
        """Create a PostgreSQL MCP client instance."""
        return PostgresMCPClient(connection_string)

    @pytest.fixture
    def mock_process(self):
        """Create a mock subprocess for real server simulation."""
        mock_proc = Mock()
        mock_proc.stdin = Mock()
        mock_proc.stdout = Mock()
        mock_proc.stderr = Mock()
        mock_proc.poll = Mock(return_value=None)
        mock_proc.terminate = Mock()
        mock_proc.kill = Mock()
        return mock_proc

    def test_real_server_command_construction(self, client):
        """Test that real server command is properly constructed."""
        # Default command should use mcp-server-postgres
        assert client.server_command[0] == "./node_modules/.bin/mcp-server-postgres"
        assert client.connection_string in client.server_command
        
        # Test with custom command
        custom_client = PostgresMCPClient(
            "postgresql://test:test@localhost/db",
            server_command=["custom-mcp-server", "--custom-arg"]
        )
        assert custom_client.server_command[0] == "custom-mcp-server"
        assert custom_client.server_command[1] == "--custom-arg"

    @pytest.mark.asyncio
    async def test_real_server_connection_success(self, client, mock_process):
        """Test successful connection to real server."""
        with patch('subprocess.Popen', return_value=mock_process):
            # Mock successful initialization response
            init_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "1.0",
                    "capabilities": {
                        "tools": True,
                        "resources": True
                    },
                    "serverInfo": {
                        "name": "postgres-mcp-server",
                        "version": "1.0.0"
                    }
                }
            }) + "\n"
            
            # Mock tools list response (real server only provides 'query' tool)
            tools_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [{
                        "name": "query",
                        "description": "Execute read-only SQL queries",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "sql": {
                                    "type": "string",
                                    "description": "The SQL query to execute"
                                }
                            },
                            "required": ["sql"]
                        }
                    }]
                }
            }) + "\n"
            
            mock_process.stdout.readline.side_effect = [init_response, tools_response]
            
            # Connect with real server mode
            connected = await client.connect(use_mock=False)
            
            assert connected is True
            assert client.use_mock is False
            assert client.mock_server is None
            assert client.process == mock_process
            
            # Verify subprocess was started with correct command
            subprocess.Popen.assert_called_once()
            call_args = subprocess.Popen.call_args[0][0]
            assert "./node_modules/.bin/mcp-server-postgres" in call_args[0]
            assert client.connection_string in call_args
            
            # Verify tools discovered
            assert len(client.tools) == 1
            assert client.tools[0]["name"] == "query"

    @pytest.mark.asyncio
    async def test_real_server_connection_failure_scenarios(self, client, mock_process):
        """Test various connection failure scenarios with real server."""
        
        # Test 1: Process fails to start
        with patch('subprocess.Popen', side_effect=FileNotFoundError("mcp-server-postgres not found")):
            connected = await client.connect(use_mock=False)
            assert connected is False
            assert client.process is None
        
        # Test 2: Server returns error during initialization
        with patch('subprocess.Popen', return_value=mock_process):
            error_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32603,
                    "message": "Failed to connect to database",
                    "data": {
                        "details": "FATAL: password authentication failed"
                    }
                }
            }) + "\n"
            
            mock_process.stdout.readline.return_value = error_response
            mock_process.stderr.read.return_value = "Connection error details"
            
            connected = await client.connect(use_mock=False)
            assert connected is False

    @pytest.mark.asyncio
    async def test_real_server_query_execution_formats(self, client, mock_process):
        """Test different response formats from real server."""
        # Setup connected client
        client.use_mock = False
        client.process = mock_process
        client.tools = [{"name": "query", "description": "Execute SQL"}]
        client._message_id = 10
        
        # Test different response formats that real server might return
        
        # Format 1: Direct result
        direct_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 11,
            "result": {
                "rows": [{"version": "PostgreSQL 14.5"}],
                "columns": ["version"]
            }
        }) + "\n"
        
        mock_process.stdout.readline.return_value = direct_response
        
        result = await client.execute_query("SELECT version()")
        assert result["success"] is True
        assert "result" in result
        
        # Format 2: Content-wrapped result (Claude format)
        content_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 12,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "rows": [{"count": 5}],
                        "columns": ["count"]
                    })
                }]
            }
        }) + "\n"
        
        mock_process.stdout.readline.return_value = content_response
        
        result = await client.execute_query("SELECT COUNT(*) FROM users")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_real_server_read_only_query_validation(self, client, mock_process):
        """Test that client doesn't pre-validate queries (server handles it)."""
        # Setup connected client
        client.use_mock = False
        client.process = mock_process
        client.tools = [{"name": "query", "description": "Execute SQL"}]
        
        # Client should send even write queries to server (server enforces read-only)
        write_queries = [
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name = 'new' WHERE id = 1",
            "DELETE FROM users WHERE id = 1",
            "DROP TABLE users"
        ]
        
        for sql in write_queries:
            # Mock server response (error)
            error_response = json.dumps({
                "jsonrpc": "2.0",
                "id": client._message_id + 1,
                "error": {
                    "code": -32603,
                    "message": "Cannot execute write query in read-only transaction"
                }
            }) + "\n"
            
            mock_process.stdout.readline.return_value = error_response
            
            # Client should attempt to execute
            result = await client.execute_query(sql)
            
            # Verify message was sent
            mock_process.stdin.write.assert_called()
            
            # Result should indicate failure from server
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_real_server_parameterized_queries(self, client, mock_process):
        """Test parameterized query handling for real server."""
        client.use_mock = False
        client.process = mock_process
        client.tools = [{"name": "query", "description": "Execute SQL"}]
        
        # Mock successful response
        param_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "rows": [{"id": 1, "name": "test"}],
                "columns": ["id", "name"]
            }
        }) + "\n"
        
        mock_process.stdout.readline.return_value = param_response
        
        # Execute parameterized query
        result = await client.execute_query(
            "SELECT * FROM users WHERE id = $1",
            [123]
        )
        
        # Verify call_tool was invoked with params
        call_args = mock_process.stdin.write.call_args[0][0]
        request = json.loads(call_args.strip())
        
        assert request["method"] == "tools/call"
        assert request["params"]["name"] == "query"
        assert request["params"]["arguments"]["sql"] == "SELECT * FROM users WHERE id = $1"
        assert request["params"]["arguments"]["params"] == [123]

    @pytest.mark.asyncio
    async def test_real_server_connection_lifecycle(self, client, mock_process):
        """Test complete connection lifecycle with real server."""
        with patch('subprocess.Popen', return_value=mock_process):
            # Setup responses
            init_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "1.0", "capabilities": {"tools": True}}
            }) + "\n"
            
            tools_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": [{"name": "query"}]}
            }) + "\n"
            
            # Execute query response
            query_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"rows": [{"test": "data"}]}
            }) + "\n"
            
            # Set up all responses including the query response
            mock_process.stdout.readline.side_effect = [init_response, tools_response, query_response]
            
            # Connect
            connected = await client.connect(use_mock=False)
            assert connected is True
            result = await client.execute_query("SELECT 'data' as test")
            assert result["success"] is True
            
            # Disconnect
            mock_process.poll.return_value = None  # Process still running
            await client.disconnect()
            
            # Verify termination sequence
            mock_process.terminate.assert_called_once()
            # The kill check happens in a separate task, so we need to wait a bit more
            await asyncio.sleep(0.6)  # Give time for kill check (disconnect waits 0.5s)
            
            # Verify kill was called if process didn't terminate
            if mock_process.poll.return_value is None:
                mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_server_error_handling(self, client, mock_process):
        """Test various error conditions with real server."""
        # Setup client properly
        client.use_mock = False
        client.process = mock_process
        client.tools = [{"name": "query"}]
        
        # Ensure stdout is set on the mock process
        mock_process.stdout = Mock()
        
        # Test 1: Network/connection errors - the method doesn't catch IOError, so we verify the exception is raised
        mock_process.stdout.readline.side_effect = IOError("Connection lost")
        
        with pytest.raises(IOError, match="Connection lost"):
            await client._receive_message()
        
        # Test 2: Invalid JSON response
        mock_process.stdout.readline.side_effect = None
        mock_process.stdout.readline.return_value = "Invalid JSON response\n"
        
        result = await client._receive_message()
        assert result is None
        
        # Test 3: Server timeout (empty response)
        mock_process.stdout.readline.return_value = ""
        
        result = await client._receive_message()
        assert result is None

    def test_real_server_tool_registration(self, client):
        """Test tool registration with real server tools."""
        # Setup client as if connected to real server
        client.use_mock = False
        client.tools = [{
            "name": "query",
            "description": "Execute read-only SQL queries",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        }]
        client.capabilities = {"tools": True, "resources": True}
        
        # Mock registry
        mock_registry = Mock(spec=ToolRegistry)
        
        # Register tools
        client.register_tools_to_registry(mock_registry)
        
        # Verify registration
        mock_registry.register_tool.assert_called_once()
        
        registered_tool = mock_registry.register_tool.call_args[0][0]
        assert registered_tool['id'] == 'postgres.query'
        assert registered_tool['name'] == 'query'
        assert registered_tool['server_type'] == 'postgres'
        assert 'input_schema' in registered_tool
        assert registered_tool['capabilities'] == client.capabilities

    @pytest.mark.asyncio
    async def test_real_server_json_rpc_protocol(self, client, mock_process):
        """Test JSON-RPC protocol compliance for real server."""
        client.process = mock_process
        
        # Test request formatting
        test_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "query", "arguments": {"sql": "SELECT 1"}},
            "id": 42
        }
        
        await client._send_message(test_request)
        
        # Verify proper JSON-RPC formatting
        written_data = mock_process.stdin.write.call_args[0][0]
        assert written_data.endswith("\n")
        
        parsed_request = json.loads(written_data.strip())
        assert parsed_request["jsonrpc"] == "2.0"
        assert parsed_request["id"] == 42
        assert parsed_request["method"] == "tools/call"
        
        # Verify stdin flush was called
        mock_process.stdin.flush.assert_called()

    @pytest.mark.asyncio
    async def test_real_server_concurrent_message_handling(self, client, mock_process):
        """Test handling of concurrent messages with real server."""
        client.process = mock_process
        client._message_id = 100
        
        # Simulate multiple concurrent requests
        messages = []
        for i in range(3):
            msg = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "query", "arguments": {"sql": f"SELECT {i}"}},
                "id": client._next_message_id()
            }
            messages.append(msg)
            await client._send_message(msg)
        
        # Verify all messages were sent
        assert mock_process.stdin.write.call_count == 3
        
        # Verify message IDs are sequential
        for i, call in enumerate(mock_process.stdin.write.call_args_list):
            data = json.loads(call[0][0].strip())
            assert data["id"] == 101 + i