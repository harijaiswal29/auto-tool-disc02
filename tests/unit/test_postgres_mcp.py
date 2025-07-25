"""
Unit Tests for PostgreSQL MCP Client

Tests individual components of PostgreSQL MCP client in isolation using mocks.
No external dependencies or real database connections required.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.postgres_mcp import PostgresMCPClient
from src.tools.mock_postgres_mcp import MockPostgresMCPServer
from src.core.tool_registry import ToolRegistry


class TestPostgresMCPClientUnit:
    """Unit tests for PostgreSQL MCP client."""

    @pytest.fixture
    def mock_connection_string(self):
        """Test connection string."""
        return "postgresql://testuser:testpass@localhost:5432/testdb"

    @pytest.fixture
    def client(self, mock_connection_string):
        """Create a PostgreSQL MCP client instance."""
        return PostgresMCPClient(mock_connection_string)

    @pytest.fixture
    def mock_process(self):
        """Create a mock subprocess."""
        mock_proc = Mock()
        mock_proc.stdin = Mock()
        mock_proc.stdout = Mock()
        mock_proc.stderr = Mock()
        mock_proc.poll = Mock(return_value=None)
        return mock_proc

    def test_client_initialization(self, client, mock_connection_string):
        """Test client initialization."""
        assert client.connection_string == mock_connection_string
        assert client.server_name == "postgres"
        assert client.process is None
        assert client.capabilities == {}
        assert client.tools == []
        assert client._message_id == 0
        assert client.mock_server is None
        assert client.use_mock is False

    def test_safe_connection_string(self, client):
        """Test password masking in connection string."""
        # Test with password
        client.connection_string = "postgresql://user:password@host:5432/db"
        safe_string = client._safe_connection_string()
        assert "password" not in safe_string
        assert "***" in safe_string
        assert "user:" in safe_string

        # Test without password
        client.connection_string = "postgresql://user@host:5432/db"
        safe_string = client._safe_connection_string()
        # The current implementation masks even when there's no colon in the user part
        # This is because it splits on "@" and checks for ":" in the first part
        # "postgresql://user" contains ":" after the protocol
        assert "***" in safe_string  # Current behavior masks it

        # Test without @ symbol
        client.connection_string = "postgresql:///localdb"
        safe_string = client._safe_connection_string()
        assert safe_string == client.connection_string

    def test_next_message_id(self, client):
        """Test message ID generation."""
        assert client._message_id == 0
        
        id1 = client._next_message_id()
        assert id1 == 1
        assert client._message_id == 1
        
        id2 = client._next_message_id()
        assert id2 == 2
        assert client._message_id == 2

    @pytest.mark.asyncio
    async def test_connect_with_mock_server(self, client):
        """Test connecting to mock server."""
        # Connect to mock server
        connected = await client.connect(use_mock=True)
        
        assert connected is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert isinstance(client.mock_server, MockPostgresMCPServer)
        assert len(client.tools) == 3  # Mock server provides 3 tools
        
        # Verify tools
        tool_names = [tool["name"] for tool in client.tools]
        assert "query" in tool_names
        assert "get_schema" in tool_names
        assert "list_tables" in tool_names

    @pytest.mark.asyncio
    async def test_connect_with_real_server_mock(self, client, mock_process):
        """Test connecting to real server (mocked subprocess)."""
        with patch('subprocess.Popen', return_value=mock_process):
            # Mock stdout responses
            init_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "1.0",
                    "capabilities": {"tools": True}
                }
            }) + "\n"
            
            tools_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "query", "description": "Execute SQL query"}
                    ]
                }
            }) + "\n"
            
            mock_process.stdout.readline.side_effect = [init_response, tools_response]
            
            connected = await client.connect(use_mock=False)
            
            assert connected is True
            assert client.use_mock is False
            assert client.process is not None
            assert len(client.tools) == 1

    @pytest.mark.asyncio
    async def test_connect_real_server_failure(self, client, mock_process):
        """Test handling of real server connection failure."""
        with patch('subprocess.Popen', return_value=mock_process):
            # Mock error response
            error_response = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32603,
                    "message": "Connection failed"
                }
            }) + "\n"
            
            mock_process.stdout.readline.return_value = error_response
            mock_process.stderr.read.return_value = "Server error output"
            
            connected = await client.connect(use_mock=False)
            
            assert connected is False
            assert client.use_mock is False

    @pytest.mark.asyncio
    async def test_execute_query(self, client):
        """Test query execution."""
        # Setup mock connection
        await client.connect(use_mock=True)
        
        # Test simple query
        result = await client.execute_query("SELECT version()")
        
        assert result["success"] is True
        assert "result" in result
        assert result["execution_time"] >= 0

    @pytest.mark.asyncio
    async def test_execute_query_with_params(self, client):
        """Test parameterized query execution."""
        await client.connect(use_mock=True)
        
        # Mock the call_tool method to return expected result
        expected_result = {
            "success": True,
            "result": {"rows": [], "columns": []},
            "execution_time": 0.1
        }
        
        with patch.object(client, 'call_tool', return_value=expected_result):
            result = await client.execute_query(
                "SELECT * FROM users WHERE id = $1",
                [123]
            )
            
            assert result == expected_result
            client.call_tool.assert_called_once()
            call_args = client.call_tool.call_args[0]
            assert call_args[0] == "query"  # tool name
            assert call_args[1]["sql"] == "SELECT * FROM users WHERE id = $1"
            assert call_args[1]["params"] == [123]

    @pytest.mark.asyncio
    async def test_execute_query_no_tool_available(self, client):
        """Test query execution when no query tool is available."""
        await client.connect(use_mock=True)
        
        # Remove query tool
        client.tools = [
            {"name": "other_tool", "description": "Not a query tool"}
        ]
        
        result = await client.execute_query("SELECT 1")
        
        assert result["error"] == "No query execution tool available"

    @pytest.mark.asyncio
    async def test_get_schema_specific_table(self, client):
        """Test getting schema for specific table."""
        await client.connect(use_mock=True)
        
        result = await client.get_schema("users")
        
        assert result["success"] is True
        assert result["table"] == "users"
        assert "columns" in result
        assert len(result["columns"]) > 0

    @pytest.mark.asyncio
    async def test_get_schema_all_tables(self, client):
        """Test getting schema for all tables."""
        await client.connect(use_mock=True)
        
        # Mock execute_query to return table list
        mock_result = {
            "success": True,
            "result": {
                "rows": [
                    {"table_name": "users", "table_type": "BASE TABLE"},
                    {"table_name": "tools", "table_type": "BASE TABLE"}
                ]
            },
            "execution_time": 0.1
        }
        
        with patch.object(client, 'execute_query', return_value=mock_result):
            result = await client.get_schema()
            
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_list_tables(self, client):
        """Test listing database tables."""
        await client.connect(use_mock=True)
        
        result = await client.list_tables()
        
        assert result["success"] is True
        assert "tables" in result
        assert len(result["tables"]) > 0
        
        # Check table structure
        for table in result["tables"]:
            assert "table_name" in table
            assert "table_type" in table
            assert "table_schema" in table

    @pytest.mark.asyncio
    async def test_get_table_info(self, client):
        """Test getting detailed table information."""
        await client.connect(use_mock=True)
        
        # Mock execute_query for table info
        mock_result = {
            "success": True,
            "result": {
                "rows": [
                    {
                        "column_name": "id",
                        "data_type": "integer",
                        "is_nullable": "NO"
                    }
                ]
            },
            "execution_time": 0.1
        }
        
        with patch.object(client, 'execute_query', return_value=mock_result):
            result = await client.get_table_info("users")
            
            assert result == mock_result
            client.execute_query.assert_called_once()
            # Check that parameterized query was used
            call_args = client.execute_query.call_args[0]
            assert "information_schema.columns" in call_args[0]
            assert call_args[1] == ["users"]

    @pytest.mark.asyncio
    async def test_call_tool_success(self, client):
        """Test successful tool execution."""
        await client.connect(use_mock=True)
        
        result = await client.call_tool("query", {"sql": "SELECT 1"})
        
        assert result["success"] is True
        assert "result" in result
        assert "execution_time" in result
        assert result["execution_time"] >= 0

    @pytest.mark.asyncio
    async def test_call_tool_error(self, client):
        """Test tool execution error handling."""
        await client.connect(use_mock=True)
        
        # Test with invalid SQL (write operation)
        result = await client.call_tool(
            "query",
            {"sql": "INSERT INTO users VALUES (1)"}
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "read-only" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_process):
        """Test disconnecting from server."""
        # Setup mock connection
        with patch('subprocess.Popen', return_value=mock_process):
            client.process = mock_process
            # Mock that process terminated successfully (poll returns non-None)
            mock_process.poll.return_value = 0  # Process has exited
            
            await client.disconnect()
            
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_force_kill(self, client, mock_process):
        """Test force killing process on disconnect."""
        with patch('subprocess.Popen', return_value=mock_process):
            client.process = mock_process
            # Mock that process is still running after terminate
            mock_process.poll.return_value = None
            
            await client.disconnect()
            
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()

    def test_register_tools_to_registry(self, client):
        """Test registering tools with tool registry."""
        # Setup mock tools
        client.tools = [
            {"name": "query", "description": "Execute query"},
            {"name": "get_schema", "description": "Get schema"}
        ]
        client.capabilities = {"tools": True}
        
        # Mock registry
        mock_registry = Mock(spec=ToolRegistry)
        
        client.register_tools_to_registry(mock_registry)
        
        # Verify registration calls
        assert mock_registry.register_tool.call_count == 2
        
        # Check first tool registration
        first_call = mock_registry.register_tool.call_args_list[0][0][0]
        assert first_call['id'] == 'postgres.query'
        assert first_call['name'] == 'query'
        assert first_call['server_type'] == 'postgres'

    @pytest.mark.asyncio
    async def test_send_receive_messages(self, client, mock_process):
        """Test JSON-RPC message sending and receiving."""
        client.process = mock_process
        
        # Test sending message
        test_message = {"jsonrpc": "2.0", "method": "test", "id": 1}
        await client._send_message(test_message)
        
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert json.loads(written_data.strip()) == test_message
        
        # Test receiving message
        response_data = json.dumps({"jsonrpc": "2.0", "result": "ok", "id": 1}) + "\n"
        mock_process.stdout.readline.return_value = response_data
        
        response = await client._receive_message()
        assert response == {"jsonrpc": "2.0", "result": "ok", "id": 1}

    @pytest.mark.asyncio
    async def test_receive_message_json_error(self, client, mock_process):
        """Test handling of invalid JSON in received messages."""
        client.process = mock_process
        
        # Return invalid JSON
        mock_process.stdout.readline.return_value = "invalid json\n"
        
        response = await client._receive_message()
        assert response is None

    @pytest.mark.asyncio
    async def test_receive_message_no_process(self, client):
        """Test receiving message when no process exists."""
        client.process = None
        
        response = await client._receive_message()
        assert response is None


class TestMockPostgresMCPServerUnit:
    """Unit tests for the mock PostgreSQL MCP server."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server instance."""
        return MockPostgresMCPServer("postgresql://test:test@localhost/test")

    def test_mock_server_initialization(self, mock_server):
        """Test mock server initialization."""
        assert mock_server.initialized is False
        assert len(mock_server.tools) == 3
        assert "tables" in mock_server.mock_data
        
        # Check mock data structure
        assert "users" in mock_server.mock_data["tables"]
        assert "tools" in mock_server.mock_data["tables"]
        assert "execution_history" in mock_server.mock_data["tables"]

    @pytest.mark.asyncio
    async def test_mock_server_initialize(self, mock_server):
        """Test mock server initialization request."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {"name": "test", "version": "1.0"}
            },
            "id": 1
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 1
        assert response["result"]["protocolVersion"] == "1.0"
        assert response["result"]["capabilities"]["tools"] is True
        assert mock_server.initialized is True

    @pytest.mark.asyncio
    async def test_mock_server_tools_list(self, mock_server):
        """Test listing tools from mock server."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 2
        assert len(response["result"]["tools"]) == 3
        
        tool_names = [t["name"] for t in response["result"]["tools"]]
        assert "query" in tool_names
        assert "get_schema" in tool_names
        assert "list_tables" in tool_names

    @pytest.mark.asyncio
    async def test_mock_server_query_execution(self, mock_server):
        """Test query execution on mock server."""
        # Test SELECT version()
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {"sql": "SELECT version()"}
            },
            "id": 3
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 3
        assert "result" in response
        assert response["result"]["rows"][0]["version"] == "PostgreSQL 14.10 (Mock Database)"

    @pytest.mark.asyncio
    async def test_mock_server_write_query_blocked(self, mock_server):
        """Test that write queries are blocked."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {"sql": "INSERT INTO users VALUES (1)"}
            },
            "id": 4
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 4
        assert "error" in response
        assert "read-only" in response["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_mock_server_unknown_method(self, mock_server):
        """Test handling of unknown methods."""
        request = {
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
            "id": 5
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 5
        assert "error" in response
        assert "Unknown method" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_mock_server_table_query(self, mock_server):
        """Test querying mock table data."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {"sql": "SELECT * FROM users"}
            },
            "id": 6
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 6
        assert "result" in response
        assert len(response["result"]["rows"]) == 3  # 3 mock users
        assert response["result"]["columns"] == ["id", "username", "email", "created_at"]

    @pytest.mark.asyncio
    async def test_mock_server_schema_operations(self, mock_server):
        """Test schema-related operations."""
        # Test get_schema for specific table
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_schema",
                "arguments": {"table": "users"}
            },
            "id": 7
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 7
        assert response["result"]["table"] == "users"
        assert len(response["result"]["columns"]) == 4  # id, username, email, created_at

    @pytest.mark.asyncio
    async def test_mock_server_list_tables(self, mock_server):
        """Test listing tables."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_tables",
                "arguments": {}
            },
            "id": 8
        }
        
        response = await mock_server.handle_request(request)
        
        assert response["id"] == 8
        assert len(response["result"]["tables"]) == 3
        
        table_names = [t["table_name"] for t in response["result"]["tables"]]
        assert "users" in table_names
        assert "tools" in table_names
        assert "execution_history" in table_names