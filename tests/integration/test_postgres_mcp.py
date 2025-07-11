"""
PostgreSQL MCP Integration Tests

Tests for PostgreSQL Model Context Protocol client implementation,
including both real server and mock server testing.
"""

import asyncio
import pytest
from pathlib import Path
import sys
import os
import subprocess
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.postgres_mcp import PostgresMCPClient
from src.tools.mock_postgres_mcp import MockPostgresMCPServer
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestPostgresMCPClient:
    """Test PostgreSQL MCP client functionality."""
    
    @pytest.fixture
    def test_connection_string(self):
        """Get test database connection string."""
        return os.environ.get(
            "POSTGRES_TEST_CONNECTION",
            "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
        )
    
    @pytest.fixture
    async def mock_client(self, test_connection_string):
        """Create a client connected to mock server."""
        client = PostgresMCPClient(test_connection_string)
        connected = await client.connect(use_mock=True)
        assert connected, "Failed to connect to mock server"
        yield client
        await client.disconnect()
    
    @pytest.fixture
    def docker_available(self):
        """Check if Docker is available and PostgreSQL container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=auto_tool_disc_postgres", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            return "auto_tool_disc_postgres" in result.stdout
        except:
            return False
    
    @pytest.fixture
    async def real_client(self, test_connection_string, docker_available):
        """Create a client connected to real server (if available)."""
        if not docker_available:
            pytest.skip("PostgreSQL Docker container not running")
        
        client = PostgresMCPClient(test_connection_string)
        connected = await client.connect(use_mock=False)
        
        if not connected:
            pytest.skip("Could not connect to real PostgreSQL MCP server")
        
        yield client
        await client.disconnect()
    
    # Mock Server Tests
    
    async def test_mock_server_connection(self, test_connection_string):
        """Test connecting to mock PostgreSQL MCP server."""
        client = PostgresMCPClient(test_connection_string)
        connected = await client.connect(use_mock=True)
        
        assert connected, "Failed to connect to mock server"
        assert client.use_mock is True
        assert client.mock_server is not None
        assert len(client.tools) > 0
        
        await client.disconnect()
    
    async def test_mock_server_tools_discovery(self, mock_client):
        """Test tool discovery with mock server."""
        # Mock server should provide 3 tools
        assert len(mock_client.tools) == 3
        
        tool_names = [tool["name"] for tool in mock_client.tools]
        assert "query" in tool_names
        assert "get_schema" in tool_names
        assert "list_tables" in tool_names
    
    async def test_mock_execute_query(self, mock_client):
        """Test query execution with mock server."""
        # Test version query
        result = await mock_client.execute_query("SELECT version()")
        assert result["success"] is True
        assert "PostgreSQL 14.10 (Mock Database)" in str(result["result"])
        
        # Test table query
        result = await mock_client.execute_query("SELECT * FROM users")
        assert result["success"] is True
        assert "result" in result
    
    async def test_mock_list_tables(self, mock_client):
        """Test listing tables with mock server."""
        result = await mock_client.list_tables()
        
        assert result["success"] is True
        assert "tables" in result
        tables = result["tables"]
        
        table_names = [t["table_name"] for t in tables]
        assert "users" in table_names
        assert "tools" in table_names
        assert "execution_history" in table_names
    
    async def test_mock_get_schema(self, mock_client):
        """Test getting schema with mock server."""
        # Test specific table schema
        result = await mock_client.get_schema("users")
        assert result["success"] is True
        assert result["table"] == "users"
        assert "columns" in result
        
        columns = result["columns"]
        column_names = [c["column_name"] for c in columns]
        assert "id" in column_names
        assert "username" in column_names
        assert "email" in column_names
    
    async def test_mock_read_only_constraint(self, mock_client):
        """Test that mock server enforces read-only constraint."""
        # Try to execute INSERT query
        result = await mock_client.execute_query(
            "INSERT INTO users (username) VALUES ('test')"
        )
        
        assert result["success"] is False
        assert "read-only" in result["error"].lower()
    
    # Real Server Tests (skipped if not available)
    
    @pytest.mark.skipif(not os.environ.get("TEST_REAL_POSTGRES"), 
                        reason="Real PostgreSQL testing not enabled")
    async def test_real_server_connection(self, real_client):
        """Test connecting to real PostgreSQL MCP server."""
        assert real_client.use_mock is False
        assert len(real_client.tools) >= 1  # At least 'query' tool
        
        # Real server only provides 'query' tool
        tool_names = [tool["name"] for tool in real_client.tools]
        assert "query" in tool_names
    
    @pytest.mark.skipif(not os.environ.get("TEST_REAL_POSTGRES"),
                        reason="Real PostgreSQL testing not enabled")
    async def test_real_execute_query(self, real_client):
        """Test query execution with real server."""
        result = await real_client.execute_query("SELECT version()")
        assert result["success"] is True
        assert "PostgreSQL" in str(result["result"])
    
    @pytest.mark.skipif(not os.environ.get("TEST_REAL_POSTGRES"),
                        reason="Real PostgreSQL testing not enabled")
    async def test_real_list_tables_via_query(self, real_client):
        """Test listing tables via SQL query with real server."""
        result = await real_client.list_tables()
        assert result["success"] is True
        assert "tables" in result or "result" in result
    
    # Tool Registry Integration Tests
    
    async def test_tool_registry_integration(self, mock_client, tmp_path):
        """Test registering PostgreSQL tools with tool registry."""
        registry_db = tmp_path / "test_registry.db"
        registry = ToolRegistry(str(registry_db))
        
        # Register tools
        mock_client.register_tools_to_registry(registry)
        
        # Check registered tools
        postgres_tools = registry.list_tools("postgres")
        assert len(postgres_tools) == 3
        
        tool_ids = [tool["id"] for tool in postgres_tools]
        assert "postgres.query" in tool_ids
        assert "postgres.get_schema" in tool_ids
        assert "postgres.list_tables" in tool_ids
    
    # Error Handling Tests
    
    async def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        # Use invalid connection string
        client = PostgresMCPClient("postgresql://invalid:invalid@nohost:5432/nodb")
        
        # Should fail to connect to real server
        connected = await client.connect(use_mock=False)
        assert connected is False
        
        # Should fall back to mock successfully
        connected = await client.connect(use_mock=True)
        assert connected is True
        await client.disconnect()
    
    async def test_invalid_query_handling(self, mock_client):
        """Test handling of invalid queries."""
        # Test with DROP query (should be blocked)
        result = await mock_client.execute_query("DROP TABLE users")
        assert result["success"] is False
        assert "read-only" in result["error"].lower()
    
    async def test_parameterized_queries(self, mock_client):
        """Test parameterized query execution."""
        result = await mock_client.get_table_info("users")
        assert "success" in result or "result" in result
    
    # Performance Tests
    
    async def test_query_performance(self, mock_client):
        """Test query execution performance tracking."""
        result = await mock_client.execute_query("SELECT * FROM tools")
        
        assert "execution_time" in result
        assert isinstance(result["execution_time"], (int, float))
        assert result["execution_time"] >= 0
    
    # Connection State Tests
    
    async def test_multiple_connections(self, test_connection_string):
        """Test multiple client connections."""
        clients = []
        
        # Create multiple clients
        for i in range(3):
            client = PostgresMCPClient(test_connection_string)
            connected = await client.connect(use_mock=True)
            assert connected
            clients.append(client)
        
        # Execute queries on all clients
        for i, client in enumerate(clients):
            result = await client.execute_query(f"SELECT {i} as client_id")
            assert result["success"] is True
        
        # Disconnect all
        for client in clients:
            await client.disconnect()


class TestMockPostgresMCPServer:
    """Test the mock PostgreSQL MCP server implementation."""
    
    async def test_mock_server_initialization(self):
        """Test mock server initialization."""
        server = MockPostgresMCPServer("postgresql://test:test@localhost/test")
        
        assert server.initialized is False
        assert len(server.tools) == 3
        assert "tables" in server.mock_data
    
    async def test_mock_server_protocol(self):
        """Test mock server JSON-RPC protocol handling."""
        server = MockPostgresMCPServer("postgresql://test:test@localhost/test")
        
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {"name": "test", "version": "1.0"}
            },
            "id": 1
        }
        
        response = await server.handle_request(init_request)
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "1.0"
        
        # Test tools listing
        tools_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        response = await server.handle_request(tools_request)
        assert response["id"] == 2
        assert len(response["result"]["tools"]) == 3
    
    async def test_mock_data_queries(self):
        """Test querying mock data."""
        server = MockPostgresMCPServer("postgresql://test:test@localhost/test")
        
        # Query users table
        query_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {"sql": "SELECT * FROM users"}
            },
            "id": 3
        }
        
        response = await server.handle_request(query_request)
        assert response["id"] == 3
        assert "result" in response
        assert len(response["result"]["rows"]) == 3  # 3 mock users


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    connection_string = "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    
    # Initialize client
    client = PostgresMCPClient(connection_string)
    
    # Connect (will use mock if real server not available)
    connected = await client.connect()
    if not connected:
        connected = await client.connect(use_mock=True)
    
    assert connected, "Failed to connect to any server"
    
    try:
        # List tables
        tables_result = await client.list_tables()
        logger.info(f"Tables: {tables_result}")
        
        # Get schema for a table
        if tables_result.get("success"):
            tables = tables_result.get("tables", [])
            if tables:
                first_table = tables[0]["table_name"]
                schema_result = await client.get_schema(first_table)
                logger.info(f"Schema for {first_table}: {schema_result}")
        
        # Execute a simple query
        query_result = await client.execute_query("SELECT current_timestamp")
        logger.info(f"Current timestamp: {query_result}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_end_to_end_workflow())