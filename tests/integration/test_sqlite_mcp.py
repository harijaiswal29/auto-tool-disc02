#!/usr/bin/env python3
"""
SQLite MCP Integration Tests

Tests the SQLite MCP tool implementation with both mock and real scenarios.
"""

import asyncio
import json
import os
import sys
import pytest
from pathlib import Path
import sqlite3
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.sqlite_mcp import SQLiteMCPClient
from src.tools.mock_sqlite_mcp import MockSQLiteMCPServer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLiteMCPClient:
    """Test SQLite MCP functionality."""
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock SQLite MCP server."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        server = MockSQLiteMCPServer(db_path)
        await server.start()
        yield server
        await server.stop()
        os.unlink(db_path)
    
    @pytest.fixture
    def test_db(self):
        """Create a test database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create test data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert test data
        users = [
            ('Alice', 'alice@example.com'),
            ('Bob', 'bob@example.com'),
            ('Charlie', 'charlie@example.com')
        ]
        cursor.executemany('INSERT INTO users (name, email) VALUES (?, ?)', users)
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_sqlite_mcp_initialization(self, test_db):
        """Test SQLite MCP initialization."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        assert sqlite_mcp.db_path == Path(test_db).resolve()
        assert sqlite_mcp.server_name == "sqlite"
    
    @pytest.mark.asyncio
    async def test_execute_query(self, test_db):
        """Test executing a query."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        # Select query
        result = await sqlite_mcp.execute_query(
            'SELECT * FROM users WHERE name = ?',
            ['Alice']
        )
        
        assert result['success'] is True
        assert len(result['data']) == 1
        assert result['data'][0]['name'] == 'Alice'
        assert result['data'][0]['email'] == 'alice@example.com'
    
    @pytest.mark.asyncio
    async def test_execute_insert(self, test_db):
        """Test executing an insert."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        # Insert query
        result = await sqlite_mcp.execute({
            'action': 'execute',
            'query': 'INSERT INTO users (name, email) VALUES (?, ?)',
            'params': ['Dave', 'dave@example.com']
        })
        
        assert result['success'] is True
        assert result['affected_rows'] == 1
        
        # Verify insert
        verify_result = await sqlite_mcp.execute({
            'action': 'query',
            'query': 'SELECT * FROM users WHERE name = ?',
            'params': ['Dave']
        })
        
        assert len(verify_result['data']) == 1
        assert verify_result['data'][0]['name'] == 'Dave'
    
    @pytest.mark.asyncio
    async def test_execute_update(self, test_db):
        """Test executing an update."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        # Update query
        result = await sqlite_mcp.execute({
            'action': 'execute',
            'query': 'UPDATE users SET email = ? WHERE name = ?',
            'params': ['alice.new@example.com', 'Alice']
        })
        
        assert result['success'] is True
        assert result['affected_rows'] == 1
        
        # Verify update
        verify_result = await sqlite_mcp.execute({
            'action': 'query',
            'query': 'SELECT * FROM users WHERE name = ?',
            'params': ['Alice']
        })
        
        assert verify_result['data'][0]['email'] == 'alice.new@example.com'
    
    @pytest.mark.asyncio
    async def test_execute_delete(self, test_db):
        """Test executing a delete."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        # Delete query
        result = await sqlite_mcp.execute({
            'action': 'execute',
            'query': 'DELETE FROM users WHERE name = ?',
            'params': ['Charlie']
        })
        
        assert result['success'] is True
        assert result['affected_rows'] == 1
        
        # Verify delete
        verify_result = await sqlite_mcp.execute({
            'action': 'query',
            'query': 'SELECT * FROM users',
            'params': []
        })
        
        assert len(verify_result['data']) == 2
        assert not any(user['name'] == 'Charlie' for user in verify_result['data'])
    
    @pytest.mark.asyncio
    async def test_get_schema(self, test_db):
        """Test getting database schema."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        result = await sqlite_mcp.execute({
            'action': 'schema'
        })
        
        assert result['success'] is True
        assert 'tables' in result
        assert 'users' in result['tables']
        
        users_schema = result['tables']['users']
        assert any(col['name'] == 'id' for col in users_schema)
        assert any(col['name'] == 'name' for col in users_schema)
        assert any(col['name'] == 'email' for col in users_schema)
    
    @pytest.mark.asyncio
    async def test_invalid_query(self, test_db):
        """Test handling invalid query."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        # Invalid SQL
        result = await sqlite_mcp.execute({
            'action': 'query',
            'query': 'SELECT * FROM non_existent_table'
        })
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_transaction(self, test_db):
        """Test transaction handling."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        # Begin transaction
        await sqlite_mcp.execute({
            'action': 'execute',
            'query': 'BEGIN TRANSACTION'
        })
        
        # Insert in transaction
        await sqlite_mcp.execute({
            'action': 'execute',
            'query': 'INSERT INTO users (name, email) VALUES (?, ?)',
            'params': ['Eve', 'eve@example.com']
        })
        
        # Rollback
        await sqlite_mcp.execute({
            'action': 'execute',
            'query': 'ROLLBACK'
        })
        
        # Verify rollback
        result = await sqlite_mcp.execute({
            'action': 'query',
            'query': 'SELECT * FROM users WHERE name = ?',
            'params': ['Eve']
        })
        
        assert len(result['data']) == 0
    
    @pytest.mark.asyncio
    async def test_capabilities(self, test_db):
        """Test getting tool capabilities."""
        sqlite_mcp = SQLiteMCPClient(db_path=test_db)
        
        capabilities = sqlite_mcp.get_capabilities()
        
        assert capabilities['name'] == 'sqlite_mcp'
        assert 'operations' in capabilities
        
        operations = capabilities['operations']
        assert any(op['name'] == 'query' for op in operations)
        assert any(op['name'] == 'execute' for op in operations)
        assert any(op['name'] == 'schema' for op in operations)
    
    @pytest.mark.asyncio
    async def test_mock_server_interaction(self, mock_server):
        """Test interaction with mock SQLite MCP server."""
        # Get server info
        response = mock_server.handle_tool_list()
        
        assert response['result'] is not None
        assert 'tools' in response['result']
        
        tools = response['result']['tools']
        assert any(tool['name'] == 'execute_query' for tool in tools)
        assert any(tool['name'] == 'get_schema' for tool in tools)
        
        # Execute query through mock
        query_response = mock_server.handle_tool_call(
            'execute_query',
            {'query': 'SELECT * FROM test_table'}
        )
        
        assert query_response['result'] is not None
        assert 'rows' in query_response['result']
        assert len(query_response['result']['rows']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])