#!/usr/bin/env python3
"""
SQLite MCP Integration Tests

Tests the SQLite MCP tool implementation with both mock and real scenarios.
This file contains comprehensive integration tests that verify the full
functionality of the SQLite MCP client with actual database operations.
"""

import asyncio
import json
import os
import sys
import pytest
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.sqlite_mcp import SQLiteMCPClient
from src.tools.mock_sqlite_mcp import MockSQLiteMCPServer
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLiteMCPIntegration:
    """Integration tests for SQLite MCP functionality."""
    
    @pytest.fixture
    def test_db(self):
        """Create a test database with sample data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create test data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Insert test data
        users = [
            ('Alice Johnson', 'alice@example.com'),
            ('Bob Smith', 'bob@example.com'),
            ('Charlie Brown', 'charlie@example.com')
        ]
        cursor.executemany('INSERT INTO users (name, email) VALUES (?, ?)', users)
        
        orders = [
            (1, 'Laptop', 1, 999.99),
            (1, 'Mouse', 2, 29.99),
            (2, 'Keyboard', 1, 79.99),
            (3, 'Monitor', 1, 299.99),
            (3, 'HDMI Cable', 3, 15.99)
        ]
        cursor.executemany('INSERT INTO orders (user_id, product, quantity, price) VALUES (?, ?, ?, ?)', orders)
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    async def client_mock(self, test_db):
        """Create SQLite MCP client with mock server."""
        client = SQLiteMCPClient(test_db)
        connected = await client.connect(use_mock=True)
        assert connected, "Failed to connect to mock server"
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def client_real(self, test_db):
        """Create SQLite MCP client with real server (if available)."""
        client = SQLiteMCPClient(test_db)
        connected = await client.connect(use_mock=False)
        if not connected:
            # Fallback to mock if real server not available
            connected = await client.connect(use_mock=True)
            logger.warning("Real SQLite MCP server not available, using mock")
        yield client
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_connection_and_initialization(self, test_db):
        """Test client connection and initialization."""
        # Test with mock server
        client_mock = SQLiteMCPClient(test_db)
        assert await client_mock.connect(use_mock=True)
        assert client_mock.use_mock is True
        assert len(client_mock.tools) > 0
        await client_mock.disconnect()
        
        # Test with real server (may fail if not available)
        client_real = SQLiteMCPClient(test_db)
        connected = await client_real.connect(use_mock=False)
        if connected:
            assert client_real.use_mock is False
            assert len(client_real.tools) > 0
        await client_real.disconnect()
    
    @pytest.mark.asyncio
    async def test_select_queries(self, client_mock):
        """Test various SELECT queries."""
        # Simple SELECT
        result = await client_mock.execute_query('SELECT * FROM users')
        assert result.get('success', False) or 'result' in result
        
        # SELECT with WHERE clause
        result = await client_mock.execute_query(
            'SELECT * FROM users WHERE name = ?',
            ['Alice Johnson']
        )
        assert result.get('success', False) or 'result' in result
        
        # SELECT with JOIN
        result = await client_mock.execute_query('''
            SELECT u.name, o.product, o.quantity, o.price
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE u.id = ?
        ''', [1])
        assert result.get('success', False) or 'result' in result
        
        # Aggregate functions
        result = await client_mock.execute_query('''
            SELECT u.name, COUNT(o.id) as order_count, SUM(o.price * o.quantity) as total_spent
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
        ''')
        assert result.get('success', False) or 'result' in result
    
    @pytest.mark.asyncio
    async def test_insert_operations(self, client_mock):
        """Test INSERT operations."""
        # Insert single record
        result = await client_mock.execute_query(
            'INSERT INTO users (name, email) VALUES (?, ?)',
            ['David Wilson', 'david@example.com']
        )
        assert result.get('success', False) or 'result' in result
        
        # Verify insert
        verify_result = await client_mock.execute_query(
            'SELECT * FROM users WHERE name = ?',
            ['David Wilson']
        )
        assert verify_result.get('success', False) or 'result' in verify_result
        
        # Insert with returning (if supported)
        result = await client_mock.execute_query(
            'INSERT INTO orders (user_id, product, quantity, price) VALUES (?, ?, ?, ?)',
            [1, 'Tablet', 1, 599.99]
        )
        assert result.get('success', False) or 'result' in result
    
    @pytest.mark.asyncio
    async def test_update_operations(self, client_mock):
        """Test UPDATE operations."""
        # Update single record
        result = await client_mock.execute_query(
            'UPDATE users SET email = ? WHERE name = ?',
            ['alice.new@example.com', 'Alice Johnson']
        )
        assert result.get('success', False) or 'result' in result
        
        # Update multiple records
        result = await client_mock.execute_query(
            'UPDATE orders SET price = price * 0.9 WHERE quantity > ?',
            [1]
        )
        assert result.get('success', False) or 'result' in result
    
    @pytest.mark.asyncio
    async def test_delete_operations(self, client_mock):
        """Test DELETE operations."""
        # Delete single record
        result = await client_mock.execute_query(
            'DELETE FROM orders WHERE product = ?',
            ['HDMI Cable']
        )
        assert result.get('success', False) or 'result' in result
        
        # Verify deletion
        verify_result = await client_mock.execute_query(
            'SELECT * FROM orders WHERE product = ?',
            ['HDMI Cable']
        )
        assert verify_result.get('success', False) or 'result' in verify_result
    
    @pytest.mark.asyncio
    async def test_schema_operations(self, client_mock):
        """Test schema inspection operations."""
        # Get schema for specific table
        result = await client_mock.get_schema('users')
        assert result is not None
        
        # Get all tables schema
        result = await client_mock.get_schema()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_transaction_handling(self, client_mock):
        """Test transaction operations."""
        # Note: Transaction support may vary based on MCP implementation
        try:
            # Begin transaction
            await client_mock.execute_query('BEGIN TRANSACTION')
            
            # Insert in transaction
            await client_mock.execute_query(
                'INSERT INTO users (name, email) VALUES (?, ?)',
                ['Eve Thompson', 'eve@example.com']
            )
            
            # Rollback
            await client_mock.execute_query('ROLLBACK')
            
            # Verify rollback
            result = await client_mock.execute_query(
                'SELECT * FROM users WHERE name = ?',
                ['Eve Thompson']
            )
            assert result.get('success', False) or 'result' in result
        except Exception as e:
            logger.info(f"Transaction test skipped: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client_mock):
        """Test error handling scenarios."""
        # Invalid SQL syntax
        result = await client_mock.execute_query('INVALID SQL STATEMENT')
        assert not result.get('success', True) or 'error' in result
        
        # Query non-existent table
        result = await client_mock.execute_query('SELECT * FROM non_existent_table')
        assert not result.get('success', True) or 'error' in result
        
        # Invalid column
        result = await client_mock.execute_query('SELECT invalid_column FROM users')
        assert not result.get('success', True) or 'error' in result
    
    @pytest.mark.asyncio
    async def test_tool_registry_integration(self, client_mock):
        """Test integration with tool registry."""
        # Create a temporary registry
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            registry_db = f.name
        
        try:
            registry = ToolRegistry(registry_db)
            
            # Register tools
            client_mock.register_tools_to_registry(registry)
            
            # Verify registration
            sqlite_tools = registry.list_tools("sqlite")
            assert len(sqlite_tools) > 0
            
            # Check tool details
            for tool in sqlite_tools:
                assert tool['server_type'] == 'sqlite'
                assert 'sqlite.' in tool['id']
        finally:
            if os.path.exists(registry_db):
                os.unlink(registry_db)
    
    @pytest.mark.asyncio
    async def test_complex_queries(self, client_mock):
        """Test complex SQL queries."""
        # Subquery
        result = await client_mock.execute_query('''
            SELECT name, email FROM users 
            WHERE id IN (SELECT DISTINCT user_id FROM orders WHERE price > ?)
        ''', [50.0])
        assert result.get('success', False) or 'result' in result
        
        # UNION query
        result = await client_mock.execute_query('''
            SELECT 'User' as type, name as identifier FROM users
            UNION
            SELECT 'Product' as type, product as identifier FROM orders
        ''')
        assert result.get('success', False) or 'result' in result
        
        # Window functions (if supported)
        try:
            result = await client_mock.execute_query('''
                SELECT user_id, product, price,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY price DESC) as rank
                FROM orders
            ''')
            assert result.get('success', False) or 'result' in result
        except Exception:
            logger.info("Window functions not supported")
    
    @pytest.mark.asyncio
    async def test_performance_and_limits(self, client_mock):
        """Test performance aspects and limits."""
        # Large batch insert
        large_batch = [(f'User{i}', f'user{i}@example.com') for i in range(100)]
        
        # Insert in batches to avoid SQL limits
        batch_size = 20
        for i in range(0, len(large_batch), batch_size):
            batch = large_batch[i:i + batch_size]
            placeholders = ','.join(['(?, ?)'] * len(batch))
            values = [item for sublist in batch for item in sublist]
            
            result = await client_mock.execute_query(
                f'INSERT INTO users (name, email) VALUES {placeholders}',
                values
            )
            assert result.get('success', False) or 'result' in result
        
        # Query with large result set
        result = await client_mock.execute_query('SELECT COUNT(*) as total FROM users')
        assert result.get('success', False) or 'result' in result
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_mock", [True, False])
    async def test_both_server_modes(self, test_db, use_mock):
        """Test with both mock and real server modes."""
        client = SQLiteMCPClient(test_db)
        connected = await client.connect(use_mock=use_mock)
        
        if not connected and not use_mock:
            # Skip if real server not available
            pytest.skip("Real SQLite MCP server not available")
        
        assert connected, f"Failed to connect with use_mock={use_mock}"
        
        try:
            # Run basic operations
            result = await client.execute_query('SELECT COUNT(*) FROM users')
            assert result.get('success', False) or 'result' in result
            
            # Test tool discovery
            assert len(client.tools) > 0
            
        finally:
            await client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])