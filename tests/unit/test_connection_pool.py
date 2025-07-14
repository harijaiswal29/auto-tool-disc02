"""
Unit tests for Connection Pool module.

Tests connection pool functionality including:
- Connection lifecycle management
- Health checking
- Connection reuse
- Idle connection cleanup
- Statistics tracking
- Concurrent access handling
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time

from src.core.connection_pool import MCPConnection, ConnectionPool


class TestMCPConnection:
    """Test cases for MCPConnection class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock MCP client."""
        client = Mock()
        client.is_connected = AsyncMock(return_value=True)
        client.disconnect = AsyncMock()
        return client
    
    @pytest.fixture
    def connection(self, mock_client):
        """Create an MCPConnection instance."""
        return MCPConnection(
            server_id="test_server",
            client=mock_client,
            server_type="sqlite"
        )
    
    def test_initialization(self, connection):
        """Test MCPConnection initialization."""
        assert connection.server_id == "test_server"
        assert connection.server_type == "sqlite"
        assert connection.usage_count == 0
        assert connection.is_healthy is True
        assert connection.last_health_check is None
        assert isinstance(connection.created_at, datetime)
        assert isinstance(connection.last_used, datetime)
    
    @pytest.mark.asyncio
    async def test_acquire_release(self, connection):
        """Test acquiring and releasing a connection."""
        # Test acquire
        await connection.acquire()
        assert connection.usage_count == 1
        assert connection._lock.locked()
        
        # Test release
        connection.release()
        assert not connection._lock.locked()
        
        # Test multiple acquire/release cycles
        await connection.acquire()
        assert connection.usage_count == 2
        connection.release()
    
    def test_is_idle(self, connection):
        """Test idle detection."""
        # Fresh connection should not be idle
        assert not connection.is_idle(10.0)
        
        # Simulate old last_used time
        connection.last_used = datetime.now() - timedelta(seconds=20)
        assert connection.is_idle(10.0)
        assert not connection.is_idle(30.0)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, connection, mock_client):
        """Test successful health check."""
        result = await connection.health_check()
        
        assert result is True
        assert connection.is_healthy is True
        assert connection.last_health_check is not None
        mock_client.is_connected.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, connection, mock_client):
        """Test failed health check."""
        mock_client.is_connected = AsyncMock(return_value=False)
        
        result = await connection.health_check()
        
        assert result is False
        assert connection.is_healthy is False
        assert connection.last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, connection, mock_client):
        """Test health check with exception."""
        mock_client.is_connected = AsyncMock(side_effect=Exception("Connection error"))
        
        result = await connection.health_check()
        
        assert result is False
        assert connection.is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_no_method(self, connection):
        """Test health check when client has no is_connected method."""
        connection.client = Mock(spec=[])  # Client without is_connected
        
        result = await connection.health_check()
        
        assert result is True  # Assumes healthy if no check method
        assert connection.is_healthy is True


class TestConnectionPool:
    """Test cases for ConnectionPool class."""
    
    @pytest.fixture
    def pool_config(self):
        """Default pool configuration for testing."""
        return {
            'max_connections': 5,
            'connection_timeout': 1.0,
            'idle_timeout': 10.0,
            'health_check_interval': 2.0
        }
    
    @pytest.fixture
    async def pool(self, pool_config):
        """Create a ConnectionPool instance."""
        pool = ConnectionPool(pool_config)
        yield pool
        # Cleanup
        if pool._running:
            await pool.stop()
    
    @pytest.fixture
    def mock_create_func(self):
        """Mock function for creating connections."""
        client = Mock()
        client.is_connected = AsyncMock(return_value=True)
        client.disconnect = AsyncMock()
        return AsyncMock(return_value=client)
    
    def test_initialization(self, pool_config):
        """Test ConnectionPool initialization."""
        pool = ConnectionPool(pool_config)
        
        assert pool.max_connections == 5
        assert pool.connection_timeout == 1.0
        assert pool.idle_timeout == 10.0
        assert pool.health_check_interval == 2.0
        assert pool._running is False
        assert len(pool.connections) == 0
    
    def test_initialization_defaults(self):
        """Test ConnectionPool with default configuration."""
        pool = ConnectionPool()
        
        assert pool.max_connections == 10
        assert pool.connection_timeout == 5.0
        assert pool.idle_timeout == 300.0
        assert pool.health_check_interval == 60.0
    
    @pytest.mark.asyncio
    async def test_start_stop(self, pool):
        """Test starting and stopping the pool."""
        # Test start
        await pool.start()
        assert pool._running is True
        assert pool._health_check_task is not None
        assert pool._cleanup_task is not None
        
        # Test stop
        await pool.stop()
        assert pool._running is False
        assert len(pool.connections) == 0
    
    @pytest.mark.asyncio
    async def test_acquire_connection_create_new(self, pool, mock_create_func):
        """Test acquiring a connection that needs to be created."""
        await pool.start()
        
        async with pool.acquire_connection(
            "server1", "sqlite", mock_create_func
        ) as conn:
            assert conn is not None
            assert conn.server_id == "server1"
            assert conn.server_type == "sqlite"
            assert pool.stats['connections_created'] == 1
            assert pool.stats['connections_reused'] == 0
    
    @pytest.mark.asyncio
    async def test_acquire_connection_reuse(self, pool, mock_create_func):
        """Test reusing an existing connection."""
        await pool.start()
        
        # First acquisition creates connection
        async with pool.acquire_connection(
            "server1", "sqlite", mock_create_func
        ) as conn1:
            assert pool.stats['connections_created'] == 1
        
        # Second acquisition reuses connection
        async with pool.acquire_connection(
            "server1", "sqlite", mock_create_func
        ) as conn2:
            assert conn1.server_id == conn2.server_id
            assert pool.stats['connections_created'] == 1
            assert pool.stats['connections_reused'] == 1
    
    @pytest.mark.asyncio
    async def test_acquire_connection_max_limit(self, pool, mock_create_func):
        """Test connection limit enforcement."""
        pool.max_connections = 2
        await pool.start()
        
        # Create max connections
        tasks = []
        for i in range(2):
            async def acquire_and_hold(server_id):
                async with pool.acquire_connection(
                    server_id, "sqlite", mock_create_func
                ):
                    await asyncio.sleep(0.5)
            
            task = asyncio.create_task(acquire_and_hold(f"server{i}"))
            tasks.append(task)
        
        # Wait a bit for connections to be acquired
        await asyncio.sleep(0.1)
        
        # Try to acquire one more (should timeout)
        with pytest.raises(TimeoutError):
            async with pool.acquire_connection(
                "server3", "sqlite", mock_create_func
            ):
                pass
        
        assert pool.stats['failed_acquisitions'] == 1
        
        # Cleanup
        await asyncio.gather(*tasks)
    
    @pytest.mark.asyncio
    async def test_acquire_connection_wait_for_available(self, pool, mock_create_func):
        """Test waiting for a connection to become available."""
        pool.max_connections = 1
        pool.connection_timeout = 2.0
        await pool.start()
        
        # Hold a connection
        async def hold_connection():
            async with pool.acquire_connection(
                "server1", "sqlite", mock_create_func
            ):
                await asyncio.sleep(0.5)
        
        # Start holding connection
        hold_task = asyncio.create_task(hold_connection())
        await asyncio.sleep(0.1)  # Let first connection be acquired
        
        # Try to acquire another connection (should wait then succeed)
        start_time = time.time()
        async with pool.acquire_connection(
            "server2", "sqlite", mock_create_func
        ) as conn:
            elapsed = time.time() - start_time
            assert elapsed >= 0.4  # Had to wait for first to release
            assert conn is not None
        
        await hold_task
    
    @pytest.mark.asyncio
    async def test_health_check_loop(self, pool, mock_create_func):
        """Test health check background loop."""
        pool.health_check_interval = 0.1  # Fast for testing
        await pool.start()
        
        # Create a connection
        async with pool.acquire_connection(
            "server1", "sqlite", mock_create_func
        ) as conn:
            conn.client.is_connected = AsyncMock(return_value=True)
        
        # Wait for health check to run
        await asyncio.sleep(0.2)
        
        # Connection should still be healthy
        assert "server1" in pool.connections
        assert pool.connections["server1"].is_healthy
    
    @pytest.mark.asyncio
    async def test_health_check_removes_unhealthy(self, pool, mock_create_func):
        """Test that unhealthy connections are removed."""
        pool.health_check_interval = 0.1
        await pool.start()
        
        # Create a connection
        async with pool.acquire_connection(
            "server1", "sqlite", mock_create_func
        ) as conn:
            # Make it unhealthy for next check
            conn.client.is_connected = AsyncMock(return_value=False)
        
        # Wait for health check to run
        await asyncio.sleep(0.2)
        
        # Unhealthy connection should be removed
        assert "server1" not in pool.connections
        assert pool.stats['health_check_failures'] > 0
        assert pool.stats['connections_closed'] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_idle_connections(self, pool, mock_create_func):
        """Test cleanup of idle connections."""
        pool.idle_timeout = 0.1  # Very short for testing
        await pool.start()
        
        # Create a connection
        async with pool.acquire_connection(
            "server1", "sqlite", mock_create_func
        ):
            pass
        
        # Make connection appear idle
        pool.connections["server1"].last_used = datetime.now() - timedelta(seconds=1)
        
        # Manually trigger cleanup
        await pool._cleanup_loop.__wrapped__(pool)
        
        # Idle connection should be removed
        assert "server1" not in pool.connections
        assert pool.stats['connections_closed'] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self, pool, mock_create_func):
        """Test concurrent connection acquisitions."""
        await pool.start()
        
        # Create multiple concurrent acquisition tasks
        async def acquire_task(i):
            async with pool.acquire_connection(
                f"server{i}", "sqlite", mock_create_func
            ) as conn:
                await asyncio.sleep(0.1)
                return conn.server_id
        
        tasks = [acquire_task(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(f"server{i}" in results for i in range(3))
    
    def test_get_statistics(self, pool):
        """Test getting pool statistics."""
        # Add some test data
        pool.stats['connections_created'] = 5
        pool.stats['connections_reused'] = 10
        pool.connections['conn1'] = Mock(server_type='sqlite')
        pool.connections['conn2'] = Mock(server_type='postgres')
        pool.available_connections['sqlite'].add('conn1')
        pool.in_use_connections['postgres'].add('conn2')
        
        stats = pool.get_statistics()
        
        assert stats['total_connections'] == 2
        assert stats['available_connections'] == 1
        assert stats['in_use_connections'] == 1
        assert stats['connections_created'] == 5
        assert stats['connections_reused'] == 10
        assert stats['connections_by_type']['sqlite'] == 1
    
    @pytest.mark.asyncio
    async def test_close_connection_methods(self, pool):
        """Test different connection close methods."""
        # Test with disconnect method
        client1 = Mock()
        client1.disconnect = AsyncMock()
        conn1 = MCPConnection("conn1", client1, "sqlite")
        pool.connections["conn1"] = conn1
        
        await pool._close_connection("conn1")
        client1.disconnect.assert_called_once()
        
        # Test with close method
        client2 = Mock(spec=['close'])
        client2.close = AsyncMock()
        conn2 = MCPConnection("conn2", client2, "postgres")
        pool.connections["conn2"] = conn2
        
        await pool._close_connection("conn2")
        client2.close.assert_called_once()
        
        # Test with no close method
        client3 = Mock(spec=[])
        conn3 = MCPConnection("conn3", client3, "other")
        pool.connections["conn3"] = conn3
        
        # Should not raise error
        await pool._close_connection("conn3")
        
        assert len(pool.connections) == 0
        assert pool.stats['connections_closed'] == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_in_loops(self, pool):
        """Test error handling in background loops."""
        pool.health_check_interval = 0.1
        await pool.start()
        
        # Create a connection that will raise error during health check
        mock_client = Mock()
        mock_client.is_connected = AsyncMock(side_effect=Exception("Test error"))
        conn = MCPConnection("error_conn", mock_client, "sqlite")
        pool.connections["error_conn"] = conn
        pool.available_connections["sqlite"].add("error_conn")
        
        # Wait for health check to run
        await asyncio.sleep(0.2)
        
        # Pool should still be running despite error
        assert pool._running is True
        # Connection should be marked unhealthy
        assert not conn.is_healthy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])