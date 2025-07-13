"""
Comprehensive test suite for retry logic, exponential backoff,
circuit breakers, and connection pooling.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.retry import (
    ExponentialBackoffRetry, FixedDelayRetry, NoRetry,
    CircuitBreaker, CircuitBreakerState, RetryManager,
    retry_async, RetryableError, NonRetryableError, CircuitBreakerOpenError
)
from src.core.connection_pool import ConnectionPool, MCPConnection
from src.core.tool_registry import ToolRegistry


class TestRetryPolicies:
    """Test different retry policy implementations."""
    
    def test_exponential_backoff_delays(self):
        """Test exponential backoff delay calculation."""
        policy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=1.0,
            max_delay=16.0,
            jitter_factor=0.0  # No jitter for predictable testing
        )
        
        # Test exponential growth
        assert policy.get_delay(0) == 1.0   # 1 * 2^0 = 1
        assert policy.get_delay(1) == 2.0   # 1 * 2^1 = 2
        assert policy.get_delay(2) == 4.0   # 1 * 2^2 = 4
        assert policy.get_delay(3) == 8.0   # 1 * 2^3 = 8
        assert policy.get_delay(4) == 16.0  # 1 * 2^4 = 16
        assert policy.get_delay(5) == 16.0  # Capped at max_delay
    
    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter."""
        policy = ExponentialBackoffRetry(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            jitter_factor=0.2
        )
        
        # Test jitter is within expected range
        for attempt in range(5):
            delay = policy.get_delay(attempt)
            base_delay = min(1.0 * (2 ** attempt), 10.0)
            min_delay = base_delay * 0.8  # -20% jitter
            max_delay = base_delay * 1.2  # +20% jitter
            assert min_delay <= delay <= max_delay
    
    def test_fixed_delay_retry(self):
        """Test fixed delay retry policy."""
        policy = FixedDelayRetry(max_attempts=3, delay=2.0)
        
        # All attempts should have same delay
        for attempt in range(5):
            assert policy.get_delay(attempt) == 2.0
    
    def test_no_retry_policy(self):
        """Test no retry policy."""
        policy = NoRetry()
        
        assert policy.max_attempts == 0
        assert policy.get_delay(0) == 0
        assert not policy.should_retry(Exception("test"), 0)
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        policy = ExponentialBackoffRetry(max_attempts=3)
        
        # Should retry normal errors
        assert policy.should_retry(Exception("test"), 0)
        assert policy.should_retry(Exception("test"), 1)
        assert policy.should_retry(Exception("test"), 2)
        
        # Should not retry after max attempts
        assert not policy.should_retry(Exception("test"), 3)
        
        # Should not retry non-retryable errors
        assert not policy.should_retry(NonRetryableError("test"), 0)
        
        # Should not retry circuit breaker open errors
        assert not policy.should_retry(CircuitBreakerOpenError("test"), 0)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_test_requests=2
        )
        
        # Initial state should be closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert not cb.is_open()
        
        # Record failures to open circuit
        for i in range(3):
            cb.record_failure(Exception("test"))
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_open()
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should transition to half-open
        assert not cb.is_open()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Record successes to close circuit
        cb.record_success()
        cb.record_success()
        
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_statistics(self):
        """Test circuit breaker statistics tracking."""
        cb = CircuitBreaker()
        
        # Record some activity
        cb.record_success()
        cb.record_success()
        cb.record_failure(Exception("test"))
        
        stats = cb.statistics
        assert stats['total_requests'] == 3
        assert stats['successful_requests'] == 2
        assert stats['failed_requests'] == 1
        assert stats['rejected_requests'] == 0
    
    def test_circuit_breaker_call_protection(self):
        """Test circuit breaker protecting function calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise Exception("Always fails")
        
        def working_function():
            return "success"
        
        # Test successful calls
        result = cb.call_with_circuit_breaker(working_function)
        assert result == "success"
        
        # Test failing calls
        with pytest.raises(Exception):
            cb.call_with_circuit_breaker(failing_function)
        
        with pytest.raises(Exception):
            cb.call_with_circuit_breaker(failing_function)
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerOpenError):
            cb.call_with_circuit_breaker(working_function)
        
        assert cb.statistics['rejected_requests'] == 1


class TestRetryAsync:
    """Test async retry decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_async_retry(self):
        """Test successful async function with retry."""
        call_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3))
        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Not yet")
            return "success"
        
        result = await eventually_succeeds()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_failed_async_retry(self):
        """Test async function that exhausts retries."""
        call_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3, base_delay=0.01))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Always fails")
        
        with pytest.raises(RetryableError):
            await always_fails()
        
        assert call_count == 4  # Initial + 3 retries
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test non-retryable errors are not retried."""
        call_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3))
        async def throws_non_retryable():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("Do not retry")
        
        with pytest.raises(NonRetryableError):
            await throws_non_retryable()
        
        assert call_count == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry decorator with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2)
        call_count = 0
        
        @retry_async(
            retry_policy=ExponentialBackoffRetry(max_attempts=5, base_delay=0.01),
            circuit_breaker=cb
        )
        async def protected_function():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Fail")
        
        # First call should fail and open circuit
        with pytest.raises(RetryableError):
            await protected_function()
        
        # Circuit should be open, preventing further calls
        with pytest.raises(CircuitBreakerOpenError):
            await protected_function()
        
        # Only 3 calls should have been made (initial + 2 to trigger circuit breaker)
        assert call_count <= 3


class TestRetryManager:
    """Test retry manager functionality."""
    
    def test_retry_manager_creation(self):
        """Test retry manager creates policies correctly."""
        config = {
            'retry_policy': {
                'type': 'exponential_backoff',
                'max_attempts': 5,
                'base_delay': 2.0
            }
        }
        
        manager = RetryManager(config)
        policy = manager.get_retry_policy('test_service')
        
        assert isinstance(policy, ExponentialBackoffRetry)
        assert policy.max_attempts == 5
        assert policy.base_delay == 2.0
    
    def test_retry_manager_circuit_breakers(self):
        """Test retry manager manages circuit breakers."""
        manager = RetryManager()
        
        # Get circuit breakers for different services
        cb1 = manager.get_circuit_breaker('service1')
        cb2 = manager.get_circuit_breaker('service2')
        
        assert cb1 is not cb2
        
        # Should return same instance for same service
        cb1_again = manager.get_circuit_breaker('service1')
        assert cb1 is cb1_again
    
    def test_retry_manager_statistics(self):
        """Test retry manager statistics collection."""
        manager = RetryManager()
        
        # Use some circuit breakers
        cb1 = manager.get_circuit_breaker('service1')
        cb1.record_failure(Exception("test"))
        
        cb2 = manager.get_circuit_breaker('service2')
        cb2.record_success()
        
        stats = manager.get_statistics()
        assert 'service1' in stats
        assert 'service2' in stats
        assert stats['service1']['statistics']['failed_requests'] == 1
        assert stats['service2']['statistics']['successful_requests'] == 1


class TestConnectionPool:
    """Test connection pool functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_basic(self):
        """Test basic connection pool operations."""
        pool = ConnectionPool({
            'max_connections': 5,
            'connection_timeout': 1.0
        })
        
        # Start pool
        await pool.start()
        
        try:
            # Mock client
            mock_client = AsyncMock()
            mock_client.is_connected = AsyncMock(return_value=True)
            
            async def create_connection():
                return mock_client
            
            # Acquire and release connection
            async with pool.acquire_connection('test_server', 'test_type', create_connection) as conn:
                assert isinstance(conn, MCPConnection)
                assert conn.client == mock_client
            
            # Check statistics
            stats = pool.get_statistics()
            assert stats['connections_created'] == 1
            assert stats['connections_reused'] == 0
            
            # Acquire again - should reuse
            async with pool.acquire_connection('test_server', 'test_type', create_connection) as conn:
                assert conn.client == mock_client
            
            stats = pool.get_statistics()
            assert stats['connections_created'] == 1
            assert stats['connections_reused'] == 1
            
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_connection_pool_limits(self):
        """Test connection pool respects limits."""
        pool = ConnectionPool({
            'max_connections': 2,
            'connection_timeout': 0.5
        })
        
        await pool.start()
        
        try:
            clients = []
            
            async def create_connection():
                client = AsyncMock()
                clients.append(client)
                return client
            
            # Acquire max connections
            conn1_ctx = pool.acquire_connection('server1', 'type1', create_connection)
            conn1 = await conn1_ctx.__aenter__()
            
            conn2_ctx = pool.acquire_connection('server2', 'type2', create_connection)
            conn2 = await conn2_ctx.__aenter__()
            
            # Try to acquire one more - should timeout
            with pytest.raises(TimeoutError):
                async with pool.acquire_connection('server3', 'type3', create_connection):
                    pass
            
            # Release one connection
            await conn1_ctx.__aexit__(None, None, None)
            
            # Now should be able to acquire
            async with pool.acquire_connection('server3', 'type3', create_connection):
                pass
            
            await conn2_ctx.__aexit__(None, None, None)
            
        finally:
            await pool.stop()


class TestToolRegistryFailureTracking:
    """Test Tool Registry failure tracking functionality."""
    
    def test_enhanced_usage_recording(self):
        """Test enhanced usage recording with retry count."""
        registry = ToolRegistry(":memory:")
        
        # Register a test tool
        tool_info = {
            'id': 'test.tool',
            'name': 'Test Tool',
            'server_type': 'test',
            'endpoint': 'test://localhost',
            'capabilities': {},
            'input_schema': {}
        }
        registry.register_tool(tool_info)
        
        # Record usage with retries
        registry.record_usage('test.tool', False, 2.5, retry_count=3)
        
        # Check failure metrics
        metrics = registry.get_failure_metrics('test.tool')
        assert metrics['failure_count'] == 1
        assert metrics['consecutive_failures'] == 1
    
    def test_circuit_breaker_state_updates(self):
        """Test circuit breaker state management in registry."""
        registry = ToolRegistry(":memory:")
        
        # Register a test tool
        tool_info = {
            'id': 'test.tool',
            'name': 'Test Tool',
            'server_type': 'test',
            'endpoint': 'test://localhost',
            'capabilities': {},
            'input_schema': {}
        }
        registry.register_tool(tool_info)
        
        # Update circuit breaker state
        registry.update_circuit_breaker_state('test.tool', 'open')
        
        # Check state was updated
        metrics = registry.get_failure_metrics('test.tool')
        assert metrics['circuit_breaker_state'] == 'open'
    
    def test_health_report_generation(self):
        """Test system health report generation."""
        registry = ToolRegistry(":memory:")
        
        # Register some test tools
        for i in range(3):
            tool_info = {
                'id': f'test.tool{i}',
                'name': f'Test Tool {i}',
                'server_type': 'test',
                'endpoint': f'test://localhost/{i}',
                'capabilities': {},
                'input_schema': {}
            }
            registry.register_tool(tool_info)
        
        # Create some usage history
        for _ in range(15):
            registry.record_usage('test.tool0', True, 0.1)
        
        for _ in range(10):
            registry.record_usage('test.tool1', False, 0.2)
        
        # Get health report
        report = registry.get_health_report()
        assert 'overall_stats' in report
        assert 'high_failure_tools' in report
        assert report['health_status'] in ['healthy', 'degraded']


def main():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()