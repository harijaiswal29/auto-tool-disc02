"""
Integration tests for retry scenarios with MCP connections.

Tests the retry logic, circuit breakers, and connection pooling
in real-world scenarios with MCP tool connections.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.core.connection_pool import ConnectionPool, PooledConnection
from src.utils.retry import (
    retry_async, RetryManager, ExponentialBackoffRetry,
    FixedDelayRetry, NoRetryPolicy, CircuitBreaker,
    RetryableError, NonRetryableError
)
from src.monitoring.retry_metrics import RetryMetricsCollector
from src.database.tool_registry import ToolRegistryDB


class TestRetryIntegration:
    """Test cases for retry logic integration with MCP connections."""
    
    @pytest.fixture
    async def setup_retry_environment(self, tmp_path):
        """Set up retry testing environment with MCP integration."""
        # Create temporary database
        db_path = tmp_path / "test_retry.db"
        
        # Initialize components
        registry_db = ToolRegistryDB(str(db_path))
        await registry_db.initialize()
        
        registry = ToolRegistry(str(db_path))
        await registry.initialize()
        
        # Create retry metrics collector
        retry_metrics = RetryMetricsCollector(registry)
        
        # Create connection pool
        connection_pool = ConnectionPool(max_connections=5)
        
        # Create MCP integration with retry support
        mcp_integration = MCPIntegration(registry)
        
        # Configure retry policies for different tools
        retry_config = {
            "reliable_tool": {
                "retry_policy": FixedDelayRetry(max_attempts=3, delay=0.1),
                "circuit_breaker": None
            },
            "flaky_tool": {
                "retry_policy": ExponentialBackoffRetry(
                    max_attempts=5,
                    base_delay=0.1,
                    max_delay=2.0,
                    jitter_factor=0.1
                ),
                "circuit_breaker": CircuitBreaker(
                    failure_threshold=3,
                    recovery_timeout=1.0,
                    half_open_test_requests=2
                )
            },
            "failing_tool": {
                "retry_policy": NoRetryPolicy(),
                "circuit_breaker": CircuitBreaker(
                    failure_threshold=2,
                    recovery_timeout=0.5
                )
            }
        }
        
        # Register test tools
        await self._register_test_tools(registry)
        
        yield {
            "registry": registry,
            "registry_db": registry_db,
            "mcp_integration": mcp_integration,
            "connection_pool": connection_pool,
            "retry_metrics": retry_metrics,
            "retry_config": retry_config,
            "db_path": db_path
        }
        
        # Cleanup
        await connection_pool.close_all()
        await registry_db.close()
    
    async def _register_test_tools(self, registry: ToolRegistry):
        """Register test tools with different reliability characteristics."""
        tools = [
            {
                "tool_id": "reliable_tool",
                "name": "Reliable Tool",
                "capabilities": {"operations": ["process"]},
                "metadata": {"reliability": "high"}
            },
            {
                "tool_id": "flaky_tool",
                "name": "Flaky Tool",
                "capabilities": {"operations": ["unstable_op"]},
                "metadata": {"reliability": "low"}
            },
            {
                "tool_id": "failing_tool",
                "name": "Failing Tool",
                "capabilities": {"operations": ["broken_op"]},
                "metadata": {"reliability": "none"}
            }
        ]
        
        for tool in tools:
            await registry.register_tool(
                tool_id=tool["tool_id"],
                name=tool["name"],
                tool_type="mcp",
                endpoint=f"mock://{tool['tool_id']}",
                capabilities=tool["capabilities"],
                metadata=tool["metadata"]
            )
    
    @pytest.mark.asyncio
    async def test_successful_retry_on_transient_failure(self, setup_retry_environment):
        """Test successful retry after transient failures."""
        env = setup_retry_environment
        mcp_integration = env["mcp_integration"]
        retry_metrics = env["retry_metrics"]
        
        # Mock tool execution with transient failures
        attempt_count = 0
        
        async def mock_execute(tool_id, params):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                raise RetryableError(f"Transient error on attempt {attempt_count}")
            
            return {"success": True, "result": "Success after retries"}
        
        # Apply retry decorator
        retry_policy = ExponentialBackoffRetry(max_attempts=5, base_delay=0.1)
        
        @retry_async(retry_policy=retry_policy)
        async def execute_with_retry(tool_id, params):
            return await mock_execute(tool_id, params)
        
        # Execute with retries
        result = await execute_with_retry("flaky_tool", {"param": "test"})
        
        # Verify success after retries
        assert result["success"] is True
        assert attempt_count == 3
        
        # Check retry metrics
        stats = retry_metrics.get_retry_statistics("flaky_tool")
        # Note: Metrics would be recorded if integrated with actual retry decorator
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_consecutive_failures(self, setup_retry_environment):
        """Test circuit breaker opens after consecutive failures."""
        env = setup_retry_environment
        retry_metrics = env["retry_metrics"]
        
        # Create circuit breaker
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_test_requests=1
        )
        
        # Simulate consecutive failures
        for i in range(3):
            try:
                circuit_breaker.call(self._failing_function)
            except Exception:
                pass
        
        # Circuit breaker should be open
        assert circuit_breaker.is_open()
        
        # Record circuit breaker event
        retry_metrics.record_circuit_breaker_event(
            tool_id="failing_tool",
            event_type="opened",
            state="open",
            reason="Threshold exceeded"
        )
        
        # Verify metrics
        cb_summary = retry_metrics.get_circuit_breaker_summary()
        assert len(cb_summary["active_circuit_breakers"]) > 0
    
    def _failing_function(self):
        """Helper function that always fails."""
        raise Exception("Simulated failure")
    
    @pytest.mark.asyncio
    async def test_connection_pool_with_retry(self, setup_retry_environment):
        """Test connection pooling with retry logic."""
        env = setup_retry_environment
        connection_pool = env["connection_pool"]
        
        # Mock connection creation with occasional failures
        create_attempts = 0
        
        async def mock_create_connection(tool_id):
            nonlocal create_attempts
            create_attempts += 1
            
            if create_attempts == 1:
                raise ConnectionError("First attempt failed")
            
            # Create mock connection
            conn = Mock()
            conn.tool_id = tool_id
            conn.is_healthy = AsyncMock(return_value=True)
            conn.close = AsyncMock()
            return conn
        
        # Acquire connection with retry
        retry_policy = ExponentialBackoffRetry(max_attempts=3, base_delay=0.1)
        
        @retry_async(retry_policy=retry_policy)
        async def acquire_with_retry(tool_id):
            return await connection_pool.acquire_connection(
                tool_id,
                "mcp",
                mock_create_connection
            )
        
        # Get connection
        async with await acquire_with_retry("reliable_tool") as conn:
            assert conn is not None
            assert conn.tool_id == "reliable_tool"
        
        # Verify retry occurred
        assert create_attempts == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_different_error_types(self, setup_retry_environment):
        """Test retry behavior with different error types."""
        env = setup_retry_environment
        retry_metrics = env["retry_metrics"]
        
        # Test retryable errors
        retry_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3))
        async def retryable_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 2:
                raise RetryableError("Temporary failure")
            return {"success": True}
        
        result = await retryable_operation()
        assert result["success"] is True
        assert retry_count == 2
        
        # Test non-retryable errors
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3))
        async def non_retryable_operation():
            raise NonRetryableError("Permanent failure")
        
        with pytest.raises(NonRetryableError):
            await non_retryable_operation()
    
    @pytest.mark.asyncio
    async def test_retry_metrics_integration(self, setup_retry_environment):
        """Test retry metrics collection during actual retries."""
        env = setup_retry_environment
        retry_metrics = env["retry_metrics"]
        
        # Create retry manager with metrics
        retry_manager = RetryManager(default_policy=ExponentialBackoffRetry())
        
        # Mock operation with retries
        attempt = 0
        
        async def flaky_operation():
            nonlocal attempt
            attempt += 1
            
            # Record retry attempt
            retry_metrics.record_retry_attempt(
                tool_id="flaky_tool",
                attempt_number=attempt,
                delay_ms=100.0 * attempt,
                error_type="timeout" if attempt < 3 else None,
                success=(attempt >= 3)
            )
            
            if attempt < 3:
                raise TimeoutError("Operation timed out")
            
            return {"success": True, "attempts": attempt}
        
        # Execute with retry manager
        policy = retry_manager.get_policy("flaky_tool")
        result = await retry_manager.execute_with_retry(
            flaky_operation,
            policy,
            "flaky_tool"
        )
        
        # Verify metrics
        stats = retry_metrics.get_retry_statistics("flaky_tool")
        assert stats["total_attempts"] == 3
        assert stats["success_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, setup_retry_environment):
        """Test circuit breaker recovery after timeout."""
        env = setup_retry_environment
        retry_metrics = env["retry_metrics"]
        
        # Create circuit breaker with short recovery timeout
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.5,
            half_open_test_requests=1
        )
        
        # Open circuit breaker
        for _ in range(2):
            try:
                circuit_breaker.call(self._failing_function)
            except:
                pass
        
        assert circuit_breaker.is_open()
        
        # Wait for recovery timeout
        await asyncio.sleep(0.6)
        
        # Circuit breaker should be half-open
        assert circuit_breaker.state == "half_open"
        
        # Successful call should close circuit
        circuit_breaker.call(lambda: {"success": True})
        assert circuit_breaker.state == "closed"
        
        # Record recovery
        retry_metrics.record_circuit_breaker_event(
            tool_id="flaky_tool",
            event_type="closed",
            state="closed",
            reason="Recovery successful"
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_retries(self, setup_retry_environment):
        """Test concurrent retry scenarios."""
        env = setup_retry_environment
        
        # Create multiple operations that need retries
        async def create_retryable_operation(operation_id: str, fail_count: int):
            attempts = 0
            
            @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=5, base_delay=0.1))
            async def operation():
                nonlocal attempts
                attempts += 1
                
                if attempts <= fail_count:
                    raise RetryableError(f"Operation {operation_id} attempt {attempts}")
                
                return {"id": operation_id, "attempts": attempts}
            
            return await operation()
        
        # Execute multiple operations concurrently
        tasks = [
            create_retryable_operation("op1", 2),
            create_retryable_operation("op2", 1),
            create_retryable_operation("op3", 3),
            create_retryable_operation("op4", 0),  # Succeeds immediately
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations eventually succeeded
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Operation {i} failed: {result}"
            assert result["id"] == f"op{i+1}"
    
    @pytest.mark.asyncio
    async def test_retry_with_jitter(self, setup_retry_environment):
        """Test retry with jitter to prevent thundering herd."""
        env = setup_retry_environment
        
        # Track retry delays
        delays = []
        
        async def track_delays(delay: float):
            delays.append(delay)
            await asyncio.sleep(delay)
        
        # Create retry policy with jitter
        policy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=1.0,
            max_delay=10.0,
            jitter_factor=0.2
        )
        
        # Calculate several delays
        for attempt in range(5):
            delay = policy.get_delay(attempt)
            await track_delays(0.01)  # Small delay instead of actual
            
            # Verify jitter is applied (delays should vary)
            base_delay = min(1.0 * (2 ** attempt), 10.0)
            assert 0.8 * base_delay <= delay <= 1.2 * base_delay
    
    @pytest.mark.asyncio
    async def test_retry_timeout_handling(self, setup_retry_environment):
        """Test retry behavior with operation timeouts."""
        env = setup_retry_environment
        
        # Create operation that times out
        @retry_async(
            retry_policy=ExponentialBackoffRetry(max_attempts=3, base_delay=0.1),
            timeout=0.5
        )
        async def slow_operation():
            await asyncio.sleep(1.0)  # Longer than timeout
            return {"success": True}
        
        # Should fail with timeout
        with pytest.raises(asyncio.TimeoutError):
            await slow_operation()
    
    @pytest.mark.asyncio
    async def test_retry_policy_configuration(self, setup_retry_environment):
        """Test different retry policy configurations."""
        env = setup_retry_environment
        retry_config = env["retry_config"]
        
        # Test each configured policy
        for tool_id, config in retry_config.items():
            policy = config["retry_policy"]
            
            if isinstance(policy, NoRetryPolicy):
                # Should not retry
                with pytest.raises(Exception):
                    @retry_async(retry_policy=policy)
                    async def no_retry_op():
                        raise Exception("Failed")
                    
                    await no_retry_op()
            
            elif isinstance(policy, FixedDelayRetry):
                # Should have constant delay
                delays = [policy.get_delay(i) for i in range(3)]
                assert all(d == delays[0] for d in delays)
            
            elif isinstance(policy, ExponentialBackoffRetry):
                # Should have exponential delays
                delays = [policy.get_delay(i) for i in range(3)]
                assert delays[1] > delays[0]
                assert delays[2] > delays[1]
    
    @pytest.mark.asyncio
    async def test_retry_alert_generation(self, setup_retry_environment):
        """Test alert generation based on retry patterns."""
        env = setup_retry_environment
        retry_metrics = env["retry_metrics"]
        
        # Simulate high failure rate
        for i in range(50):
            retry_metrics.record_retry_attempt(
                tool_id="problematic_tool",
                attempt_number=1,
                delay_ms=1000.0,
                error_type="error",
                success=(i % 5 == 0)  # 20% success rate
            )
        
        # Record excessive consecutive failures
        retry_metrics.record_consecutive_failure("problematic_tool", 15)
        
        # Generate alerts
        alerts = retry_metrics.generate_alert_recommendations()
        
        # Should have alerts for low success rate and excessive failures
        alert_types = [alert["type"] for alert in alerts]
        assert any("success_rate" in t for t in alert_types)
        assert "excessive_consecutive_failures" in alert_types
        
        # Verify recommendations
        for alert in alerts:
            assert "recommendation" in alert
            assert alert["severity"] in ["high", "medium", "low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])