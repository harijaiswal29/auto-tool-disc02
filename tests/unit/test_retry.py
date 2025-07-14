"""
Unit tests for Retry utilities.

Tests retry logic functionality including:
- Retry policies (exponential backoff, fixed delay, no retry)
- Circuit breaker implementation
- Retry decorator for async functions
- Retry manager
- Error handling and classification
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.utils.retry import (
    RetryPolicy, ExponentialBackoffRetry, FixedDelayRetry, NoRetry,
    CircuitBreaker, CircuitBreakerState, CircuitBreakerOpenError,
    RetryableError, NonRetryableError,
    retry_async, RetryManager
)


class TestRetryPolicies:
    """Test cases for retry policy implementations."""
    
    def test_exponential_backoff_initialization(self):
        """Test ExponentialBackoffRetry initialization."""
        policy = ExponentialBackoffRetry(
            max_attempts=3,
            base_delay=2.0,
            max_delay=10.0,
            jitter_factor=0.1
        )
        
        assert policy.max_attempts == 3
        assert policy.base_delay == 2.0
        assert policy.max_delay == 10.0
        assert policy.jitter_factor == 0.1
    
    def test_exponential_backoff_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        policy = ExponentialBackoffRetry(
            base_delay=1.0,
            max_delay=16.0,
            jitter_factor=0.0  # No jitter for predictable testing
        )
        
        # Test exponential growth
        assert policy.get_delay(0) == 1.0   # 1 * 2^0 = 1
        assert policy.get_delay(1) == 2.0   # 1 * 2^1 = 2
        assert policy.get_delay(2) == 4.0   # 1 * 2^2 = 4
        assert policy.get_delay(3) == 8.0   # 1 * 2^3 = 8
        assert policy.get_delay(4) == 16.0  # 1 * 2^4 = 16 (capped at max)
        assert policy.get_delay(5) == 16.0  # Still capped at max
    
    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter."""
        policy = ExponentialBackoffRetry(
            base_delay=1.0,
            jitter_factor=0.2
        )
        
        # Get multiple delays for same attempt to test jitter
        delays = [policy.get_delay(2) for _ in range(10)]
        
        # Base delay for attempt 2 should be 4.0
        # With 0.2 jitter factor, range should be [3.2, 4.8]
        assert all(3.2 <= d <= 4.8 for d in delays)
        # Delays should vary due to jitter
        assert len(set(delays)) > 1
    
    def test_fixed_delay_policy(self):
        """Test FixedDelayRetry policy."""
        policy = FixedDelayRetry(max_attempts=5, delay=2.5)
        
        assert policy.max_attempts == 5
        assert policy.delay == 2.5
        
        # Delay should be constant regardless of attempt
        assert policy.get_delay(0) == 2.5
        assert policy.get_delay(1) == 2.5
        assert policy.get_delay(10) == 2.5
    
    def test_no_retry_policy(self):
        """Test NoRetry policy."""
        policy = NoRetry()
        
        assert policy.max_attempts == 0
        assert policy.get_delay(0) == 0
        assert policy.should_retry(Exception("test"), 0) is False
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        policy = ExponentialBackoffRetry(max_attempts=3)
        
        # Should retry regular exceptions
        assert policy.should_retry(Exception("test"), 0) is True
        assert policy.should_retry(Exception("test"), 1) is True
        assert policy.should_retry(Exception("test"), 2) is True
        assert policy.should_retry(Exception("test"), 3) is False  # Max attempts reached
        
        # Should not retry NonRetryableError
        assert policy.should_retry(NonRetryableError("test"), 0) is False
        
        # Should not retry CircuitBreakerOpenError
        assert policy.should_retry(CircuitBreakerOpenError("test"), 0) is False


class TestCircuitBreaker:
    """Test cases for CircuitBreaker implementation."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker instance."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_test_requests=2
        )
    
    def test_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 5.0
        assert circuit_breaker.half_open_test_requests == 2
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    def test_is_open_when_closed(self, circuit_breaker):
        """Test is_open returns False when circuit is closed."""
        assert circuit_breaker.is_open() is False
    
    def test_record_success_in_closed_state(self, circuit_breaker):
        """Test recording success in closed state."""
        circuit_breaker.failure_count = 2  # Some previous failures
        circuit_breaker.record_success()
        
        assert circuit_breaker.failure_count == 0  # Reset on success
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.statistics['successful_requests'] == 1
    
    def test_record_failure_opens_circuit(self, circuit_breaker):
        """Test that circuit opens after threshold failures."""
        # Record failures up to threshold
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(Exception("test"))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.failure_threshold
        assert circuit_breaker.is_open() is True
    
    def test_recovery_timeout_transitions_to_half_open(self, circuit_breaker):
        """Test that circuit transitions to half-open after recovery timeout."""
        # Open the circuit
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(Exception("test"))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Simulate passage of time
        circuit_breaker.last_failure_time = time.time() - circuit_breaker.recovery_timeout - 1
        
        # Check if open should now return False and state changes
        assert circuit_breaker.is_open() is False
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
    
    def test_half_open_to_closed_transition(self, circuit_breaker):
        """Test successful recovery from half-open to closed."""
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        
        # Record successful requests
        for i in range(circuit_breaker.half_open_test_requests):
            circuit_breaker.record_success()
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test failure in half-open state reopens circuit."""
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        
        circuit_breaker.record_failure(Exception("test"))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.failure_threshold
    
    def test_call_with_circuit_breaker_success(self, circuit_breaker):
        """Test calling function through circuit breaker successfully."""
        mock_func = Mock(return_value="success")
        
        result = circuit_breaker.call_with_circuit_breaker(mock_func, 1, 2, key="value")
        
        assert result == "success"
        mock_func.assert_called_once_with(1, 2, key="value")
        assert circuit_breaker.statistics['successful_requests'] == 1
    
    def test_call_with_circuit_breaker_failure(self, circuit_breaker):
        """Test calling function through circuit breaker with failure."""
        mock_func = Mock(side_effect=Exception("test error"))
        
        with pytest.raises(Exception) as exc_info:
            circuit_breaker.call_with_circuit_breaker(mock_func)
        
        assert str(exc_info.value) == "test error"
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.statistics['failed_requests'] == 1
    
    def test_call_with_open_circuit_breaker(self, circuit_breaker):
        """Test calling function when circuit is open."""
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.last_failure_time = time.time()  # Recent failure
        
        mock_func = Mock()
        
        with pytest.raises(CircuitBreakerOpenError):
            circuit_breaker.call_with_circuit_breaker(mock_func)
        
        mock_func.assert_not_called()
        assert circuit_breaker.statistics['rejected_requests'] == 1
    
    def test_custom_error_types(self):
        """Test circuit breaker with custom error types."""
        class CustomError(Exception):
            pass
        
        cb = CircuitBreaker(
            failure_threshold=2,
            error_types=[CustomError]
        )
        
        # Regular exceptions shouldn't count
        cb.record_failure(Exception("regular"))
        assert cb.failure_count == 0
        
        # Custom errors should count
        cb.record_failure(CustomError("custom"))
        assert cb.failure_count == 1
    
    def test_statistics_tracking(self, circuit_breaker):
        """Test statistics tracking."""
        # Record some activity
        circuit_breaker.record_success()
        circuit_breaker.record_failure(Exception("test"))
        
        stats = circuit_breaker.statistics
        assert stats['total_requests'] == 2
        assert stats['successful_requests'] == 1
        assert stats['failed_requests'] == 1
        assert stats['rejected_requests'] == 0


class TestRetryDecorator:
    """Test cases for retry_async decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self):
        """Test successful execution without retry."""
        call_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3))
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await success_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failures."""
        call_count = 0
        
        @retry_async(
            retry_policy=ExponentialBackoffRetry(
                max_attempts=3,
                base_delay=0.01  # Short delay for testing
            )
        )
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return "success"
        
        result = await failing_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        call_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=2))
        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent error")
        
        with pytest.raises(Exception) as exc_info:
            await always_failing_func()
        
        assert str(exc_info.value) == "Persistent error"
        assert call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        call_count = 0
        
        @retry_async(retry_policy=ExponentialBackoffRetry(max_attempts=3))
        async def non_retryable_func():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("Should not retry")
        
        with pytest.raises(NonRetryableError):
            await non_retryable_func()
        
        assert call_count == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_retry_with_callback(self):
        """Test retry with on_retry callback."""
        retry_attempts = []
        
        def on_retry_callback(error, attempt):
            retry_attempts.append((str(error), attempt))
        
        @retry_async(
            retry_policy=ExponentialBackoffRetry(max_attempts=2, base_delay=0.01),
            on_retry=on_retry_callback
        )
        async def failing_func():
            raise Exception("Test error")
        
        with pytest.raises(Exception):
            await failing_func()
        
        assert len(retry_attempts) == 2
        assert retry_attempts[0] == ("Test error", 0)
        assert retry_attempts[1] == ("Test error", 1)
    
    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry decorator with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2)
        call_count = 0
        
        @retry_async(
            retry_policy=ExponentialBackoffRetry(max_attempts=5),
            circuit_breaker=cb
        )
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Test error")
        
        # First call should fail and increment circuit breaker
        with pytest.raises(Exception):
            await failing_func()
        
        # Circuit should be open after threshold failures
        assert cb.failure_count == 1
        
        # Continue failing until circuit opens
        with pytest.raises(Exception):
            await failing_func()
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Next call should fail immediately with CircuitBreakerOpenError
        with pytest.raises(Exception):
            await failing_func()


class TestRetryManager:
    """Test cases for RetryManager."""
    
    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager instance."""
        config = {
            'retry_policy': {
                'type': 'exponential_backoff',
                'max_attempts': 3,
                'base_delay': 1.0
            },
            'circuit_breaker': {
                'failure_threshold': 5,
                'recovery_timeout': 30.0
            }
        }
        return RetryManager(config)
    
    def test_initialization(self, retry_manager):
        """Test RetryManager initialization."""
        assert retry_manager.default_config is not None
        assert len(retry_manager.retry_policies) == 0
        assert len(retry_manager.circuit_breakers) == 0
    
    def test_create_exponential_backoff_policy(self, retry_manager):
        """Test creating exponential backoff policy."""
        config = {
            'type': 'exponential_backoff',
            'max_attempts': 5,
            'base_delay': 2.0,
            'max_delay': 32.0,
            'jitter_factor': 0.1
        }
        
        policy = retry_manager.create_retry_policy(config)
        
        assert isinstance(policy, ExponentialBackoffRetry)
        assert policy.max_attempts == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 32.0
        assert policy.jitter_factor == 0.1
    
    def test_create_fixed_delay_policy(self, retry_manager):
        """Test creating fixed delay policy."""
        config = {
            'type': 'fixed_delay',
            'max_attempts': 3,
            'delay': 1.5
        }
        
        policy = retry_manager.create_retry_policy(config)
        
        assert isinstance(policy, FixedDelayRetry)
        assert policy.max_attempts == 3
        assert policy.delay == 1.5
    
    def test_create_no_retry_policy(self, retry_manager):
        """Test creating no retry policy."""
        config = {'type': 'no_retry'}
        
        policy = retry_manager.create_retry_policy(config)
        
        assert isinstance(policy, NoRetry)
    
    def test_create_unknown_policy_type(self, retry_manager):
        """Test creating policy with unknown type."""
        config = {'type': 'unknown_type'}
        
        with pytest.raises(ValueError) as exc_info:
            retry_manager.create_retry_policy(config)
        
        assert "Unknown retry policy type" in str(exc_info.value)
    
    def test_get_retry_policy_creates_default(self, retry_manager):
        """Test getting retry policy creates default if not exists."""
        policy = retry_manager.get_retry_policy('service1')
        
        assert isinstance(policy, ExponentialBackoffRetry)
        assert 'service1' in retry_manager.retry_policies
        
        # Getting again should return same instance
        policy2 = retry_manager.get_retry_policy('service1')
        assert policy is policy2
    
    def test_get_circuit_breaker_creates_default(self, retry_manager):
        """Test getting circuit breaker creates default if not exists."""
        cb = retry_manager.get_circuit_breaker('service1')
        
        assert isinstance(cb, CircuitBreaker)
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert 'service1' in retry_manager.circuit_breakers
        
        # Getting again should return same instance
        cb2 = retry_manager.get_circuit_breaker('service1')
        assert cb is cb2
    
    def test_get_statistics(self, retry_manager):
        """Test getting statistics from all circuit breakers."""
        # Create some circuit breakers
        cb1 = retry_manager.get_circuit_breaker('service1')
        cb2 = retry_manager.get_circuit_breaker('service2')
        
        # Record some activity
        cb1.record_success()
        cb2.record_failure(Exception("test"))
        
        stats = retry_manager.get_statistics()
        
        assert 'service1' in stats
        assert 'service2' in stats
        assert stats['service1']['state'] == 'closed'
        assert stats['service1']['statistics']['successful_requests'] == 1
        assert stats['service2']['statistics']['failed_requests'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])