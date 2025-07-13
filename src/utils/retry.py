"""
Retry utilities for handling transient failures with exponential backoff,
circuit breakers, and configurable retry policies.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should not trigger retry."""
    pass


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds for the given attempt number."""
        pass
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry based on error and attempt."""
        if attempt >= self.max_attempts:
            return False
        
        # Don't retry non-retryable errors
        if isinstance(error, NonRetryableError):
            return False
        
        # Don't retry if circuit breaker is open
        if isinstance(error, CircuitBreakerOpenError):
            return False
        
        return True


class ExponentialBackoffRetry(RetryPolicy):
    """Exponential backoff retry policy with jitter."""
    
    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.2
    ):
        super().__init__(max_attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        # Exponential delay: base * 2^attempt
        delay = self.base_delay * (2 ** attempt)
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Add jitter (±jitter_factor)
        jitter = delay * self.jitter_factor * (2 * random.random() - 1)
        delay = delay + jitter
        
        # Ensure non-negative delay
        return max(0, delay)


class FixedDelayRetry(RetryPolicy):
    """Fixed delay retry policy."""
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        super().__init__(max_attempts)
        self.delay = delay
    
    def get_delay(self, attempt: int) -> float:
        """Return fixed delay."""
        return self.delay


class NoRetry(RetryPolicy):
    """No retry policy - always fail immediately."""
    
    def __init__(self):
        super().__init__(max_attempts=0)
    
    def get_delay(self, attempt: int) -> float:
        """No delay needed as we don't retry."""
        return 0
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Never retry."""
        return False


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_test_requests: int = 3,
        error_types: Optional[List[Type[Exception]]] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_test_requests = half_open_test_requests
        self.error_types = error_types or [Exception]
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_request_count = 0
        self.statistics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rejected_requests': 0,
            'state_changes': []
        }
    
    def _should_count_error(self, error: Exception) -> bool:
        """Check if error should count towards circuit breaker."""
        return any(isinstance(error, error_type) for error_type in self.error_types)
    
    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state and log it."""
        old_state = self.state
        self.state = new_state
        self.statistics['state_changes'].append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self._change_state(CircuitBreakerState.HALF_OPEN)
                self.half_open_request_count = 0
                return False
            return True
        return False
    
    def record_success(self):
        """Record a successful request."""
        self.statistics['total_requests'] += 1
        self.statistics['successful_requests'] += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_request_count += 1
            if self.half_open_request_count >= self.half_open_test_requests:
                # Enough successful requests, close the circuit
                self._change_state(CircuitBreakerState.CLOSED)
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self, error: Exception):
        """Record a failed request."""
        self.statistics['total_requests'] += 1
        self.statistics['failed_requests'] += 1
        
        if not self._should_count_error(error):
            return
        
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._change_state(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery test, open again
            self._change_state(CircuitBreakerState.OPEN)
            self.failure_count = self.failure_threshold
    
    def call_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.is_open():
            self.statistics['rejected_requests'] += 1
            raise CircuitBreakerOpenError(f"Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise


def retry_async(
    retry_policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for async functions with retry logic and optional circuit breaker.
    
    Args:
        retry_policy: Retry policy to use (defaults to ExponentialBackoffRetry)
        circuit_breaker: Optional circuit breaker instance
        on_retry: Optional callback called before each retry
    """
    if retry_policy is None:
        retry_policy = ExponentialBackoffRetry()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(retry_policy.max_attempts + 1):
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        # For async functions, we need to await inside the circuit breaker
                        return await circuit_breaker.call_with_circuit_breaker(
                            func, *args, **kwargs
                        )
                    else:
                        return await func(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    
                    # Check if we should retry
                    if not retry_policy.should_retry(e, attempt):
                        logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                        raise
                    
                    if attempt < retry_policy.max_attempts:
                        delay = retry_policy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{retry_policy.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {str(e)}"
                        )
                        
                        # Call retry callback if provided
                        if on_retry:
                            on_retry(e, attempt)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({retry_policy.max_attempts}) exceeded for {func.__name__}"
                        )
            
            # All retries exhausted
            raise last_error
        
        return wrapper
    return decorator


class RetryManager:
    """Manages retry policies and circuit breakers for different services."""
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.default_config = default_config or {
            'retry_policy': {
                'type': 'exponential_backoff',
                'max_attempts': 5,
                'base_delay': 1.0,
                'max_delay': 16.0,
                'jitter_factor': 0.2
            },
            'circuit_breaker': {
                'failure_threshold': 5,
                'recovery_timeout': 30.0,
                'half_open_test_requests': 3
            }
        }
        
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def create_retry_policy(self, config: Dict[str, Any]) -> RetryPolicy:
        """Create retry policy from configuration."""
        policy_type = config.get('type', 'exponential_backoff')
        
        if policy_type == 'exponential_backoff':
            return ExponentialBackoffRetry(
                max_attempts=config.get('max_attempts', 5),
                base_delay=config.get('base_delay', 1.0),
                max_delay=config.get('max_delay', 60.0),
                jitter_factor=config.get('jitter_factor', 0.2)
            )
        elif policy_type == 'fixed_delay':
            return FixedDelayRetry(
                max_attempts=config.get('max_attempts', 3),
                delay=config.get('delay', 1.0)
            )
        elif policy_type == 'no_retry':
            return NoRetry()
        else:
            raise ValueError(f"Unknown retry policy type: {policy_type}")
    
    def get_retry_policy(self, service_id: str) -> RetryPolicy:
        """Get or create retry policy for a service."""
        if service_id not in self.retry_policies:
            config = self.default_config.get('retry_policy', {})
            self.retry_policies[service_id] = self.create_retry_policy(config)
        return self.retry_policies[service_id]
    
    def get_circuit_breaker(self, service_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_id not in self.circuit_breakers:
            config = self.default_config.get('circuit_breaker', {})
            self.circuit_breakers[service_id] = CircuitBreaker(
                failure_threshold=config.get('failure_threshold', 5),
                recovery_timeout=config.get('recovery_timeout', 30.0),
                half_open_test_requests=config.get('half_open_test_requests', 3)
            )
        return self.circuit_breakers[service_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers."""
        stats = {}
        for service_id, cb in self.circuit_breakers.items():
            stats[service_id] = {
                'state': cb.state.value,
                'statistics': cb.statistics
            }
        return stats