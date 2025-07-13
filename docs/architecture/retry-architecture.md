# Retry and Resilience Architecture

## Overview

The Auto Tool Discovery system implements a comprehensive retry and resilience layer to handle transient failures, network issues, and service disruptions. This architecture ensures high availability and graceful degradation through exponential backoff, circuit breakers, and intelligent retry policies.

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Retry & Resilience Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │   Retry     │ │   Circuit    │ │    Connection       │   │
│  │  Policies   │ │   Breakers   │ │      Pool           │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │   Retry     │ │   Failure    │ │      Retry          │   │
│  │  Manager    │ │   Recovery   │ │     Metrics         │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Retry Policies

### 1. Exponential Backoff Retry

The default retry policy implements exponential backoff with jitter to prevent thundering herd problems.

```python
class ExponentialBackoffRetry(RetryPolicy):
    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.2
    )
```

**Delay Calculation:**
- Base formula: `delay = base_delay * (2 ^ attempt)`
- With jitter: `delay = delay ± (delay * jitter_factor)`
- Capped at `max_delay`

**Default Configuration:**
- Max attempts: 5
- Base delay: 1s
- Max delay: 16s
- Jitter: ±20%
- Sequence: ~1s, ~2s, ~4s, ~8s, ~16s (with jitter)

### 2. Fixed Delay Retry

Simple retry policy with constant delay between attempts.

```python
class FixedDelayRetry(RetryPolicy):
    def __init__(self, max_attempts: int = 3, delay: float = 1.0)
```

**Use Cases:**
- Rate-limited APIs
- Services with predictable recovery times
- Testing scenarios

### 3. No Retry Policy

Fail immediately without retry attempts.

```python
class NoRetry(RetryPolicy):
    def __init__(self)
```

**Use Cases:**
- Non-idempotent operations
- User-facing operations requiring immediate feedback
- Critical path operations

## Circuit Breaker Pattern

### States

```
CLOSED → OPEN → HALF_OPEN → CLOSED
   ↑                ↓
   └────────────────┘
```

1. **CLOSED** (Normal Operation)
   - All requests pass through
   - Failures are counted
   - Transitions to OPEN after threshold

2. **OPEN** (Failing Fast)
   - All requests rejected immediately
   - No load on failing service
   - Transitions to HALF_OPEN after recovery timeout

3. **HALF_OPEN** (Testing Recovery)
   - Limited requests allowed through
   - Success → CLOSED
   - Failure → OPEN

### Configuration

```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_test_requests: int = 3,
        error_types: Optional[List[Type[Exception]]] = None
    )
```

**Default Settings:**
- Failure threshold: 5 consecutive failures
- Recovery timeout: 30 seconds
- Half-open test requests: 3
- Error types: All exceptions

### State Transitions

1. **CLOSED → OPEN**
   - Triggered when `failure_count >= failure_threshold`
   - Immediate rejection of new requests
   - Logs state change with timestamp

2. **OPEN → HALF_OPEN**
   - Automatic after `recovery_timeout` expires
   - Allows limited test traffic
   - Resets test counter

3. **HALF_OPEN → CLOSED**
   - After `half_open_test_requests` succeed
   - Full traffic restored
   - Failure count reset

4. **HALF_OPEN → OPEN**
   - Any failure during test period
   - Returns to full rejection mode
   - Updates last failure time

## Retry Manager

Centralized management of retry policies and circuit breakers per service.

```python
class RetryManager:
    def __init__(self, default_config: Optional[Dict[str, Any]] = None)
```

### Default Configuration

```json
{
    "retry_policy": {
        "type": "exponential_backoff",
        "max_attempts": 5,
        "base_delay": 1.0,
        "max_delay": 16.0,
        "jitter_factor": 0.2
    },
    "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout": 30.0,
        "half_open_test_requests": 3
    }
}
```

### Per-Service Configuration

Services can have custom retry configurations:

```json
{
    "services": {
        "filesystem_mcp": {
            "retry_policy": {
                "type": "fixed_delay",
                "max_attempts": 3,
                "delay": 0.5
            }
        },
        "external_api": {
            "retry_policy": {
                "type": "exponential_backoff",
                "max_attempts": 10,
                "base_delay": 2.0,
                "max_delay": 60.0
            },
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 60.0
            }
        }
    }
}
```

## Connection Pool Integration

The retry system integrates with connection pooling for efficient resource usage.

### Features

1. **Connection Reuse**
   - Reduces connection overhead
   - Maintains healthy connections
   - Automatic cleanup of idle connections

2. **Health Checking**
   - Periodic health checks (default: 60s)
   - Automatic removal of unhealthy connections
   - Prevents retry storms on dead connections

3. **Pool Configuration**
   ```python
   {
       'max_connections': 10,
       'connection_timeout': 5.0,
       'idle_timeout': 300.0,
       'health_check_interval': 60.0
   }
   ```

## Retry Decorator Usage

### Basic Usage

```python
@retry_async()
async def call_mcp_tool(tool_id: str, params: dict):
    # Tool execution logic
    pass
```

### Custom Policy

```python
@retry_async(
    retry_policy=ExponentialBackoffRetry(max_attempts=3),
    circuit_breaker=CircuitBreaker(failure_threshold=3)
)
async def critical_operation():
    # Critical logic
    pass
```

### With Retry Callback

```python
@retry_async(
    on_retry=lambda e, attempt: logger.warning(f"Retry {attempt}: {e}")
)
async def monitored_operation():
    # Operation with retry monitoring
    pass
```

## Error Classification

### Retryable Errors

Errors that trigger retry attempts:
- Network timeouts
- Connection errors
- Temporary service unavailability (503)
- Rate limit errors (429) with delay

### Non-Retryable Errors

Errors that fail immediately:
- Authentication failures (401)
- Authorization errors (403)
- Not found errors (404)
- Bad request errors (400)
- `NonRetryableError` exceptions

### Circuit Breaker Errors

Errors that count toward circuit breaker:
- Connection timeouts
- Service errors (5xx)
- Repeated failures
- Custom error types

## Metrics and Monitoring

### Retry Metrics

The `RetryMetricsCollector` tracks:

1. **Attempt Metrics**
   - Total retry attempts
   - Success/failure rates
   - Retry delays
   - Error type distribution

2. **Circuit Breaker Metrics**
   - State changes
   - Open/close events
   - Current states
   - Rejection counts

3. **Performance Metrics**
   - Average retry delay
   - P95/P99 delays
   - Time series data
   - Hourly aggregations

### Monitoring Dashboard

Key metrics displayed:
- Active circuit breakers
- Retry success rates by service
- Common failure patterns
- Performance impact of retries

### Alerting

Automatic alerts for:
- Low retry success rate (<50%)
- Multiple open circuit breakers
- Excessive consecutive failures
- Unusual error patterns

## Best Practices

### 1. Idempotency
- Ensure operations are safe to retry
- Use unique request IDs
- Implement deduplication

### 2. Timeout Configuration
- Set appropriate timeouts
- Consider total retry time
- Align with user expectations

### 3. Error Handling
- Log all retry attempts
- Preserve original error context
- Track failure patterns

### 4. Resource Management
- Monitor connection pool usage
- Set reasonable retry limits
- Implement backpressure

### 5. Testing
- Test retry logic explicitly
- Simulate failures
- Verify circuit breaker behavior

## Performance Considerations

### 1. Retry Storms
- Jitter prevents synchronized retries
- Circuit breakers limit cascade failures
- Connection pools prevent resource exhaustion

### 2. Latency Impact
- Monitor total request time including retries
- Set aggressive timeouts for user-facing operations
- Use async operations to prevent blocking

### 3. Resource Usage
- Connection pool limits prevent exhaustion
- Circuit breakers reduce load on failing services
- Metrics collection has minimal overhead

## Configuration Examples

### High-Reliability Service

```json
{
    "retry_policy": {
        "type": "exponential_backoff",
        "max_attempts": 10,
        "base_delay": 0.5,
        "max_delay": 30.0,
        "jitter_factor": 0.3
    },
    "circuit_breaker": {
        "failure_threshold": 10,
        "recovery_timeout": 60.0,
        "half_open_test_requests": 5
    }
}
```

### Fast-Fail Service

```json
{
    "retry_policy": {
        "type": "fixed_delay",
        "max_attempts": 2,
        "delay": 0.2
    },
    "circuit_breaker": {
        "failure_threshold": 3,
        "recovery_timeout": 10.0,
        "half_open_test_requests": 1
    }
}
```

### No-Retry Critical Path

```json
{
    "retry_policy": {
        "type": "no_retry"
    },
    "circuit_breaker": null
}
```

## Future Enhancements

1. **Adaptive Retry Strategies**
   - ML-based retry delay optimization
   - Historical success rate consideration
   - Dynamic policy selection

2. **Advanced Circuit Breakers**
   - Gradual traffic increase in half-open
   - Multiple failure thresholds
   - Service dependency awareness

3. **Distributed Coordination**
   - Shared circuit breaker state
   - Cross-instance retry coordination
   - Global rate limiting

4. **Enhanced Monitoring**
   - Real-time retry visualization
   - Predictive failure detection
   - Cost analysis of retries