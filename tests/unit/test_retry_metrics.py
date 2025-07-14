"""
Unit tests for Retry Metrics module.

Tests the retry metrics collection, analysis, and alerting functionality
for monitoring retry attempts and circuit breaker states.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict, deque

from src.monitoring.retry_metrics import RetryMetricsCollector


class TestRetryMetricsCollector:
    """Test cases for RetryMetricsCollector class."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock ToolRegistry."""
        registry = Mock()
        registry.record_retry_metric = Mock()
        registry.update_circuit_breaker_state = Mock()
        registry.get_failure_metrics = Mock(return_value={"failures": 0, "successes": 0})
        return registry
    
    @pytest.fixture
    def collector(self, mock_registry):
        """Create a RetryMetricsCollector instance."""
        return RetryMetricsCollector(mock_registry, window_size=100)
    
    def test_initialization(self, collector, mock_registry):
        """Test collector initialization."""
        assert collector.registry == mock_registry
        assert collector.window_size == 100
        assert collector.metrics_summary['total_retry_attempts'] == 0
        assert collector.metrics_summary['successful_retries'] == 0
        assert collector.metrics_summary['failed_retries'] == 0
        assert isinstance(collector.retry_attempts, defaultdict)
        assert isinstance(collector.circuit_breaker_events, defaultdict)
    
    def test_record_retry_attempt_success(self, collector, mock_registry):
        """Test recording a successful retry attempt."""
        collector.record_retry_attempt(
            tool_id="test_tool",
            attempt_number=2,
            delay_ms=1000.0,
            error_type="timeout",
            success=True
        )
        
        # Verify in-memory storage
        assert len(collector.retry_attempts["test_tool"]) == 1
        attempt = collector.retry_attempts["test_tool"][0]
        assert attempt['attempt_number'] == 2
        assert attempt['delay_ms'] == 1000.0
        assert attempt['error_type'] == "timeout"
        assert attempt['success'] is True
        
        # Verify summary metrics
        assert collector.metrics_summary['total_retry_attempts'] == 1
        assert collector.metrics_summary['successful_retries'] == 1
        assert collector.metrics_summary['failed_retries'] == 0
        
        # Verify failure patterns
        assert collector.failure_patterns["test_tool"]["timeout"] == 1
        
        # Verify registry call
        mock_registry.record_retry_metric.assert_called_once_with(
            "test_tool", 2, 1000.0, "timeout", "Retry attempt 2"
        )
    
    def test_record_retry_attempt_failure(self, collector, mock_registry):
        """Test recording a failed retry attempt."""
        collector.record_retry_attempt(
            tool_id="test_tool",
            attempt_number=3,
            delay_ms=2000.0,
            error_type="connection_error",
            success=False
        )
        
        # Verify summary metrics
        assert collector.metrics_summary['total_retry_attempts'] == 1
        assert collector.metrics_summary['successful_retries'] == 0
        assert collector.metrics_summary['failed_retries'] == 1
        
        # Verify failure patterns
        assert collector.failure_patterns["test_tool"]["connection_error"] == 1
    
    def test_record_circuit_breaker_event(self, collector, mock_registry):
        """Test recording circuit breaker events."""
        # Record circuit breaker opened
        collector.record_circuit_breaker_event(
            tool_id="test_tool",
            event_type="opened",
            state="open",
            reason="Too many failures"
        )
        
        # Verify event storage
        assert len(collector.circuit_breaker_events["test_tool"]) == 1
        event = collector.circuit_breaker_events["test_tool"][0]
        assert event['event_type'] == "opened"
        assert event['state'] == "open"
        assert event['reason'] == "Too many failures"
        
        # Verify summary
        assert collector.metrics_summary['circuit_breaker_opens'] == 1
        assert collector.metrics_summary['circuit_breaker_closes'] == 0
        
        # Verify registry update
        mock_registry.update_circuit_breaker_state.assert_called_once_with("test_tool", "open")
        
        # Record circuit breaker closed
        collector.record_circuit_breaker_event(
            tool_id="test_tool",
            event_type="closed",
            state="closed",
            reason="Recovery successful"
        )
        
        assert collector.metrics_summary['circuit_breaker_closes'] == 1
    
    def test_record_consecutive_failure(self, collector):
        """Test tracking consecutive failures."""
        # Record increasing consecutive failures
        collector.record_consecutive_failure("test_tool", 5)
        assert collector.metrics_summary['max_consecutive_failures']["test_tool"] == 5
        
        # Lower count should not update max
        collector.record_consecutive_failure("test_tool", 3)
        assert collector.metrics_summary['max_consecutive_failures']["test_tool"] == 5
        
        # Higher count should update
        collector.record_consecutive_failure("test_tool", 10)
        assert collector.metrics_summary['max_consecutive_failures']["test_tool"] == 10
    
    def test_get_retry_statistics_single_tool(self, collector):
        """Test getting retry statistics for a specific tool."""
        # Add some retry attempts
        for i in range(5):
            collector.record_retry_attempt(
                tool_id="test_tool",
                attempt_number=i + 1,
                delay_ms=1000.0 * (i + 1),
                error_type="timeout",
                success=(i >= 3)  # Last 2 attempts succeed
            )
        
        stats = collector.get_retry_statistics("test_tool")
        
        assert stats['tool_id'] == "test_tool"
        assert stats['total_attempts'] == 5
        assert stats['success_rate'] == 0.4  # 2 out of 5
        assert stats['avg_delay_ms'] == 3000.0  # (1000+2000+3000+4000+5000)/5
        assert stats['median_delay_ms'] == 3000.0
        assert stats['max_delay_ms'] == 5000.0
        assert len(stats['recent_attempts']) == 5
    
    def test_get_retry_statistics_all_tools(self, collector):
        """Test getting aggregate retry statistics."""
        # Add attempts for multiple tools
        collector.record_retry_attempt("tool1", 1, 1000.0, "error1", True)
        collector.record_retry_attempt("tool1", 2, 2000.0, "error1", False)
        collector.record_retry_attempt("tool2", 1, 1500.0, "error2", True)
        
        stats = collector.get_retry_statistics()
        
        assert stats['total_tools_with_retries'] == 2
        assert stats['total_attempts'] == 3
        assert stats['overall_success_rate'] == 2/3
        assert stats['avg_delay_ms'] == 1500.0  # (1000+2000+1500)/3
        assert 'summary' in stats
    
    def test_get_retry_statistics_no_data(self, collector):
        """Test statistics when no data exists."""
        # Single tool with no data
        stats = collector.get_retry_statistics("nonexistent_tool")
        assert stats['tool_id'] == "nonexistent_tool"
        assert stats['no_data'] is True
        
        # All tools with no data
        stats = collector.get_retry_statistics()
        assert stats['no_data'] is True
    
    def test_get_failure_patterns_single_tool(self, collector):
        """Test getting failure patterns for a specific tool."""
        # Record various failures
        collector.failure_patterns["test_tool"]["timeout"] = 5
        collector.failure_patterns["test_tool"]["connection_error"] = 3
        collector.failure_patterns["test_tool"]["rate_limit"] = 2
        
        patterns = collector.get_failure_patterns("test_tool")
        
        assert patterns['tool_id'] == "test_tool"
        assert patterns['total_failures'] == 10
        assert patterns['error_distribution']['timeout'] == 5
        assert patterns['most_common_error'] == "timeout"
    
    def test_get_failure_patterns_all_tools(self, collector):
        """Test getting aggregate failure patterns."""
        # Add failures for multiple tools
        collector.failure_patterns["tool1"]["timeout"] = 5
        collector.failure_patterns["tool1"]["connection_error"] = 2
        collector.failure_patterns["tool2"]["timeout"] = 3
        collector.failure_patterns["tool2"]["rate_limit"] = 4
        
        patterns = collector.get_failure_patterns()
        
        assert patterns['total_failures'] == 14
        assert patterns['error_distribution']['timeout'] == 8
        assert patterns['error_distribution']['connection_error'] == 2
        assert patterns['error_distribution']['rate_limit'] == 4
        
        # Check top errors
        top_errors = patterns['top_errors']
        assert top_errors[0][0] == "timeout"
        assert top_errors[0][1] == 8
    
    def test_get_circuit_breaker_summary(self, collector):
        """Test circuit breaker summary generation."""
        # Add circuit breaker events
        now = datetime.now()
        
        # Tool1: currently open
        collector.circuit_breaker_events["tool1"].append({
            'timestamp': now.isoformat(),
            'event_type': 'opened',
            'state': 'open',
            'reason': 'Too many failures'
        })
        
        # Tool2: was open but now closed
        collector.circuit_breaker_events["tool2"].append({
            'timestamp': (now - timedelta(hours=2)).isoformat(),
            'event_type': 'opened',
            'state': 'open',
            'reason': 'Connection issues'
        })
        collector.circuit_breaker_events["tool2"].append({
            'timestamp': (now - timedelta(hours=1)).isoformat(),
            'event_type': 'closed',
            'state': 'closed',
            'reason': 'Recovered'
        })
        
        collector.metrics_summary['circuit_breaker_opens'] = 2
        collector.metrics_summary['circuit_breaker_closes'] = 1
        
        summary = collector.get_circuit_breaker_summary()
        
        # Check active circuit breakers
        assert len(summary['active_circuit_breakers']) == 1
        assert summary['active_circuit_breakers'][0]['tool_id'] == "tool1"
        
        # Check lifetime counts
        assert summary['total_opens_lifetime'] == 2
        assert summary['total_closes_lifetime'] == 1
        
        # Check recent events (within 24 hours)
        assert len(summary['recent_events']) >= 3
    
    def test_get_time_series_metrics(self, collector):
        """Test time series metrics generation."""
        # Add hourly metrics
        now = datetime.now()
        current_hour = now.strftime('%Y-%m-%d %H:00')
        last_hour = (now - timedelta(hours=1)).strftime('%Y-%m-%d %H:00')
        
        collector.hourly_metrics[last_hour] = {
            'retry_attempts': 10,
            'failures': 3,
            'circuit_breaker_events': 1
        }
        
        collector.hourly_metrics[current_hour] = {
            'retry_attempts': 5,
            'failures': 1,
            'circuit_breaker_events': 0
        }
        
        time_series = collector.get_time_series_metrics(hours=2)
        
        assert len(time_series) >= 1
        assert time_series[-1]['hour'] == current_hour
        assert time_series[-1]['retry_attempts'] == 5
        assert time_series[-1]['retry_rate'] == 5/3600  # per second
    
    def test_export_metrics(self, collector, tmp_path):
        """Test metrics export functionality."""
        # Add some data
        collector.record_retry_attempt("test_tool", 1, 1000.0, "timeout", True)
        collector.record_circuit_breaker_event("test_tool", "opened", "open", "Test")
        
        # Export to file
        export_path = tmp_path / "retry_metrics.json"
        result = collector.export_metrics(str(export_path))
        
        # Verify file was created
        assert export_path.exists()
        
        # Verify export structure
        assert 'timestamp' in result
        assert 'retry_statistics' in result
        assert 'failure_patterns' in result
        assert 'circuit_breaker_summary' in result
        assert 'time_series_24h' in result
        assert 'tool_specific_stats' in result
        
        # Verify JSON validity
        with open(export_path, 'r') as f:
            data = json.load(f)
        assert data['retry_statistics']['total_attempts'] >= 0
    
    def test_generate_alert_recommendations(self, collector):
        """Test alert recommendations generation."""
        # Test low retry success rate alert
        for i in range(150):
            collector.record_retry_attempt(
                tool_id=f"tool_{i % 5}",
                attempt_number=1,
                delay_ms=1000.0,
                error_type="error",
                success=(i % 3 == 0)  # ~33% success rate
            )
        
        # Test excessive consecutive failures
        collector.metrics_summary['max_consecutive_failures']["failing_tool"] = 15
        
        # Test multiple circuit breakers open
        for i in range(5):
            collector.circuit_breaker_events[f"cb_tool_{i}"].append({
                'timestamp': datetime.now().isoformat(),
                'event_type': 'opened',
                'state': 'open',
                'reason': 'Test'
            })
        
        alerts = collector.generate_alert_recommendations()
        
        # Should have alerts for all three conditions
        alert_types = [alert['type'] for alert in alerts]
        assert 'low_retry_success_rate' in alert_types
        assert 'excessive_consecutive_failures' in alert_types
        assert 'multiple_circuit_breakers_open' in alert_types
        
        # Verify alert details
        for alert in alerts:
            assert 'severity' in alert
            assert 'message' in alert
            assert 'recommendation' in alert
    
    def test_window_size_enforcement(self, collector):
        """Test that window size is enforced for deques."""
        # Add more attempts than window size
        for i in range(150):  # Window size is 100
            collector.record_retry_attempt(
                tool_id="test_tool",
                attempt_number=i + 1,
                delay_ms=1000.0,
                error_type="error",
                success=True
            )
        
        # Should only keep last 100
        assert len(collector.retry_attempts["test_tool"]) == 100
        
        # First attempt should be #51 (150-100+1)
        first_attempt = collector.retry_attempts["test_tool"][0]
        assert first_attempt['attempt_number'] == 51
    
    def test_hourly_metrics_update(self, collector):
        """Test hourly metrics are properly updated."""
        # Record some retry attempts
        collector.record_retry_attempt("tool1", 1, 1000.0, "error", False)
        collector.record_retry_attempt("tool2", 1, 2000.0, "error", True)
        
        # Record circuit breaker event
        collector.record_circuit_breaker_event("tool1", "opened", "open", "Test")
        
        # Check hourly metrics
        hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
        assert collector.hourly_metrics[hour_key]['retry_attempts'] == 2
        assert collector.hourly_metrics[hour_key]['circuit_breaker_events'] == 1
    
    def test_empty_patterns_handling(self, collector):
        """Test handling of empty failure patterns."""
        # Get patterns for tool with no failures
        patterns = collector.get_failure_patterns("empty_tool")
        assert patterns['tool_id'] == "empty_tool"
        assert patterns['total_failures'] == 0
        assert patterns['error_distribution'] == {}
        assert patterns['most_common_error'] is None
    
    def test_percentile_calculation(self, collector):
        """Test percentile calculation in statistics."""
        # Add enough retry attempts for percentile calculation
        for i in range(25):
            collector.record_retry_attempt(
                tool_id="test_tool",
                attempt_number=i + 1,
                delay_ms=100.0 * (i + 1),  # 100, 200, ..., 2500
                error_type="error",
                success=True
            )
        
        stats = collector.get_retry_statistics()
        
        # With 25 values, p95 should be around 2400
        assert 'p95_delay_ms' in stats
        assert 2300 <= stats['p95_delay_ms'] <= 2500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])