"""
Retry metrics monitoring for tracking retry attempts, circuit breaker states,
and failure patterns across the system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import statistics

from ..utils.logger import get_logger
from ..core.tool_registry import ToolRegistry

logger = get_logger(__name__)


class RetryMetricsCollector:
    """
    Collects and aggregates retry metrics for monitoring and analysis.
    
    Features:
    - Real-time retry tracking
    - Circuit breaker state monitoring
    - Failure pattern detection
    - Performance impact analysis
    """
    
    def __init__(self, registry: ToolRegistry, window_size: int = 1000):
        """
        Initialize retry metrics collector.
        
        Args:
            registry: Tool registry for persisting metrics
            window_size: Size of sliding window for metrics
        """
        self.registry = registry
        self.window_size = window_size
        
        # In-memory metrics storage
        self.retry_attempts = defaultdict(lambda: deque(maxlen=window_size))
        self.circuit_breaker_events = defaultdict(list)
        self.failure_patterns = defaultdict(lambda: defaultdict(int))
        
        # Aggregated metrics
        self.metrics_summary = {
            'total_retry_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'circuit_breaker_opens': 0,
            'circuit_breaker_closes': 0,
            'avg_retry_delay_ms': 0,
            'max_consecutive_failures': defaultdict(int)
        }
        
        # Time-based metrics
        self.hourly_metrics = defaultdict(lambda: {
            'retry_attempts': 0,
            'failures': 0,
            'circuit_breaker_events': 0
        })
        
        logger.info("[INIT] Retry metrics collector initialized")
    
    def record_retry_attempt(self, tool_id: str, attempt_number: int, 
                           delay_ms: float, error_type: Optional[str] = None,
                           success: bool = False):
        """Record a retry attempt."""
        timestamp = datetime.now()
        
        # Create retry record
        retry_record = {
            'timestamp': timestamp.isoformat(),
            'attempt_number': attempt_number,
            'delay_ms': delay_ms,
            'error_type': error_type,
            'success': success
        }
        
        # Add to in-memory storage
        self.retry_attempts[tool_id].append(retry_record)
        
        # Update summary metrics
        self.metrics_summary['total_retry_attempts'] += 1
        if success:
            self.metrics_summary['successful_retries'] += 1
        else:
            self.metrics_summary['failed_retries'] += 1
        
        # Update hourly metrics
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        self.hourly_metrics[hour_key]['retry_attempts'] += 1
        
        # Track failure patterns
        if error_type:
            self.failure_patterns[tool_id][error_type] += 1
        
        # Persist to registry
        self.registry.record_retry_metric(
            tool_id, attempt_number, delay_ms, error_type, 
            f"Retry attempt {attempt_number}"
        )
        
        logger.debug(
            f"[RETRY] Tool: {tool_id}, Attempt: {attempt_number}, "
            f"Delay: {delay_ms}ms, Success: {success}"
        )
    
    def record_circuit_breaker_event(self, tool_id: str, event_type: str, 
                                   state: str, reason: Optional[str] = None):
        """Record circuit breaker state change."""
        timestamp = datetime.now()
        
        event = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'state': state,
            'reason': reason
        }
        
        # Add to event log
        self.circuit_breaker_events[tool_id].append(event)
        
        # Update summary
        if event_type == 'opened':
            self.metrics_summary['circuit_breaker_opens'] += 1
        elif event_type == 'closed':
            self.metrics_summary['circuit_breaker_closes'] += 1
        
        # Update hourly metrics
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        self.hourly_metrics[hour_key]['circuit_breaker_events'] += 1
        
        # Update registry
        self.registry.update_circuit_breaker_state(tool_id, state)
        
        logger.info(
            f"[CIRCUIT_BREAKER] Tool: {tool_id}, Event: {event_type}, "
            f"State: {state}, Reason: {reason}"
        )
    
    def record_consecutive_failure(self, tool_id: str, failure_count: int):
        """Track consecutive failures for a tool."""
        current_max = self.metrics_summary['max_consecutive_failures'][tool_id]
        if failure_count > current_max:
            self.metrics_summary['max_consecutive_failures'][tool_id] = failure_count
    
    def get_retry_statistics(self, tool_id: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics for a specific tool or all tools."""
        if tool_id:
            attempts = list(self.retry_attempts[tool_id])
            if not attempts:
                return {'tool_id': tool_id, 'no_data': True}
            
            delays = [a['delay_ms'] for a in attempts]
            success_rate = sum(1 for a in attempts if a['success']) / len(attempts)
            
            return {
                'tool_id': tool_id,
                'total_attempts': len(attempts),
                'success_rate': success_rate,
                'avg_delay_ms': statistics.mean(delays),
                'median_delay_ms': statistics.median(delays),
                'max_delay_ms': max(delays),
                'recent_attempts': attempts[-10:]  # Last 10 attempts
            }
        else:
            # Aggregate statistics for all tools
            all_delays = []
            total_attempts = 0
            successful_attempts = 0
            
            for tool_attempts in self.retry_attempts.values():
                for attempt in tool_attempts:
                    all_delays.append(attempt['delay_ms'])
                    total_attempts += 1
                    if attempt['success']:
                        successful_attempts += 1
            
            if not all_delays:
                return {'no_data': True}
            
            return {
                'total_tools_with_retries': len(self.retry_attempts),
                'total_attempts': total_attempts,
                'overall_success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
                'avg_delay_ms': statistics.mean(all_delays),
                'median_delay_ms': statistics.median(all_delays),
                'p95_delay_ms': statistics.quantiles(all_delays, n=20)[18] if len(all_delays) >= 20 else max(all_delays),
                'summary': self.metrics_summary
            }
    
    def get_failure_patterns(self, tool_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze failure patterns."""
        if tool_id:
            patterns = dict(self.failure_patterns[tool_id])
            total_failures = sum(patterns.values())
            
            return {
                'tool_id': tool_id,
                'total_failures': total_failures,
                'error_distribution': patterns,
                'most_common_error': max(patterns.items(), key=lambda x: x[1])[0] if patterns else None
            }
        else:
            # Aggregate failure patterns
            all_patterns = defaultdict(int)
            for tool_patterns in self.failure_patterns.values():
                for error_type, count in tool_patterns.items():
                    all_patterns[error_type] += count
            
            total_failures = sum(all_patterns.values())
            
            return {
                'total_failures': total_failures,
                'error_distribution': dict(all_patterns),
                'top_errors': sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            }
    
    def get_circuit_breaker_summary(self) -> Dict[str, Any]:
        """Get circuit breaker status summary."""
        active_circuit_breakers = []
        recent_events = []
        
        for tool_id, events in self.circuit_breaker_events.items():
            if events:
                latest_event = events[-1]
                if latest_event['state'] == 'open':
                    active_circuit_breakers.append({
                        'tool_id': tool_id,
                        'opened_at': latest_event['timestamp'],
                        'reason': latest_event['reason']
                    })
                
                # Get recent events (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                for event in events:
                    if datetime.fromisoformat(event['timestamp']) > cutoff_time:
                        recent_events.append({
                            'tool_id': tool_id,
                            **event
                        })
        
        return {
            'active_circuit_breakers': active_circuit_breakers,
            'total_opens_lifetime': self.metrics_summary['circuit_breaker_opens'],
            'total_closes_lifetime': self.metrics_summary['circuit_breaker_closes'],
            'recent_events': sorted(recent_events, key=lambda x: x['timestamp'], reverse=True)[:20]
        }
    
    def get_time_series_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get time series metrics for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        time_series = []
        
        for hour_key in sorted(self.hourly_metrics.keys()):
            hour_time = datetime.strptime(hour_key, '%Y-%m-%d %H:00')
            if hour_time >= cutoff_time:
                metrics = self.hourly_metrics[hour_key]
                time_series.append({
                    'hour': hour_key,
                    'retry_attempts': metrics['retry_attempts'],
                    'failures': metrics['failures'],
                    'circuit_breaker_events': metrics['circuit_breaker_events'],
                    'retry_rate': metrics['retry_attempts'] / 3600  # Per second
                })
        
        return time_series
    
    def export_metrics(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export all metrics to JSON format."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'retry_statistics': self.get_retry_statistics(),
            'failure_patterns': self.get_failure_patterns(),
            'circuit_breaker_summary': self.get_circuit_breaker_summary(),
            'time_series_24h': self.get_time_series_metrics(24),
            'tool_specific_stats': {}
        }
        
        # Add tool-specific statistics
        for tool_id in self.retry_attempts.keys():
            export_data['tool_specific_stats'][tool_id] = {
                'retry_stats': self.get_retry_statistics(tool_id),
                'failure_patterns': self.get_failure_patterns(tool_id),
                'registry_metrics': self.registry.get_failure_metrics(tool_id)
            }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"[EXPORT] Metrics exported to {filepath}")
        
        return export_data
    
    def generate_alert_recommendations(self) -> List[Dict[str, Any]]:
        """Generate alert recommendations based on metrics."""
        alerts = []
        
        # Check for high retry rates
        recent_stats = self.get_retry_statistics()
        if recent_stats.get('total_attempts', 0) > 100:
            success_rate = recent_stats.get('overall_success_rate', 1)
            if success_rate < 0.5:
                alerts.append({
                    'severity': 'high',
                    'type': 'low_retry_success_rate',
                    'message': f"Overall retry success rate is {success_rate:.1%}",
                    'recommendation': "Investigate common failure patterns and consider increasing retry delays"
                })
        
        # Check for tools with excessive consecutive failures
        for tool_id, failures in self.metrics_summary['max_consecutive_failures'].items():
            if failures >= 10:
                alerts.append({
                    'severity': 'high',
                    'type': 'excessive_consecutive_failures',
                    'tool_id': tool_id,
                    'message': f"Tool {tool_id} had {failures} consecutive failures",
                    'recommendation': "Consider implementing circuit breaker or increasing failure threshold"
                })
        
        # Check for frequently opening circuit breakers
        cb_summary = self.get_circuit_breaker_summary()
        if len(cb_summary['active_circuit_breakers']) > 3:
            alerts.append({
                'severity': 'high',
                'type': 'multiple_circuit_breakers_open',
                'message': f"{len(cb_summary['active_circuit_breakers'])} circuit breakers are currently open",
                'recommendation': "System may be experiencing widespread issues"
            })
        
        return alerts