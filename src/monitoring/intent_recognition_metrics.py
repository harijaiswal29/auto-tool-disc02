"""
Performance monitoring and metrics collection for Intent Recognition.

This module provides comprehensive metrics tracking for the Intent Recognition
Agent including performance, accuracy, and usage statistics.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import json
import os

from src.utils.logger import get_logger
from src.agents.intent_models import IntentResult


class IntentRecognitionMetrics:
    """
    Metrics collector for Intent Recognition Agent.
    
    Tracks:
    - Processing time metrics
    - Classification accuracy
    - Cache hit rates
    - Pipeline stage performance
    - Error rates
    - Usage patterns
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.logger = get_logger(__name__)
        self.window_size = window_size
        
        # Performance metrics
        self.processing_times = deque(maxlen=window_size)
        self.stage_times = defaultdict(lambda: deque(maxlen=window_size))
        
        # Accuracy metrics
        self.intent_classifications = deque(maxlen=window_size)
        self.confidence_scores = deque(maxlen=window_size)
        self.false_positives = 0
        self.false_negatives = 0
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Error tracking
        self.errors = defaultdict(int)
        self.error_details = deque(maxlen=100)
        
        # Usage patterns
        self.intent_frequency = defaultdict(int)
        self.query_patterns = defaultdict(int)
        self.multi_intent_queries = 0
        self.total_queries = 0
        
        # Time-based metrics
        self.hourly_metrics = defaultdict(lambda: defaultdict(int))
        self.start_time = datetime.now()
    
    def record_query_processing(self, result: IntentResult, 
                              stage_timings: Optional[Dict[str, float]] = None):
        """
        Record metrics for a processed query.
        
        Args:
            result: The IntentResult from processing
            stage_timings: Optional timing for each pipeline stage
        """
        self.total_queries += 1
        
        # Record processing time
        self.processing_times.append(result.processing_time_ms)
        
        # Record stage timings if provided
        if stage_timings:
            for stage, timing in stage_timings.items():
                self.stage_times[stage].append(timing)
        
        # Record intent classification
        self.intent_classifications.append({
            'intent': result.primary_intent.type,
            'confidence': result.primary_intent.confidence,
            'passed': result.confidence_passed,
            'timestamp': datetime.now()
        })
        
        # Track confidence scores
        self.confidence_scores.append(result.primary_intent.confidence)
        
        # Track intent frequency
        self.intent_frequency[result.primary_intent.type] += 1
        
        # Track multi-intent queries
        if len(result.all_intents) > 1:
            self.multi_intent_queries += 1
        
        # Update hourly metrics
        hour_key = datetime.now().strftime("%Y-%m-%d_%H")
        self.hourly_metrics[hour_key]['queries'] += 1
        self.hourly_metrics[hour_key]['avg_time'] = self.get_average_processing_time()
    
    def record_cache_access(self, hit: bool):
        """Record cache hit or miss."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_error(self, error_type: str, error_details: Dict[str, Any]):
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error
            error_details: Detailed error information
        """
        self.errors[error_type] += 1
        self.error_details.append({
            'type': error_type,
            'details': error_details,
            'timestamp': datetime.now()
        })
    
    def record_feedback(self, intent: str, correct: bool):
        """
        Record user feedback on intent classification.
        
        Args:
            intent: The classified intent
            correct: Whether the classification was correct
        """
        if not correct:
            if intent:
                self.false_positives += 1
            else:
                self.false_negatives += 1
    
    # Metric calculations
    
    def get_average_processing_time(self) -> float:
        """Get average processing time in ms."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_percentile_processing_time(self, percentile: float) -> float:
        """Get percentile processing time (e.g., p95)."""
        if not self.processing_times:
            return 0.0
        
        sorted_times = sorted(self.processing_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_stage_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each pipeline stage."""
        stage_metrics = {}
        
        for stage, times in self.stage_times.items():
            if times:
                stage_metrics[stage] = {
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'p95_ms': self._calculate_percentile(list(times), 95)
                }
        
        return stage_metrics
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100
    
    def get_classification_accuracy(self) -> float:
        """Calculate classification accuracy based on feedback."""
        total_feedback = len(self.intent_classifications)
        if total_feedback == 0:
            return 0.0
        
        errors = self.false_positives + self.false_negatives
        accuracy = (total_feedback - errors) / total_feedback
        return accuracy * 100
    
    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence scores."""
        if not self.confidence_scores:
            return {}
        
        distribution = {
            'very_low': 0,    # < 0.3
            'low': 0,         # 0.3-0.5
            'medium': 0,      # 0.5-0.7
            'high': 0,        # 0.7-0.9
            'very_high': 0    # > 0.9
        }
        
        for score in self.confidence_scores:
            if score < 0.3:
                distribution['very_low'] += 1
            elif score < 0.5:
                distribution['low'] += 1
            elif score < 0.7:
                distribution['medium'] += 1
            elif score < 0.9:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1
        
        return distribution
    
    def get_intent_distribution(self) -> Dict[str, float]:
        """Get distribution of classified intents."""
        if self.total_queries == 0:
            return {}
        
        distribution = {}
        for intent, count in self.intent_frequency.items():
            distribution[intent] = (count / self.total_queries) * 100
        
        return distribution
    
    def get_error_rate(self) -> float:
        """Get overall error rate."""
        if self.total_queries == 0:
            return 0.0
        
        total_errors = sum(self.errors.values())
        return (total_errors / self.total_queries) * 100
    
    def get_multi_intent_rate(self) -> float:
        """Get percentage of multi-intent queries."""
        if self.total_queries == 0:
            return 0.0
        return (self.multi_intent_queries / self.total_queries) * 100
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        uptime = datetime.now() - self.start_time
        
        return {
            'performance': {
                'avg_processing_time_ms': self.get_average_processing_time(),
                'p50_processing_time_ms': self.get_percentile_processing_time(50),
                'p95_processing_time_ms': self.get_percentile_processing_time(95),
                'p99_processing_time_ms': self.get_percentile_processing_time(99),
                'stage_performance': self.get_stage_performance()
            },
            'accuracy': {
                'classification_accuracy': self.get_classification_accuracy(),
                'confidence_distribution': self.get_confidence_distribution(),
                'avg_confidence': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
            },
            'cache': {
                'hit_rate': self.get_cache_hit_rate(),
                'total_hits': self.cache_hits,
                'total_misses': self.cache_misses
            },
            'usage': {
                'total_queries': self.total_queries,
                'queries_per_hour': self.total_queries / max(uptime.total_seconds() / 3600, 1),
                'intent_distribution': self.get_intent_distribution(),
                'multi_intent_rate': self.get_multi_intent_rate()
            },
            'reliability': {
                'error_rate': self.get_error_rate(),
                'error_types': dict(self.errors),
                'uptime_hours': uptime.total_seconds() / 3600
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics = self.get_summary_metrics()
        metrics['export_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.processing_times.clear()
        self.stage_times.clear()
        self.intent_classifications.clear()
        self.confidence_scores.clear()
        self.false_positives = 0
        self.false_negatives = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors.clear()
        self.error_details.clear()
        self.intent_frequency.clear()
        self.query_patterns.clear()
        self.multi_intent_queries = 0
        self.total_queries = 0
        self.hourly_metrics.clear()
        self.start_time = datetime.now()
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value from list."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class MetricsAggregator:
    """
    Aggregates metrics over time for reporting and analysis.
    """
    
    def __init__(self, metrics: IntentRecognitionMetrics):
        self.metrics = metrics
        self.logger = get_logger(__name__)
        self.aggregation_interval = 60  # seconds
        self.aggregation_task = None
    
    async def start_aggregation(self):
        """Start periodic metrics aggregation."""
        self.aggregation_task = asyncio.create_task(self._aggregate_periodically())
        self.logger.info("Started metrics aggregation")
    
    async def stop_aggregation(self):
        """Stop metrics aggregation."""
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped metrics aggregation")
    
    async def _aggregate_periodically(self):
        """Periodically aggregate and log metrics."""
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval)
                
                # Get summary metrics
                summary = self.metrics.get_summary_metrics()
                
                # Log key metrics
                self.logger.info(
                    f"Intent Recognition Metrics - "
                    f"Avg Time: {summary['performance']['avg_processing_time_ms']:.2f}ms, "
                    f"P95: {summary['performance']['p95_processing_time_ms']:.2f}ms, "
                    f"Cache Hit Rate: {summary['cache']['hit_rate']:.1f}%, "
                    f"Error Rate: {summary['reliability']['error_rate']:.2f}%"
                )
                
                # Check for performance issues
                if summary['performance']['p95_processing_time_ms'] > 200:
                    self.logger.warning(
                        f"High P95 processing time: {summary['performance']['p95_processing_time_ms']:.2f}ms"
                    )
                
                if summary['reliability']['error_rate'] > 5:
                    self.logger.warning(
                        f"High error rate: {summary['reliability']['error_rate']:.2f}%"
                    )
                
            except Exception as e:
                self.logger.error(f"Error in metrics aggregation: {e}")


# Global metrics instance
_metrics_instance = None


def get_metrics() -> IntentRecognitionMetrics:
    """Get or create global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = IntentRecognitionMetrics()
    return _metrics_instance