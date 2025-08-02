"""
Cache Metrics Monitoring Module.

This module provides comprehensive monitoring and tracking of cache performance metrics,
including time-series data, trend analysis, and performance indicators essential for
demonstrating learning effectiveness in the dissertation.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from src.agents.result_cache import ResultCache


@dataclass
class CacheMetricSnapshot:
    """Represents a point-in-time cache metric snapshot."""
    timestamp: datetime
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    expirations: int
    current_size: int
    max_size: int
    cache_size_bytes: int
    avg_retrieval_time_ms: float
    total_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class QueryPatternMetrics:
    """Metrics for specific query patterns."""
    pattern: str
    query_count: int
    cache_hits: int
    cache_misses: int
    avg_response_time_ms: float
    last_accessed: datetime
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate for this pattern."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class CacheMetricsMonitor:
    """
    Comprehensive cache metrics monitoring system.
    
    Features:
    - Time-series tracking of cache metrics
    - Query pattern analysis
    - Performance trend detection
    - Alert generation for anomalies
    - Integration with learning system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the cache metrics monitor."""
        self.logger = get_logger(__name__)
        
        # Configuration
        self.config = config or {}
        self.monitor_config = self.config.get('cache_monitoring', {})
        
        # Monitoring settings
        self.collection_interval = self.monitor_config.get('collection_interval', 10.0)  # seconds
        self.history_window = self.monitor_config.get('history_window', 3600)  # 1 hour
        self.max_history_size = self.monitor_config.get('max_history_size', 1000)
        self.pattern_tracking_enabled = self.monitor_config.get('track_patterns', True)
        
        # Metrics storage
        self.metrics_history: Deque[CacheMetricSnapshot] = deque(maxlen=self.max_history_size)
        self.query_patterns: Dict[str, QueryPatternMetrics] = {}
        self.hourly_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.performance_indicators = {
            'hit_rate_trend': 0.0,  # Positive = improving, negative = degrading
            'avg_response_improvement': 0.0,  # % improvement in response time
            'cache_efficiency': 0.0,  # Ratio of cache benefit vs memory usage
            'warming_effectiveness': 0.0  # How well cache warming works
        }
        
        # Alert thresholds
        self.alert_thresholds = self.monitor_config.get('alert_thresholds', {
            'min_hit_rate': 0.3,  # Alert if hit rate drops below 30%
            'max_eviction_rate': 0.2,  # Alert if >20% evictions per interval
            'performance_degradation': -0.1  # Alert if 10% performance drop
        })
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.cache_ref: Optional[ResultCache] = None
        self.last_snapshot: Optional[CacheMetricSnapshot] = None
        
        # Data persistence
        self.metrics_file = Path(self.monitor_config.get(
            'metrics_file', 
            'data/monitoring/cache_metrics.json'
        ))
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Cache metrics monitor initialized")
    
    def attach_cache(self, cache: ResultCache):
        """Attach a cache instance to monitor."""
        self.cache_ref = cache
        self.logger.info("Cache attached for monitoring")
    
    async def start_monitoring(self):
        """Start the monitoring process."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        if not self.cache_ref:
            raise ValueError("No cache attached for monitoring")
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Cache monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring process."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Save final metrics
        await self.save_metrics()
        self.logger.info("Cache monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                snapshot = await self._collect_metrics()
                
                # Update history
                self.metrics_history.append(snapshot)
                
                # Analyze trends
                self._analyze_trends()
                
                # Check for alerts
                alerts = self._check_alerts(snapshot)
                if alerts:
                    await self._handle_alerts(alerts)
                
                # Update performance indicators
                self._update_performance_indicators()
                
                # Periodic save
                if len(self.metrics_history) % 10 == 0:
                    await self.save_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_metrics(self) -> CacheMetricSnapshot:
        """Collect current cache metrics."""
        if not self.cache_ref:
            raise ValueError("No cache attached")
        
        metrics = self.cache_ref.get_metrics()
        
        snapshot = CacheMetricSnapshot(
            timestamp=datetime.now(),
            hits=metrics.get('hits', 0),
            misses=metrics.get('misses', 0),
            hit_rate=metrics.get('hit_rate', 0.0),
            evictions=metrics.get('evictions', 0),
            expirations=metrics.get('expirations', 0),
            current_size=metrics.get('current_size', 0),
            max_size=metrics.get('max_size', 0),
            cache_size_bytes=metrics.get('cache_size_bytes', 0),
            avg_retrieval_time_ms=metrics.get('avg_retrieval_time_ms', 0.0),
            total_queries=metrics.get('hits', 0) + metrics.get('misses', 0)
        )
        
        self.last_snapshot = snapshot
        return snapshot
    
    def _analyze_trends(self):
        """Analyze performance trends from historical data."""
        if len(self.metrics_history) < 10:
            return
        
        # Get recent history
        recent_metrics = list(self.metrics_history)[-20:]
        
        # Calculate hit rate trend
        hit_rates = [m.hit_rate for m in recent_metrics]
        if len(hit_rates) >= 2:
            # Simple linear regression for trend
            x = np.arange(len(hit_rates))
            coeffs = np.polyfit(x, hit_rates, 1)
            self.performance_indicators['hit_rate_trend'] = coeffs[0]
        
        # Calculate response time improvement
        if len(recent_metrics) >= 2:
            old_avg = recent_metrics[0].avg_retrieval_time_ms
            new_avg = recent_metrics[-1].avg_retrieval_time_ms
            if old_avg > 0:
                improvement = (old_avg - new_avg) / old_avg
                self.performance_indicators['avg_response_improvement'] = improvement
        
        # Calculate cache efficiency (benefit per byte)
        if self.last_snapshot and self.last_snapshot.cache_size_bytes > 0:
            time_saved = (self.last_snapshot.hits * self.last_snapshot.avg_retrieval_time_ms)
            efficiency = time_saved / (self.last_snapshot.cache_size_bytes / 1024)  # ms saved per KB
            self.performance_indicators['cache_efficiency'] = efficiency
    
    def _check_alerts(self, snapshot: CacheMetricSnapshot) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        # Check hit rate threshold
        if snapshot.hit_rate < self.alert_thresholds['min_hit_rate']:
            alerts.append({
                'type': 'low_hit_rate',
                'severity': 'warning',
                'message': f"Cache hit rate ({snapshot.hit_rate:.2%}) below threshold",
                'value': snapshot.hit_rate,
                'threshold': self.alert_thresholds['min_hit_rate']
            })
        
        # Check eviction rate
        if len(self.metrics_history) >= 2:
            prev_evictions = self.metrics_history[-2].evictions
            eviction_rate = (snapshot.evictions - prev_evictions) / snapshot.max_size
            if eviction_rate > self.alert_thresholds['max_eviction_rate']:
                alerts.append({
                    'type': 'high_eviction_rate',
                    'severity': 'warning',
                    'message': f"High eviction rate ({eviction_rate:.2%})",
                    'value': eviction_rate,
                    'threshold': self.alert_thresholds['max_eviction_rate']
                })
        
        # Check performance degradation
        if self.performance_indicators['hit_rate_trend'] < self.alert_thresholds['performance_degradation']:
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'critical',
                'message': "Cache performance is degrading",
                'trend': self.performance_indicators['hit_rate_trend']
            })
        
        return alerts
    
    async def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle generated alerts."""
        for alert in alerts:
            self.logger.warning(f"Cache alert: {alert['message']}")
            # Could integrate with external alerting systems here
    
    def _update_performance_indicators(self):
        """Update high-level performance indicators."""
        if not self.metrics_history:
            return
        
        # Update hourly aggregates
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hour_key = current_hour.isoformat()
        
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp >= current_hour]
        
        if recent_metrics:
            self.hourly_metrics[hour_key] = {
                'avg_hit_rate': np.mean([m.hit_rate for m in recent_metrics]),
                'total_queries': sum(m.total_queries for m in recent_metrics),
                'avg_response_time': np.mean([m.avg_retrieval_time_ms for m in recent_metrics])
            }
    
    def track_query_pattern(self, query: str, pattern: str, cache_hit: bool, 
                           response_time_ms: float):
        """Track metrics for specific query patterns."""
        if not self.pattern_tracking_enabled:
            return
        
        if pattern not in self.query_patterns:
            self.query_patterns[pattern] = QueryPatternMetrics(
                pattern=pattern,
                query_count=0,
                cache_hits=0,
                cache_misses=0,
                avg_response_time_ms=0.0,
                last_accessed=datetime.now()
            )
        
        metrics = self.query_patterns[pattern]
        metrics.query_count += 1
        if cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1
        
        # Update moving average of response time
        alpha = 0.1  # Smoothing factor
        metrics.avg_response_time_ms = (
            alpha * response_time_ms + 
            (1 - alpha) * metrics.avg_response_time_ms
        )
        metrics.last_accessed = datetime.now()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current cache metrics summary."""
        if not self.last_snapshot:
            return {}
        
        return {
            'current': self.last_snapshot.to_dict(),
            'performance_indicators': self.performance_indicators.copy(),
            'hourly_metrics': dict(self.hourly_metrics),
            'top_patterns': self._get_top_patterns()
        }
    
    def _get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top query patterns by hit rate."""
        patterns = sorted(
            self.query_patterns.values(),
            key=lambda p: p.hit_rate,
            reverse=True
        )[:limit]
        
        return [
            {
                'pattern': p.pattern,
                'hit_rate': p.hit_rate,
                'query_count': p.query_count,
                'avg_response_time_ms': p.avg_response_time_ms
            }
            for p in patterns
        ]
    
    def get_historical_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            snapshot.to_dict()
            for snapshot in self.metrics_history
            if snapshot.timestamp >= cutoff_time
        ]
    
    async def save_metrics(self):
        """Save metrics to disk for persistence."""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'metrics_history': [s.to_dict() for s in self.metrics_history],
                'performance_indicators': self.performance_indicators,
                'query_patterns': {
                    pattern: {
                        'pattern': metrics.pattern,
                        'query_count': metrics.query_count,
                        'cache_hits': metrics.cache_hits,
                        'cache_misses': metrics.cache_misses,
                        'hit_rate': metrics.hit_rate,
                        'avg_response_time_ms': metrics.avg_response_time_ms,
                        'last_accessed': metrics.last_accessed.isoformat()
                    }
                    for pattern, metrics in self.query_patterns.items()
                },
                'hourly_metrics': dict(self.hourly_metrics)
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    async def load_metrics(self):
        """Load historical metrics from disk."""
        if not self.metrics_file.exists():
            return
        
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            # Restore metrics history
            for metric_dict in data.get('metrics_history', []):
                metric_dict['timestamp'] = datetime.fromisoformat(metric_dict['timestamp'])
                self.metrics_history.append(CacheMetricSnapshot(**metric_dict))
            
            # Restore performance indicators
            self.performance_indicators.update(data.get('performance_indicators', {}))
            
            # Restore query patterns
            for pattern, pattern_data in data.get('query_patterns', {}).items():
                pattern_data['last_accessed'] = datetime.fromisoformat(
                    pattern_data['last_accessed']
                )
                self.query_patterns[pattern] = QueryPatternMetrics(**pattern_data)
            
            # Restore hourly metrics
            self.hourly_metrics.update(data.get('hourly_metrics', {}))
            
            self.logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
    
    def calculate_learning_effectiveness(self) -> Dict[str, float]:
        """
        Calculate metrics that demonstrate learning effectiveness.
        Important for dissertation evaluation.
        """
        if len(self.metrics_history) < 20:
            return {
                'learning_rate': 0.0,
                'performance_gain': 0.0,
                'stability_score': 0.0
            }
        
        # Calculate learning rate (improvement in hit rate over time)
        early_metrics = list(self.metrics_history)[:10]
        recent_metrics = list(self.metrics_history)[-10:]
        
        early_hit_rate = np.mean([m.hit_rate for m in early_metrics])
        recent_hit_rate = np.mean([m.hit_rate for m in recent_metrics])
        
        learning_rate = (recent_hit_rate - early_hit_rate) / max(early_hit_rate, 0.01)
        
        # Calculate performance gain (response time improvement)
        early_response = np.mean([m.avg_retrieval_time_ms for m in early_metrics])
        recent_response = np.mean([m.avg_retrieval_time_ms for m in recent_metrics])
        
        performance_gain = (early_response - recent_response) / max(early_response, 0.01)
        
        # Calculate stability (low variance in recent performance)
        recent_hit_rates = [m.hit_rate for m in recent_metrics]
        stability_score = 1.0 - min(np.std(recent_hit_rates), 1.0)
        
        return {
            'learning_rate': learning_rate,
            'performance_gain': performance_gain,
            'stability_score': stability_score,
            'overall_effectiveness': (learning_rate + performance_gain + stability_score) / 3
        }