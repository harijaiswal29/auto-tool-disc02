"""
Performance regression detection module using statistical algorithms.

This module provides real-time detection of performance regressions using
multiple statistical methods including CUSUM, EWMA, and Z-score detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Performance regression alert details."""
    timestamp: datetime
    metric_name: str
    detection_method: str
    severity: str  # 'info', 'warning', 'critical'
    current_value: float
    baseline_value: float
    deviation: float
    confidence: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for a metric."""
    metric_name: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: datetime
    window_size: int = 100
    
    def update(self, value: float):
        """Update baseline with exponential moving average."""
        alpha = 2 / (self.window_size + 1)
        self.mean = alpha * value + (1 - alpha) * self.mean
        
        # Update std dev using Welford's method approximation
        diff = value - self.mean
        self.std_dev = np.sqrt(alpha * diff**2 + (1 - alpha) * self.std_dev**2)
        
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.sample_count += 1
        self.last_updated = datetime.now()


class PerformanceRegressionDetector:
    """Detects performance regressions using multiple statistical methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.metric_history: Dict[str, deque] = {}
        self.alerts: List[RegressionAlert] = []
        
        # Detection thresholds
        self.z_score_threshold = self.config.get('z_score_threshold', 2.5)
        self.cusum_threshold = self.config.get('cusum_threshold', 5.0)
        self.ewma_lambda = self.config.get('ewma_lambda', 0.2)
        self.ewma_L = self.config.get('ewma_L', 3.0)  # Control limit multiplier
        
        # Baseline settings
        self.baseline_window = self.config.get('baseline_window', 100)
        self.min_samples_for_detection = self.config.get('min_samples', 20)
        
        # CUSUM parameters
        self.cusum_states: Dict[str, Dict[str, float]] = {}
        
        # EWMA parameters
        self.ewma_states: Dict[str, Dict[str, float]] = {}
        
    def update_metric(self, metric_name: str, value: float, 
                     timestamp: Optional[datetime] = None) -> List[RegressionAlert]:
        """
        Update metric and check for regressions.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            List of regression alerts generated
        """
        timestamp = timestamp or datetime.now()
        
        # Initialize tracking if needed
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = deque(maxlen=self.baseline_window * 2)
            
        # Add to history
        self.metric_history[metric_name].append((timestamp, value))
        
        # Update or create baseline
        if metric_name not in self.baselines:
            if len(self.metric_history[metric_name]) >= self.min_samples_for_detection:
                self._initialize_baseline(metric_name)
            else:
                return []  # Not enough data yet
        else:
            self.baselines[metric_name].update(value)
        
        # Check for regressions
        alerts = []
        
        # Skip detection if not enough samples
        if len(self.metric_history[metric_name]) < self.min_samples_for_detection:
            return alerts
        
        # Z-score detection
        z_alert = self._detect_zscore_regression(metric_name, value)
        if z_alert:
            alerts.append(z_alert)
        
        # CUSUM detection
        cusum_alert = self._detect_cusum_regression(metric_name, value)
        if cusum_alert:
            alerts.append(cusum_alert)
        
        # EWMA detection
        ewma_alert = self._detect_ewma_regression(metric_name, value)
        if ewma_alert:
            alerts.append(ewma_alert)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    def _initialize_baseline(self, metric_name: str):
        """Initialize baseline from historical data."""
        values = [v for _, v in self.metric_history[metric_name]]
        
        self.baselines[metric_name] = PerformanceBaseline(
            metric_name=metric_name,
            mean=np.mean(values),
            std_dev=np.std(values),
            min_value=np.min(values),
            max_value=np.max(values),
            sample_count=len(values),
            last_updated=datetime.now(),
            window_size=self.baseline_window
        )
        
        # Initialize CUSUM state
        self.cusum_states[metric_name] = {
            'S_high': 0.0,
            'S_low': 0.0
        }
        
        # Initialize EWMA state
        self.ewma_states[metric_name] = {
            'ewma': self.baselines[metric_name].mean,
            'sigma': self.baselines[metric_name].std_dev
        }
    
    def _detect_zscore_regression(self, metric_name: str, value: float) -> Optional[RegressionAlert]:
        """Detect regression using Z-score method."""
        baseline = self.baselines[metric_name]
        
        if baseline.std_dev == 0:
            return None
        
        z_score = (value - baseline.mean) / baseline.std_dev
        
        # For performance metrics, lower is usually better, so positive z-score is bad
        if z_score > self.z_score_threshold:
            severity = self._calculate_severity(z_score, self.z_score_threshold)
            return RegressionAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                detection_method='z-score',
                severity=severity,
                current_value=value,
                baseline_value=baseline.mean,
                deviation=z_score,
                confidence=min(0.99, abs(z_score) / 5.0),
                message=f"Performance regression detected: {metric_name} is {z_score:.2f} std devs above baseline",
                metadata={'z_score': z_score, 'threshold': self.z_score_threshold}
            )
        
        return None
    
    def _detect_cusum_regression(self, metric_name: str, value: float) -> Optional[RegressionAlert]:
        """Detect regression using CUSUM (Cumulative Sum) method."""
        baseline = self.baselines[metric_name]
        state = self.cusum_states[metric_name]
        
        if baseline.std_dev == 0:
            return None
        
        # CUSUM parameters
        k = 0.5 * baseline.std_dev  # Slack parameter
        h = self.cusum_threshold * baseline.std_dev  # Decision threshold
        
        # Update CUSUM statistics
        state['S_high'] = max(0, state['S_high'] + (value - baseline.mean - k))
        state['S_low'] = max(0, state['S_low'] + (baseline.mean - k - value))
        
        # Check for regression (performance degradation = higher values)
        if state['S_high'] > h:
            severity = self._calculate_severity(state['S_high'] / h, 1.0)
            
            # Reset CUSUM after detection
            state['S_high'] = 0
            
            return RegressionAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                detection_method='CUSUM',
                severity=severity,
                current_value=value,
                baseline_value=baseline.mean,
                deviation=state['S_high'] / baseline.std_dev,
                confidence=min(0.95, state['S_high'] / (2 * h)),
                message=f"CUSUM detected sustained performance regression in {metric_name}",
                metadata={
                    'cusum_high': state['S_high'],
                    'threshold': h,
                    'k': k
                }
            )
        
        return None
    
    def _detect_ewma_regression(self, metric_name: str, value: float) -> Optional[RegressionAlert]:
        """Detect regression using EWMA (Exponentially Weighted Moving Average) method."""
        baseline = self.baselines[metric_name]
        state = self.ewma_states[metric_name]
        
        if baseline.std_dev == 0:
            return None
        
        # Update EWMA
        state['ewma'] = self.ewma_lambda * value + (1 - self.ewma_lambda) * state['ewma']
        
        # Calculate control limits
        n = baseline.sample_count
        sigma_ewma = baseline.std_dev * np.sqrt(self.ewma_lambda / (2 - self.ewma_lambda) * 
                                               (1 - (1 - self.ewma_lambda)**(2 * n)))
        
        upper_limit = baseline.mean + self.ewma_L * sigma_ewma
        lower_limit = baseline.mean - self.ewma_L * sigma_ewma
        
        # Check for regression
        if state['ewma'] > upper_limit:
            deviation = (state['ewma'] - baseline.mean) / sigma_ewma
            severity = self._calculate_severity(deviation, self.ewma_L)
            
            return RegressionAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                detection_method='EWMA',
                severity=severity,
                current_value=value,
                baseline_value=baseline.mean,
                deviation=deviation,
                confidence=min(0.90, deviation / (2 * self.ewma_L)),
                message=f"EWMA detected trending performance regression in {metric_name}",
                metadata={
                    'ewma': state['ewma'],
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit,
                    'sigma_ewma': sigma_ewma
                }
            )
        
        return None
    
    def _calculate_severity(self, deviation: float, threshold: float) -> str:
        """Calculate alert severity based on deviation magnitude."""
        ratio = abs(deviation) / threshold
        
        if ratio < 1.5:
            return 'info'
        elif ratio < 2.0:
            return 'warning'
        else:
            return 'critical'
    
    def get_baseline_stats(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get baseline statistics for a metric."""
        if metric_name not in self.baselines:
            return None
        
        baseline = self.baselines[metric_name]
        return {
            'mean': baseline.mean,
            'std_dev': baseline.std_dev,
            'min': baseline.min_value,
            'max': baseline.max_value,
            'sample_count': baseline.sample_count,
            'last_updated': baseline.last_updated.isoformat()
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[RegressionAlert]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]
    
    def reset_baseline(self, metric_name: str):
        """Reset baseline for a metric."""
        if metric_name in self.baselines:
            del self.baselines[metric_name]
        if metric_name in self.cusum_states:
            del self.cusum_states[metric_name]
        if metric_name in self.ewma_states:
            del self.ewma_states[metric_name]
        
        logger.info(f"Reset baseline for metric: {metric_name}")