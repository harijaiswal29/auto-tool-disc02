"""
Alert management system for performance regression notifications.

This module handles alert generation, routing, suppression, and delivery
through multiple notification channels.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import aiofiles
import aiohttp

from src.evaluation.performance_regression_detector import RegressionAlert

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert routing and suppression rule."""
    name: str
    condition: Callable[[RegressionAlert], bool]
    channels: List[str]
    suppression_window: Optional[timedelta] = None
    severity_filter: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertHistory:
    """Historical record of an alert."""
    alert: RegressionAlert
    delivered_at: datetime
    channels: List[str]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class AlertManager:
    """Manages alert delivery and suppression."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_history: deque = deque(maxlen=1000)
        self.suppression_cache: Dict[str, datetime] = {}
        self.rules: List[AlertRule] = []
        self.channels: Dict[str, Callable] = {}
        
        # Configure channels
        self._setup_channels()
        
        # Load rules
        self._load_rules()
        
        # Alert statistics
        self.stats = defaultdict(int)
        
    def _setup_channels(self):
        """Setup notification channels."""
        # Log channel (always enabled)
        self.channels['log'] = self._send_to_log
        
        # File channel
        if self.config.get('file_channel', {}).get('enabled', True):
            self.channels['file'] = self._send_to_file
        
        # Webhook channel
        if self.config.get('webhook_channel', {}).get('enabled', False):
            self.channels['webhook'] = self._send_to_webhook
        
        # Email channel (placeholder)
        if self.config.get('email_channel', {}).get('enabled', False):
            self.channels['email'] = self._send_to_email
    
    def _load_rules(self):
        """Load alert routing rules."""
        # Default rule: all alerts to log
        self.add_rule(AlertRule(
            name='default',
            condition=lambda alert: True,
            channels=['log'],
            suppression_window=timedelta(minutes=5)
        ))
        
        # Critical alerts to all channels
        self.add_rule(AlertRule(
            name='critical',
            condition=lambda alert: alert.severity == 'critical',
            channels=['log', 'file', 'webhook'],
            suppression_window=timedelta(minutes=1),
            severity_filter={'critical'}
        ))
        
        # Warning alerts
        self.add_rule(AlertRule(
            name='warning',
            condition=lambda alert: alert.severity == 'warning',
            channels=['log', 'file'],
            suppression_window=timedelta(minutes=10),
            severity_filter={'warning'}
        ))
        
        # Info alerts
        self.add_rule(AlertRule(
            name='info',
            condition=lambda alert: alert.severity == 'info',
            channels=['log'],
            suppression_window=timedelta(minutes=30),
            severity_filter={'info'}
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert routing rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    async def process_alert(self, alert: RegressionAlert) -> bool:
        """
        Process an alert through routing rules.
        
        Args:
            alert: The alert to process
            
        Returns:
            True if alert was delivered, False if suppressed
        """
        # Check suppression
        if self._is_suppressed(alert):
            self.stats['suppressed'] += 1
            logger.debug(f"Alert suppressed: {alert.metric_name} - {alert.detection_method}")
            return False
        
        # Find matching rules
        matching_rules = [rule for rule in self.rules if rule.condition(alert)]
        
        if not matching_rules:
            logger.warning(f"No matching rules for alert: {alert}")
            return False
        
        # Collect unique channels
        channels = set()
        for rule in matching_rules:
            channels.update(rule.channels)
        
        # Send to channels
        delivered = False
        for channel in channels:
            if channel in self.channels:
                try:
                    await self.channels[channel](alert)
                    delivered = True
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel}: {e}")
        
        if delivered:
            # Update suppression cache
            self._update_suppression_cache(alert)
            
            # Record history
            self.alert_history.append(AlertHistory(
                alert=alert,
                delivered_at=datetime.now(),
                channels=list(channels)
            ))
            
            self.stats['delivered'] += 1
            self.stats[f'severity_{alert.severity}'] += 1
        
        return delivered
    
    def _is_suppressed(self, alert: RegressionAlert) -> bool:
        """Check if alert should be suppressed."""
        # Create suppression key
        key = f"{alert.metric_name}:{alert.detection_method}:{alert.severity}"
        
        if key in self.suppression_cache:
            last_sent = self.suppression_cache[key]
            
            # Find applicable suppression window
            for rule in self.rules:
                if rule.condition(alert) and rule.suppression_window:
                    if datetime.now() - last_sent < rule.suppression_window:
                        return True
        
        return False
    
    def _update_suppression_cache(self, alert: RegressionAlert):
        """Update suppression cache after sending alert."""
        key = f"{alert.metric_name}:{alert.detection_method}:{alert.severity}"
        self.suppression_cache[key] = datetime.now()
        
        # Clean old entries
        cutoff = datetime.now() - timedelta(hours=24)
        self.suppression_cache = {
            k: v for k, v in self.suppression_cache.items()
            if v > cutoff
        }
    
    async def _send_to_log(self, alert: RegressionAlert):
        """Send alert to log."""
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.ERROR
        }.get(alert.severity, logging.INFO)
        
        logger.log(
            log_level,
            f"PERFORMANCE ALERT: {alert.message} | "
            f"Method: {alert.detection_method} | "
            f"Current: {alert.current_value:.3f} | "
            f"Baseline: {alert.baseline_value:.3f} | "
            f"Deviation: {alert.deviation:.2f}"
        )
    
    async def _send_to_file(self, alert: RegressionAlert):
        """Send alert to file."""
        file_config = self.config.get('file_channel', {})
        file_path = file_config.get('path', 'alerts.log')
        
        alert_data = {
            'timestamp': alert.timestamp.isoformat(),
            'metric_name': alert.metric_name,
            'detection_method': alert.detection_method,
            'severity': alert.severity,
            'current_value': alert.current_value,
            'baseline_value': alert.baseline_value,
            'deviation': alert.deviation,
            'confidence': alert.confidence,
            'message': alert.message,
            'metadata': alert.metadata
        }
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(json.dumps(alert_data) + '\n')
    
    async def _send_to_webhook(self, alert: RegressionAlert):
        """Send alert to webhook."""
        webhook_config = self.config.get('webhook_channel', {})
        url = webhook_config.get('url')
        
        if not url:
            logger.error("Webhook URL not configured")
            return
        
        alert_data = asdict(alert)
        alert_data['timestamp'] = alert.timestamp.isoformat()
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=alert_data) as response:
                    if response.status != 200:
                        logger.error(f"Webhook returned status {response.status}")
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")
    
    async def _send_to_email(self, alert: RegressionAlert):
        """Send alert via email (placeholder)."""
        # This would integrate with an email service
        logger.info(f"Email alert (not implemented): {alert.message}")
    
    def acknowledge_alert(self, alert_index: int, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_index: Index in alert history
            acknowledged_by: User acknowledging the alert
            
        Returns:
            True if acknowledged successfully
        """
        try:
            if 0 <= alert_index < len(self.alert_history):
                history_item = self.alert_history[alert_index]
                history_item.acknowledged = True
                history_item.acknowledged_by = acknowledged_by
                history_item.acknowledged_at = datetime.now()
                
                self.stats['acknowledged'] += 1
                return True
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
        
        return False
    
    def get_active_alerts(self, hours: int = 24) -> List[AlertHistory]:
        """Get active (unacknowledged) alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            item for item in self.alert_history
            if item.delivered_at >= cutoff and not item.acknowledged
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_delivered': self.stats['delivered'],
            'total_suppressed': self.stats['suppressed'],
            'total_acknowledged': self.stats['acknowledged'],
            'active_alerts': len(active_alerts),
            'by_severity': {
                'info': self.stats.get('severity_info', 0),
                'warning': self.stats.get('severity_warning', 0),
                'critical': self.stats.get('severity_critical', 0)
            },
            'recent_alerts': [
                {
                    'timestamp': item.alert.timestamp.isoformat(),
                    'metric': item.alert.metric_name,
                    'severity': item.alert.severity,
                    'message': item.alert.message,
                    'acknowledged': item.acknowledged
                }
                for item in list(self.alert_history)[-10:]
            ]
        }
    
    def clear_history(self, before: Optional[datetime] = None):
        """Clear alert history before a given time."""
        if before:
            self.alert_history = deque(
                (item for item in self.alert_history if item.delivered_at >= before),
                maxlen=1000
            )
        else:
            self.alert_history.clear()
        
        logger.info("Cleared alert history")