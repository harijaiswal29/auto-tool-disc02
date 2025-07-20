"""
Real-time monitoring service for continuous performance analysis.

This module provides a background service that continuously monitors
system performance and detects regressions in real-time.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import aiohttp
import websockets

from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.performance_regression_detector import PerformanceRegressionDetector, RegressionAlert
from src.evaluation.alert_manager import AlertManager
from src.evaluation.metrics_collector import MetricsCollector
from src.agents.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger(__name__)


class RealtimeMonitor:
    """Service for real-time performance monitoring and regression detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_enabled = False
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Core components
        self.evaluation_engine = EvaluationEngine(config)
        self.orchestrator_agent = None  # Will be set when monitoring starts
        
        # Monitoring configuration
        self.monitor_config = config.get('realtime_monitoring', {})
        self.update_interval = self.monitor_config.get('update_interval', 1.0)
        self.metrics_window = self.monitor_config.get('metrics_window', 300)  # 5 minutes
        self.anomaly_threshold = self.monitor_config.get('anomaly_threshold', 3.0)
        
        # Performance tracking
        self.performance_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history = deque(maxlen=100)
        self.monitoring_task = None
        
        # WebSocket server configuration
        self.ws_host = self.monitor_config.get('ws_host', 'localhost')
        self.ws_port = self.monitor_config.get('ws_port', 8765)
        self.ws_server = None
        
    async def start(self, orchestrator_agent: OrchestratorAgent):
        """Start the real-time monitoring service."""
        self.orchestrator_agent = orchestrator_agent
        self.monitoring_enabled = True
        
        logger.info("Starting real-time monitoring service")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start WebSocket server for real-time dashboard
        if self.monitor_config.get('enable_websocket', True):
            self.ws_server = await websockets.serve(
                self._handle_websocket_client,
                self.ws_host,
                self.ws_port
            )
            logger.info(f"WebSocket server started on ws://{self.ws_host}:{self.ws_port}")
        
        # Enable online monitoring in evaluation engine
        self.evaluation_engine.enable_online_monitoring()
        
        # Start evaluation engine's online evaluation
        asyncio.create_task(
            self.evaluation_engine.run_online_evaluation(
                orchestrator_callback=self._get_orchestrator_performance,
                update_interval=self.update_interval
            )
        )
    
    async def stop(self):
        """Stop the real-time monitoring service."""
        self.monitoring_enabled = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connections
        for client in self.websocket_clients.copy():
            await client.close()
        
        # Stop WebSocket server
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
        
        # Disable online monitoring
        self.evaluation_engine.disable_online_monitoring()
        
        logger.info("Real-time monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Collect current metrics
                metrics = await self._collect_current_metrics()
                
                # Update performance buffers
                self._update_performance_buffers(metrics)
                
                # Check for anomalies
                anomalies = self._detect_anomalies(metrics)
                
                # Process anomalies
                for anomaly in anomalies:
                    await self._process_anomaly(anomaly)
                
                # Send updates to WebSocket clients
                await self._broadcast_metrics(metrics)
                
                # Sleep until next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {}
        }
        
        # Get metrics for each strategy
        for strategy_name in self.evaluation_engine.strategies:
            strategy_metrics = {
                'performance': await self._get_strategy_performance(strategy_name),
                'resource_usage': await self._get_resource_usage(strategy_name),
                'trends': self.evaluation_engine.metrics_collector.get_performance_trends(
                    strategy_name
                )
            }
            metrics['strategies'][strategy_name] = strategy_metrics
        
        # Get system-wide metrics
        metrics['system'] = {
            'active_alerts': len(self.evaluation_engine.get_regression_alerts(1)),
            'total_episodes': sum(
                len(self.evaluation_engine.metrics_collector.episode_metrics[s])
                for s in self.evaluation_engine.strategies
            )
        }
        
        return metrics
    
    async def _get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """Get current performance metrics for a strategy."""
        # Get latest performance from metrics collector
        recent_metrics = self.evaluation_engine.metrics_collector.get_real_time_metrics(
            f"{strategy_name}_reward",
            time_window=timedelta(seconds=60)
        )
        
        if recent_metrics:
            recent_rewards = [m['value'] for m in recent_metrics]
            return {
                'current_reward': recent_rewards[-1] if recent_rewards else 0,
                'avg_reward': sum(recent_rewards) / len(recent_rewards),
                'min_reward': min(recent_rewards),
                'max_reward': max(recent_rewards)
            }
        
        return {
            'current_reward': 0,
            'avg_reward': 0,
            'min_reward': 0,
            'max_reward': 0
        }
    
    async def _get_resource_usage(self, strategy_name: str) -> Dict[str, float]:
        """Get resource usage metrics for a strategy."""
        # This would integrate with actual resource monitoring
        # For now, return mock data
        return {
            'cpu_percent': 0,
            'memory_mb': 0,
            'execution_time_ms': 0
        }
    
    async def _get_orchestrator_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get performance data from orchestrator."""
        if not self.orchestrator_agent:
            return None
        
        # Get latest execution results
        # This would integrate with the actual orchestrator
        # For now, return mock data for demonstration
        return {
            'reward': 0.85,
            'selection_time': 0.05,
            'regret': 0.15,
            'tools_selected': ['tool1', 'tool2']
        }
    
    def _update_performance_buffers(self, metrics: Dict[str, Any]):
        """Update performance tracking buffers."""
        timestamp = datetime.now()
        
        for strategy_name, strategy_metrics in metrics['strategies'].items():
            perf = strategy_metrics['performance']
            
            # Store in buffer
            self.performance_buffer[strategy_name].append({
                'timestamp': timestamp,
                'reward': perf['current_reward'],
                'avg_reward': perf['avg_reward']
            })
    
    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        for strategy_name, strategy_metrics in metrics['strategies'].items():
            # Check each metric for anomalies
            metric_name = f"{strategy_name}_reward"
            current_value = strategy_metrics['performance']['current_reward']
            
            # Calculate deviation
            deviation = self.evaluation_engine.metrics_collector.calculate_metric_deviation(
                metric_name, current_value
            )
            
            if deviation and abs(deviation) > self.anomaly_threshold:
                anomalies.append({
                    'strategy': strategy_name,
                    'metric': 'reward',
                    'value': current_value,
                    'deviation': deviation,
                    'timestamp': datetime.now()
                })
        
        return anomalies
    
    async def _process_anomaly(self, anomaly: Dict[str, Any]):
        """Process detected anomaly."""
        # Create alert
        alert = RegressionAlert(
            timestamp=anomaly['timestamp'],
            metric_name=f"{anomaly['strategy']}_{anomaly['metric']}",
            detection_method='anomaly_detection',
            severity='warning' if abs(anomaly['deviation']) < 4 else 'critical',
            current_value=anomaly['value'],
            baseline_value=0,  # Will be filled by baseline
            deviation=anomaly['deviation'],
            confidence=min(0.99, abs(anomaly['deviation']) / 5),
            message=f"Anomaly detected in {anomaly['strategy']} {anomaly['metric']}: "
                   f"{anomaly['deviation']:.2f} std devs from baseline",
            metadata={'anomaly': anomaly}
        )
        
        # Process through alert manager
        await self.evaluation_engine.alert_manager.process_alert(alert)
        
        # Store in history
        self.alert_history.append(alert)
    
    async def _broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics to WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            'type': 'metrics_update',
            'data': metrics
        })
        
        # Send to all connected clients
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending to WebSocket client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def _handle_websocket_client(self, websocket, path):
        """Handle WebSocket client connections."""
        # Register client
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected from {websocket.remote_address}")
        
        try:
            # Send initial state
            await websocket.send(json.dumps({
                'type': 'connection',
                'status': 'connected',
                'monitoring_enabled': self.monitoring_enabled
            }))
            
            # Handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Unregister client
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected from {websocket.remote_address}")
    
    async def _handle_client_message(self, websocket, data: Dict[str, Any]):
        """Handle message from WebSocket client."""
        msg_type = data.get('type')
        
        if msg_type == 'get_alerts':
            # Send recent alerts
            alerts = self.evaluation_engine.get_regression_alerts(
                hours=data.get('hours', 24)
            )
            await websocket.send(json.dumps({
                'type': 'alerts',
                'data': [self._serialize_alert(a) for a in alerts]
            }))
            
        elif msg_type == 'get_stats':
            # Send alert statistics
            stats = self.evaluation_engine.get_alert_statistics()
            await websocket.send(json.dumps({
                'type': 'stats',
                'data': stats
            }))
            
        elif msg_type == 'acknowledge_alert':
            # Acknowledge alert
            alert_index = data.get('alert_index')
            acknowledged_by = data.get('user', 'unknown')
            
            success = self.evaluation_engine.alert_manager.acknowledge_alert(
                alert_index, acknowledged_by
            )
            
            await websocket.send(json.dumps({
                'type': 'acknowledge_result',
                'success': success
            }))
            
        elif msg_type == 'reset_baseline':
            # Reset baseline for strategy
            strategy = data.get('strategy')
            if strategy:
                self.evaluation_engine.reset_performance_baseline(strategy)
                
            await websocket.send(json.dumps({
                'type': 'baseline_reset',
                'strategy': strategy
            }))
    
    def _serialize_alert(self, alert: RegressionAlert) -> Dict[str, Any]:
        """Serialize alert for JSON transmission."""
        return {
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
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'enabled': self.monitoring_enabled,
            'websocket_clients': len(self.websocket_clients),
            'strategies_monitored': len(self.evaluation_engine.strategies),
            'active_alerts': len(self.evaluation_engine.get_regression_alerts(1)),
            'performance_buffer_size': {
                name: len(buffer) 
                for name, buffer in self.performance_buffer.items()
            }
        }