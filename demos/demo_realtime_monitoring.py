#!/usr/bin/env python3
"""
Demo script for real-time performance monitoring and regression detection.

This script demonstrates:
1. Starting the real-time monitoring service
2. Simulating performance degradation
3. Detecting regressions using multiple algorithms
4. Generating and handling alerts
5. WebSocket-based real-time dashboard
"""

import asyncio
import sys
import os
import random
import numpy as np
from datetime import datetime, timedelta
import json
import websockets
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.realtime_monitor import RealtimeMonitor
from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.performance_regression_detector import PerformanceRegressionDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceSimulator:
    """Simulates performance metrics with various patterns."""
    
    def __init__(self):
        self.episode = 0
        self.degradation_start = 50
        self.recovery_start = 100
        
    def get_normal_performance(self) -> float:
        """Generate normal performance with some noise."""
        return 0.85 + random.gauss(0, 0.05)
    
    def get_degraded_performance(self) -> float:
        """Generate degraded performance."""
        base = 0.65 + random.gauss(0, 0.08)
        # Add gradual degradation
        degradation = (self.episode - self.degradation_start) * 0.002
        return max(0.3, base - degradation)
    
    def get_performance(self, strategy: str) -> float:
        """Get simulated performance for a strategy."""
        self.episode += 1
        
        if strategy == "q_learning":
            # Simulate sudden performance drop
            if 50 <= self.episode < 100:
                return self.get_degraded_performance()
            elif self.episode >= 100:
                # Recovery phase
                recovery = min(0.2, (self.episode - 100) * 0.004)
                return min(0.85, self.get_degraded_performance() + recovery)
            else:
                return self.get_normal_performance()
                
        elif strategy == "random":
            # Stable but low performance
            return 0.4 + random.gauss(0, 0.1)
            
        else:
            # Other strategies - stable performance
            return 0.7 + random.gauss(0, 0.06)


async def simulate_performance_data(monitor: RealtimeMonitor, simulator: PerformanceSimulator):
    """Simulate performance data for demonstration."""
    strategies = ["q_learning", "random", "greedy", "popular"]
    
    logger.info("Starting performance simulation...")
    
    for i in range(150):
        # Simulate metrics for each strategy
        for strategy in strategies:
            performance = simulator.get_performance(strategy)
            selection_time = 0.05 + random.gauss(0, 0.01)
            
            # Record metrics
            monitor.evaluation_engine.metrics_collector.record_real_time_metric(
                f"{strategy}_reward", performance
            )
            monitor.evaluation_engine.metrics_collector.record_real_time_metric(
                f"{strategy}_selection_time", selection_time
            )
            
            # Also update regression detector directly
            alerts = monitor.evaluation_engine.regression_detector.update_metric(
                f"{strategy}_reward", performance
            )
            
            # Process alerts
            for alert in alerts:
                await monitor.evaluation_engine.alert_manager.process_alert(alert)
                logger.info(f"🚨 Alert: {alert.message}")
        
        # Log progress
        if i % 10 == 0:
            logger.info(f"Episode {i}: Q-learning performance = {performance:.3f}")
        
        await asyncio.sleep(0.1)  # Small delay for demonstration


async def websocket_client_demo():
    """Demonstrate WebSocket client interaction."""
    await asyncio.sleep(5)  # Wait for server to start
    
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            logger.info("WebSocket client connected")
            
            # Get alerts
            await websocket.send(json.dumps({
                "type": "get_alerts",
                "hours": 1
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "alerts":
                logger.info(f"Received {len(data['data'])} alerts")
                for alert in data["data"][:3]:  # Show first 3
                    logger.info(f"  - {alert['message']}")
            
            # Get statistics
            await websocket.send(json.dumps({
                "type": "get_stats"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "stats":
                stats = data["data"]
                logger.info(f"Alert Statistics:")
                logger.info(f"  - Total delivered: {stats.get('total_delivered', 0)}")
                logger.info(f"  - Active alerts: {stats.get('active_alerts', 0)}")
                
    except Exception as e:
        logger.error(f"WebSocket client error: {e}")


async def main():
    """Main demo function."""
    logger.info("=== Real-time Performance Monitoring Demo ===")
    
    # Load configuration
    config_path = Path("config/config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = {}
    
    # Configure for demo
    config_dict['regression_detection'] = {
        'z_score_threshold': 2.0,  # More sensitive for demo
        'cusum_threshold': 3.0,
        'ewma_lambda': 0.3,
        'min_samples': 10  # Faster detection for demo
    }
    
    config_dict['alert_config'] = {
        'file_channel': {
            'enabled': True,
            'path': 'demo_alerts.log'
        },
        'webhook_channel': {
            'enabled': False
        }
    }
    
    config_dict['realtime_monitoring'] = {
        'update_interval': 0.5,
        'enable_websocket': True,
        'ws_host': 'localhost',
        'ws_port': 8765
    }
    
    # Create monitoring service
    monitor = RealtimeMonitor(config_dict)
    
    # Create performance simulator
    simulator = PerformanceSimulator()
    
    # Start monitoring service (mock orchestrator)
    class MockOrchestrator:
        pass
    
    mock_orchestrator = MockOrchestrator()
    await monitor.start(mock_orchestrator)
    
    logger.info("Monitoring service started")
    logger.info(f"WebSocket server at ws://localhost:8765")
    
    # Run simulation and client demo concurrently
    await asyncio.gather(
        simulate_performance_data(monitor, simulator),
        websocket_client_demo()
    )
    
    # Show final statistics
    logger.info("\n=== Final Statistics ===")
    
    # Get alert statistics
    stats = monitor.evaluation_engine.get_alert_statistics()
    logger.info(f"Total alerts delivered: {stats['total_delivered']}")
    logger.info(f"Alerts by severity:")
    for severity, count in stats['by_severity'].items():
        logger.info(f"  - {severity}: {count}")
    
    # Get recent alerts
    recent_alerts = monitor.evaluation_engine.get_regression_alerts(1)
    logger.info(f"\nRecent alerts ({len(recent_alerts)} total):")
    for alert in recent_alerts[:5]:
        logger.info(f"  - [{alert.severity}] {alert.detection_method}: {alert.message}")
    
    # Check baselines
    logger.info("\nPerformance baselines:")
    for strategy in ["q_learning", "random", "greedy"]:
        baseline = monitor.evaluation_engine.regression_detector.get_baseline_stats(
            f"{strategy}_reward"
        )
        if baseline:
            logger.info(f"  - {strategy}: mean={baseline['mean']:.3f}, "
                       f"std={baseline['std_dev']:.3f}")
    
    # Stop monitoring service
    await monitor.stop()
    logger.info("\nMonitoring service stopped")
    
    # Show alerts log
    if os.path.exists('demo_alerts.log'):
        logger.info("\nSample alerts from log file:")
        with open('demo_alerts.log', 'r') as f:
            lines = f.readlines()[-5:]  # Last 5 alerts
            for line in lines:
                alert_data = json.loads(line.strip())
                logger.info(f"  - {alert_data['timestamp']}: {alert_data['message']}")


if __name__ == "__main__":
    asyncio.run(main())