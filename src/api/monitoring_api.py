"""
API endpoints for real-time performance monitoring.

This module provides RESTful and WebSocket endpoints for accessing
real-time performance metrics and managing alerts.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio

from src.evaluation.realtime_monitor import RealtimeMonitor
from src.evaluation.evaluation_engine import EvaluationEngine

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Global reference to monitoring service
monitor_service: Optional[RealtimeMonitor] = None
evaluation_engine: Optional[EvaluationEngine] = None


def set_monitor_service(service: RealtimeMonitor):
    """Set the global monitor service instance."""
    global monitor_service
    monitor_service = service
    if service:
        global evaluation_engine
        evaluation_engine = service.evaluation_engine


@router.get("/performance")
async def get_current_performance() -> Dict[str, Any]:
    """
    Get current performance metrics for all strategies.
    
    Returns:
        Current performance data including rewards, trends, and resource usage
    """
    if not monitor_service:
        raise HTTPException(status_code=503, detail="Monitoring service not available")
    
    metrics = await monitor_service._collect_current_metrics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "monitoring_enabled": monitor_service.monitoring_enabled,
        "strategies": metrics.get("strategies", {}),
        "system": metrics.get("system", {})
    }


@router.get("/performance/{strategy_name}")
async def get_strategy_performance(
    strategy_name: str,
    time_window: int = 300  # seconds
) -> Dict[str, Any]:
    """
    Get performance metrics for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy
        time_window: Time window in seconds (default: 300)
        
    Returns:
        Performance metrics for the strategy
    """
    if not evaluation_engine:
        raise HTTPException(status_code=503, detail="Evaluation engine not available")
    
    # Check if strategy exists
    if strategy_name not in evaluation_engine.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
    
    # Get real-time metrics
    reward_metrics = evaluation_engine.metrics_collector.get_real_time_metrics(
        f"{strategy_name}_reward",
        time_window=timedelta(seconds=time_window)
    )
    
    time_metrics = evaluation_engine.metrics_collector.get_real_time_metrics(
        f"{strategy_name}_selection_time",
        time_window=timedelta(seconds=time_window)
    )
    
    # Get trends
    trends = evaluation_engine.metrics_collector.get_performance_trends(strategy_name)
    
    # Get baseline
    baseline = evaluation_engine.metrics_collector.get_metric_baseline(
        f"{strategy_name}_reward"
    )
    
    return {
        "strategy": strategy_name,
        "time_window": time_window,
        "metrics": {
            "reward": {
                "current": reward_metrics[-1]["value"] if reward_metrics else None,
                "average": sum(m["value"] for m in reward_metrics) / len(reward_metrics) if reward_metrics else 0,
                "count": len(reward_metrics)
            },
            "selection_time": {
                "current": time_metrics[-1]["value"] if time_metrics else None,
                "average": sum(m["value"] for m in time_metrics) / len(time_metrics) if time_metrics else 0,
                "count": len(time_metrics)
            }
        },
        "trends": trends,
        "baseline": baseline
    }


@router.get("/alerts")
async def get_alerts(
    hours: int = 24,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Get performance regression alerts.
    
    Args:
        hours: Time window in hours (default: 24)
        severity: Filter by severity (info, warning, critical)
        acknowledged: Filter by acknowledgment status
        
    Returns:
        List of alerts and statistics
    """
    if not evaluation_engine:
        raise HTTPException(status_code=503, detail="Evaluation engine not available")
    
    # Get alerts
    all_alerts = evaluation_engine.get_regression_alerts(hours)
    
    # Filter alerts
    filtered_alerts = []
    for alert in all_alerts:
        if severity and alert.severity != severity:
            continue
        
        # Check acknowledgment status if needed
        # (would need to track this in alert manager)
        
        filtered_alerts.append({
            "timestamp": alert.timestamp.isoformat(),
            "metric_name": alert.metric_name,
            "detection_method": alert.detection_method,
            "severity": alert.severity,
            "current_value": alert.current_value,
            "baseline_value": alert.baseline_value,
            "deviation": alert.deviation,
            "confidence": alert.confidence,
            "message": alert.message,
            "metadata": alert.metadata
        })
    
    # Get statistics
    stats = evaluation_engine.get_alert_statistics()
    
    return {
        "alerts": filtered_alerts,
        "statistics": stats,
        "filter": {
            "hours": hours,
            "severity": severity,
            "acknowledged": acknowledged
        }
    }


@router.post("/alerts/acknowledge")
async def acknowledge_alert(
    alert_index: int,
    user: str = "api_user"
) -> Dict[str, Any]:
    """
    Acknowledge a performance alert.
    
    Args:
        alert_index: Index of alert to acknowledge
        user: User acknowledging the alert
        
    Returns:
        Success status
    """
    if not evaluation_engine:
        raise HTTPException(status_code=503, detail="Evaluation engine not available")
    
    success = evaluation_engine.alert_manager.acknowledge_alert(alert_index, user)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    
    return {
        "success": True,
        "alert_index": alert_index,
        "acknowledged_by": user,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/baseline/reset")
async def reset_baseline(strategy_name: str) -> Dict[str, Any]:
    """
    Reset performance baseline for a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Success status
    """
    if not evaluation_engine:
        raise HTTPException(status_code=503, detail="Evaluation engine not available")
    
    if strategy_name not in evaluation_engine.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
    
    evaluation_engine.reset_performance_baseline(strategy_name)
    
    return {
        "success": True,
        "strategy": strategy_name,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/anomalies")
async def get_anomalies(
    metric_name: str,
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Get detected anomalies for a metric.
    
    Args:
        metric_name: Name of the metric
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        List of detected anomalies
    """
    if not evaluation_engine:
        raise HTTPException(status_code=503, detail="Evaluation engine not available")
    
    anomalies = evaluation_engine.metrics_collector.detect_anomalies(
        metric_name, threshold
    )
    
    return {
        "metric": metric_name,
        "threshold": threshold,
        "anomalies": [
            {
                "timestamp": a["timestamp"].isoformat(),
                "value": a["value"],
                "z_score": a["z_score"],
                "baseline_mean": a["baseline_mean"],
                "baseline_std": a["baseline_std"]
            }
            for a in anomalies
        ]
    }


@router.get("/status")
async def get_monitoring_status() -> Dict[str, Any]:
    """
    Get monitoring service status.
    
    Returns:
        Current status of the monitoring service
    """
    if not monitor_service:
        return {
            "status": "unavailable",
            "message": "Monitoring service not initialized"
        }
    
    status = monitor_service.get_monitoring_status()
    
    return {
        "status": "active" if status["enabled"] else "inactive",
        "details": status
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metric streaming.
    
    Clients can subscribe to real-time updates and interact with the monitoring system.
    """
    await websocket.accept()
    
    if not monitor_service:
        await websocket.send_json({
            "type": "error",
            "message": "Monitoring service not available"
        })
        await websocket.close()
        return
    
    # Add to monitor service's clients
    monitor_service.websocket_clients.add(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_json()
                
                # Process different message types
                if data.get("type") == "subscribe":
                    # Subscribe to specific metrics
                    metrics = data.get("metrics", [])
                    interval = data.get("interval", 1.0)
                    
                    # Start streaming metrics
                    async def stream_callback(updates):
                        await websocket.send_json({
                            "type": "metrics",
                            "data": updates
                        })
                    
                    asyncio.create_task(
                        evaluation_engine.metrics_collector.stream_metrics(
                            stream_callback, metrics, interval
                        )
                    )
                    
                elif data.get("type") == "get_alerts":
                    # Send recent alerts
                    alerts = evaluation_engine.get_regression_alerts(
                        data.get("hours", 24)
                    )
                    
                    await websocket.send_json({
                        "type": "alerts",
                        "data": [
                            {
                                "timestamp": a.timestamp.isoformat(),
                                "metric_name": a.metric_name,
                                "severity": a.severity,
                                "message": a.message
                            }
                            for a in alerts
                        ]
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    finally:
        # Remove from clients
        monitor_service.websocket_clients.discard(websocket)