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
from src.monitoring.cache_metrics_monitor import CacheMetricsMonitor

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Global reference to monitoring service
monitor_service: Optional[RealtimeMonitor] = None
evaluation_engine: Optional[EvaluationEngine] = None
cache_monitor: Optional[CacheMetricsMonitor] = None


def set_monitor_service(service: RealtimeMonitor):
    """Set the global monitor service instance."""
    global monitor_service
    monitor_service = service
    if service:
        global evaluation_engine
        evaluation_engine = service.evaluation_engine


def set_cache_monitor(monitor: CacheMetricsMonitor):
    """Set the global cache monitor instance."""
    global cache_monitor
    cache_monitor = monitor


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


# Cache Monitoring Endpoints

@router.get("/cache/metrics")
async def get_cache_metrics() -> Dict[str, Any]:
    """
    Get current cache performance metrics.
    
    Returns:
        Current cache metrics including hit rate, performance indicators, and patterns
    """
    if not cache_monitor:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    metrics = cache_monitor.get_current_metrics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "monitoring_active": cache_monitor.monitoring_active,
        **metrics
    }


@router.get("/cache/history")
async def get_cache_history(
    hours: int = 1,
    resolution: Optional[str] = "raw"  # raw, hourly, daily
) -> Dict[str, Any]:
    """
    Get historical cache performance data.
    
    Args:
        hours: Number of hours of history to retrieve
        resolution: Data resolution (raw, hourly, daily)
        
    Returns:
        Historical cache metrics
    """
    if not cache_monitor:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    historical_data = cache_monitor.get_historical_metrics(hours=hours)
    
    # Apply resolution if needed
    if resolution == "hourly":
        # Group by hour
        hourly_data = {}
        for metric in historical_data:
            hour_key = datetime.fromisoformat(metric['timestamp']).replace(
                minute=0, second=0, microsecond=0
            ).isoformat()
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            hourly_data[hour_key].append(metric)
        
        # Average metrics per hour
        aggregated = []
        for hour, metrics in hourly_data.items():
            if metrics:
                aggregated.append({
                    'timestamp': hour,
                    'hit_rate': sum(m['hit_rate'] for m in metrics) / len(metrics),
                    'avg_retrieval_time_ms': sum(m['avg_retrieval_time_ms'] for m in metrics) / len(metrics),
                    'total_queries': sum(m['total_queries'] for m in metrics)
                })
        historical_data = aggregated
    
    return {
        "hours": hours,
        "resolution": resolution,
        "data_points": len(historical_data),
        "data": historical_data
    }


@router.get("/cache/patterns")
async def get_cache_patterns() -> Dict[str, Any]:
    """
    Get cache performance metrics grouped by query patterns.
    
    Returns:
        Pattern-based cache metrics
    """
    if not cache_monitor or not cache_monitor.cache_ref:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    pattern_metrics = cache_monitor.cache_ref.get_pattern_metrics()
    
    # Sort by hit rate
    sorted_patterns = sorted(
        pattern_metrics.items(),
        key=lambda x: x[1]['hit_rate'],
        reverse=True
    )
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_patterns": len(pattern_metrics),
        "patterns": [
            {
                "pattern": pattern,
                **metrics
            }
            for pattern, metrics in sorted_patterns[:20]  # Top 20 patterns
        ]
    }


@router.get("/cache/learning")
async def get_cache_learning_metrics() -> Dict[str, Any]:
    """
    Get cache learning effectiveness metrics.
    
    Returns:
        Learning effectiveness indicators for dissertation evaluation
    """
    if not cache_monitor:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    effectiveness = cache_monitor.calculate_learning_effectiveness()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "effectiveness_metrics": effectiveness,
        "interpretation": {
            "learning_rate": "Positive values indicate improving hit rate over time",
            "performance_gain": "Positive values indicate faster response times",
            "stability_score": "Higher values indicate more consistent performance",
            "overall_effectiveness": "Combined score (0-1) of all metrics"
        }
    }


@router.get("/cache/alerts")
async def get_cache_alerts(
    minutes: int = 60
) -> Dict[str, Any]:
    """
    Get recent cache performance alerts.
    
    Args:
        minutes: Number of minutes of alert history
        
    Returns:
        Recent cache alerts and their status
    """
    if not cache_monitor:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    # This would integrate with the alert system
    # For now, check current thresholds
    if cache_monitor.last_snapshot:
        alerts = cache_monitor._check_alerts(cache_monitor.last_snapshot)
    else:
        alerts = []
    
    return {
        "timestamp": datetime.now().isoformat(),
        "alert_count": len(alerts),
        "alerts": alerts,
        "thresholds": cache_monitor.alert_thresholds
    }


@router.post("/cache/monitoring/{action}")
async def control_cache_monitoring(action: str) -> Dict[str, Any]:
    """
    Control cache monitoring (start/stop).
    
    Args:
        action: "start" or "stop"
        
    Returns:
        Success status
    """
    if not cache_monitor:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    if action == "start":
        await cache_monitor.start_monitoring()
        return {
            "success": True,
            "action": "started",
            "monitoring_active": True,
            "timestamp": datetime.now().isoformat()
        }
    elif action == "stop":
        await cache_monitor.stop_monitoring()
        return {
            "success": True,
            "action": "stopped",
            "monitoring_active": False,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}")


@router.websocket("/cache/live")
async def cache_metrics_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time cache metrics.
    
    Sends cache metrics updates every few seconds while connected.
    """
    await websocket.accept()
    
    if not cache_monitor:
        await websocket.send_json({
            "error": "Cache monitor not available"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Get current metrics
            metrics = cache_monitor.get_current_metrics()
            
            # Add timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['type'] = 'cache_metrics_update'
            
            # Send to client
            await websocket.send_json(metrics)
            
            # Wait before next update
            await asyncio.sleep(5)  # 5 second updates
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "error": str(e)
        })
        await websocket.close()


@router.get("/cache/export")
async def export_cache_metrics(
    format: str = "json"  # json, csv
) -> Dict[str, Any]:
    """
    Export cache metrics for analysis.
    
    Args:
        format: Export format (json or csv)
        
    Returns:
        File path or data depending on format
    """
    if not cache_monitor:
        raise HTTPException(status_code=503, detail="Cache monitor not available")
    
    # Save current metrics
    await cache_monitor.save_metrics()
    
    if format == "json":
        return {
            "format": "json",
            "file_path": str(cache_monitor.metrics_file),
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


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