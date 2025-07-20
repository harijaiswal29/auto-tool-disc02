"""
API Endpoints for A/B Testing Management

This module provides REST API endpoints for creating, managing, and analyzing
A/B test experiments in the Auto Tool Discovery system.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from src.evaluation.ab_testing_framework import (
    ExperimentConfig, AssignmentStrategy, StatisticalMethod
)
from src.evaluation.ab_test_manager import ABTestManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ab-testing", tags=["ab-testing"])

# Global A/B test manager instance (initialized in main app)
ab_test_manager: Optional[ABTestManager] = None


# Pydantic models for API requests/responses
class CreateExperimentRequest(BaseModel):
    """Request model for creating an experiment."""
    name: str = Field(..., description="Unique experiment name")
    description: str = Field(..., description="Experiment description")
    variants: List[str] = Field(..., description="List of variant names")
    primary_metric: str = Field(..., description="Primary metric to track")
    secondary_metrics: List[str] = Field(default_factory=list, description="Secondary metrics")
    assignment_strategy: str = Field("random", description="Assignment strategy")
    assignment_weights: Optional[Dict[str, float]] = Field(None, description="Variant weights")
    statistical_method: str = Field("frequentist", description="Statistical method")
    target_sample_size: Optional[int] = Field(None, description="Target sample size")
    min_sample_size: int = Field(100, description="Minimum sample size")
    confidence_level: float = Field(0.95, description="Confidence level")
    power: float = Field(0.8, description="Statistical power")
    minimum_detectable_effect: float = Field(0.05, description="MDE")
    max_duration_days: Optional[int] = Field(None, description="Max duration in days")
    enable_early_stopping: bool = Field(True, description="Enable early stopping")
    enable_multi_armed_bandit: bool = Field(False, description="Enable MAB")


class AssignmentRequest(BaseModel):
    """Request model for user assignment."""
    user_id: str = Field(..., description="User ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="User context")


class RecordMetricRequest(BaseModel):
    """Request model for recording metrics."""
    user_id: str = Field(..., description="User ID")
    metric_name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    success: bool = Field(True, description="Success indicator")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status."""
    name: str
    status: str
    start_time: Optional[str]
    total_users: int
    variant_metrics: List[Dict[str, Any]]


class ExperimentResultsResponse(BaseModel):
    """Response model for experiment results."""
    experiment_name: str
    status: str
    duration_days: Optional[int]
    total_users: int
    variant_results: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    winner: Optional[str]
    confidence: Optional[float]
    analysis_timestamp: Optional[str]


def get_ab_test_manager() -> ABTestManager:
    """Dependency to get A/B test manager."""
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test manager not initialized")
    return ab_test_manager


@router.post("/experiments", response_model=Dict[str, str])
async def create_experiment(
    request: CreateExperimentRequest,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Create a new A/B test experiment."""
    try:
        # Convert request to ExperimentConfig
        config = ExperimentConfig(
            name=request.name,
            description=request.description,
            variants=request.variants,
            primary_metric=request.primary_metric,
            secondary_metrics=request.secondary_metrics,
            assignment_strategy=AssignmentStrategy(request.assignment_strategy),
            assignment_weights=request.assignment_weights,
            statistical_method=StatisticalMethod(request.statistical_method),
            target_sample_size=request.target_sample_size,
            min_sample_size=request.min_sample_size,
            confidence_level=request.confidence_level,
            power=request.power,
            minimum_detectable_effect=request.minimum_detectable_effect,
            max_duration_days=request.max_duration_days,
            enable_early_stopping=request.enable_early_stopping,
            enable_multi_armed_bandit=request.enable_multi_armed_bandit
        )
        
        experiment_id = await manager.create_experiment(config)
        return {"experiment_id": experiment_id, "status": "created"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create experiment")


@router.post("/experiments/{experiment_name}/start")
async def start_experiment(
    experiment_name: str,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Start an experiment."""
    try:
        await manager.start_experiment(experiment_name)
        return {"experiment_name": experiment_name, "status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to start experiment")


@router.post("/experiments/{experiment_name}/stop")
async def stop_experiment(
    experiment_name: str,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Stop an experiment and get results."""
    try:
        results = await manager.stop_experiment(experiment_name)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop experiment")


@router.post("/experiments/{experiment_name}/assign")
async def assign_user(
    experiment_name: str,
    request: AssignmentRequest,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Assign a user to a variant in an experiment."""
    try:
        variant = await manager.get_variant_for_user(
            experiment_name, request.user_id, request.context
        )
        if not variant:
            raise HTTPException(status_code=404, detail="Experiment not found or not running")
        
        return {"user_id": request.user_id, "variant": variant}
    except Exception as e:
        logger.error(f"Failed to assign user: {e}")
        raise HTTPException(status_code=500, detail="Failed to assign user")


@router.post("/experiments/{experiment_name}/metrics")
async def record_metric(
    experiment_name: str,
    request: RecordMetricRequest,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Record a metric for an experiment."""
    try:
        await manager.record_metric(
            experiment_name,
            request.user_id,
            request.metric_name,
            request.value,
            request.success,
            request.metadata
        )
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to record metric")


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments(
    status: Optional[str] = None,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """List all experiments with optional status filter."""
    try:
        experiments = await manager.list_experiments(status)
        return experiments
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to list experiments")


@router.get("/experiments/{experiment_name}/status", response_model=ExperimentStatusResponse)
async def get_experiment_status(
    experiment_name: str,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Get current status of an experiment."""
    try:
        status = await manager.get_experiment_status(experiment_name)
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        return ExperimentStatusResponse(**status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get experiment status")


@router.get("/experiments/{experiment_name}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(
    experiment_name: str,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Get results for a completed experiment."""
    try:
        results = await manager.get_experiment_results(experiment_name)
        if not results:
            raise HTTPException(status_code=404, detail="No results found for experiment")
        return ExperimentResultsResponse(**results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get experiment results")


@router.post("/experiments/{experiment_name}/analyze")
async def analyze_experiment(
    experiment_name: str,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Analyze a running or completed experiment."""
    try:
        # Get experiment from framework
        experiment = await manager.framework.get_experiment(experiment_name)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Analyze without stopping
        results = await experiment.analyze()
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze experiment")


@router.get("/sample-size-calculator")
async def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    confidence_level: float = 0.95,
    power: float = 0.8,
    manager: ABTestManager = Depends(get_ab_test_manager)
):
    """Calculate required sample size for given parameters."""
    try:
        alpha = 1 - confidence_level
        sample_size = manager.framework.calculate_sample_size(
            baseline_rate, minimum_detectable_effect, alpha, power
        )
        return {
            "baseline_rate": baseline_rate,
            "minimum_detectable_effect": minimum_detectable_effect,
            "confidence_level": confidence_level,
            "power": power,
            "required_sample_size_per_variant": sample_size
        }
    except Exception as e:
        logger.error(f"Failed to calculate sample size: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate sample size")


# Initialize function to be called from main app
def initialize_ab_testing_api(config: Dict[str, Any]):
    """Initialize the A/B testing API with configuration."""
    global ab_test_manager
    ab_test_manager = ABTestManager(config)
    asyncio.create_task(ab_test_manager.initialize())
    logger.info("Initialized A/B testing API")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if A/B testing service is healthy."""
    if ab_test_manager:
        return {"status": "healthy", "service": "ab-testing"}
    else:
        raise HTTPException(status_code=503, detail="A/B testing service not initialized")