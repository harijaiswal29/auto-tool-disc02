"""
Pipeline infrastructure for modular data processing.
"""

from .base import (
    PipelineData,
    PipelineStage,
    Pipeline,
    ConditionalStage,
    ParallelStage
)

__all__ = [
    'PipelineData',
    'PipelineStage', 
    'Pipeline',
    'ConditionalStage',
    'ParallelStage'
]