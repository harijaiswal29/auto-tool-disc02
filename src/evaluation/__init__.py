"""Evaluation framework for automated baseline comparisons.

This module provides tools for comparing Q-learning and DQN approaches against
various baseline strategies to demonstrate improvement and track performance.
"""

from .baseline_strategies import (
    BaselineStrategy,
    RandomSelectionBaseline,
    MostPopularToolsBaseline,
    FixedPolicyBaseline,
    GreedySingleToolBaseline,
    ContextAgnosticQLearningBaseline
)
from .evaluation_engine import EvaluationEngine
from .metrics_collector import MetricsCollector
from .comparison_visualizer import ComparisonVisualizer

__all__ = [
    'BaselineStrategy',
    'RandomSelectionBaseline',
    'MostPopularToolsBaseline',
    'FixedPolicyBaseline',
    'GreedySingleToolBaseline',
    'ContextAgnosticQLearningBaseline',
    'EvaluationEngine',
    'MetricsCollector',
    'ComparisonVisualizer'
]