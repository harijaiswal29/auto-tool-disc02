"""Advanced reward calculation strategies for Q-learning.

This module provides sophisticated reward calculation strategies that go beyond
basic success/failure rewards, including temporal difference rewards, hierarchical
goal-based rewards, information-theoretic rewards, and adaptive reward shaping.
"""

from .base_strategy import BaseRewardStrategy, RewardStrategyResult
from .temporal_rewards import TemporalRewardCalculator
from .hierarchical_rewards import HierarchicalRewardCalculator
from .adaptive_shaping import AdaptiveRewardShaper
from .information_theoretic import InformationTheoreticRewardCalculator
from .strategy_manager import StrategyManager

__all__ = [
    'BaseRewardStrategy',
    'RewardStrategyResult',
    'TemporalRewardCalculator',
    'HierarchicalRewardCalculator',
    'AdaptiveRewardShaper',
    'InformationTheoreticRewardCalculator',
    'StrategyManager'
]