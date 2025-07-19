"""Base class for advanced reward calculation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewardStrategyResult:
    """Result from a reward strategy calculation."""
    reward: float
    components: Dict[str, float]
    metadata: Dict[str, Any]
    computation_time_ms: float = 0.0


class BaseRewardStrategy(ABC):
    """Abstract base class for reward calculation strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the reward strategy.
        
        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 1.0)
        self._initialize_strategy()
        
    @abstractmethod
    def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        pass
    
    @abstractmethod
    def calculate(self, 
                  state: np.ndarray,
                  action: List[str],
                  next_state: np.ndarray,
                  execution_results: List[Any],
                  context: Dict[str, Any]) -> RewardStrategyResult:
        """Calculate reward using this strategy.
        
        Args:
            state: Current state vector
            action: Action taken (list of tool IDs)
            next_state: Next state vector
            execution_results: Results from tool execution
            context: Additional context information
            
        Returns:
            RewardStrategyResult with calculated reward and breakdown
        """
        pass
    
    @abstractmethod
    def update_parameters(self, feedback: Dict[str, Any]):
        """Update strategy parameters based on feedback.
        
        Args:
            feedback: Feedback data for parameter adjustment
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this strategy is enabled."""
        return self.enabled
    
    def get_weight(self) -> float:
        """Get the weight for this strategy in ensemble calculations."""
        return self.weight
    
    def get_name(self) -> str:
        """Get the name of this strategy."""
        return self.__class__.__name__
    
    def _clip_reward(self, reward: float, min_val: float = -2.0, max_val: float = 2.0) -> float:
        """Clip reward to reasonable range."""
        return np.clip(reward, min_val, max_val)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        vec1_norm = self._normalize_vector(vec1)
        vec2_norm = self._normalize_vector(vec2)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def save_state(self) -> Dict[str, Any]:
        """Save strategy state for persistence."""
        return {
            'config': self.config,
            'enabled': self.enabled,
            'weight': self.weight
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load strategy state from saved data."""
        self.config = state.get('config', self.config)
        self.enabled = state.get('enabled', self.enabled)
        self.weight = state.get('weight', self.weight)
        self._initialize_strategy()