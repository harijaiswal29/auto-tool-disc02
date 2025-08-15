"""DQN Strategy wrapper for evaluation.

This module provides a wrapper that forces DQN mode for Q-learning,
allowing it to run as a separate strategy alongside traditional Q-learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict, Any, Tuple
import copy

from evaluation.baseline_strategies import BaselineStrategy
from learning.q_learning_engine import QLearningEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class QLearningDQNStrategy(BaselineStrategy):
    """Q-Learning strategy that forces DQN mode."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DQN strategy with forced DQN configuration."""
        super().__init__("q_learning_dqn", config)
        
        # Create a copy of config and force DQN enabled
        dqn_config = copy.deepcopy(config)
        dqn_config['dqn'] = dqn_config.get('dqn', {})
        dqn_config['dqn']['enabled'] = True
        
        # Ensure all DQN parameters are set
        dqn_defaults = {
            'network_type': 'standard',
            'network_architecture': [512, 256, 128],
            'dropout_rate': 0.2,
            'learning_rate': 0.0001,
            'discount_factor': 0.99,
            'tau': 0.001,
            'target_update_frequency': 1000,
            'double_dqn': True,
            'dueling_dqn': False,
            'gradient_clip': 1.0,
            'batch_size': 64,
            'memory_size': 100000,
            'exploration_rate': 0.1,
            'exploration_decay': 0.995,
            'min_exploration_rate': 0.01,
            'prioritized_replay': True,
            'prioritized_replay_alpha': 0.6,
            'prioritized_replay_beta': 0.4,
            'prioritized_replay_beta_increment': 0.001,
            'device': 'cpu'
        }
        
        for key, value in dqn_defaults.items():
            if key not in dqn_config['dqn']:
                dqn_config['dqn'][key] = value
        
        # Create Q-learning engine with DQN forced on
        self.q_learning = QLearningEngine(dqn_config)
        
        # Verify DQN is enabled
        if not self.q_learning.use_dqn:
            raise RuntimeError("Failed to enable DQN in QLearningDQNStrategy")
        
        logger.info("Initialized Q-Learning with DQN strategy")
        logger.info(f"State dimensions: {self.q_learning.state_encoder.total_dimensions}")
        logger.info(f"DQN architecture: {dqn_config['dqn']['network_architecture']}")
    
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select tools using DQN-based Q-learning.
        
        Args:
            state: Current state representation
            available_tools: List of available tool IDs
            constraints: Tool constraints
            
        Returns:
            List of selected tool IDs
        """
        # Use the Q-learning engine's select_action method
        selected_combination = await self.q_learning.select_action(
            state, available_tools, constraints
        )
        
        # Convert tuple to list
        return list(selected_combination)
    
    async def learn_from_experience(self, state: np.ndarray, action: List[str],
                                   reward: float, next_state: np.ndarray,
                                   next_available: List[str]) -> None:
        """Update DQN based on experience.
        
        Args:
            state: Current state
            action: Action taken (tools selected)
            reward: Reward received
            next_state: Next state
            next_available: Available tools in next state
        """
        # Convert action list to tuple for Q-learning engine
        action_tuple = tuple(action)
        
        # Let the Q-learning engine handle the learning
        await self.q_learning.learn_from_experience(
            state, action_tuple, reward, next_state, next_available
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy."""
        return {
            'name': 'Q-Learning with DQN',
            'type': 'neural_network',
            'state_dimensions': self.q_learning.state_encoder.total_dimensions,
            'uses_dqn': True,
            'network_architecture': self.config['dqn'].get('network_architecture', [512, 256, 128]),
            'learning_rate': self.config['dqn'].get('learning_rate', 0.0001),
            'memory_size': self.config['dqn'].get('memory_size', 100000)
        }


class QLearningTabularStrategy(BaselineStrategy):
    """Q-Learning strategy that forces tabular (non-DQN) mode."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tabular Q-learning strategy with DQN disabled."""
        super().__init__("q_learning_tabular", config)
        
        # Create a copy of config and force DQN disabled
        tabular_config = copy.deepcopy(config)
        tabular_config['dqn'] = tabular_config.get('dqn', {})
        tabular_config['dqn']['enabled'] = False
        
        # Create Q-learning engine with DQN forced off
        self.q_learning = QLearningEngine(tabular_config)
        
        # Verify DQN is disabled
        if self.q_learning.use_dqn:
            raise RuntimeError("Failed to disable DQN in QLearningTabularStrategy")
        
        logger.info("Initialized Q-Learning with Tabular (Q-table) strategy")
        logger.info(f"State dimensions: {self.q_learning.state_encoder.total_dimensions}")
    
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select tools using tabular Q-learning.
        
        Args:
            state: Current state representation
            available_tools: List of available tool IDs
            constraints: Tool constraints
            
        Returns:
            List of selected tool IDs
        """
        # Use the Q-learning engine's select_action method
        selected_combination = await self.q_learning.select_action(
            state, available_tools, constraints
        )
        
        # Convert tuple to list
        return list(selected_combination)
    
    async def learn_from_experience(self, state: np.ndarray, action: List[str],
                                   reward: float, next_state: np.ndarray,
                                   next_available: List[str]) -> None:
        """Update Q-table based on experience.
        
        Args:
            state: Current state
            action: Action taken (tools selected)
            reward: Reward received
            next_state: Next state
            next_available: Available tools in next state
        """
        # Convert action list to tuple for Q-learning engine
        action_tuple = tuple(action)
        
        # Let the Q-learning engine handle the learning
        await self.q_learning.learn_from_experience(
            state, action_tuple, reward, next_state, next_available
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy."""
        return {
            'name': 'Q-Learning with Q-Table',
            'type': 'tabular',
            'state_dimensions': self.q_learning.state_encoder.total_dimensions,
            'uses_dqn': False,
            'learning_rate': self.q_learning.learning_rate,
            'discount_factor': self.q_learning.discount_factor,
            'exploration_rate': self.q_learning.exploration_rate
        }