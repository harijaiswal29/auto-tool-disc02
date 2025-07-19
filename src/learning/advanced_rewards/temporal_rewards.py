"""Temporal difference based reward calculation.

This module implements TD(λ) and n-step return calculations for
more sophisticated credit assignment over time.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Deque
from collections import deque
import time
import logging

from .base_strategy import BaseRewardStrategy, RewardStrategyResult

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger(__name__)


class TemporalRewardCalculator(BaseRewardStrategy):
    """Calculates rewards using temporal difference methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize temporal reward calculator.
        
        Args:
            config: Configuration with keys:
                - lambda: TD(λ) trace decay parameter (default: 0.9)
                - n_steps: Number of steps for n-step returns (default: 5)
                - gamma: Discount factor (default: 0.9)
                - use_gae: Use Generalized Advantage Estimation (default: False)
        """
        super().__init__(config)
        
    def _initialize_strategy(self):
        """Initialize TD-specific components."""
        self.lambda_param = self.config.get('lambda', 0.9)
        self.n_steps = self.config.get('n_steps', 5)
        self.gamma = self.config.get('gamma', 0.9)
        self.use_gae = self.config.get('use_gae', False)
        
        # Eligibility traces for TD(λ)
        self.eligibility_traces = {}
        
        # Buffer for n-step returns
        self.experience_buffer: Deque[Tuple[np.ndarray, List[str], float, np.ndarray]] = deque(maxlen=self.n_steps)
        
        # Value function estimates (would be populated from Q-learning engine)
        self.value_estimates = {}
        
        logger.info(f"Initialized TemporalRewardCalculator with λ={self.lambda_param}, n_steps={self.n_steps}")
    
    def calculate(self, 
                  state: np.ndarray,
                  action: List[str],
                  next_state: np.ndarray,
                  execution_results: List[Any],
                  context: Dict[str, Any]) -> RewardStrategyResult:
        """Calculate temporal difference based reward.
        
        This implements both TD(λ) with eligibility traces and n-step returns
        for better credit assignment across time.
        """
        start_time = time.time()
        
        # Get immediate reward from execution results
        immediate_reward = self._calculate_immediate_reward(execution_results)
        
        # Store experience for n-step calculation
        self.experience_buffer.append((state, action, immediate_reward, next_state))
        
        # Calculate TD error
        td_error = self._calculate_td_error(state, action, immediate_reward, next_state, context)
        
        # Calculate n-step return if we have enough experiences
        n_step_return = self._calculate_n_step_return() if len(self.experience_buffer) >= self.n_steps else immediate_reward
        
        # Update eligibility traces
        self._update_eligibility_traces(state, action)
        
        # Calculate TD(λ) reward
        td_lambda_reward = self._calculate_td_lambda_reward(td_error)
        
        # Combine different temporal components
        if self.use_gae:
            final_reward = self._calculate_gae(immediate_reward, td_error, n_step_return)
        else:
            # Weighted combination of immediate, n-step, and TD(λ) rewards
            final_reward = (
                0.3 * immediate_reward +
                0.4 * n_step_return +
                0.3 * td_lambda_reward
            )
        
        # Clip to reasonable range
        final_reward = self._clip_reward(final_reward)
        
        computation_time = (time.time() - start_time) * 1000
        
        return RewardStrategyResult(
            reward=final_reward,
            components={
                'immediate_reward': immediate_reward,
                'td_error': td_error,
                'n_step_return': n_step_return,
                'td_lambda_reward': td_lambda_reward
            },
            metadata={
                'buffer_size': len(self.experience_buffer),
                'trace_size': len(self.eligibility_traces),
                'lambda': self.lambda_param,
                'n_steps': self.n_steps
            },
            computation_time_ms=computation_time
        )
    
    def _calculate_immediate_reward(self, execution_results: List[Any]) -> float:
        """Calculate immediate reward from execution results."""
        if not execution_results:
            return -0.5
        
        success_count = sum(1 for r in execution_results if hasattr(r, 'success') and r.success)
        total_count = len(execution_results)
        
        return (success_count / total_count) * 2.0 - 1.0  # Scale to [-1, 1]
    
    def _calculate_td_error(self, state: np.ndarray, action: List[str], 
                           reward: float, next_state: np.ndarray, 
                           context: Dict[str, Any]) -> float:
        """Calculate temporal difference error."""
        # Get value estimates (would come from Q-learning engine in practice)
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_value = self.value_estimates.get(state_key, 0.0)
        next_value = self.value_estimates.get(next_state_key, 0.0)
        
        # TD error = r + γV(s') - V(s)
        td_error = reward + self.gamma * next_value - current_value
        
        return td_error
    
    def _calculate_n_step_return(self) -> float:
        """Calculate n-step return from buffered experiences."""
        if len(self.experience_buffer) < self.n_steps:
            return 0.0
        
        n_step_return = 0.0
        discount = 1.0
        
        for i, (_, _, reward, _) in enumerate(self.experience_buffer):
            n_step_return += discount * reward
            discount *= self.gamma
        
        # Add bootstrap value from final state
        final_state = self.experience_buffer[-1][3]
        final_state_key = self._state_to_key(final_state)
        bootstrap_value = self.value_estimates.get(final_state_key, 0.0)
        n_step_return += discount * bootstrap_value
        
        return n_step_return
    
    def _update_eligibility_traces(self, state: np.ndarray, action: List[str]):
        """Update eligibility traces for TD(λ)."""
        state_action_key = self._state_action_to_key(state, action)
        
        # Decay all traces
        for key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[key] *= self.gamma * self.lambda_param
            # Remove small traces to save memory
            if self.eligibility_traces[key] < 0.01:
                del self.eligibility_traces[key]
        
        # Increment trace for current state-action
        if state_action_key not in self.eligibility_traces:
            self.eligibility_traces[state_action_key] = 0.0
        self.eligibility_traces[state_action_key] += 1.0
    
    def _calculate_td_lambda_reward(self, td_error: float) -> float:
        """Calculate TD(λ) reward using eligibility traces."""
        td_lambda_reward = 0.0
        
        # Update all state-action values proportional to their traces
        for state_action_key, trace in self.eligibility_traces.items():
            # In practice, this would update Q-values
            # Here we accumulate the reward signal
            td_lambda_reward += trace * td_error * 0.1  # Scale factor
        
        return td_lambda_reward
    
    def _calculate_gae(self, immediate_reward: float, td_error: float, 
                      n_step_return: float) -> float:
        """Calculate Generalized Advantage Estimation."""
        # Simplified GAE calculation
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        advantage = 0.0
        if len(self.experience_buffer) > 1:
            # Calculate advantages for each step
            for i in range(len(self.experience_buffer) - 1):
                _, _, reward, _ = self.experience_buffer[i]
                advantage = reward + self.gamma * gae_lambda * advantage
        
        return immediate_reward + 0.5 * advantage
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key."""
        return str(hash(state.tobytes()))
    
    def _state_action_to_key(self, state: np.ndarray, action: List[str]) -> str:
        """Convert state-action pair to hashable key."""
        state_key = self._state_to_key(state)
        action_key = '_'.join(sorted(action))
        return f"{state_key}_{action_key}"
    
    def update_parameters(self, feedback: Dict[str, Any]):
        """Update temporal parameters based on feedback."""
        # Adapt lambda based on performance
        if 'performance_improvement' in feedback:
            improvement = feedback['performance_improvement']
            if improvement > 0.1:
                # Increase lambda for better long-term credit assignment
                self.lambda_param = min(0.95, self.lambda_param * 1.05)
            elif improvement < -0.1:
                # Decrease lambda to focus on immediate rewards
                self.lambda_param = max(0.5, self.lambda_param * 0.95)
        
        # Adapt n_steps based on task complexity
        if 'task_complexity' in feedback:
            complexity = feedback['task_complexity']
            if complexity > 0.7:
                self.n_steps = min(10, self.n_steps + 1)
            elif complexity < 0.3:
                self.n_steps = max(3, self.n_steps - 1)
        
        logger.debug(f"Updated temporal parameters: λ={self.lambda_param}, n_steps={self.n_steps}")
    
    def set_value_estimates(self, value_function: Dict[str, float]):
        """Set value function estimates from Q-learning engine."""
        self.value_estimates = value_function
    
    def reset_traces(self):
        """Reset eligibility traces."""
        self.eligibility_traces.clear()
        logger.debug("Reset eligibility traces")
    
    def save_state(self) -> Dict[str, Any]:
        """Save strategy state including traces and buffer."""
        state = super().save_state()
        state.update({
            'lambda_param': self.lambda_param,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'eligibility_traces': dict(self.eligibility_traces),
            'buffer_size': len(self.experience_buffer)
        })
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load strategy state."""
        super().load_state(state)
        self.lambda_param = state.get('lambda_param', self.lambda_param)
        self.n_steps = state.get('n_steps', self.n_steps)
        self.gamma = state.get('gamma', self.gamma)
        self.eligibility_traces = state.get('eligibility_traces', {})