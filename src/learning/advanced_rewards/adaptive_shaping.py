"""Adaptive reward shaping with dynamic weight adjustment.

This module implements reward shaping that adapts based on learning
progress and performance metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time
import logging

from .base_strategy import BaseRewardStrategy, RewardStrategyResult

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveRewardShaper(BaseRewardStrategy):
    """Dynamically adjusts reward weights based on learning progress."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive reward shaper.
        
        Args:
            config: Configuration with keys:
                - adaptation_rate: Learning rate for weight updates (default: 0.01)
                - curriculum_stages: Number of curriculum stages (default: 3)
                - performance_window: Window size for performance tracking (default: 100)
                - meta_learning_rate: Rate for meta-parameter updates (default: 0.001)
        """
        super().__init__(config)
        
    def _initialize_strategy(self):
        """Initialize adaptive shaping components."""
        self.adaptation_rate = self.config.get('adaptation_rate', 0.01)
        self.curriculum_stages = self.config.get('curriculum_stages', 3)
        self.performance_window = self.config.get('performance_window', 100)
        self.meta_learning_rate = self.config.get('meta_learning_rate', 0.001)
        
        # Current curriculum stage
        self.current_stage = 0
        
        # Adaptive weights for different reward components
        self.component_weights = {
            'success': 1.0,
            'efficiency': 0.5,
            'exploration': 0.3,
            'complexity': 0.2,
            'consistency': 0.4
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.performance_window)
        self.weight_history = deque(maxlen=self.performance_window)
        
        # Curriculum thresholds
        self.stage_thresholds = self._compute_stage_thresholds()
        
        # Meta-parameters for adaptation
        self.meta_params = {
            'weight_momentum': 0.9,
            'weight_decay': 0.999,
            'exploration_bonus_decay': 0.995
        }
        
        # Weight gradients for momentum
        self.weight_gradients = {k: 0.0 for k in self.component_weights}
        
        logger.info(f"Initialized AdaptiveRewardShaper with {self.curriculum_stages} stages")
    
    def _compute_stage_thresholds(self) -> List[float]:
        """Compute performance thresholds for curriculum stages."""
        return [0.4 + (0.4 / self.curriculum_stages) * i for i in range(self.curriculum_stages)]
    
    def calculate(self, 
                  state: np.ndarray,
                  action: List[str],
                  next_state: np.ndarray,
                  execution_results: List[Any],
                  context: Dict[str, Any]) -> RewardStrategyResult:
        """Calculate adaptively shaped reward."""
        start_time = time.time()
        
        # Calculate component rewards
        success_reward = self._calculate_success_component(execution_results)
        efficiency_reward = self._calculate_efficiency_component(execution_results)
        exploration_reward = self._calculate_exploration_component(state, action, context)
        complexity_reward = self._calculate_complexity_component(action, execution_results)
        consistency_reward = self._calculate_consistency_component(state, next_state, action)
        
        # Apply current weights
        weighted_reward = (
            self.component_weights['success'] * success_reward +
            self.component_weights['efficiency'] * efficiency_reward +
            self.component_weights['exploration'] * exploration_reward +
            self.component_weights['complexity'] * complexity_reward +
            self.component_weights['consistency'] * consistency_reward
        )
        
        # Apply curriculum scaling
        curriculum_multiplier = self._get_curriculum_multiplier()
        shaped_reward = weighted_reward * curriculum_multiplier
        
        # Update performance tracking
        performance_score = self._compute_performance_score(execution_results, shaped_reward)
        self.performance_history.append(performance_score)
        
        # Adapt weights based on performance
        self._adapt_weights(performance_score, {
            'success': success_reward,
            'efficiency': efficiency_reward,
            'exploration': exploration_reward,
            'complexity': complexity_reward,
            'consistency': consistency_reward
        })
        
        # Check for stage progression
        self._check_stage_progression()
        
        # Clip final reward
        final_reward = self._clip_reward(shaped_reward)
        
        computation_time = (time.time() - start_time) * 1000
        
        return RewardStrategyResult(
            reward=final_reward,
            components={
                'success': success_reward * self.component_weights['success'],
                'efficiency': efficiency_reward * self.component_weights['efficiency'],
                'exploration': exploration_reward * self.component_weights['exploration'],
                'complexity': complexity_reward * self.component_weights['complexity'],
                'consistency': consistency_reward * self.component_weights['consistency'],
                'curriculum_multiplier': curriculum_multiplier
            },
            metadata={
                'current_stage': self.current_stage,
                'avg_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
                'weights': dict(self.component_weights)
            },
            computation_time_ms=computation_time
        )
    
    def _calculate_success_component(self, execution_results: List[Any]) -> float:
        """Calculate success-based reward component."""
        if not execution_results:
            return -0.5
        
        success_count = sum(1 for r in execution_results if hasattr(r, 'success') and r.success)
        partial_count = sum(1 for r in execution_results 
                           if hasattr(r, 'partial_success') and r.partial_success and not r.success)
        
        success_rate = success_count / len(execution_results)
        partial_rate = partial_count / len(execution_results)
        
        return success_rate + 0.5 * partial_rate
    
    def _calculate_efficiency_component(self, execution_results: List[Any]) -> float:
        """Calculate efficiency-based reward component."""
        if not execution_results:
            return 0.0
        
        # Time efficiency
        avg_time = np.mean([getattr(r, 'execution_time_ms', 1000) for r in execution_results])
        time_efficiency = 1.0 / (1.0 + avg_time / 1000.0)  # Normalize to [0, 1]
        
        # Resource efficiency
        resource_scores = []
        for r in execution_results:
            if hasattr(r, 'resource_usage') and r.resource_usage:
                cpu_score = 1.0 - min(r.resource_usage.get('cpu_percent', 0) / 100.0, 1.0)
                memory_score = 1.0 - min(r.resource_usage.get('memory_mb', 0) / 1000.0, 1.0)
                resource_scores.append((cpu_score + memory_score) / 2)
        
        resource_efficiency = np.mean(resource_scores) if resource_scores else 0.5
        
        return (time_efficiency + resource_efficiency) / 2
    
    def _calculate_exploration_component(self, state: np.ndarray, action: List[str], 
                                       context: Dict[str, Any]) -> float:
        """Calculate exploration-based reward component."""
        # Reward for trying new combinations
        exploration_bonus = 0.0
        
        # Check if this is a novel state-action pair
        state_action_hash = self._compute_state_action_hash(state, action)
        visit_count = context.get('state_action_visits', {}).get(state_action_hash, 0)
        
        if visit_count == 0:
            # First visit bonus
            exploration_bonus = 0.5
        else:
            # Decreasing bonus for repeated visits
            exploration_bonus = 0.1 / (1 + visit_count)
        
        # Bonus for diverse tool usage
        tool_diversity = len(set(action)) / max(len(action), 1)
        exploration_bonus += 0.3 * tool_diversity
        
        # Apply exploration decay based on stage
        decay_factor = self.meta_params['exploration_bonus_decay'] ** self.current_stage
        
        return exploration_bonus * decay_factor
    
    def _calculate_complexity_component(self, action: List[str], 
                                      execution_results: List[Any]) -> float:
        """Calculate complexity-handling reward component."""
        # Reward for handling complex tasks successfully
        task_complexity = len(action) / 3.0  # Normalize by max tools
        
        if execution_results:
            success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
            # Higher reward for succeeding on complex tasks
            complexity_reward = task_complexity * success_rate
        else:
            complexity_reward = -task_complexity * 0.5
        
        return complexity_reward
    
    def _calculate_consistency_component(self, state: np.ndarray, next_state: np.ndarray, 
                                       action: List[str]) -> float:
        """Calculate consistency reward component."""
        # Reward for smooth state transitions
        state_distance = np.linalg.norm(next_state - state)
        consistency_score = 1.0 / (1.0 + state_distance)
        
        # Penalty for erratic tool selection
        if hasattr(self, 'previous_action'):
            action_similarity = len(set(action).intersection(self.previous_action)) / max(len(action), 1)
            consistency_score = (consistency_score + action_similarity) / 2
        
        self.previous_action = set(action)
        
        return consistency_score
    
    def _compute_performance_score(self, execution_results: List[Any], reward: float) -> float:
        """Compute overall performance score."""
        if not execution_results:
            return reward
        
        # Combine multiple metrics
        success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
        quality_score = np.mean([getattr(r, 'result_quality', 0.5) for r in execution_results])
        
        # Weighted combination
        performance = 0.4 * success_rate + 0.3 * quality_score + 0.3 * (reward + 1) / 2
        
        return np.clip(performance, 0, 1)
    
    def _adapt_weights(self, performance: float, component_rewards: Dict[str, float]):
        """Adapt component weights based on performance."""
        # Calculate performance gradient
        if len(self.performance_history) > 10:
            recent_performance = list(self.performance_history)[-10:]
            performance_gradient = np.gradient(recent_performance)[-1]
        else:
            performance_gradient = 0.0
        
        # Update weights based on component contributions
        for component, reward in component_rewards.items():
            # Calculate gradient with momentum
            gradient = performance_gradient * reward
            self.weight_gradients[component] = (
                self.meta_params['weight_momentum'] * self.weight_gradients[component] +
                (1 - self.meta_params['weight_momentum']) * gradient
            )
            
            # Update weight
            self.component_weights[component] += self.adaptation_rate * self.weight_gradients[component]
            
            # Apply weight decay
            self.component_weights[component] *= self.meta_params['weight_decay']
            
            # Ensure weights stay positive and normalized
            self.component_weights[component] = max(0.1, self.component_weights[component])
        
        # Normalize weights
        total_weight = sum(self.component_weights.values())
        for component in self.component_weights:
            self.component_weights[component] /= total_weight
        
        # Store weight history
        self.weight_history.append(dict(self.component_weights))
    
    def _get_curriculum_multiplier(self) -> float:
        """Get curriculum-based reward multiplier."""
        # Different multipliers for different stages
        stage_multipliers = [0.5, 0.75, 1.0]
        
        if self.current_stage < len(stage_multipliers):
            return stage_multipliers[self.current_stage]
        
        return 1.0
    
    def _check_stage_progression(self):
        """Check if we should progress to next curriculum stage."""
        if not self.performance_history or self.current_stage >= self.curriculum_stages - 1:
            return
        
        avg_performance = np.mean(self.performance_history)
        
        if avg_performance > self.stage_thresholds[self.current_stage]:
            self.current_stage += 1
            logger.info(f"Progressed to curriculum stage {self.current_stage}")
            
            # Adjust weights for new stage
            self._adjust_weights_for_stage()
    
    def _adjust_weights_for_stage(self):
        """Adjust weights when entering new curriculum stage."""
        if self.current_stage == 1:
            # Intermediate stage: balance all components
            target_weights = {
                'success': 0.3,
                'efficiency': 0.2,
                'exploration': 0.2,
                'complexity': 0.15,
                'consistency': 0.15
            }
        elif self.current_stage >= 2:
            # Advanced stage: focus on efficiency and complexity
            target_weights = {
                'success': 0.25,
                'efficiency': 0.3,
                'exploration': 0.1,
                'complexity': 0.25,
                'consistency': 0.1
            }
        else:
            return
        
        # Gradually move towards target weights
        for component, target in target_weights.items():
            self.component_weights[component] = (
                0.7 * self.component_weights[component] + 0.3 * target
            )
    
    def _compute_state_action_hash(self, state: np.ndarray, action: List[str]) -> str:
        """Compute hash for state-action pair."""
        state_str = str(hash(state.tobytes()))
        action_str = '_'.join(sorted(action))
        return f"{state_str}_{action_str}"
    
    def update_parameters(self, feedback: Dict[str, Any]):
        """Update adaptive parameters based on feedback."""
        # Update meta-parameters based on learning progress
        if 'learning_rate_performance' in feedback:
            lr_perf = feedback['learning_rate_performance']
            if lr_perf < 0.3:
                # Learning too slow, increase adaptation rate
                self.adaptation_rate = min(0.1, self.adaptation_rate * 1.1)
            elif lr_perf > 0.7:
                # Learning too fast, decrease adaptation rate
                self.adaptation_rate = max(0.001, self.adaptation_rate * 0.9)
        
        # Adjust curriculum based on overall progress
        if 'overall_progress' in feedback:
            progress = feedback['overall_progress']
            if progress > 0.8 and self.current_stage < self.curriculum_stages - 1:
                # Force progression if doing very well
                self.current_stage += 1
                self._adjust_weights_for_stage()
    
    def reset_to_stage(self, stage: int):
        """Reset to specific curriculum stage."""
        self.current_stage = max(0, min(stage, self.curriculum_stages - 1))
        self._adjust_weights_for_stage()
        logger.info(f"Reset to curriculum stage {self.current_stage}")
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get metrics about adaptation process."""
        return {
            'current_weights': dict(self.component_weights),
            'weight_changes': self._compute_weight_changes(),
            'performance_trend': self._compute_performance_trend(),
            'stage_progress': self.current_stage / (self.curriculum_stages - 1)
        }
    
    def _compute_weight_changes(self) -> Dict[str, float]:
        """Compute recent weight changes."""
        if len(self.weight_history) < 2:
            return {k: 0.0 for k in self.component_weights}
        
        recent = self.weight_history[-1]
        previous = self.weight_history[-10] if len(self.weight_history) >= 10 else self.weight_history[0]
        
        return {k: recent[k] - previous[k] for k in self.component_weights}
    
    def _compute_performance_trend(self) -> float:
        """Compute performance trend."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = list(self.performance_history)[-10:]
        return float(np.polyfit(range(len(recent)), recent, 1)[0])
    
    def save_state(self) -> Dict[str, Any]:
        """Save strategy state."""
        state = super().save_state()
        state.update({
            'current_stage': self.current_stage,
            'component_weights': dict(self.component_weights),
            'weight_gradients': dict(self.weight_gradients),
            'meta_params': dict(self.meta_params),
            'performance_stats': {
                'avg_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
                'performance_trend': self._compute_performance_trend()
            }
        })
        return state