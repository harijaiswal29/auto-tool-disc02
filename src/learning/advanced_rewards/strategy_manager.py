"""Strategy manager for coordinating multiple reward calculation strategies.

This module manages the ensemble of reward strategies, handles strategy
selection, combination, and A/B testing.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import time
import logging
import json

from .base_strategy import BaseRewardStrategy, RewardStrategyResult
from .temporal_rewards import TemporalRewardCalculator
from .hierarchical_rewards import HierarchicalRewardCalculator
from .adaptive_shaping import AdaptiveRewardShaper
from .information_theoretic import InformationTheoreticRewardCalculator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger
from database.database import DatabaseManager

logger = get_logger(__name__)


class StrategyManager:
    """Manages multiple reward calculation strategies."""
    
    def __init__(self, config: Dict[str, Any], db_manager: Optional[DatabaseManager] = None):
        """Initialize strategy manager.
        
        Args:
            config: Configuration for all strategies
            db_manager: Database manager for persistence
        """
        self.config = config
        self.db_manager = db_manager
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Strategy weights for ensemble
        self.strategy_weights = config.get('strategy_weights', {
            'temporal': 0.25,
            'hierarchical': 0.25,
            'adaptive': 0.25,
            'information_theoretic': 0.25
        })
        
        # A/B testing configuration
        self.ab_testing_enabled = config.get('ab_testing_enabled', False)
        self.ab_test_groups = defaultdict(list)
        self.ab_test_results = defaultdict(lambda: {'rewards': [], 'performance': []})
        
        # Performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'total_reward': 0.0,
            'execution_count': 0,
            'avg_computation_time': 0.0,
            'success_correlation': 0.0
        })
        
        # Combination method
        self.combination_method = config.get('combination_method', 'weighted_average')
        
        logger.info(f"Initialized StrategyManager with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> Dict[str, BaseRewardStrategy]:
        """Initialize all reward strategies."""
        strategies = {}
        
        # Get advanced strategies config
        adv_config = self.config.get('advanced_reward_strategies', {})
        
        if adv_config.get('enabled', True):
            # Temporal strategy
            if adv_config.get('strategies', {}).get('temporal_difference', {}).get('enabled', True):
                strategies['temporal'] = TemporalRewardCalculator(
                    adv_config['strategies']['temporal_difference']
                )
            
            # Hierarchical strategy
            if adv_config.get('strategies', {}).get('hierarchical', {}).get('enabled', True):
                strategies['hierarchical'] = HierarchicalRewardCalculator(
                    adv_config['strategies']['hierarchical']
                )
            
            # Adaptive shaping strategy
            if adv_config.get('strategies', {}).get('adaptive_shaping', {}).get('enabled', True):
                strategies['adaptive'] = AdaptiveRewardShaper(
                    adv_config['strategies']['adaptive_shaping']
                )
            
            # Information-theoretic strategy
            if adv_config.get('strategies', {}).get('information_theoretic', {}).get('enabled', True):
                strategies['information_theoretic'] = InformationTheoreticRewardCalculator(
                    adv_config['strategies']['information_theoretic']
                )
        
        return strategies
    
    def calculate_reward(self,
                        state: np.ndarray,
                        action: List[str],
                        next_state: np.ndarray,
                        execution_results: List[Any],
                        context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate reward using ensemble of strategies.
        
        Returns:
            Tuple of (total_reward, detailed_breakdown)
        """
        start_time = time.time()
        
        # Determine which strategies to use
        active_strategies = self._select_active_strategies(context)
        
        # Calculate rewards from each strategy
        strategy_results = {}
        for name, strategy in active_strategies.items():
            if strategy.is_enabled():
                try:
                    result = strategy.calculate(state, action, next_state, execution_results, context)
                    strategy_results[name] = result
                    
                    # Update performance tracking
                    self._update_strategy_performance(name, result, execution_results)
                except Exception as e:
                    logger.error(f"Error in {name} strategy: {e}")
                    strategy_results[name] = RewardStrategyResult(
                        reward=0.0,
                        components={},
                        metadata={'error': str(e)},
                        computation_time_ms=0.0
                    )
        
        # Combine strategy rewards
        if self.combination_method == 'weighted_average':
            final_reward = self._weighted_average_combination(strategy_results)
        elif self.combination_method == 'max':
            final_reward = self._max_combination(strategy_results)
        elif self.combination_method == 'voting':
            final_reward = self._voting_combination(strategy_results)
        else:
            final_reward = self._weighted_average_combination(strategy_results)
        
        # A/B testing tracking
        if self.ab_testing_enabled:
            self._track_ab_test_results(context, final_reward, strategy_results)
        
        # Store results in database if available
        if self.db_manager:
            self._store_strategy_metrics(strategy_results, final_reward)
        
        total_time = (time.time() - start_time) * 1000
        
        # Build detailed breakdown
        breakdown = {
            'final_reward': final_reward,
            'strategy_rewards': {name: result.reward for name, result in strategy_results.items()},
            'strategy_components': {name: result.components for name, result in strategy_results.items()},
            'strategy_metadata': {name: result.metadata for name, result in strategy_results.items()},
            'combination_method': self.combination_method,
            'active_strategies': list(active_strategies.keys()),
            'computation_time_ms': total_time
        }
        
        return final_reward, breakdown
    
    def _select_active_strategies(self, context: Dict[str, Any]) -> Dict[str, BaseRewardStrategy]:
        """Select which strategies to use based on context."""
        if self.ab_testing_enabled:
            # A/B testing mode - select based on test group
            test_group = context.get('ab_test_group', 'control')
            if test_group == 'control':
                # Use all strategies
                return self.strategies
            else:
                # Use specific strategy for test group
                if test_group in self.strategies:
                    return {test_group: self.strategies[test_group]}
        
        # Normal mode - use all enabled strategies
        return {name: strategy for name, strategy in self.strategies.items() 
                if strategy.is_enabled()}
    
    def _weighted_average_combination(self, results: Dict[str, RewardStrategyResult]) -> float:
        """Combine rewards using weighted average."""
        if not results:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, result in results.items():
            weight = self.strategy_weights.get(name, 0.25)
            weighted_sum += weight * result.reward
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _max_combination(self, results: Dict[str, RewardStrategyResult]) -> float:
        """Combine rewards by taking maximum."""
        if not results:
            return 0.0
        
        return max(result.reward for result in results.values())
    
    def _voting_combination(self, results: Dict[str, RewardStrategyResult]) -> float:
        """Combine rewards using voting mechanism."""
        if not results:
            return 0.0
        
        # Categorize rewards
        positive_votes = sum(1 for r in results.values() if r.reward > 0.1)
        negative_votes = sum(1 for r in results.values() if r.reward < -0.1)
        
        # Weighted by vote strength
        vote_strength = np.mean([abs(r.reward) for r in results.values()])
        
        if positive_votes > negative_votes:
            return vote_strength
        elif negative_votes > positive_votes:
            return -vote_strength
        else:
            # Tie - use average
            return np.mean([r.reward for r in results.values()])
    
    def _update_strategy_performance(self, name: str, result: RewardStrategyResult, 
                                   execution_results: List[Any]):
        """Update performance metrics for a strategy."""
        perf = self.strategy_performance[name]
        
        # Update reward tracking
        perf['total_reward'] += result.reward
        perf['execution_count'] += 1
        
        # Update computation time
        n = perf['execution_count']
        perf['avg_computation_time'] = (
            (perf['avg_computation_time'] * (n - 1) + result.computation_time_ms) / n
        )
        
        # Update success correlation
        if execution_results:
            success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
            # Simple correlation approximation
            reward_sign = 1 if result.reward > 0 else -1
            success_sign = 1 if success_rate > 0.5 else -1
            correlation_update = reward_sign * success_sign
            
            perf['success_correlation'] = (
                (perf['success_correlation'] * (n - 1) + correlation_update) / n
            )
    
    def _track_ab_test_results(self, context: Dict[str, Any], reward: float, 
                              strategy_results: Dict[str, RewardStrategyResult]):
        """Track results for A/B testing."""
        test_group = context.get('ab_test_group', 'control')
        execution_id = context.get('execution_id', 'unknown')
        
        # Track rewards
        self.ab_test_results[test_group]['rewards'].append(reward)
        
        # Track performance metric (simplified)
        if 'execution_results' in context:
            success_rate = sum(1 for r in context['execution_results'] 
                             if hasattr(r, 'success') and r.success) / len(context['execution_results'])
            self.ab_test_results[test_group]['performance'].append(success_rate)
        
        # Store detailed results
        self.ab_test_groups[test_group].append({
            'execution_id': execution_id,
            'reward': reward,
            'strategy_results': {name: result.reward for name, result in strategy_results.items()}
        })
    
    def _store_strategy_metrics(self, strategy_results: Dict[str, RewardStrategyResult], 
                               final_reward: float):
        """Store strategy metrics in database."""
        if not self.db_manager:
            return
        
        try:
            # Store individual strategy results
            for name, result in strategy_results.items():
                self.db_manager.execute("""
                    INSERT INTO reward_strategy_metrics 
                    (strategy_name, execution_id, reward_contribution, computation_time_ms)
                    VALUES (?, ?, ?, ?)
                """, (name, time.time(), result.reward, result.computation_time_ms))
            
            # Commit changes
            self.db_manager.commit()
        except Exception as e:
            logger.error(f"Failed to store strategy metrics: {e}")
    
    def update_strategy_weights(self, performance_data: Dict[str, float]):
        """Update strategy weights based on performance."""
        # Normalize performance scores
        total_performance = sum(performance_data.values())
        if total_performance <= 0:
            return
        
        # Update weights proportional to performance
        for strategy, performance in performance_data.items():
            if strategy in self.strategy_weights:
                # Exponential moving average update
                alpha = 0.1
                new_weight = performance / total_performance
                self.strategy_weights[strategy] = (
                    alpha * new_weight + (1 - alpha) * self.strategy_weights[strategy]
                )
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] /= total_weight
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all strategies."""
        report = {}
        
        for name, perf in self.strategy_performance.items():
            avg_reward = perf['total_reward'] / perf['execution_count'] if perf['execution_count'] > 0 else 0
            report[name] = {
                'average_reward': avg_reward,
                'execution_count': perf['execution_count'],
                'avg_computation_time_ms': perf['avg_computation_time'],
                'success_correlation': perf['success_correlation'],
                'current_weight': self.strategy_weights.get(name, 0.0)
            }
        
        return report
    
    def get_ab_test_results(self) -> Dict[str, Any]:
        """Get A/B test results summary."""
        if not self.ab_testing_enabled:
            return {'error': 'A/B testing not enabled'}
        
        results = {}
        for group, data in self.ab_test_results.items():
            if data['rewards']:
                results[group] = {
                    'avg_reward': np.mean(data['rewards']),
                    'std_reward': np.std(data['rewards']),
                    'avg_performance': np.mean(data['performance']) if data['performance'] else 0,
                    'sample_size': len(data['rewards'])
                }
        
        # Statistical significance test (simplified t-test)
        if len(results) >= 2 and 'control' in results:
            control_rewards = self.ab_test_results['control']['rewards']
            for group in results:
                if group != 'control':
                    test_rewards = self.ab_test_results[group]['rewards']
                    if len(control_rewards) > 10 and len(test_rewards) > 10:
                        # Simple t-test approximation
                        t_stat = (np.mean(test_rewards) - np.mean(control_rewards)) / (
                            np.sqrt(np.var(test_rewards) / len(test_rewards) + 
                                   np.var(control_rewards) / len(control_rewards))
                        )
                        results[group]['t_statistic'] = t_stat
                        results[group]['significant'] = abs(t_stat) > 1.96  # 95% confidence
        
        return results
    
    def set_combination_method(self, method: str):
        """Set the method for combining strategy rewards."""
        valid_methods = ['weighted_average', 'max', 'voting']
        if method in valid_methods:
            self.combination_method = method
            logger.info(f"Set combination method to: {method}")
        else:
            logger.warning(f"Invalid combination method: {method}")
    
    def enable_ab_testing(self, groups: List[str]):
        """Enable A/B testing with specified groups."""
        self.ab_testing_enabled = True
        # Reset test data
        self.ab_test_groups.clear()
        self.ab_test_results.clear()
        logger.info(f"Enabled A/B testing with groups: {groups}")
    
    def disable_ab_testing(self):
        """Disable A/B testing."""
        self.ab_testing_enabled = False
        logger.info("Disabled A/B testing")
    
    def save_state(self) -> Dict[str, Any]:
        """Save manager state."""
        state = {
            'strategy_weights': dict(self.strategy_weights),
            'combination_method': self.combination_method,
            'strategy_performance': dict(self.strategy_performance),
            'ab_testing_enabled': self.ab_testing_enabled
        }
        
        # Save individual strategy states
        state['strategy_states'] = {}
        for name, strategy in self.strategies.items():
            state['strategy_states'][name] = strategy.save_state()
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load manager state."""
        self.strategy_weights = state.get('strategy_weights', self.strategy_weights)
        self.combination_method = state.get('combination_method', self.combination_method)
        self.strategy_performance = defaultdict(
            lambda: {'total_reward': 0.0, 'execution_count': 0, 
                    'avg_computation_time': 0.0, 'success_correlation': 0.0},
            state.get('strategy_performance', {})
        )
        self.ab_testing_enabled = state.get('ab_testing_enabled', False)
        
        # Load individual strategy states
        if 'strategy_states' in state:
            for name, strategy_state in state['strategy_states'].items():
                if name in self.strategies:
                    self.strategies[name].load_state(strategy_state)