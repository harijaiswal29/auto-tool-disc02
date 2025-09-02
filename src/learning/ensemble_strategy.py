"""
Ensemble Strategy System for combining multiple learning approaches.

This module implements an ensemble of strategies that dynamically
adjusts weights based on performance to leverage strengths of different
approaches.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import random

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Track performance metrics for a strategy."""
    name: str
    weight: float
    min_weight: float
    max_weight: float
    selections: deque = field(default_factory=lambda: deque(maxlen=100))
    successes: deque = field(default_factory=lambda: deque(maxlen=100))
    rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_outcome(self, selected: bool, success: bool, reward: float, confidence: float):
        """Record strategy outcome."""
        self.selections.append(selected)
        if selected:
            self.successes.append(success)
            self.rewards.append(reward)
            self.confidence_scores.append(confidence)
    
    def get_success_rate(self, window: int = 50) -> float:
        """Calculate recent success rate."""
        if not self.successes:
            return 0.5  # Default neutral rate
        
        recent = list(self.successes)[-window:]
        return sum(recent) / len(recent) if recent else 0.5
    
    def get_average_reward(self, window: int = 50) -> float:
        """Calculate average recent reward."""
        if not self.rewards:
            return 0.0
        
        recent = list(self.rewards)[-window:]
        return np.mean(recent) if recent else 0.0
    
    def get_selection_rate(self, window: int = 50) -> float:
        """Calculate how often this strategy was selected."""
        if not self.selections:
            return 0.0
        
        recent = list(self.selections)[-window:]
        return sum(recent) / len(recent) if recent else 0.0


class BaseStrategy(ABC):
    """Base class for ensemble strategies."""
    
    @abstractmethod
    def select_action(self, state: np.ndarray, available_actions: List[Any],
                     context: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Select an action given state and context.
        
        Args:
            state: Current state vector
            available_actions: List of available actions
            context: Additional context information
            
        Returns:
            Tuple of (selected_action, confidence_score)
        """
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: Any, reward: float,
              next_state: np.ndarray, done: bool):
        """Update strategy based on experience."""
        pass


class RandomStrategy(BaseStrategy):
    """Random selection strategy for baseline and exploration."""
    
    def select_action(self, state: np.ndarray, available_actions: List[Any],
                     context: Dict[str, Any]) -> Tuple[Any, float]:
        """Select random action."""
        if not available_actions:
            return None, 0.0
        
        action = random.choice(available_actions)
        confidence = 1.0 / len(available_actions)  # Uniform confidence
        return action, confidence
    
    def update(self, state: np.ndarray, action: Any, reward: float,
              next_state: np.ndarray, done: bool):
        """No learning for random strategy."""
        pass


class RuleBasedStrategy(BaseStrategy):
    """Rule-based strategy using domain heuristics."""
    
    def __init__(self):
        """Initialize rule-based strategy."""
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize domain-specific rules."""
        return [
            {
                'condition': lambda ctx: 'search' in ctx.get('query', '').lower(),
                'preferred_tools': ['search_mcp', 'brave_search'],
                'priority': 0.9
            },
            {
                'condition': lambda ctx: 'file' in ctx.get('query', '').lower(),
                'preferred_tools': ['filesystem_mcp'],
                'priority': 0.8
            },
            {
                'condition': lambda ctx: 'database' in ctx.get('query', '').lower() or 
                                       'sql' in ctx.get('query', '').lower(),
                'preferred_tools': ['postgres_mcp', 'sqlite_mcp'],
                'priority': 0.85
            }
        ]
    
    def select_action(self, state: np.ndarray, available_actions: List[Any],
                     context: Dict[str, Any]) -> Tuple[Any, float]:
        """Select action based on rules."""
        if not available_actions:
            return None, 0.0
        
        # Apply rules
        scores = {}
        for action in available_actions:
            scores[action] = self._score_action(action, context)
        
        # Select highest scoring action
        if scores:
            best_action = max(scores, key=scores.get)
            confidence = scores[best_action]
            return best_action, confidence
        
        # Fallback to random
        return random.choice(available_actions), 0.5
    
    def _score_action(self, action: Any, context: Dict[str, Any]) -> float:
        """Score an action based on rules."""
        score = 0.5  # Base score
        
        # Convert action to tool names if needed
        tool_names = action if isinstance(action, list) else [str(action)]
        
        for rule in self.rules:
            if rule['condition'](context):
                for tool in tool_names:
                    if any(pref in str(tool) for pref in rule['preferred_tools']):
                        score = max(score, rule['priority'])
        
        return score
    
    def update(self, state: np.ndarray, action: Any, reward: float,
              next_state: np.ndarray, done: bool):
        """Could update rule weights based on outcomes."""
        pass


class ExplorationFocusedStrategy(BaseStrategy):
    """Strategy focused on exploration and discovering new combinations."""
    
    def __init__(self):
        """Initialize exploration strategy."""
        self.action_counts = {}
        self.action_values = {}
        self.total_selections = 0
        self.exploration_bonus = 0.5
    
    def select_action(self, state: np.ndarray, available_actions: List[Any],
                     context: Dict[str, Any]) -> Tuple[Any, float]:
        """Select action with UCB-like exploration bonus."""
        if not available_actions:
            return None, 0.0
        
        self.total_selections += 1
        
        # Calculate UCB scores
        scores = {}
        for action in available_actions:
            action_key = str(action)
            
            # Get action statistics
            count = self.action_counts.get(action_key, 0)
            value = self.action_values.get(action_key, 0.5)
            
            # UCB formula
            if count == 0:
                score = float('inf')  # Unexplored action
            else:
                exploration_term = self.exploration_bonus * np.sqrt(
                    np.log(self.total_selections) / count
                )
                score = value + exploration_term
            
            scores[action] = score
        
        # Select action with highest UCB score
        best_action = max(scores, key=scores.get)
        
        # Confidence based on exploration vs exploitation
        count = self.action_counts.get(str(best_action), 0)
        if count == 0:
            confidence = 0.3  # Low confidence for unexplored
        else:
            confidence = min(0.9, 0.5 + count * 0.02)  # Increase with experience
        
        return best_action, confidence
    
    def update(self, state: np.ndarray, action: Any, reward: float,
              next_state: np.ndarray, done: bool):
        """Update action statistics."""
        action_key = str(action)
        
        # Update counts
        if action_key not in self.action_counts:
            self.action_counts[action_key] = 0
            self.action_values[action_key] = 0.5
        
        self.action_counts[action_key] += 1
        
        # Update value estimate (running average)
        count = self.action_counts[action_key]
        old_value = self.action_values[action_key]
        
        # Normalize reward to 0-1 range
        normalized_reward = (reward + 100) / 200  # Assuming -100 to 100 range
        normalized_reward = np.clip(normalized_reward, 0, 1)
        
        # Incremental update
        new_value = old_value + (normalized_reward - old_value) / count
        self.action_values[action_key] = new_value


class EnsembleManager:
    """
    Manages ensemble of strategies with dynamic weighting.
    
    Features:
    - Multiple strategy instances
    - Dynamic weight adjustment
    - Weighted voting for action selection
    - Performance tracking per strategy
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble manager.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Performance tracking
        self.strategy_performance = {}
        for strategy_config in self.config.get('strategies', []):
            self.strategy_performance[strategy_config['name']] = StrategyPerformance(
                name=strategy_config['name'],
                weight=strategy_config.get('weight', 0.25),
                min_weight=strategy_config.get('min_weight', 0.1),
                max_weight=strategy_config.get('max_weight', 0.5)
            )
        
        # Ensemble parameters
        self.voting_method = self.config.get('voting_method', 'weighted_average')
        self.weight_update_rate = self.config.get('weight_update_rate', 0.01)
        self.performance_window = self.config.get('performance_tracking_window', 50)
        self.diversity_bonus = self.config.get('diversity_bonus', 0.1)
        
        # Tracking
        self.episode_count = 0
        self.last_selected_strategy = None
        
        logger.info(f"Ensemble manager initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> Dict[str, BaseStrategy]:
        """Initialize strategy instances."""
        strategies = {}
        
        # Note: In real implementation, would initialize actual Q-learning instances
        # For now, using placeholder strategies
        strategy_configs = self.config.get('strategies', [])
        
        for config in strategy_configs:
            name = config['name']
            
            if name == 'random':
                strategies[name] = RandomStrategy()
            elif name == 'rule_based':
                strategies[name] = RuleBasedStrategy()
            elif name == 'exploration_focused':
                strategies[name] = ExplorationFocusedStrategy()
            else:
                # Placeholder for Q-learning strategies
                strategies[name] = RandomStrategy()  # Would be actual implementation
        
        return strategies
    
    def select_action(self, state: np.ndarray, available_actions: List[Any],
                     context: Dict[str, Any]) -> Tuple[Any, float, str]:
        """
        Select action using ensemble voting.
        
        Args:
            state: Current state
            available_actions: Available actions
            context: Additional context
            
        Returns:
            Tuple of (selected_action, confidence, strategy_name)
        """
        if not self.enabled or not available_actions:
            # Fallback to random
            return random.choice(available_actions) if available_actions else None, 0.0, "random"
        
        # Collect votes from each strategy
        votes = {}
        confidences = {}
        
        for name, strategy in self.strategies.items():
            if name in self.strategy_performance:
                action, confidence = strategy.select_action(state, available_actions, context)
                votes[name] = action
                confidences[name] = confidence
        
        # Apply voting method
        if self.voting_method == 'weighted_average':
            selected_action, selected_strategy = self._weighted_voting(votes, confidences)
        elif self.voting_method == 'majority':
            selected_action, selected_strategy = self._majority_voting(votes)
        else:
            selected_action, selected_strategy = self._weighted_voting(votes, confidences)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(
            selected_action, votes, confidences
        )
        
        self.last_selected_strategy = selected_strategy
        
        # Mark which strategies agreed
        for name in votes:
            agreed = (votes[name] == selected_action)
            self.strategy_performance[name].add_outcome(
                agreed, False, 0.0, confidences.get(name, 0.5)
            )
        
        return selected_action, ensemble_confidence, selected_strategy
    
    def _weighted_voting(self, votes: Dict[str, Any], 
                        confidences: Dict[str, float]) -> Tuple[Any, str]:
        """
        Perform weighted voting.
        
        Args:
            votes: Strategy votes
            confidences: Strategy confidences
            
        Returns:
            Tuple of (selected_action, strategy_name)
        """
        # Calculate weighted scores for each unique action
        action_scores = {}
        action_strategies = {}
        
        for name, action in votes.items():
            if action is None:
                continue
            
            weight = self.strategy_performance[name].weight
            confidence = confidences.get(name, 0.5)
            score = weight * confidence
            
            action_key = str(action)
            if action_key not in action_scores:
                action_scores[action_key] = 0
                action_strategies[action_key] = []
            
            action_scores[action_key] += score
            action_strategies[action_key].append(name)
        
        if not action_scores:
            return None, "none"
        
        # Select action with highest weighted score
        best_action_key = max(action_scores, key=action_scores.get)
        
        # Find actual action object
        best_action = None
        for name, action in votes.items():
            if str(action) == best_action_key:
                best_action = action
                break
        
        # Determine primary strategy
        primary_strategy = action_strategies[best_action_key][0]
        
        return best_action, primary_strategy
    
    def _majority_voting(self, votes: Dict[str, Any]) -> Tuple[Any, str]:
        """
        Perform majority voting.
        
        Args:
            votes: Strategy votes
            
        Returns:
            Tuple of (selected_action, strategy_name)
        """
        # Count votes for each action
        action_counts = {}
        action_strategies = {}
        
        for name, action in votes.items():
            if action is None:
                continue
            
            action_key = str(action)
            if action_key not in action_counts:
                action_counts[action_key] = 0
                action_strategies[action_key] = []
            
            action_counts[action_key] += 1
            action_strategies[action_key].append(name)
        
        if not action_counts:
            return None, "none"
        
        # Select action with most votes
        best_action_key = max(action_counts, key=action_counts.get)
        
        # Find actual action object
        best_action = None
        for name, action in votes.items():
            if str(action) == best_action_key:
                best_action = action
                break
        
        primary_strategy = action_strategies[best_action_key][0]
        
        return best_action, primary_strategy
    
    def _calculate_ensemble_confidence(self, selected_action: Any,
                                      votes: Dict[str, Any],
                                      confidences: Dict[str, float]) -> float:
        """
        Calculate ensemble confidence score.
        
        Args:
            selected_action: Selected action
            votes: Strategy votes
            confidences: Strategy confidences
            
        Returns:
            Ensemble confidence score
        """
        if selected_action is None:
            return 0.0
        
        # Calculate agreement level
        total_weight = 0
        agreement_weight = 0
        
        for name, action in votes.items():
            if name in self.strategy_performance:
                weight = self.strategy_performance[name].weight
                total_weight += weight
                
                if action == selected_action:
                    confidence = confidences.get(name, 0.5)
                    agreement_weight += weight * confidence
        
        if total_weight == 0:
            return 0.5
        
        # Base confidence on weighted agreement
        ensemble_confidence = agreement_weight / total_weight
        
        # Apply diversity bonus if strategies disagree (encourages exploration)
        unique_actions = len(set(str(a) for a in votes.values() if a is not None))
        if unique_actions > 1:
            diversity_factor = 1.0 - self.diversity_bonus * (unique_actions - 1) / len(votes)
            ensemble_confidence *= diversity_factor
        
        return np.clip(ensemble_confidence, 0.0, 1.0)
    
    def update(self, state: np.ndarray, action: Any, reward: float,
              next_state: np.ndarray, done: bool):
        """
        Update strategies and weights based on outcome.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Episode complete flag
        """
        self.episode_count += 1
        
        # Update all strategies
        for strategy in self.strategies.values():
            strategy.update(state, action, reward, next_state, done)
        
        # Update performance for the strategy that was selected
        if self.last_selected_strategy and self.last_selected_strategy in self.strategy_performance:
            perf = self.strategy_performance[self.last_selected_strategy]
            success = reward > 0  # Simple success criterion
            perf.add_outcome(True, success, reward, 1.0)
        
        # Periodically update weights
        if self.episode_count % 10 == 0:
            self._update_weights()
    
    def _update_weights(self):
        """Update strategy weights based on performance."""
        # Calculate performance metrics for each strategy
        performances = {}
        
        for name, perf in self.strategy_performance.items():
            # Combine success rate and average reward
            success_rate = perf.get_success_rate(self.performance_window)
            avg_reward = perf.get_average_reward(self.performance_window)
            
            # Normalize reward to 0-1 scale
            normalized_reward = (avg_reward + 100) / 200
            normalized_reward = np.clip(normalized_reward, 0, 1)
            
            # Combined performance score
            performance_score = 0.6 * success_rate + 0.4 * normalized_reward
            performances[name] = performance_score
        
        # Update weights based on relative performance
        if performances:
            avg_performance = np.mean(list(performances.values()))
            
            for name, score in performances.items():
                if name in self.strategy_performance:
                    perf = self.strategy_performance[name]
                    
                    # Calculate weight adjustment
                    performance_diff = score - avg_performance
                    weight_change = self.weight_update_rate * performance_diff
                    
                    # Update weight with bounds
                    new_weight = perf.weight + weight_change
                    new_weight = np.clip(new_weight, perf.min_weight, perf.max_weight)
                    perf.weight = new_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(p.weight for p in self.strategy_performance.values())
            if total_weight > 0:
                for perf in self.strategy_performance.values():
                    perf.weight /= total_weight
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get ensemble status.
        
        Returns:
            Status dictionary
        """
        status = {
            'enabled': self.enabled,
            'episode_count': self.episode_count,
            'voting_method': self.voting_method,
            'strategies': {}
        }
        
        for name, perf in self.strategy_performance.items():
            status['strategies'][name] = {
                'weight': perf.weight,
                'success_rate': perf.get_success_rate(),
                'average_reward': perf.get_average_reward(),
                'selection_rate': perf.get_selection_rate()
            }
        
        return status