"""Information-theoretic reward calculation.

This module implements curiosity-driven rewards based on information gain,
entropy, and novelty detection.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import time
import logging
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

from .base_strategy import BaseRewardStrategy, RewardStrategyResult

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger(__name__)


class InformationTheoreticRewardCalculator(BaseRewardStrategy):
    """Calculates rewards based on information gain and curiosity."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize information-theoretic reward calculator.
        
        Args:
            config: Configuration with keys:
                - curiosity_weight: Weight for curiosity bonus (default: 0.1)
                - entropy_bonus: Bonus for high-entropy actions (default: 0.05)
                - novelty_threshold: Threshold for novelty detection (default: 0.7)
                - state_visit_decay: Decay rate for state visit counts (default: 0.99)
                - mutual_info_weight: Weight for mutual information reward (default: 0.15)
        """
        super().__init__(config)
        
    def _initialize_strategy(self):
        """Initialize information-theoretic components."""
        self.curiosity_weight = self.config.get('curiosity_weight', 0.1)
        self.entropy_bonus = self.config.get('entropy_bonus', 0.05)
        self.novelty_threshold = self.config.get('novelty_threshold', 0.7)
        self.state_visit_decay = self.config.get('state_visit_decay', 0.99)
        self.mutual_info_weight = self.config.get('mutual_info_weight', 0.15)
        
        # State visitation tracking
        self.state_visits = defaultdict(float)
        self.state_action_visits = defaultdict(float)
        
        # State embedding history for novelty detection
        self.state_history = []
        self.max_history_size = 1000
        
        # Action probability distributions
        self.action_probs = defaultdict(lambda: defaultdict(float))
        self.state_transition_counts = defaultdict(lambda: defaultdict(int))
        
        # Information gain tracking
        self.information_gains = []
        
        # Predictive model for state transitions (simplified)
        self.transition_predictions = {}
        
        logger.info(f"Initialized InformationTheoreticRewardCalculator with curiosity_weight={self.curiosity_weight}")
    
    def calculate(self, 
                  state: np.ndarray,
                  action: List[str],
                  next_state: np.ndarray,
                  execution_results: List[Any],
                  context: Dict[str, Any]) -> RewardStrategyResult:
        """Calculate information-theoretic rewards."""
        start_time = time.time()
        
        # Calculate various information-theoretic components
        curiosity_reward = self._calculate_curiosity_reward(state, action, next_state)
        entropy_reward = self._calculate_entropy_reward(state, action)
        novelty_reward = self._calculate_novelty_reward(state, next_state)
        information_gain = self._calculate_information_gain(state, action, next_state)
        mutual_info_reward = self._calculate_mutual_information_reward(state, action, execution_results)
        
        # Calculate prediction error (surprise)
        surprise_reward = self._calculate_surprise_reward(state, action, next_state)
        
        # Update visitation counts and history
        self._update_visitation_counts(state, action)
        self._update_state_history(state, next_state)
        
        # Combine all information-theoretic rewards
        final_reward = (
            self.curiosity_weight * curiosity_reward +
            self.entropy_bonus * entropy_reward +
            novelty_reward * 0.2 +
            information_gain * 0.15 +
            self.mutual_info_weight * mutual_info_reward +
            surprise_reward * 0.1
        )
        
        # Scale based on execution success to encourage useful exploration
        if execution_results:
            success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
            final_reward *= (0.5 + 0.5 * success_rate)  # Scale between 0.5x and 1x
        
        # Clip to reasonable range
        final_reward = self._clip_reward(final_reward, -0.5, 1.0)
        
        computation_time = (time.time() - start_time) * 1000
        
        return RewardStrategyResult(
            reward=final_reward,
            components={
                'curiosity': curiosity_reward * self.curiosity_weight,
                'entropy': entropy_reward * self.entropy_bonus,
                'novelty': novelty_reward * 0.2,
                'information_gain': information_gain * 0.15,
                'mutual_information': mutual_info_reward * self.mutual_info_weight,
                'surprise': surprise_reward * 0.1
            },
            metadata={
                'state_visits': self.state_visits[self._state_to_key(state)],
                'state_novelty': self._compute_state_novelty(state),
                'action_entropy': self._compute_action_entropy(state),
                'total_states_seen': len(self.state_visits)
            },
            computation_time_ms=computation_time
        )
    
    def _calculate_curiosity_reward(self, state: np.ndarray, action: List[str], 
                                   next_state: np.ndarray) -> float:
        """Calculate curiosity-driven exploration reward."""
        # Reward for visiting rarely seen state-action pairs
        state_action_key = self._state_action_to_key(state, action)
        visit_count = self.state_action_visits[state_action_key]
        
        # UCB-style curiosity bonus
        if visit_count == 0:
            curiosity = 1.0
        else:
            total_visits = sum(self.state_action_visits.values())
            curiosity = np.sqrt(2 * np.log(max(total_visits, 1)) / visit_count)
        
        # Bonus for state transition novelty
        transition_key = f"{self._state_to_key(state)}_{self._state_to_key(next_state)}"
        if transition_key not in self.state_transition_counts[self._state_to_key(state)]:
            curiosity += 0.3
        
        return min(curiosity, 1.0)
    
    def _calculate_entropy_reward(self, state: np.ndarray, action: List[str]) -> float:
        """Reward high-entropy (diverse) action selection."""
        state_key = self._state_to_key(state)
        
        # Calculate action distribution entropy for this state
        action_counts = self.action_probs[state_key]
        if not action_counts:
            return 1.0  # Maximum entropy for unexplored state
        
        # Convert to probability distribution
        total_count = sum(action_counts.values())
        if total_count == 0:
            return 1.0
        
        probs = np.array([count / total_count for count in action_counts.values()])
        
        # Calculate entropy
        action_entropy = entropy(probs) if len(probs) > 1 else 0.0
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(action_counts)) if len(action_counts) > 1 else 1.0
        normalized_entropy = action_entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _calculate_novelty_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """Reward for discovering novel states."""
        # Check novelty of current state
        state_novelty = self._compute_state_novelty(state)
        next_state_novelty = self._compute_state_novelty(next_state)
        
        # Reward for reaching novel states
        novelty_reward = 0.0
        if state_novelty > self.novelty_threshold:
            novelty_reward += 0.5
        if next_state_novelty > self.novelty_threshold:
            novelty_reward += 0.5
        
        # Bonus for large state space exploration
        state_distance = np.linalg.norm(next_state - state)
        if state_distance > np.mean([np.linalg.norm(s) for s in self.state_history[-10:]] or [1.0]):
            novelty_reward += 0.3
        
        return novelty_reward
    
    def _calculate_information_gain(self, state: np.ndarray, action: List[str], 
                                   next_state: np.ndarray) -> float:
        """Calculate information gain from state transition."""
        # Estimate reduction in uncertainty about environment
        state_key = self._state_to_key(state)
        
        # Prior entropy (uncertainty before action)
        prior_transitions = self.state_transition_counts[state_key]
        if not prior_transitions:
            prior_entropy = 1.0  # Maximum uncertainty
        else:
            total = sum(prior_transitions.values())
            probs = [count / total for count in prior_transitions.values()]
            prior_entropy = entropy(probs) if len(probs) > 1 else 0.0
        
        # Posterior entropy (uncertainty after observing transition)
        # Update counts temporarily
        next_state_key = self._state_to_key(next_state)
        prior_transitions[next_state_key] += 1
        total = sum(prior_transitions.values())
        probs = [count / total for count in prior_transitions.values()]
        posterior_entropy = entropy(probs) if len(probs) > 1 else 0.0
        prior_transitions[next_state_key] -= 1  # Restore
        
        # Information gain is reduction in entropy
        info_gain = prior_entropy - posterior_entropy
        
        return max(0, info_gain)
    
    def _calculate_mutual_information_reward(self, state: np.ndarray, action: List[str], 
                                           execution_results: List[Any]) -> float:
        """Calculate mutual information between actions and outcomes."""
        # Simplified mutual information calculation
        # I(Action; Outcome) = H(Outcome) - H(Outcome|Action)
        
        if not execution_results:
            return 0.0
        
        # Outcome entropy
        success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
        outcome_probs = [success_rate, 1 - success_rate]
        outcome_entropy = entropy(outcome_probs) if success_rate not in [0, 1] else 0.0
        
        # Conditional entropy (simplified)
        action_key = '_'.join(sorted(action))
        if action_key in self.action_probs:
            # Use historical success rates for this action
            conditional_entropy = outcome_entropy * 0.8  # Simplified
        else:
            conditional_entropy = outcome_entropy
        
        mutual_info = outcome_entropy - conditional_entropy
        
        return mutual_info
    
    def _calculate_surprise_reward(self, state: np.ndarray, action: List[str], 
                                   next_state: np.ndarray) -> float:
        """Calculate reward based on prediction error (surprise)."""
        # Create transition key
        transition_key = f"{self._state_to_key(state)}_{self._action_to_key(action)}"
        
        if transition_key in self.transition_predictions:
            # Calculate prediction error
            predicted_state = self.transition_predictions[transition_key]
            prediction_error = np.linalg.norm(next_state - predicted_state)
            
            # Reward is proportional to prediction error (surprise)
            surprise = 1.0 / (1.0 + np.exp(-prediction_error))
        else:
            # First time seeing this transition - maximum surprise
            surprise = 1.0
        
        # Update prediction (simple exponential moving average)
        if transition_key in self.transition_predictions:
            self.transition_predictions[transition_key] = (
                0.9 * self.transition_predictions[transition_key] + 0.1 * next_state
            )
        else:
            self.transition_predictions[transition_key] = next_state.copy()
        
        return surprise
    
    def _compute_state_novelty(self, state: np.ndarray) -> float:
        """Compute novelty score for a state."""
        if not self.state_history:
            return 1.0
        
        # Compare with recent states
        recent_states = self.state_history[-100:]
        
        # Compute similarities
        state_normalized = state / (np.linalg.norm(state) + 1e-8)
        similarities = []
        
        for hist_state in recent_states:
            hist_normalized = hist_state / (np.linalg.norm(hist_state) + 1e-8)
            sim = np.dot(state_normalized, hist_normalized)
            similarities.append(sim)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        return novelty
    
    def _compute_action_entropy(self, state: np.ndarray) -> float:
        """Compute entropy of action distribution for a state."""
        state_key = self._state_to_key(state)
        action_counts = self.action_probs[state_key]
        
        if not action_counts:
            return 0.0
        
        total = sum(action_counts.values())
        if total == 0:
            return 0.0
        
        probs = [count / total for count in action_counts.values()]
        return entropy(probs) if len(probs) > 1 else 0.0
    
    def _update_visitation_counts(self, state: np.ndarray, action: List[str]):
        """Update state and state-action visitation counts."""
        # Decay existing counts
        for key in list(self.state_visits.keys()):
            self.state_visits[key] *= self.state_visit_decay
            if self.state_visits[key] < 0.01:
                del self.state_visits[key]
        
        for key in list(self.state_action_visits.keys()):
            self.state_action_visits[key] *= self.state_visit_decay
            if self.state_action_visits[key] < 0.01:
                del self.state_action_visits[key]
        
        # Increment current visits
        state_key = self._state_to_key(state)
        state_action_key = self._state_action_to_key(state, action)
        
        self.state_visits[state_key] += 1.0
        self.state_action_visits[state_action_key] += 1.0
        
        # Update action probabilities
        action_key = self._action_to_key(action)
        self.action_probs[state_key][action_key] += 1.0
    
    def _update_state_history(self, state: np.ndarray, next_state: np.ndarray):
        """Update state history for novelty detection."""
        self.state_history.append(state.copy())
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
        
        # Update transition counts
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        self.state_transition_counts[state_key][next_state_key] += 1
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key."""
        # Use a coarser discretization for better generalization
        discretized = np.round(state, decimals=2)
        return str(hash(discretized.tobytes()))
    
    def _action_to_key(self, action: List[str]) -> str:
        """Convert action to hashable key."""
        return '_'.join(sorted(action))
    
    def _state_action_to_key(self, state: np.ndarray, action: List[str]) -> str:
        """Convert state-action pair to hashable key."""
        return f"{self._state_to_key(state)}_{self._action_to_key(action)}"
    
    def update_parameters(self, feedback: Dict[str, Any]):
        """Update information-theoretic parameters based on feedback."""
        # Adjust curiosity weight based on exploration effectiveness
        if 'exploration_quality' in feedback:
            quality = feedback['exploration_quality']
            if quality > 0.7:
                # Good exploration, maintain or increase curiosity
                self.curiosity_weight = min(0.3, self.curiosity_weight * 1.05)
            elif quality < 0.3:
                # Poor exploration, reduce curiosity
                self.curiosity_weight = max(0.05, self.curiosity_weight * 0.95)
        
        # Adjust novelty threshold based on state space coverage
        if 'state_space_coverage' in feedback:
            coverage = feedback['state_space_coverage']
            if coverage < 0.2:
                # Low coverage, reduce novelty threshold
                self.novelty_threshold = max(0.5, self.novelty_threshold * 0.95)
            elif coverage > 0.8:
                # High coverage, increase novelty threshold
                self.novelty_threshold = min(0.9, self.novelty_threshold * 1.05)
    
    def get_exploration_metrics(self) -> Dict[str, Any]:
        """Get metrics about exploration progress."""
        return {
            'unique_states': len(self.state_visits),
            'unique_state_actions': len(self.state_action_visits),
            'avg_state_novelty': np.mean([self._compute_state_novelty(s) for s in self.state_history[-10:]]) if self.state_history else 0.0,
            'exploration_entropy': np.mean([self._compute_action_entropy(s) for s in self.state_history[-10:]]) if self.state_history else 0.0,
            'information_gains': np.mean(self.information_gains[-100:]) if self.information_gains else 0.0
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save strategy state."""
        state = super().save_state()
        state.update({
            'state_visits_count': len(self.state_visits),
            'state_action_visits_count': len(self.state_action_visits),
            'curiosity_weight': self.curiosity_weight,
            'novelty_threshold': self.novelty_threshold,
            'exploration_metrics': self.get_exploration_metrics()
        })
        return state