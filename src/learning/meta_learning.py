"""
Meta-Learning Optimizer for adaptive hyperparameter tuning.

This module implements meta-learning capabilities to dynamically adjust
hyperparameters based on learning progress and performance trends.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterTrend(Enum):
    """Parameter adjustment trends."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"


@dataclass
class ParameterHistory:
    """Track parameter values and performance."""
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    performances: deque = field(default_factory=lambda: deque(maxlen=100))
    gradients: deque = field(default_factory=lambda: deque(maxlen=50))
    best_value: float = 0.0
    best_performance: float = 0.0
    current_value: float = 0.0
    
    def add_sample(self, value: float, performance: float):
        """Add a parameter-performance sample."""
        self.values.append(value)
        self.performances.append(performance)
        
        # Track best
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_value = value
        
        self.current_value = value
        
        # Calculate gradient
        if len(self.values) >= 2:
            gradient = (self.performances[-1] - self.performances[-2]) / \
                      (self.values[-1] - self.values[-2] + 1e-8)
            self.gradients.append(gradient)
    
    def get_trend(self, window: int = 10) -> ParameterTrend:
        """Analyze parameter trend."""
        if len(self.gradients) < window:
            return ParameterTrend.STABLE
        
        recent_gradients = list(self.gradients)[-window:]
        avg_gradient = np.mean(recent_gradients)
        std_gradient = np.std(recent_gradients)
        
        if std_gradient > abs(avg_gradient) * 2:
            return ParameterTrend.OSCILLATING
        elif avg_gradient > 0.01:
            return ParameterTrend.INCREASING
        elif avg_gradient < -0.01:
            return ParameterTrend.DECREASING
        else:
            return ParameterTrend.STABLE


class MetaLearningOptimizer:
    """
    Meta-learning optimizer for dynamic hyperparameter adaptation.
    
    Features:
    - Performance-based parameter adjustment
    - Gradient-based optimization
    - Exploration-exploitation balance
    - Transfer learning support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize meta-learning optimizer.
        
        Args:
            config: Meta-learning configuration
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Learning parameters
        self.adaptation_rate = self.config.get('adaptation_rate', 0.001)
        self.update_interval = self.config.get('parameter_update_interval', 50)
        self.performance_window = self.config.get('performance_tracking_window', 100)
        
        # Parameter configuration
        self.adaptive_parameters = self.config.get('adaptive_parameters', [
            'learning_rate',
            'exploration_rate',
            'pattern_weight'
        ])
        
        self.parameter_bounds = self.config.get('hyperparameter_bounds', {
            'learning_rate': [0.0001, 0.1],
            'exploration_rate': [0.05, 0.5],
            'pattern_weight': [0.1, 0.6],
            'batch_size': [32, 256],
            'tau': [0.001, 0.01]
        })
        
        # Initialize parameter tracking
        self.parameter_history = {
            param: ParameterHistory() for param in self.adaptive_parameters
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.performance_window)
        self.episode_count = 0
        self.updates_performed = 0
        
        # Optimization state
        self.momentum = {param: 0.0 for param in self.adaptive_parameters}
        self.velocity = {param: 0.0 for param in self.adaptive_parameters}
        self.adam_m = {param: 0.0 for param in self.adaptive_parameters}
        self.adam_v = {param: 0.0 for param in self.adaptive_parameters}
        self.adam_t = 0
        
        # Transfer learning
        self.transfer_enabled = self.config.get('transfer_learning_enabled', True)
        self.knowledge_base = {}
        
        # Exploration parameters for meta-learning
        self.meta_exploration_rate = 0.1
        self.meta_exploration_decay = 0.999
        
        logger.info(f"Meta-learning optimizer initialized for parameters: {self.adaptive_parameters}")
    
    def update_performance(self, metrics: Dict[str, float]):
        """
        Update performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
        """
        if not self.enabled:
            return
        
        self.episode_count += 1
        
        # Calculate composite performance score
        performance_score = self._calculate_performance_score(metrics)
        self.performance_history.append(performance_score)
        
        # Check if should update parameters
        if self.episode_count % self.update_interval == 0:
            self._update_parameters(performance_score)
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite performance score.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Composite score
        """
        # Weighted combination of metrics
        weights = {
            'completion_rate': 0.4,
            'accuracy': 0.3,
            'reward': 0.2,
            'efficiency': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize reward to 0-1 scale
                if metric == 'reward':
                    # Assume reward range of -100 to 100
                    normalized_value = (metrics[metric] + 100) / 200
                else:
                    normalized_value = metrics[metric]
                
                score += normalized_value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _update_parameters(self, current_performance: float):
        """
        Update hyperparameters based on performance.
        
        Args:
            current_performance: Current performance score
        """
        self.updates_performed += 1
        
        for param_name in self.adaptive_parameters:
            if param_name not in self.parameter_history:
                continue
            
            # Get current parameter value (would be retrieved from actual system)
            current_value = self._get_current_parameter_value(param_name)
            
            # Record current state
            self.parameter_history[param_name].add_sample(
                current_value, current_performance
            )
            
            # Calculate adjustment
            new_value = self._calculate_parameter_update(
                param_name, current_value, current_performance
            )
            
            # Apply bounds
            new_value = self._apply_bounds(param_name, new_value)
            
            # Apply update (would update actual system)
            self._apply_parameter_update(param_name, new_value)
            
            logger.debug(f"Updated {param_name}: {current_value:.6f} -> {new_value:.6f}")
    
    def _calculate_parameter_update(self, param_name: str, 
                                   current_value: float,
                                   current_performance: float) -> float:
        """
        Calculate parameter update using adaptive optimization.
        
        Args:
            param_name: Parameter name
            current_value: Current parameter value
            current_performance: Current performance
            
        Returns:
            New parameter value
        """
        history = self.parameter_history[param_name]
        
        # Get performance gradient
        if len(history.performances) < 2:
            return current_value
        
        # Estimate gradient using finite differences
        gradient = self._estimate_gradient(param_name, current_value, current_performance)
        
        # Apply Adam optimizer
        new_value = self._adam_update(param_name, current_value, gradient)
        
        # Add exploration noise occasionally
        if np.random.random() < self.meta_exploration_rate:
            noise_scale = (self.parameter_bounds[param_name][1] - 
                          self.parameter_bounds[param_name][0]) * 0.05
            new_value += np.random.normal(0, noise_scale)
        
        # Decay exploration
        self.meta_exploration_rate *= self.meta_exploration_decay
        
        return new_value
    
    def _estimate_gradient(self, param_name: str, 
                          current_value: float,
                          current_performance: float) -> float:
        """
        Estimate performance gradient with respect to parameter.
        
        Args:
            param_name: Parameter name
            current_value: Current parameter value
            current_performance: Current performance
            
        Returns:
            Estimated gradient
        """
        history = self.parameter_history[param_name]
        
        if len(history.values) < 2:
            return 0.0
        
        # Use recent samples for gradient estimation
        window = min(10, len(history.values))
        recent_values = list(history.values)[-window:]
        recent_performances = list(history.performances)[-window:]
        
        # Linear regression to estimate gradient
        if len(set(recent_values)) > 1:  # Need variation in values
            X = np.array(recent_values).reshape(-1, 1)
            y = np.array(recent_performances)
            
            # Add small regularization
            X_mean = X.mean()
            y_mean = y.mean()
            
            numerator = np.sum((X - X_mean) * (y - y_mean))
            denominator = np.sum((X - X_mean) ** 2) + 1e-8
            
            gradient = numerator / denominator
        else:
            # No variation, use simple difference
            gradient = (recent_performances[-1] - recent_performances[-2]) / \
                      (recent_values[-1] - recent_values[-2] + 1e-8)
        
        return gradient
    
    def _adam_update(self, param_name: str, current_value: float, gradient: float) -> float:
        """
        Apply Adam optimizer update.
        
        Args:
            param_name: Parameter name
            current_value: Current value
            gradient: Estimated gradient
            
        Returns:
            Updated value
        """
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        self.adam_t += 1
        
        # Update biased first moment estimate
        self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * gradient
        
        # Update biased second raw moment estimate
        self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * gradient ** 2
        
        # Compute bias-corrected first moment estimate
        m_hat = self.adam_m[param_name] / (1 - beta1 ** self.adam_t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.adam_v[param_name] / (1 - beta2 ** self.adam_t)
        
        # Update parameter
        update = self.adaptation_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Gradient ascent (we want to maximize performance)
        new_value = current_value + update
        
        return new_value
    
    def _apply_bounds(self, param_name: str, value: float) -> float:
        """
        Apply parameter bounds.
        
        Args:
            param_name: Parameter name
            value: Proposed value
            
        Returns:
            Bounded value
        """
        if param_name in self.parameter_bounds:
            min_val, max_val = self.parameter_bounds[param_name]
            return np.clip(value, min_val, max_val)
        return value
    
    def _get_current_parameter_value(self, param_name: str) -> float:
        """
        Get current parameter value from system.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Current value
        """
        # This would interface with actual system
        # For now, return from history or default
        if param_name in self.parameter_history:
            history = self.parameter_history[param_name]
            if history.current_value > 0:
                return history.current_value
        
        # Default values
        defaults = {
            'learning_rate': 0.01,
            'exploration_rate': 0.3,
            'pattern_weight': 0.3,
            'batch_size': 64,
            'tau': 0.005
        }
        
        return defaults.get(param_name, 0.1)
    
    def _apply_parameter_update(self, param_name: str, new_value: float):
        """
        Apply parameter update to system.
        
        Args:
            param_name: Parameter name
            new_value: New value
        """
        # This would interface with actual system
        # For now, just log
        self.parameter_history[param_name].current_value = new_value
        
        # Store in knowledge base for transfer learning
        if self.transfer_enabled:
            self._update_knowledge_base(param_name, new_value)
    
    def _update_knowledge_base(self, param_name: str, value: float):
        """
        Update knowledge base for transfer learning.
        
        Args:
            param_name: Parameter name
            value: Parameter value
        """
        if param_name not in self.knowledge_base:
            self.knowledge_base[param_name] = []
        
        # Store with context
        entry = {
            'value': value,
            'performance': self.performance_history[-1] if self.performance_history else 0.0,
            'episode': self.episode_count,
            'context': self._get_learning_context()
        }
        
        self.knowledge_base[param_name].append(entry)
        
        # Keep only recent entries
        max_entries = 1000
        if len(self.knowledge_base[param_name]) > max_entries:
            self.knowledge_base[param_name] = self.knowledge_base[param_name][-max_entries:]
    
    def _get_learning_context(self) -> Dict[str, Any]:
        """
        Get current learning context for knowledge base.
        
        Returns:
            Context dictionary
        """
        return {
            'episode': self.episode_count,
            'updates': self.updates_performed,
            'avg_performance': np.mean(list(self.performance_history)) 
                if self.performance_history else 0.0,
            'performance_trend': self._get_performance_trend()
        }
    
    def _get_performance_trend(self) -> str:
        """
        Analyze performance trend.
        
        Returns:
            Trend description
        """
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = list(self.performance_history)[-10:]
        older = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def get_recommended_parameters(self) -> Dict[str, float]:
        """
        Get recommended parameter values.
        
        Returns:
            Dictionary of recommended parameters
        """
        recommendations = {}
        
        for param_name in self.adaptive_parameters:
            if param_name in self.parameter_history:
                history = self.parameter_history[param_name]
                
                # Use best performing value with some exploration
                if history.best_value > 0:
                    recommendations[param_name] = history.best_value
                else:
                    recommendations[param_name] = self._get_current_parameter_value(param_name)
            else:
                recommendations[param_name] = self._get_current_parameter_value(param_name)
        
        return recommendations
    
    def transfer_knowledge(self, source_knowledge: Dict[str, Any]):
        """
        Transfer knowledge from previous learning.
        
        Args:
            source_knowledge: Knowledge from source task
        """
        if not self.transfer_enabled:
            return
        
        # Merge knowledge bases
        for param_name, entries in source_knowledge.items():
            if param_name not in self.knowledge_base:
                self.knowledge_base[param_name] = []
            
            # Add relevant entries
            for entry in entries:
                # Filter based on relevance (could be more sophisticated)
                if entry['performance'] > 0.5:  # Only transfer successful parameters
                    self.knowledge_base[param_name].append(entry)
        
        # Initialize parameters based on transferred knowledge
        for param_name in self.adaptive_parameters:
            if param_name in self.knowledge_base and self.knowledge_base[param_name]:
                # Use best performing parameters from knowledge base
                best_entry = max(self.knowledge_base[param_name], 
                               key=lambda x: x['performance'])
                self.parameter_history[param_name].current_value = best_entry['value']
        
        logger.info(f"Transferred knowledge for {len(self.knowledge_base)} parameters")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get meta-learning status.
        
        Returns:
            Status dictionary
        """
        status = {
            'enabled': self.enabled,
            'episode_count': self.episode_count,
            'updates_performed': self.updates_performed,
            'current_parameters': self.get_recommended_parameters(),
            'performance_trend': self._get_performance_trend(),
            'parameter_trends': {}
        }
        
        # Add parameter trends
        for param_name in self.adaptive_parameters:
            if param_name in self.parameter_history:
                history = self.parameter_history[param_name]
                status['parameter_trends'][param_name] = {
                    'current': history.current_value,
                    'best': history.best_value,
                    'trend': history.get_trend().value,
                    'samples': len(history.values)
                }
        
        return status
    
    def save_state(self, filepath: str):
        """Save meta-learning state."""
        state = {
            'episode_count': self.episode_count,
            'updates_performed': self.updates_performed,
            'parameter_history': {},
            'knowledge_base': self.knowledge_base,
            'adam_state': {
                'adam_m': self.adam_m,
                'adam_v': self.adam_v,
                'adam_t': self.adam_t
            }
        }
        
        # Save parameter histories
        for param_name, history in self.parameter_history.items():
            state['parameter_history'][param_name] = {
                'values': list(history.values),
                'performances': list(history.performances),
                'best_value': history.best_value,
                'best_performance': history.best_performance,
                'current_value': history.current_value
            }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load meta-learning state."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.episode_count = state['episode_count']
            self.updates_performed = state['updates_performed']
            self.knowledge_base = state.get('knowledge_base', {})
            
            # Restore Adam state
            adam_state = state.get('adam_state', {})
            self.adam_m = adam_state.get('adam_m', {})
            self.adam_v = adam_state.get('adam_v', {})
            self.adam_t = adam_state.get('adam_t', 0)
            
            # Restore parameter histories
            for param_name, history_data in state['parameter_history'].items():
                history = ParameterHistory()
                history.values = deque(history_data['values'], maxlen=100)
                history.performances = deque(history_data['performances'], maxlen=100)
                history.best_value = history_data['best_value']
                history.best_performance = history_data['best_performance']
                history.current_value = history_data['current_value']
                self.parameter_history[param_name] = history
            
            logger.info(f"Loaded meta-learning state from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load meta-learning state: {e}")