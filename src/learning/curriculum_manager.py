"""
Curriculum Learning Manager for progressive difficulty training.

This module implements a curriculum learning system that progressively
increases task difficulty based on agent performance, enabling more
efficient learning and better final performance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np
from collections import deque
import json

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Task difficulty levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class CurriculumStage:
    """Represents a stage in the curriculum."""
    name: str
    difficulty: float  # 0.0 to 1.0
    min_performance: float  # Minimum performance to advance
    min_episodes: int  # Minimum episodes before advancement
    query_distribution: Dict[str, float]  # Distribution of query types
    performance_window: int = 50  # Episodes to consider for performance
    
    def __post_init__(self):
        """Validate stage configuration."""
        total = sum(self.query_distribution.values())
        if abs(total - 1.0) > 0.01:
            # Normalize if not exactly 1.0
            for key in self.query_distribution:
                self.query_distribution[key] /= total


@dataclass
class PerformanceMetrics:
    """Track performance metrics for curriculum decisions."""
    completion_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_count: int = 0
    stage_episode_count: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    
    def add_episode(self, completion: float, accuracy: float, reward: float):
        """Add episode results."""
        self.completion_rates.append(completion)
        self.accuracy_scores.append(accuracy)
        self.rewards.append(reward)
        self.episode_count += 1
        self.stage_episode_count += 1
        
        # Track consecutive performance
        if completion >= 0.5:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
    
    def get_recent_performance(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance statistics."""
        window = min(window, len(self.completion_rates))
        if window == 0:
            return {
                'completion_rate': 0.0,
                'accuracy': 0.0,
                'average_reward': 0.0
            }
        
        recent_completions = list(self.completion_rates)[-window:]
        recent_accuracies = list(self.accuracy_scores)[-window:]
        recent_rewards = list(self.rewards)[-window:]
        
        return {
            'completion_rate': np.mean(recent_completions),
            'accuracy': np.mean(recent_accuracies),
            'average_reward': np.mean(recent_rewards),
            'std_completion': np.std(recent_completions) if len(recent_completions) > 1 else 0.0,
            'std_reward': np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
        }
    
    def reset_stage_metrics(self):
        """Reset metrics for new stage."""
        self.stage_episode_count = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0


class CurriculumManager:
    """
    Manages curriculum learning progression.
    
    Features:
    - Progressive difficulty stages
    - Performance-based advancement
    - Adaptive query selection
    - Performance tracking and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize curriculum manager.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Initialize stages
        self.stages = self._initialize_stages()
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.stage_history = []
        
        # Adaptive parameters
        self.adaptive_progression = self.config.get('adaptive_progression', True)
        self.performance_window = self.config.get('performance_window', 50)
        self.stage_transition_threshold = self.config.get('stage_transition_threshold', 0.65)
        
        # Query pools by difficulty
        self.query_pools = self._initialize_query_pools()
        
        # Advancement criteria
        self.advancement_checks = {
            'performance': True,
            'episodes': True,
            'stability': True
        }
        
        logger.info(f"Curriculum manager initialized with {len(self.stages)} stages")
    
    def _initialize_stages(self) -> List[CurriculumStage]:
        """Initialize curriculum stages from config."""
        stages_config = self.config.get('stages', [])
        
        if not stages_config:
            # Default stages if not configured
            stages_config = [
                {
                    'name': 'simple',
                    'difficulty': 0.3,
                    'min_performance': 0.6,
                    'min_episodes': 100,
                    'query_distribution': {
                        'simple': 0.8,
                        'medium': 0.2,
                        'complex': 0.0
                    }
                },
                {
                    'name': 'medium',
                    'difficulty': 0.6,
                    'min_performance': 0.5,
                    'min_episodes': 200,
                    'query_distribution': {
                        'simple': 0.3,
                        'medium': 0.5,
                        'complex': 0.2
                    }
                },
                {
                    'name': 'complex',
                    'difficulty': 1.0,
                    'min_performance': 0.0,
                    'min_episodes': 0,
                    'query_distribution': {
                        'simple': 0.2,
                        'medium': 0.4,
                        'complex': 0.4
                    }
                }
            ]
        
        stages = []
        for stage_config in stages_config:
            stage = CurriculumStage(
                name=stage_config['name'],
                difficulty=stage_config['difficulty'],
                min_performance=stage_config.get('min_performance', 0.0),
                min_episodes=stage_config.get('min_episodes', 0),
                query_distribution=stage_config.get('query_distribution', {}),
                performance_window=stage_config.get('performance_window', self.performance_window)
            )
            stages.append(stage)
        
        return stages
    
    def _initialize_query_pools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize query pools by difficulty."""
        # This would typically load from a file or database
        # For now, create placeholder structure
        return {
            'simple': [],
            'medium': [],
            'complex': []
        }
    
    def select_query(self, available_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select next query based on current curriculum stage.
        
        Args:
            available_queries: List of available queries with difficulty labels
            
        Returns:
            Selected query
        """
        if not self.enabled or not available_queries:
            return random.choice(available_queries) if available_queries else {}
        
        # Categorize queries by difficulty
        queries_by_difficulty = {
            'simple': [],
            'medium': [],
            'complex': []
        }
        
        for query in available_queries:
            difficulty = query.get('difficulty', 'medium')
            if difficulty in queries_by_difficulty:
                queries_by_difficulty[difficulty].append(query)
        
        # Select based on stage distribution
        selected_difficulty = self._sample_difficulty()
        
        # Get queries for selected difficulty
        candidate_queries = queries_by_difficulty.get(selected_difficulty, [])
        
        # Fallback to any available queries if none match
        if not candidate_queries:
            candidate_queries = available_queries
        
        # Apply additional selection criteria
        query = self._apply_selection_strategy(candidate_queries)
        
        return query
    
    def _sample_difficulty(self) -> str:
        """Sample difficulty level based on current stage distribution."""
        distribution = self.current_stage.query_distribution
        
        # Create probability array
        difficulties = list(distribution.keys())
        probabilities = list(distribution.values())
        
        # Sample
        if difficulties and probabilities:
            return np.random.choice(difficulties, p=probabilities)
        
        return 'medium'  # Default
    
    def _apply_selection_strategy(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply selection strategy to choose from candidate queries.
        
        Args:
            queries: Candidate queries
            
        Returns:
            Selected query
        """
        if not queries:
            return {}
        
        # Strategy 1: Prioritize unseen queries
        unseen_queries = [q for q in queries if not q.get('seen', False)]
        if unseen_queries:
            selected = random.choice(unseen_queries)
            selected['seen'] = True
            return selected
        
        # Strategy 2: Prioritize least recently used
        queries_with_usage = [(q, q.get('last_used', 0)) for q in queries]
        queries_with_usage.sort(key=lambda x: x[1])
        
        # Select from least recently used (with some randomness)
        n_candidates = min(5, len(queries_with_usage))
        candidates = [q for q, _ in queries_with_usage[:n_candidates]]
        
        return random.choice(candidates)
    
    def update_performance(self, completion: float, accuracy: float, reward: float):
        """
        Update performance metrics after episode.
        
        Args:
            completion: Task completion rate (0-1)
            accuracy: Tool selection accuracy (0-1)
            reward: Episode reward
        """
        self.metrics.add_episode(completion, accuracy, reward)
        
        # Check for stage advancement
        if self.should_advance_stage():
            self.advance_stage()
        # Check for stage regression (optional)
        elif self.should_regress_stage():
            self.regress_stage()
    
    def should_advance_stage(self) -> bool:
        """
        Check if criteria met for advancing to next stage.
        
        Returns:
            True if should advance
        """
        if not self.enabled or self.current_stage_idx >= len(self.stages) - 1:
            return False
        
        # Check minimum episodes
        if self.metrics.stage_episode_count < self.current_stage.min_episodes:
            return False
        
        # Check performance threshold
        recent_performance = self.metrics.get_recent_performance(
            self.current_stage.performance_window
        )
        
        completion_rate = recent_performance['completion_rate']
        if completion_rate < self.current_stage.min_performance:
            return False
        
        # Additional adaptive checks
        if self.adaptive_progression:
            # Check performance stability
            std_completion = recent_performance['std_completion']
            if std_completion > 0.2:  # High variance indicates instability
                return False
            
            # Check consistent success
            if self.metrics.consecutive_failures > 5:
                return False
            
            # Check if significantly exceeding threshold
            if completion_rate > self.stage_transition_threshold:
                return True
        
        return True
    
    def should_regress_stage(self) -> bool:
        """
        Check if should move back to easier stage.
        
        Returns:
            True if should regress
        """
        if not self.enabled or self.current_stage_idx == 0:
            return False
        
        # Only regress if struggling significantly
        if self.metrics.consecutive_failures > 20:
            return True
        
        # Check if performance dropped significantly
        recent_performance = self.metrics.get_recent_performance(25)
        if recent_performance['completion_rate'] < 0.2:
            return True
        
        return False
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            # Record stage completion
            self.stage_history.append({
                'stage': self.current_stage.name,
                'episodes': self.metrics.stage_episode_count,
                'final_performance': self.metrics.get_recent_performance()
            })
            
            # Move to next stage
            self.current_stage_idx += 1
            self.current_stage = self.stages[self.current_stage_idx]
            self.metrics.reset_stage_metrics()
            
            logger.info(f"Advanced to curriculum stage: {self.current_stage.name}")
    
    def regress_stage(self):
        """Move back to previous stage."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            self.current_stage = self.stages[self.current_stage_idx]
            self.metrics.reset_stage_metrics()
            
            logger.warning(f"Regressed to curriculum stage: {self.current_stage.name}")
    
    def get_current_difficulty(self) -> float:
        """
        Get current difficulty level.
        
        Returns:
            Difficulty value (0-1)
        """
        return self.current_stage.difficulty
    
    def get_stage_info(self) -> Dict[str, Any]:
        """
        Get current stage information.
        
        Returns:
            Stage information dictionary
        """
        return {
            'stage_name': self.current_stage.name,
            'stage_index': self.current_stage_idx,
            'difficulty': self.current_stage.difficulty,
            'episodes_in_stage': self.metrics.stage_episode_count,
            'total_episodes': self.metrics.episode_count,
            'recent_performance': self.metrics.get_recent_performance(),
            'query_distribution': self.current_stage.query_distribution
        }
    
    def get_progress_report(self) -> Dict[str, Any]:
        """
        Get comprehensive progress report.
        
        Returns:
            Progress report dictionary
        """
        return {
            'enabled': self.enabled,
            'current_stage': self.get_stage_info(),
            'total_stages': len(self.stages),
            'stage_history': self.stage_history,
            'overall_metrics': {
                'total_episodes': self.metrics.episode_count,
                'average_completion': np.mean(list(self.metrics.completion_rates)) 
                    if self.metrics.completion_rates else 0.0,
                'average_accuracy': np.mean(list(self.metrics.accuracy_scores))
                    if self.metrics.accuracy_scores else 0.0,
                'average_reward': np.mean(list(self.metrics.rewards))
                    if self.metrics.rewards else 0.0
            }
        }
    
    def save_state(self, filepath: str):
        """Save curriculum state to file."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'metrics': {
                'completion_rates': list(self.metrics.completion_rates),
                'accuracy_scores': list(self.metrics.accuracy_scores),
                'rewards': list(self.metrics.rewards),
                'episode_count': self.metrics.episode_count,
                'stage_episode_count': self.metrics.stage_episode_count
            },
            'stage_history': self.stage_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load curriculum state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_stage_idx = state['current_stage_idx']
            self.current_stage = self.stages[self.current_stage_idx]
            
            # Restore metrics
            metrics_data = state['metrics']
            self.metrics.completion_rates = deque(
                metrics_data['completion_rates'], 
                maxlen=100
            )
            self.metrics.accuracy_scores = deque(
                metrics_data['accuracy_scores'],
                maxlen=100
            )
            self.metrics.rewards = deque(
                metrics_data['rewards'],
                maxlen=100
            )
            self.metrics.episode_count = metrics_data['episode_count']
            self.metrics.stage_episode_count = metrics_data['stage_episode_count']
            
            self.stage_history = state.get('stage_history', [])
            
            logger.info(f"Loaded curriculum state from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load curriculum state: {e}")