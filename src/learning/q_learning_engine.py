"""Q-Learning Engine for autonomous tool discovery and optimization.

This module implements the core Q-learning functionality including state representation,
action space management, Q-table updates, and experience replay.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hashlib
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, deque
from datetime import datetime
import logging
import random
import itertools

from database.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StateRepresentation:
    """Encodes states from intent vectors, context, and history."""
    
    def __init__(self):
        self.state_dimensions = {
            'intent_vector': 384,      # Sentence transformer output
            'context_features': 10,    # Domain, user history, etc.
            'tool_history': 20,        # Recent tool usage
            'performance_metrics': 5,   # Success rate, response time, etc.
            'failure_rates': 10,       # Per-tool failure rates
            'failure_types': 5,        # Network, permission, timeout, rate_limit, other
            'retry_patterns': 5,       # Retry statistics and patterns
            'user_expertise': 3,       # One-hot: novice, intermediate, expert
            'domain_context': 5        # One-hot: general, engineering, data_science, web_dev, devops
        }
        self.total_dimensions = sum(self.state_dimensions.values())
    
    def encode_state(self, intent: Any, context: Dict[str, Any], 
                     history: List[str]) -> np.ndarray:
        """Combine all features into state vector.
        
        Args:
            intent: Intent object with embedding
            context: Context dictionary with domain, session, etc.
            history: List of recently used tool IDs
            
        Returns:
            State vector as numpy array
        """
        state_components = []
        
        # Intent embedding
        if hasattr(intent, 'embedding'):
            intent_embedding = np.array(intent.embedding)
            if len(intent_embedding) != self.state_dimensions['intent_vector']:
                # Resize if needed
                intent_embedding = np.resize(intent_embedding, 
                                           self.state_dimensions['intent_vector'])
        else:
            intent_embedding = np.zeros(self.state_dimensions['intent_vector'], dtype=np.float32)
        state_components.append(intent_embedding)
        
        # Context features
        context_features = self._encode_context(context)
        state_components.append(context_features)
        
        # Tool history
        history_features = self._encode_history(history)
        state_components.append(history_features)
        
        # Performance metrics
        metrics_features = self._encode_metrics(context.get('metrics', {}))
        state_components.append(metrics_features)
        
        # Failure rates
        failure_features = self._encode_failure_rates(context.get('failure_rates', {}))
        state_components.append(failure_features)
        
        # Failure types
        failure_type_features = self._encode_failure_types(context.get('failure_types', {}))
        state_components.append(failure_type_features)
        
        # Retry patterns
        retry_features = self._encode_retry_patterns(context.get('retry_patterns', {}))
        state_components.append(retry_features)
        
        # User expertise
        expertise_features = self._encode_user_expertise(context.get('user_expertise', 'intermediate'))
        state_components.append(expertise_features)
        
        # Domain context
        domain_features = self._encode_domain_context(context.get('domain', 'general'))
        state_components.append(domain_features)
        
        # Combine all components
        state_vector = np.concatenate(state_components).astype(np.float32)
        return state_vector
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context features."""
        features = np.zeros(self.state_dimensions['context_features'], dtype=np.float32)
        
        # Domain encoding (one-hot)
        domains = ['engineering', 'data_science', 'general', 'system']
        domain = context.get('domain', 'general')
        if domain in domains:
            features[domains.index(domain)] = 1.0
        
        # Session features
        features[4] = min(context.get('query_count', 0) / 10.0, 1.0)  # Normalized
        features[5] = min(context.get('session_duration', 0) / 3600.0, 1.0)  # Hours
        
        # User history features
        features[6] = min(context.get('total_queries', 0) / 100.0, 1.0)
        features[7] = context.get('success_rate', 0.5)
        
        # Time features
        hour = datetime.now().hour
        features[8] = np.sin(2 * np.pi * hour / 24)  # Cyclic encoding
        features[9] = np.cos(2 * np.pi * hour / 24)
        
        return features
    
    def _encode_history(self, history: List[str]) -> np.ndarray:
        """Encode tool usage history."""
        features = np.zeros(self.state_dimensions['tool_history'], dtype=np.float32)
        
        # Frequency encoding for recent tools
        tool_counts = defaultdict(int)
        for tool in history[-20:]:  # Last 20 tools
            tool_counts[tool] += 1
        
        # Common tools get dedicated features
        common_tools = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp', 
                       'postgres_mcp', 'github_mcp']
        
        for i, tool in enumerate(common_tools[:10]):
            if tool in tool_counts:
                features[i] = min(tool_counts[tool] / 5.0, 1.0)
        
        # General statistics
        features[10] = len(set(history[-10:])) / 10.0  # Diversity
        features[11] = len(history) / 50.0 if history else 0  # Volume
        
        # Pattern features
        if len(history) >= 2:
            # Consecutive same tool usage
            consecutive = sum(1 for i in range(1, len(history)) 
                            if history[i] == history[i-1])
            features[12] = min(consecutive / len(history), 1.0)
        
        return features
    
    def _encode_metrics(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Encode performance metrics."""
        features = np.zeros(self.state_dimensions['performance_metrics'], dtype=np.float32)
        
        features[0] = metrics.get('avg_response_time', 1000) / 5000.0  # Normalized to 5s
        features[1] = metrics.get('success_rate', 0.5)
        features[2] = metrics.get('error_rate', 0.1)
        features[3] = min(metrics.get('tools_invoked', 1) / 5.0, 1.0)
        features[4] = metrics.get('cache_hit_rate', 0.0)
        
        return features
    
    def _encode_failure_rates(self, failure_rates: Dict[str, float]) -> np.ndarray:
        """Encode per-tool failure rates."""
        features = np.zeros(self.state_dimensions['failure_rates'], dtype=np.float32)
        
        # Common tools get dedicated features
        common_tools = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp', 
                       'postgres_mcp', 'github_mcp', 'weather_mcp']
        
        for i, tool in enumerate(common_tools[:6]):
            if tool in failure_rates:
                features[i] = failure_rates[tool]
        
        # Aggregate statistics
        if failure_rates:
            all_rates = list(failure_rates.values())
            features[6] = np.mean(all_rates)  # Average failure rate
            features[7] = np.max(all_rates)   # Max failure rate
            features[8] = np.std(all_rates)   # Failure rate variance
            features[9] = len([r for r in all_rates if r > 0.5])  # High failure count
        
        return features
    
    def _encode_failure_types(self, failure_types: Dict[str, int]) -> np.ndarray:
        """Encode failure type distribution."""
        features = np.zeros(self.state_dimensions['failure_types'], dtype=np.float32)
        
        # Map failure types to indices
        type_mapping = {
            'network_timeout': 0,
            'permission_error': 1,
            'rate_limit': 2,
            'connection_error': 3,
            'other': 4
        }
        
        total_failures = sum(failure_types.values()) if failure_types else 1
        
        for failure_type, index in type_mapping.items():
            count = failure_types.get(failure_type, 0)
            features[index] = count / total_failures  # Normalized frequency
        
        return features
    
    def _encode_retry_patterns(self, retry_patterns: Dict[str, Any]) -> np.ndarray:
        """Encode retry statistics and patterns."""
        features = np.zeros(self.state_dimensions['retry_patterns'], dtype=np.float32)
        
        features[0] = retry_patterns.get('avg_retry_count', 0) / 5.0  # Normalized
        features[1] = retry_patterns.get('retry_success_rate', 0.5)
        features[2] = min(retry_patterns.get('avg_retry_delay_ms', 1000) / 10000, 1.0)
        features[3] = retry_patterns.get('circuit_breaker_triggers', 0) / 10.0
        features[4] = min(retry_patterns.get('max_consecutive_failures', 0) / 5.0, 1.0)
        
        return features
    
    def _encode_user_expertise(self, expertise: str) -> np.ndarray:
        """Encode user expertise level as one-hot vector."""
        features = np.zeros(self.state_dimensions['user_expertise'], dtype=np.float32)
        
        expertise_map = {
            'novice': 0,
            'intermediate': 1,
            'expert': 2
        }
        
        if expertise in expertise_map:
            features[expertise_map[expertise]] = 1.0
        else:
            # Default to intermediate
            features[1] = 1.0
        
        return features
    
    def _encode_domain_context(self, domain: str) -> np.ndarray:
        """Encode domain context as one-hot vector."""
        features = np.zeros(self.state_dimensions['domain_context'], dtype=np.float32)
        
        domain_map = {
            'general': 0,
            'engineering': 1,
            'data_science': 2,
            'web_dev': 3,
            'devops': 4
        }
        
        if domain in domain_map:
            features[domain_map[domain]] = 1.0
        else:
            # Default to general
            features[0] = 1.0
        
        return features
    
    def encode_to_hash(self, state_vector: np.ndarray) -> str:
        """Convert state vector to hash for sparse storage."""
        # Discretize continuous values
        discretized = np.round(state_vector, decimals=2)
        state_str = json.dumps(discretized.tolist())
        return hashlib.md5(state_str.encode()).hexdigest()


class ActionSpace:
    """Defines and manages valid tool combinations."""
    
    def __init__(self, max_tools: int = 3):
        self.max_tools = max_tools
        self.tool_combinations_cache = {}
        self.constraint_validators = {
            'conflicts': self._validate_no_conflicts,
            'requires': self._validate_requirements,
            'max_tools': self._validate_max_tools
        }
    
    def get_valid_actions(self, available_tools: List[str], 
                         constraints: Dict[str, Any]) -> List[Tuple[str, ...]]:
        """Generate all valid tool combinations.
        
        Args:
            available_tools: List of available tool IDs
            constraints: Dictionary of constraints (conflicts, requirements, etc.)
            
        Returns:
            List of valid tool combinations as tuples
        """
        # Check cache
        cache_key = (tuple(sorted(available_tools)), 
                    json.dumps(constraints, sort_keys=True))
        if cache_key in self.tool_combinations_cache:
            return self.tool_combinations_cache[cache_key]
        
        # Generate all possible combinations
        actions = []
        for r in range(1, min(len(available_tools), self.max_tools) + 1):
            for combo in itertools.combinations(available_tools, r):
                if self.validate_combination(combo, constraints):
                    actions.append(combo)
        
        # Cache results
        self.tool_combinations_cache[cache_key] = actions
        return actions
    
    def validate_combination(self, combination: Tuple[str, ...], 
                           constraints: Dict[str, Any]) -> bool:
        """Validate if a tool combination satisfies all constraints."""
        for validator_name, validator_func in self.constraint_validators.items():
            if not validator_func(combination, constraints):
                return False
        return True
    
    def _validate_no_conflicts(self, combination: Tuple[str, ...], 
                              constraints: Dict[str, Any]) -> bool:
        """Check for conflicting tools."""
        conflicts = constraints.get('conflicts', {})
        for tool1, tool2 in itertools.combinations(combination, 2):
            if tool2 in conflicts.get(tool1, []) or tool1 in conflicts.get(tool2, []):
                return False
        return True
    
    def _validate_requirements(self, combination: Tuple[str, ...], 
                             constraints: Dict[str, Any]) -> bool:
        """Check if required dependencies are met."""
        requirements = constraints.get('requires', {})
        combo_set = set(combination)
        
        for tool in combination:
            required_tools = requirements.get(tool, [])
            if not all(req in combo_set for req in required_tools):
                return False
        return True
    
    def _validate_max_tools(self, combination: Tuple[str, ...], 
                           constraints: Dict[str, Any]) -> bool:
        """Check maximum tools constraint."""
        max_allowed = constraints.get('max_tools', self.max_tools)
        return len(combination) <= max_allowed
    
    def encode_action(self, action: Tuple[str, ...]) -> str:
        """Encode action (tool combination) as string for Q-table."""
        return '|'.join(sorted(action))
    
    def decode_action(self, action_str: str) -> Tuple[str, ...]:
        """Decode action string back to tool combination."""
        return tuple(action_str.split('|'))


class QTable:
    """Sparse Q-table implementation with learning capabilities."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.q_values = {}  # Sparse representation: (state_hash, action_str) -> q_value
        self.update_count = defaultdict(int)
        self.state_encoder = StateRepresentation()
        self.action_space = ActionSpace()
        self.lock = asyncio.Lock()
    
    async def get_q_value(self, state: np.ndarray, action: Tuple[str, ...]) -> float:
        """Get Q-value for state-action pair."""
        state_hash = self.state_encoder.encode_to_hash(state)
        action_str = self.action_space.encode_action(action)
        
        async with self.lock:
            return self.q_values.get((state_hash, action_str), 0.0)
    
    async def get_all_q_values(self, state: np.ndarray, 
                              actions: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], float]:
        """Get Q-values for all possible actions in a state."""
        state_hash = self.state_encoder.encode_to_hash(state)
        q_values = {}
        
        async with self.lock:
            for action in actions:
                action_str = self.action_space.encode_action(action)
                q_values[action] = self.q_values.get((state_hash, action_str), 0.0)
        
        return q_values
    
    async def update(self, state: np.ndarray, action: Tuple[str, ...], 
                    reward: float, next_state: np.ndarray, 
                    next_actions: List[Tuple[str, ...]]):
        """Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        """
        state_hash = self.state_encoder.encode_to_hash(state)
        action_str = self.action_space.encode_action(action)
        
        # Get current Q-value
        current_q = await self.get_q_value(state, action)
        
        # Get max Q-value for next state
        if next_actions:
            next_q_values = await self.get_all_q_values(next_state, next_actions)
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        async with self.lock:
            self.q_values[(state_hash, action_str)] = new_q
            self.update_count[(state_hash, action_str)] += 1
        
        logger.debug(f"Q-update: state={state_hash[:8]}, action={action_str}, "
                    f"reward={reward:.2f}, Q: {current_q:.3f} -> {new_q:.3f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Q-table statistics."""
        q_values_list = list(self.q_values.values())
        return {
            'total_entries': len(self.q_values),
            'total_updates': sum(self.update_count.values()),
            'avg_q_value': np.mean(q_values_list) if q_values_list else 0.0,
            'max_q_value': max(q_values_list) if q_values_list else 0.0,
            'min_q_value': min(q_values_list) if q_values_list else 0.0,
            'most_updated': max(self.update_count.items(), 
                               key=lambda x: x[1])[0] if self.update_count else None
        }


class ExperienceReplayBuffer:
    """Store and sample experiences for learning."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, experience: Dict[str, Any], priority: Optional[float] = None):
        """Add experience to buffer.
        
        Args:
            experience: Dictionary containing state, action, reward, next_state, etc.
            priority: Optional priority for prioritized replay
        """
        if priority is None:
            priority = self._calculate_priority(experience)
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, prioritized: bool = True) -> List[Dict[str, Any]]:
        """Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            prioritized: Whether to use prioritized sampling
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized and self.priorities:
            # Prioritized experience replay
            priorities = np.array(self.priorities)
            # Add small epsilon to avoid zero probabilities
            priorities = priorities + 1e-6
            probs = priorities / priorities.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
        
        return [self.buffer[i] for i in indices]
    
    def _calculate_priority(self, experience: Dict[str, Any]) -> float:
        """Calculate priority based on TD error or reward magnitude."""
        # Simple priority based on reward magnitude and success
        reward = abs(experience.get('reward', 0))
        success = 1.0 if experience.get('success', False) else 0.5
        
        # Higher priority for rare successes or large rewards
        priority = reward * success
        
        # Boost priority for experiences with errors (to learn from mistakes)
        if experience.get('error', False):
            priority *= 2.0
            
        return priority
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for specific experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class QLearningEngine:
    """Main Q-learning engine coordinating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        # Extract Q-learning parameters from config
        q_config = config.get('q_learning', {})
        self.learning_rate = q_config.get('learning_rate', q_config.get('alpha', 0.1))
        self.discount_factor = q_config.get('discount_factor', q_config.get('gamma', 0.9))
        self.exploration_rate = q_config.get('exploration_rate', q_config.get('epsilon', 0.2))
        self.exploration_decay = q_config.get('exploration_decay', 0.995)
        self.min_exploration_rate = q_config.get('min_exploration_rate', 0.01)
        
        # Initialize components
        self.state_encoder = StateRepresentation()
        self.action_space = ActionSpace(max_tools=q_config.get('max_tools', 3))
        
        # Check if DQN is enabled
        self.use_dqn = config.get('dqn', {}).get('enabled', False)
        
        if self.use_dqn:
            # Initialize DQN agent
            from .dqn_agent import DQNAgent
            # Get maximum possible action space size
            # This is an upper bound - actual valid actions may be fewer
            max_action_combinations = self._estimate_max_actions(q_config.get('max_tools', 3))
            self.dqn_agent = DQNAgent(
                config, 
                self.state_encoder.total_dimensions,
                max_action_combinations
            )
            # DQN uses its own experience replay
            self.experience_buffer = None
            logger.info("Using Deep Q-Network (DQN) for value function approximation")
        else:
            # Use traditional tabular Q-learning
            self.q_table = QTable(self.learning_rate, self.discount_factor)
            self.experience_buffer = ExperienceReplayBuffer(
                capacity=q_config.get('buffer_capacity', 10000)
            )
            self.dqn_agent = None
            logger.info("Using tabular Q-learning")
        
        # Initialize pattern miner
        # Import here to avoid circular import
        from .pattern_miner import PatternMiner
        self.pattern_miner = PatternMiner(
            config,
            min_support=q_config.get('pattern_min_support', 0.1),
            min_confidence=q_config.get('pattern_min_confidence', 0.8)
        )
        self.use_patterns = q_config.get('use_patterns', True)
        self.use_incremental_patterns = q_config.get('use_incremental_patterns', True)
        self.pattern_batch_size = q_config.get('pattern_batch_size', 1000)
        self.pattern_decay_factor = q_config.get('pattern_decay_factor', 0.95)
        self.pattern_weight = q_config.get('pattern_weight', 0.3)  # Weight for pattern suggestions
        
        # Training parameters
        self.batch_size = q_config.get('batch_size', 32)
        self.update_frequency = q_config.get('update_frequency', 4)
        self.steps_since_update = 0
        
        # Database for persistence
        self.db_manager = DatabaseManager()
        
        # Metrics tracking
        self.episode_count = 0
        self.total_reward = 0
        self.success_count = 0
        
        logger.info(f"Q-Learning Engine initialized with α={self.learning_rate}, "
                   f"γ={self.discount_factor}, ε={self.exploration_rate}")
        
        # Initialize patterns asynchronously after creation
        self._patterns_loaded = False
        
        # Current context for context-aware patterns
        self.current_context = {}
    
    def _estimate_max_actions(self, max_tools: int) -> int:
        """Estimate maximum number of possible action combinations.
        
        This provides an upper bound for the DQN output layer size.
        Actual valid actions may be fewer due to constraints.
        """
        # Assume we have at most 10 different tools available
        # and can combine up to max_tools at a time
        max_available_tools = 10
        
        # Calculate sum of combinations: C(n,1) + C(n,2) + ... + C(n,max_tools)
        from math import comb
        total = sum(comb(max_available_tools, i) for i in range(1, min(max_tools + 1, max_available_tools + 1)))
        
        # Add some buffer
        return min(total + 10, 1000)  # Cap at 1000 to avoid too large networks
    
    async def select_action(self, state: np.ndarray, available_tools: List[str],
                          constraints: Dict[str, Any], 
                          current_tools: Optional[List[str]] = None,
                          context: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
        """Select action using epsilon-greedy strategy with pattern guidance.
        
        Args:
            state: Current state vector
            available_tools: List of available tool IDs
            constraints: Tool combination constraints
            current_tools: Tools already used in current sequence (for pattern matching)
            context: Optional context with user_expertise and domain
            
        Returns:
            Selected tool combination
        """
        # Store context for pattern matching
        if context:
            self.current_context = context
        # Get valid actions
        valid_actions = self.action_space.get_valid_actions(available_tools, constraints)
        
        if not valid_actions:
            logger.warning("No valid actions available")
            return ()
        
        if self.use_dqn:
            # Use DQN for action selection
            action = self.dqn_agent.select_action(state, valid_actions, training=True)
            logger.debug(f"DQN selected action: {action}")
        else:
            # Use tabular Q-learning
            # Epsilon-greedy selection
            if random.random() < self.exploration_rate:
                # Exploration: random action
                action = random.choice(valid_actions)
                logger.debug(f"Exploration: selected {action}")
            else:
                # Exploitation: best known action with pattern guidance
                action = await self._select_best_action_with_patterns(
                    state, valid_actions, current_tools
                )
            
        return action
    
    async def _select_best_action_with_patterns(self, state: np.ndarray, 
                                              valid_actions: List[Tuple[str, ...]],
                                              current_tools: Optional[List[str]] = None) -> Tuple[str, ...]:
        """Select best action considering both Q-values and patterns.
        
        Args:
            state: Current state vector
            valid_actions: List of valid tool combinations
            current_tools: Tools already used in sequence
            
        Returns:
            Best action based on combined scoring
        """
        # Get Q-values
        q_values = await self.q_table.get_all_q_values(state, valid_actions)
        
        # If using patterns and we have current tools
        if self.use_patterns and current_tools:
            # Ensure patterns are loaded
            if not self._patterns_loaded:
                await self.initialize_patterns()
            # Get pattern-based scores
            pattern_scores = {}
            
            for action in valid_actions:
                # Create potential sequence
                potential_sequence = current_tools + list(action)
                
                # Get matching patterns with context
                # Extract context from state (assumes context is passed in select_action)
                user_expertise = self.current_context.get('user_expertise', 'intermediate')
                domain = self.current_context.get('domain', 'general')
                
                matching_patterns = self.pattern_miner.get_context_matching_patterns(
                    potential_sequence, user_expertise, domain
                )
                
                # Calculate pattern score
                if matching_patterns:
                    # Use the highest scoring pattern
                    best_pattern = matching_patterns[0]
                    pattern_score = best_pattern.confidence * best_pattern.lift
                    pattern_scores[action] = pattern_score
                else:
                    pattern_scores[action] = 0.0
            
            # Combine Q-values and pattern scores
            combined_scores = {}
            for action in valid_actions:
                q_value = q_values.get(action, 0.0)
                pattern_score = pattern_scores.get(action, 0.0)
                
                # Weighted combination
                combined_score = (
                    (1 - self.pattern_weight) * q_value + 
                    self.pattern_weight * pattern_score
                )
                combined_scores[action] = combined_score
            
            # Select action with highest combined score
            if combined_scores:
                action = max(combined_scores.items(), key=lambda x: x[1])[0]
                logger.debug(f"Pattern-guided selection: {action} "
                           f"(Q={q_values.get(action, 0):.3f}, "
                           f"Pattern={pattern_scores.get(action, 0):.3f})")
            else:
                action = random.choice(valid_actions)
                
        else:
            # Use only Q-values
            if q_values:
                action = max(q_values.items(), key=lambda x: x[1])[0]
                logger.debug(f"Q-value selection: {action} with Q={q_values[action]:.3f}")
            else:
                action = random.choice(valid_actions)
                logger.debug(f"Random selection (no Q-values): {action}")
        
        return action
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Convert state array to hashable string representation.
        
        Args:
            state: State array
            
        Returns:
            Hashable string representation
        """
        # Convert to tuple for hashing, round to avoid floating point issues
        state_tuple = tuple(np.round(state, decimals=4))
        return str(hash(state_tuple))
    
    async def learn_from_experience(self, state: np.ndarray, action: Tuple[str, ...],
                                  reward: float, next_state: np.ndarray,
                                  next_available_tools: List[str],
                                  constraints: Dict[str, Any],
                                  done: bool = False):
        """Learn from a single experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_available_tools: Available tools in next state
            constraints: Tool constraints
            done: Whether episode is complete
        """
        if self.use_dqn:
            # DQN learning
            # Get valid next actions for DQN
            next_valid_actions = self.action_space.get_valid_actions(
                next_available_tools, constraints
            ) if not done else []
            
            # Store transition in DQN's replay buffer
            self.dqn_agent.store_transition(
                state, action, reward, next_state, next_valid_actions, done
            )
            
            # Perform training step
            loss = self.dqn_agent.train_step()
            if loss is not None:
                logger.debug(f"DQN training loss: {loss:.4f}")
        else:
            # Tabular Q-learning
            # Get valid next actions
            next_actions = self.action_space.get_valid_actions(
                next_available_tools, constraints
            ) if not done else []
            
            # Update Q-table
            await self.q_table.update(state, action, reward, next_state, next_actions)
            
            # Store experience
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'next_actions': next_actions,
                'done': done,
                'timestamp': datetime.now()
            }
            self.experience_buffer.add(experience)
            
            # Periodic batch learning from replay buffer
            self.steps_since_update += 1
            if self.steps_since_update >= self.update_frequency:
                await self._replay_experiences()
                self.steps_since_update = 0
        
        # Update metrics (common for both)
        self.total_reward += reward
        if reward > 0:
            self.success_count += 1
    
    async def _replay_experiences(self):
        """Learn from batch of experiences in replay buffer."""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.experience_buffer.sample(self.batch_size)
        
        # Learn from each experience
        for exp in batch:
            await self.q_table.update(
                exp['state'], exp['action'], exp['reward'],
                exp['next_state'], exp['next_actions']
            )
    
    def decay_exploration(self):
        """Decay exploration rate over time."""
        if self.use_dqn:
            # DQN manages its own exploration
            self.dqn_agent.decay_epsilon()
            self.exploration_rate = self.dqn_agent.epsilon  # Keep in sync for metrics
        else:
            # Tabular Q-learning exploration decay
            self.exploration_rate = max(
                self.exploration_rate * self.exploration_decay,
                self.min_exploration_rate
            )
        self.episode_count += 1
    
    async def save_model(self, version: Optional[str] = None):
        """Save Q-table/DQN and learning state to database."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_dqn:
            # Save DQN model
            checkpoint_path = f"models/dqn_checkpoint_{version}.pt"
            self.dqn_agent.save_checkpoint(checkpoint_path)
            
            model_data = {
                'model_type': 'dqn',
                'checkpoint_path': checkpoint_path,
                'metadata': {
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'episode_count': self.episode_count,
                    'total_reward': self.total_reward,
                    'success_count': self.success_count,
                    'dqn_metrics': self.dqn_agent.get_metrics()
                }
            }
        else:
            # Save tabular Q-learning model
            # Prepare model data - convert tuple keys to strings for JSON
            q_values_serialized = {}
            for (state_hash, action_str), value in self.q_table.q_values.items():
                key = f"{state_hash}|{action_str}"
                q_values_serialized[key] = value
            
            update_counts_serialized = {}
            for (state_hash, action_str), count in self.q_table.update_count.items():
                key = f"{state_hash}|{action_str}"
                update_counts_serialized[key] = count
            
            model_data = {
                'model_type': 'tabular',
                'q_values': q_values_serialized,
                'update_counts': update_counts_serialized,
                'metadata': {
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'exploration_rate': self.exploration_rate,
                    'episode_count': self.episode_count,
                    'total_reward': self.total_reward,
                    'success_count': self.success_count,
                    'q_table_stats': self.q_table.get_statistics()
                }
            }
        
        # Save to database
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """INSERT INTO model_snapshots (version, model_type, model_data, created_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (version, 'dqn' if self.use_dqn else 'q_learning', json.dumps(model_data))
            )
            await conn.commit()
        
        logger.info(f"{'DQN' if self.use_dqn else 'Q-Learning'} model saved with version: {version}")
    
    async def load_model(self, version: Optional[str] = None):
        """Load Q-table/DQN and learning state from database."""
        async with self.db_manager.get_connection() as conn:
            model_type = 'dqn' if self.use_dqn else 'q_learning'
            
            if version:
                query = """SELECT model_data FROM model_snapshots 
                          WHERE version = ? AND model_type = ?"""
                params = (version, model_type)
            else:
                query = """SELECT model_data FROM model_snapshots 
                          WHERE model_type = ? 
                          ORDER BY created_at DESC LIMIT 1"""
                params = (model_type,)
            
            async with conn.execute(query, params) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    model_data = json.loads(row[0])
                    
                    if self.use_dqn:
                        # Load DQN model
                        if 'checkpoint_path' in model_data:
                            checkpoint_path = model_data['checkpoint_path']
                            if os.path.exists(checkpoint_path):
                                self.dqn_agent.load_checkpoint(checkpoint_path)
                            else:
                                logger.warning(f"DQN checkpoint not found: {checkpoint_path}")
                    else:
                        # Load tabular Q-learning model
                        # Restore Q-values - deserialize string keys back to tuples
                        self.q_table.q_values = {}
                        for key_str, value in model_data.get('q_values', {}).items():
                            parts = key_str.split('|', 1)  # Split only on first |
                            if len(parts) == 2:
                                state_hash, action_str = parts
                                self.q_table.q_values[(state_hash, action_str)] = value
                        
                        # Restore update counts
                        self.q_table.update_count = defaultdict(int)
                        for key_str, count in model_data.get('update_counts', {}).items():
                            parts = key_str.split('|', 1)
                            if len(parts) == 2:
                                state_hash, action_str = parts
                                self.q_table.update_count[(state_hash, action_str)] = count
                    
                    # Restore metadata (common for both)
                    metadata = model_data['metadata']
                    self.episode_count = metadata.get('episode_count', 0)
                    self.total_reward = metadata.get('total_reward', 0)
                    self.success_count = metadata.get('success_count', 0)
                    
                    if self.use_dqn:
                        # Sync exploration rate from DQN
                        self.exploration_rate = self.dqn_agent.epsilon
                    else:
                        self.exploration_rate = metadata.get(
                            'exploration_rate', self.exploration_rate
                        )
                    
                    logger.info(f"{'DQN' if self.use_dqn else 'Q-Learning'} model loaded: "
                              f"version={metadata['version']}, episodes={self.episode_count}")
                else:
                    logger.warning(f"No saved {'DQN' if self.use_dqn else 'Q-learning'} model found")
    
    async def initialize_patterns(self):
        """Load existing patterns from database."""
        if not self._patterns_loaded:
            try:
                await self.pattern_miner.load_patterns()
                self._patterns_loaded = True
                logger.info(f"Loaded {len(self.pattern_miner.discovered_patterns)} patterns")
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")
    
    async def update_patterns(self, use_incremental: Optional[bool] = None, batch_size: Optional[int] = None):
        """Mine new patterns from recent execution history.
        
        Args:
            use_incremental: If True, use incremental update; if False, do full mining
            batch_size: Maximum number of sequences to process in incremental update
        """
        try:
            # Use configuration if not explicitly specified
            if use_incremental is None:
                use_incremental = self.use_incremental_patterns
            if batch_size is None:
                batch_size = self.pattern_batch_size
                
            if use_incremental:
                logger.info("Starting incremental pattern mining update...")
                patterns = await self.pattern_miner.incremental_update(
                    batch_size=batch_size,
                    decay_factor=self.pattern_decay_factor
                )
            else:
                logger.info("Starting full pattern mining update...")
                patterns = await self.pattern_miner.mine_patterns()
            
            total_patterns = sum(len(p) for p in patterns.values())
            logger.info(f"Pattern mining complete. Discovered/updated {total_patterns} patterns")
            
            # Update patterns loaded flag
            self._patterns_loaded = True
            
            # Patterns are automatically stored and loaded by pattern_miner
            return patterns
        except Exception as e:
            logger.error(f"Pattern mining failed: {e}")
            return {}
    
    async def suggest_tools_from_patterns(self, current_tools: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """Get tool suggestions based on discovered patterns.
        
        Args:
            current_tools: Tools already used in sequence
            k: Number of suggestions to return
            
        Returns:
            List of (tool, score) tuples
        """
        if not self._patterns_loaded:
            await self.initialize_patterns()
            
        return self.pattern_miner.suggest_next_tools(current_tools, k)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics."""
        base_metrics = {
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.episode_count, 1),
            'success_rate': self.success_count / max(self.episode_count, 1),
            'exploration_rate': self.exploration_rate,
            'patterns_loaded': self._patterns_loaded,
            'pattern_count': len(self.pattern_miner.discovered_patterns) if self._patterns_loaded else 0,
            'learning_type': 'dqn' if self.use_dqn else 'tabular'
        }
        
        if self.use_dqn:
            # Add DQN-specific metrics
            dqn_metrics = self.dqn_agent.get_metrics()
            base_metrics.update({
                'dqn_metrics': dqn_metrics,
                'buffer_size': dqn_metrics.get('memory_size', 0)
            })
        else:
            # Add tabular Q-learning specific metrics
            base_metrics.update({
                'q_table_stats': self.q_table.get_statistics(),
                'buffer_size': len(self.experience_buffer)
            })
        
        return base_metrics