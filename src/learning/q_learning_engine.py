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
from utils.logger import get_logger

logger = get_logger(__name__)


class StateRepresentation:
    """Encodes states from intent vectors, context, and history."""
    
    def __init__(self):
        self.state_dimensions = {
            'intent_vector': 384,  # Sentence transformer output
            'context_features': 10,  # Domain, user history, etc.
            'tool_history': 20,     # Recent tool usage
            'performance_metrics': 5  # Success rate, response time, etc.
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
            intent_embedding = np.zeros(self.state_dimensions['intent_vector'])
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
        
        # Combine all components
        state_vector = np.concatenate(state_components)
        return state_vector
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context features."""
        features = np.zeros(self.state_dimensions['context_features'])
        
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
        features = np.zeros(self.state_dimensions['tool_history'])
        
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
        features = np.zeros(self.state_dimensions['performance_metrics'])
        
        features[0] = metrics.get('avg_response_time', 1000) / 5000.0  # Normalized to 5s
        features[1] = metrics.get('success_rate', 0.5)
        features[2] = metrics.get('error_rate', 0.1)
        features[3] = min(metrics.get('tools_invoked', 1) / 5.0, 1.0)
        features[4] = metrics.get('cache_hit_rate', 0.0)
        
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
        self.learning_rate = q_config.get('learning_rate', 0.1)
        self.discount_factor = q_config.get('discount_factor', 0.9)
        self.exploration_rate = q_config.get('exploration_rate', 0.2)
        self.exploration_decay = q_config.get('exploration_decay', 0.995)
        self.min_exploration_rate = q_config.get('min_exploration_rate', 0.01)
        
        # Initialize components
        self.state_encoder = StateRepresentation()
        self.action_space = ActionSpace(max_tools=q_config.get('max_tools', 3))
        self.q_table = QTable(self.learning_rate, self.discount_factor)
        self.experience_buffer = ExperienceReplayBuffer(
            capacity=q_config.get('buffer_capacity', 10000)
        )
        
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
    
    async def select_action(self, state: np.ndarray, available_tools: List[str],
                          constraints: Dict[str, Any]) -> Tuple[str, ...]:
        """Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state vector
            available_tools: List of available tool IDs
            constraints: Tool combination constraints
            
        Returns:
            Selected tool combination
        """
        # Get valid actions
        valid_actions = self.action_space.get_valid_actions(available_tools, constraints)
        
        if not valid_actions:
            logger.warning("No valid actions available")
            return ()
        
        # Epsilon-greedy selection
        if random.random() < self.exploration_rate:
            # Exploration: random action
            action = random.choice(valid_actions)
            logger.debug(f"Exploration: selected {action}")
        else:
            # Exploitation: best known action
            q_values = await self.q_table.get_all_q_values(state, valid_actions)
            
            if q_values:
                # Select action with highest Q-value
                action = max(q_values.items(), key=lambda x: x[1])[0]
                logger.debug(f"Exploitation: selected {action} with Q={q_values[action]:.3f}")
            else:
                # No Q-values yet, random selection
                action = random.choice(valid_actions)
                logger.debug(f"No Q-values, random selection: {action}")
        
        return action
    
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
        
        # Update metrics
        self.total_reward += reward
        if reward > 0:
            self.success_count += 1
        
        # Periodic batch learning from replay buffer
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_frequency:
            await self._replay_experiences()
            self.steps_since_update = 0
    
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
        self.exploration_rate = max(
            self.exploration_rate * self.exploration_decay,
            self.min_exploration_rate
        )
        self.episode_count += 1
    
    async def save_model(self, version: Optional[str] = None):
        """Save Q-table and learning state to database."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
                   VALUES (?, 'q_learning', ?, datetime('now'))""",
                (version, json.dumps(model_data))
            )
            await conn.commit()
        
        logger.info(f"Model saved with version: {version}")
    
    async def load_model(self, version: Optional[str] = None):
        """Load Q-table and learning state from database."""
        async with self.db_manager.get_connection() as conn:
            if version:
                query = """SELECT model_data FROM model_snapshots 
                          WHERE version = ? AND model_type = 'q_learning'"""
                params = (version,)
            else:
                query = """SELECT model_data FROM model_snapshots 
                          WHERE model_type = 'q_learning' 
                          ORDER BY created_at DESC LIMIT 1"""
                params = ()
            
            async with conn.execute(query, params) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    model_data = json.loads(row[0])
                    
                    # Restore Q-values - deserialize string keys back to tuples
                    self.q_table.q_values = {}
                    for key_str, value in model_data['q_values'].items():
                        parts = key_str.split('|', 1)  # Split only on first |
                        if len(parts) == 2:
                            state_hash, action_str = parts
                            self.q_table.q_values[(state_hash, action_str)] = value
                    
                    # Restore update counts
                    self.q_table.update_count = defaultdict(int)
                    for key_str, count in model_data['update_counts'].items():
                        parts = key_str.split('|', 1)
                        if len(parts) == 2:
                            state_hash, action_str = parts
                            self.q_table.update_count[(state_hash, action_str)] = count
                    
                    # Restore metadata
                    metadata = model_data['metadata']
                    self.exploration_rate = metadata.get(
                        'exploration_rate', self.exploration_rate
                    )
                    self.episode_count = metadata.get('episode_count', 0)
                    self.total_reward = metadata.get('total_reward', 0)
                    self.success_count = metadata.get('success_count', 0)
                    
                    logger.info(f"Model loaded: version={metadata['version']}, "
                              f"episodes={self.episode_count}")
                else:
                    logger.warning("No saved model found")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics."""
        return {
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.episode_count, 1),
            'success_rate': self.success_count / max(self.episode_count, 1),
            'exploration_rate': self.exploration_rate,
            'q_table_stats': self.q_table.get_statistics(),
            'buffer_size': len(self.experience_buffer)
        }