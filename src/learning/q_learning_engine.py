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
    """Encodes states from intent vectors, context, and history with dimensionality reduction."""
    
    def __init__(self, use_pca=True, target_dim=128, use_encoder=False, encoder_path=None):
        self.state_dimensions = {
            'intent_vector': 384,      # Sentence transformer output
            'context_features': 10,    # Domain, user history, etc.
            'tool_history': 20,        # Recent tool usage
            'performance_metrics': 5,   # Success rate, response time, etc.
            'failure_rates': 10,       # Per-tool failure rates
            'failure_types': 5,        # Network, permission, timeout, rate_limit, other
            'retry_patterns': 5,       # Retry statistics and patterns
            'user_expertise': 3,       # One-hot: novice, intermediate, expert
            'domain_context': 5,       # One-hot: general, engineering, data_science, web_dev, devops
            'tool_categories': 10,     # Tool category features for semantic matching
            'query_complexity': 5,     # NEW: Query complexity indicators
            'temporal_features': 4,    # NEW: Episode progress, learning phase
            'attention_weights': 10    # NEW: Attention mechanism for relevant features
        }
        self.total_dimensions = sum(self.state_dimensions.values())
        
        # Dimensionality reduction settings
        self.use_pca = use_pca
        self.target_dim = target_dim
        self.pca_fitted = False
        self.pca_components = None
        self.feature_importance = None
        
        # Supervised encoder settings
        self.use_encoder = use_encoder
        self.encoder = None
        self.encoder_dim = 50  # Default encoder output dimension
        
        # Load encoder if specified
        if use_encoder and encoder_path:
            self._load_encoder(encoder_path)
        
        # Initialize feature importance weights
        self._initialize_feature_importance()
    
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
        
        # Tool categories
        tool_category_features = self._encode_tool_categories(context.get('available_tool_categories', []))
        state_components.append(tool_category_features)
        
        # Query complexity features (NEW)
        complexity_features = self._encode_query_complexity(context)
        state_components.append(complexity_features)
        
        # Temporal features (NEW)
        temporal_features = self._encode_temporal_features(context)
        state_components.append(temporal_features)
        
        # Attention weights (NEW)
        attention_features = self._calculate_attention_weights(intent, context)
        state_components.append(attention_features)
        
        # Combine all components
        state_vector = np.concatenate(state_components).astype(np.float32)
        
        # Apply feature importance weighting
        if self.feature_importance is not None:
            state_vector = state_vector * self.feature_importance
        
        # Apply dimensionality reduction
        if self.use_encoder and self.encoder is not None:
            # Use supervised encoder for dimensionality reduction
            state_vector = self._encode_with_supervised_encoder(state_vector)
        elif self.use_pca and self.target_dim < len(state_vector):
            # Fall back to PCA if encoder not available
            state_vector = self._apply_pca(state_vector)
        
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
        """Enhanced encoding of tool usage history with sequence and temporal information."""
        features = np.zeros(self.state_dimensions['tool_history'], dtype=np.float32)
        
        # ENHANCED: Track last 5 tools in sequence with positional encoding
        last_5_tools = history[-5:] if history else []
        
        # Encode tool sequence (features 0-9: 2 features per tool)
        for i, tool in enumerate(last_5_tools):
            if i * 2 < 10:
                # Recency weight (more recent = higher weight)
                features[i * 2] = 1.0 / (len(last_5_tools) - i)
                # Tool identifier (simple hash encoding)
                features[i * 2 + 1] = (hash(tool) % 100) / 100.0
        
        # Frequency encoding for recent tools (features 10-14)
        tool_counts = defaultdict(int)
        for tool in history[-20:]:  # Last 20 tools
            tool_counts[tool] += 1
        
        # Most frequent tools
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (tool, count) in enumerate(sorted_tools):
            if 10 + i < 13:
                features[10 + i] = min(count / 5.0, 1.0)
        
        # Tool pattern features (features 13-19)
        features[13] = len(set(history[-10:])) / 10.0 if history else 0  # Diversity
        features[14] = len(history) / 50.0 if history else 0  # Volume
        features[15] = 1.0 if len(last_5_tools) > 1 and last_5_tools[-1] == last_5_tools[-2] else 0.0  # Repetition
        features[16] = len([t for t in last_5_tools if 'search' in t.lower()]) / max(len(last_5_tools), 1)  # Search tools ratio
        features[17] = len([t for t in last_5_tools if 'database' in t.lower() or 'sql' in t.lower()]) / max(len(last_5_tools), 1)  # DB tools ratio
        features[18] = 1.0 if any('github' in t or 'filesystem' in t for t in last_5_tools) else 0.0  # Code tools used
        features[19] = min(len(history) / 10.0, 1.0) if history else 0.0  # Experience level
        
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
        
        # Ensure domain is a string
        if not isinstance(domain, str):
            logger.warning(f"Domain is not a string: {domain} (type: {type(domain)}), converting to 'general'")
            domain = 'general'
        
        # Updated domain map to match actual test query domains
        # Map similar domains to the same category
        domain_map = {
            'general': 0,
            'file_operations': 1,  # File/system operations
            'information_retrieval': 2,  # Search/info tasks
            'web_search': 2,  # Group with info retrieval
            'data_analysis': 3,  # Data/analysis tasks
            'data_collection': 3,  # Group with data analysis
            'business_analysis': 3,  # Group with analysis
            'development': 4,  # Dev/engineering tasks
            'engineering': 4,  # Group with development
            'web_dev': 4,  # Group with development
            'devops': 4  # Group with development
        }
        
        if domain in domain_map:
            features[domain_map[domain]] = 1.0
        else:
            # Default to general
            features[0] = 1.0
        
        return features
    
    def _encode_tool_categories(self, available_categories: List[str]) -> np.ndarray:
        """Encode available tool categories as features.
        
        This helps Q-learning understand what types of tools are available
        and learn category-specific patterns.
        """
        features = np.zeros(self.state_dimensions['tool_categories'], dtype=np.float32)
        
        # Define category mapping
        category_map = {
            'search': 0,
            'code': 1,
            'database': 2,
            'filesystem': 3,
            'finance': 4,
            'weather': 5,
            'productivity': 6,
            'general': 7
        }
        
        # Set binary features for available categories
        for category in available_categories:
            if category in category_map:
                features[category_map[category]] = 1.0
        
        # Add diversity metric
        features[8] = len(available_categories) / len(category_map) if category_map else 0  # Category diversity
        
        # Add primary category indicator (most relevant)
        if available_categories:
            primary_category = available_categories[0]  # First is most relevant
            if primary_category in category_map:
                features[9] = category_map[primary_category] / len(category_map)  # Normalized
        
        return features
    
    def _encode_query_complexity(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode query complexity indicators."""
        features = np.zeros(self.state_dimensions['query_complexity'], dtype=np.float32)
        
        # Query type (simple/complex/mixed)
        query_type = context.get('query_type', 'simple')
        if query_type == 'simple':
            features[0] = 1.0
        elif query_type == 'complex':
            features[1] = 1.0
        else:  # mixed
            features[2] = 1.0
        
        # Number of intents
        num_intents = context.get('num_intents', 1)
        features[3] = min(num_intents / 5.0, 1.0)  # Normalized
        
        # Expected tools needed
        expected_tools = context.get('expected_tools', 1)
        features[4] = min(expected_tools / 3.0, 1.0)  # Normalized
        
        return features
    
    def _encode_temporal_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode temporal features for learning phase awareness."""
        features = np.zeros(self.state_dimensions['temporal_features'], dtype=np.float32)
        
        # Episode progress
        episode = context.get('episode', 0)
        total_episodes = context.get('total_episodes', 10000)
        features[0] = min(episode / total_episodes, 1.0)
        
        # Learning phase (early/mid/late)
        if episode < 1000:
            features[1] = 1.0  # Early exploration
        elif episode < 5000:
            features[2] = 1.0  # Mid exploitation
        else:
            features[3] = 1.0  # Late refinement
        
        return features
    
    def _calculate_attention_weights(self, intent: Any, context: Dict[str, Any]) -> np.ndarray:
        """Calculate attention weights for feature relevance."""
        features = np.zeros(self.state_dimensions['attention_weights'], dtype=np.float32)
        
        # Simple attention mechanism based on intent confidence
        if hasattr(intent, 'confidence_scores'):
            scores = intent.confidence_scores[:10]  # Top 10 scores
            features[:len(scores)] = scores
        else:
            # Default uniform attention
            features[:] = 0.1
        
        return features
    
    def _load_encoder(self, encoder_path: str):
        """Load the supervised encoder model."""
        try:
            import torch
            from src.learning.state_encoder import load_encoder
            
            self.encoder = load_encoder(encoder_path, device='cpu')
            self.encoder.eval()
            
            # Get encoder output dimension from model
            if hasattr(self.encoder, 'latent_dim'):
                self.encoder_dim = self.encoder.latent_dim
            
            logger.info(f"Loaded encoder from {encoder_path}, output dim: {self.encoder_dim}")
        except Exception as e:
            logger.error(f"Failed to load encoder from {encoder_path}: {e}")
            logger.warning("Falling back to raw state representation")
            self.use_encoder = False
            self.encoder = None
    
    def _encode_with_supervised_encoder(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply supervised encoder for dimensionality reduction."""
        if self.encoder is None:
            logger.warning("Encoder not loaded, returning raw state")
            return state_vector
        
        try:
            import torch
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)  # Add batch dimension
            
            # Encode
            with torch.no_grad():
                encoded = self.encoder(state_tensor)
                encoded_np = encoded.squeeze(0).numpy()  # Remove batch dimension
            
            return encoded_np
            
        except Exception as e:
            logger.error(f"Error encoding state: {e}")
            logger.warning("Falling back to raw state")
            return state_vector
    
    def _initialize_feature_importance(self):
        """Initialize feature importance weights for weighted state representation."""
        # Start with uniform importance
        self.feature_importance = np.ones(self.total_dimensions, dtype=np.float32)
        
        # Increase importance for key features
        # Intent vector is most important
        start_idx = 0
        end_idx = self.state_dimensions['intent_vector']
        self.feature_importance[start_idx:end_idx] *= 1.5
        
        # Context and performance metrics are important
        start_idx = end_idx
        end_idx = start_idx + self.state_dimensions['context_features']
        self.feature_importance[start_idx:end_idx] *= 1.2
        
        # Normalize
        self.feature_importance = self.feature_importance / np.mean(self.feature_importance)
    
    def _apply_pca(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        # For now, simple truncation (full PCA would require sklearn)
        # In production, would use sklearn.decomposition.PCA
        if len(state_vector) > self.target_dim:
            # Select most important features based on importance weights
            if self.feature_importance is not None:
                # Get indices of top features
                top_indices = np.argsort(self.feature_importance)[-self.target_dim:]
                return state_vector[top_indices]
            else:
                # Simple truncation
                return state_vector[:self.target_dim]
        return state_vector
    
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
    """Sparse Q-table implementation with double Q-learning capabilities."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, use_double_q: bool = True):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.use_double_q = use_double_q
        
        if self.use_double_q:
            # Double Q-learning: maintain two Q-tables
            self.q_values_a = {}  # Q-table A: (state_hash, action_str) -> q_value
            self.q_values_b = {}  # Q-table B: (state_hash, action_str) -> q_value
            # Optimistic initialization for better exploration
            self.initial_q_value = 0.5
        else:
            # Single Q-table
            self.q_values = {}
            self.initial_q_value = 0.0
        
        self.update_count = defaultdict(int)
        self.state_encoder = StateRepresentation(use_pca=False)
        self.action_space = ActionSpace()
        self.lock = asyncio.Lock()
    
    async def get_q_value(self, state: np.ndarray, action: Tuple[str, ...]) -> float:
        """Get Q-value for state-action pair (averaged for double Q-learning)."""
        state_hash = self.state_encoder.encode_to_hash(state)
        action_str = self.action_space.encode_action(action)
        
        async with self.lock:
            if self.use_double_q:
                # Return average of both Q-tables for action selection
                q_a = self.q_values_a.get((state_hash, action_str), self.initial_q_value)
                q_b = self.q_values_b.get((state_hash, action_str), self.initial_q_value)
                return (q_a + q_b) / 2.0
            else:
                return self.q_values.get((state_hash, action_str), self.initial_q_value)
    
    async def get_all_q_values(self, state: np.ndarray, 
                              actions: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], float]:
        """Get Q-values for all possible actions in a state."""
        state_hash = self.state_encoder.encode_to_hash(state)
        q_values = {}
        
        async with self.lock:
            for action in actions:
                action_str = self.action_space.encode_action(action)
                if self.use_double_q:
                    # Average of both Q-tables
                    q_a = self.q_values_a.get((state_hash, action_str), self.initial_q_value)
                    q_b = self.q_values_b.get((state_hash, action_str), self.initial_q_value)
                    q_values[action] = (q_a + q_b) / 2.0
                else:
                    q_values[action] = self.q_values.get((state_hash, action_str), self.initial_q_value)
        
        return q_values
    
    async def update(self, state: np.ndarray, action: Tuple[str, ...], 
                    reward: float, next_state: np.ndarray, 
                    next_actions: List[Tuple[str, ...]]):
        """Update Q-value using double Q-learning to reduce overestimation bias.
        
        For double Q-learning:
        - Randomly choose which Q-table to update
        - Use the other Q-table to select the best action
        - This reduces overestimation bias
        """
        state_hash = self.state_encoder.encode_to_hash(state)
        action_str = self.action_space.encode_action(action)
        next_state_hash = self.state_encoder.encode_to_hash(next_state)
        
        async with self.lock:
            if self.use_double_q:
                # Double Q-learning update
                import random
                if random.random() < 0.5:
                    # Update Q_A using Q_B for action selection
                    current_q = self.q_values_a.get((state_hash, action_str), self.initial_q_value)
                    
                    if next_actions:
                        # Find best action according to Q_A
                        best_action = None
                        best_q_a = float('-inf')
                        for next_action in next_actions:
                            next_action_str = self.action_space.encode_action(next_action)
                            q_a_val = self.q_values_a.get((next_state_hash, next_action_str), self.initial_q_value)
                            if q_a_val > best_q_a:
                                best_q_a = q_a_val
                                best_action = next_action_str
                        
                        # Use Q_B to evaluate the best action
                        if best_action:
                            next_q = self.q_values_b.get((next_state_hash, best_action), self.initial_q_value)
                        else:
                            next_q = 0.0
                    else:
                        next_q = 0.0
                    
                    # Update Q_A
                    new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                    self.q_values_a[(state_hash, action_str)] = new_q
                    
                else:
                    # Update Q_B using Q_A for action selection
                    current_q = self.q_values_b.get((state_hash, action_str), self.initial_q_value)
                    
                    if next_actions:
                        # Find best action according to Q_B
                        best_action = None
                        best_q_b = float('-inf')
                        for next_action in next_actions:
                            next_action_str = self.action_space.encode_action(next_action)
                            q_b_val = self.q_values_b.get((next_state_hash, next_action_str), self.initial_q_value)
                            if q_b_val > best_q_b:
                                best_q_b = q_b_val
                                best_action = next_action_str
                        
                        # Use Q_A to evaluate the best action
                        if best_action:
                            next_q = self.q_values_a.get((next_state_hash, best_action), self.initial_q_value)
                        else:
                            next_q = 0.0
                    else:
                        next_q = 0.0
                    
                    # Update Q_B
                    new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                    self.q_values_b[(state_hash, action_str)] = new_q
                
            else:
                # Standard Q-learning update
                current_q = self.q_values.get((state_hash, action_str), self.initial_q_value)
                
                # Get max Q-value for next state
                if next_actions:
                    max_next_q = float('-inf')
                    for next_action in next_actions:
                        next_action_str = self.action_space.encode_action(next_action)
                        q_val = self.q_values.get((next_state_hash, next_action_str), self.initial_q_value)
                        max_next_q = max(max_next_q, q_val)
                else:
                    max_next_q = 0.0
                
                # Q-learning update
                new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
                self.q_values[(state_hash, action_str)] = new_q
            
            self.update_count[(state_hash, action_str)] += 1
        
        logger.debug(f"Q-update: state={state_hash[:8]}, action={action_str}, "
                    f"reward={reward:.2f}, double_q={self.use_double_q}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Q-table statistics."""
        if self.use_double_q:
            # Combine values from both Q-tables
            q_values_list = list(self.q_values_a.values()) + list(self.q_values_b.values())
            total_entries = len(self.q_values_a) + len(self.q_values_b)
        else:
            q_values_list = list(self.q_values.values())
            total_entries = len(self.q_values)
        
        return {
            'total_entries': total_entries,
            'total_updates': sum(self.update_count.values()),
            'avg_q_value': np.mean(q_values_list) if q_values_list else 0.0,
            'max_q_value': max(q_values_list) if q_values_list else 0.0,
            'min_q_value': min(q_values_list) if q_values_list else 0.0,
            'most_updated': max(self.update_count.items(), 
                               key=lambda x: x[1])[0] if self.update_count else None,
            'using_double_q': self.use_double_q
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
        # ENHANCED: Start with higher exploration for better initial learning
        self.exploration_rate = q_config.get('exploration_rate', q_config.get('epsilon', 0.5))
        self.initial_exploration_rate = self.exploration_rate
        self.exploration_decay = q_config.get('exploration_decay', 0.995)
        self.min_exploration_rate = q_config.get('min_exploration_rate', 0.01)
        
        # Adaptive decay settings
        self.adaptive_decay = q_config.get('adaptive_decay', False)
        self.decay_schedule = q_config.get('decay_schedule', 'exponential')
        self.decay_milestones = q_config.get('decay_milestones', [1000, 3000, 5000, 7000])
        self.performance_based_decay = q_config.get('performance_based_decay', False)
        self.initial_exploration_rate = self.exploration_rate
        self.recent_performance = deque(maxlen=100)  # Track recent rewards
        
        # Initialize components
        # Check for encoder configuration
        encoder_config = config.get('state_encoder', {})
        use_encoder = encoder_config.get('enabled', False)
        encoder_path = encoder_config.get('model_path', None)
        
        # Initialize state encoder with supervised encoder if configured
        if use_encoder and encoder_path and os.path.exists(encoder_path):
            logger.info(f"Initializing with supervised encoder from {encoder_path}")
            self.state_encoder = StateRepresentation(
                use_pca=False,
                use_encoder=True,
                encoder_path=encoder_path
            )
        else:
            # Disable PCA for DQN to maintain full state dimensions (476)
            self.state_encoder = StateRepresentation(use_pca=False)
            if use_encoder:
                logger.warning(f"Encoder enabled but model not found at {encoder_path}")
        
        self.action_space = ActionSpace(max_tools=q_config.get('max_tools', 3))
        
        # Check if DQN is enabled
        self.use_dqn = config.get('dqn', {}).get('enabled', False)
        
        if self.use_dqn:
            # Initialize DQN agent
            from .dqn_agent import DQNAgent
            # Get maximum possible action space size
            # This is an upper bound - actual valid actions may be fewer
            max_action_combinations = self._estimate_max_actions(q_config.get('max_tools', 3))
            
            # Get actual state dimension (either encoder output or raw)
            if self.state_encoder.use_encoder and self.state_encoder.encoder:
                state_dim = self.state_encoder.encoder_dim
                logger.info(f"DQN using encoded state dimension: {state_dim}")
            else:
                state_dim = self.state_encoder.total_dimensions
                logger.info(f"DQN using raw state dimension: {state_dim}")
            
            self.dqn_agent = DQNAgent(
                config, 
                state_dim,
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
        
        # Track performance for adaptive decay
        if self.performance_based_decay:
            self.recent_performance.append(1.0 if reward > 0 else 0.0)
    
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
            if self.adaptive_decay:
                self._apply_adaptive_decay()
            else:
                # Standard exponential decay
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
                        # Load DQN model with backward compatibility
                        if 'checkpoint_path' in model_data:
                            checkpoint_path = model_data['checkpoint_path']
                            if os.path.exists(checkpoint_path):
                                try:
                                    self.dqn_agent.load_checkpoint(checkpoint_path)
                                    logger.info(f"Successfully loaded DQN checkpoint from {checkpoint_path}")
                                except RuntimeError as e:
                                    if "Error(s) in loading state_dict" in str(e):
                                        logger.warning(f"Model architecture mismatch when loading checkpoint: {e}")
                                        logger.warning("This usually happens when the saved model has a different architecture.")
                                        logger.warning("Creating new model with current architecture. Training will start fresh.")
                                        # Model is already initialized, just don't load the old weights
                                    else:
                                        # Re-raise if it's a different error
                                        raise
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
    
    def _apply_adaptive_decay(self):
        """Apply adaptive epsilon decay based on schedule and performance."""
        if self.decay_schedule == 'exponential':
            # ENHANCED: More aggressive exponential decay for better exploitation
            # Start at 0.5, decay to 0.01 over episodes
            initial_epsilon = 0.5
            decay_rate = 0.995  # Faster decay rate
            
            # Calculate new epsilon with exponential decay
            new_epsilon = initial_epsilon * (decay_rate ** self.episode_count)
            
            # Check for milestone adjustments
            if self.episode_count in self.decay_milestones:
                # Stronger decay at milestones
                new_epsilon *= 0.7
                logger.info(f"Milestone {self.episode_count}: Extra decay applied, ε={new_epsilon:.4f}")
            
            self.exploration_rate = max(new_epsilon, self.min_exploration_rate)
            
        elif self.decay_schedule == 'linear':
            # Linear decay over total episodes
            total_episodes = 10000  # Expected total episodes
            decay_per_episode = (self.initial_exploration_rate - self.min_exploration_rate) / total_episodes
            self.exploration_rate = max(
                self.initial_exploration_rate - decay_per_episode * self.episode_count,
                self.min_exploration_rate
            )
            
        elif self.decay_schedule == 'cosine':
            # Cosine annealing schedule
            import math
            total_episodes = 10000
            if self.episode_count < total_episodes:
                cosine_decay = 0.5 * (1 + math.cos(math.pi * self.episode_count / total_episodes))
                self.exploration_rate = self.min_exploration_rate + \
                    (self.initial_exploration_rate - self.min_exploration_rate) * cosine_decay
            else:
                self.exploration_rate = self.min_exploration_rate
        
        # Performance-based adjustment
        if self.performance_based_decay and len(self.recent_performance) >= 50:
            avg_performance = sum(self.recent_performance) / len(self.recent_performance)
            if avg_performance > 0.5:  # Good performance, reduce exploration
                self.exploration_rate *= 0.95
            elif avg_performance < 0.2:  # Poor performance, increase exploration slightly
                self.exploration_rate = min(self.exploration_rate * 1.05, 0.3)
        
        # Ensure within bounds
        self.exploration_rate = max(min(self.exploration_rate, 1.0), self.min_exploration_rate)
    
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