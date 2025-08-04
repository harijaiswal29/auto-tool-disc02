"""Unit tests for Q-Learning Engine."""

import pytest
import asyncio
import numpy as np
import json
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.learning.q_learning_engine import (
    QLearningEngine, StateRepresentation, ActionSpace, 
    QTable, ExperienceReplayBuffer
)


class TestStateRepresentation:
    """Test cases for StateRepresentation class."""
    
    def setup_method(self):
        """Set up test instance."""
        self.state_encoder = StateRepresentation()
    
    def test_initialization(self):
        """Test StateRepresentation initialization."""
        assert self.state_encoder.state_dimensions['intent_vector'] == 384
        assert self.state_encoder.state_dimensions['context_features'] == 10
        assert self.state_encoder.state_dimensions['tool_history'] == 20
        assert self.state_encoder.state_dimensions['performance_metrics'] == 5
        assert self.state_encoder.state_dimensions['failure_rates'] == 10
        assert self.state_encoder.state_dimensions['failure_types'] == 5
        assert self.state_encoder.state_dimensions['retry_patterns'] == 5
        assert self.state_encoder.state_dimensions['user_expertise'] == 3
        assert self.state_encoder.state_dimensions['domain_context'] == 5
        assert self.state_encoder.total_dimensions == 447  # Updated to correct value
    
    def test_encode_state(self):
        """Test state encoding."""
        # Mock intent with embedding
        intent = MagicMock()
        intent.embedding = np.random.rand(384)
        
        context = {
            'domain': 'engineering',
            'query_count': 5,
            'session_duration': 1800,
            'metrics': {'avg_response_time': 500}
        }
        
        history = ['tool1', 'tool2']
        
        state = self.state_encoder.encode_state(intent, context, history)
        
        assert isinstance(state, np.ndarray)
        assert state.shape[0] == self.state_encoder.total_dimensions
    
    def test_encode_state_without_embedding(self):
        """Test state encoding when intent has no embedding."""
        intent = MagicMock()
        intent.embedding = None
        del intent.embedding  # Remove attribute
        
        context = {}
        history = []
        
        state = self.state_encoder.encode_state(intent, context, history)
        
        assert isinstance(state, np.ndarray)
        assert state.shape[0] == self.state_encoder.total_dimensions
    
    def test_encode_to_hash(self):
        """Test state vector hashing."""
        state_vector = np.random.rand(447)
        
        hash1 = self.state_encoder.encode_to_hash(state_vector)
        hash2 = self.state_encoder.encode_to_hash(state_vector)
        
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        assert hash1 == hash2  # Same input should produce same hash
    
    def test_context_encoding(self):
        """Test context feature encoding."""
        context = {
            'domain': 'data_science',
            'query_count': 10,
            'session_duration': 7200,
            'total_queries': 100,
            'success_rate': 0.9
        }
        
        features = self.state_encoder._encode_context(context)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == self.state_encoder.state_dimensions['context_features']
        assert features[1] == 1.0  # data_science domain
        assert features[4] == 1.0  # query_count normalized to max
    
    def test_history_encoding(self):
        """Test tool history encoding."""
        history = ['filesystem_mcp'] * 5 + ['sqlite_mcp'] * 3
        
        features = self.state_encoder._encode_history(history)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == self.state_encoder.state_dimensions['tool_history']
        assert features[0] == 1.0  # filesystem_mcp frequency maxed out


class TestActionSpace:
    """Test cases for ActionSpace class."""
    
    def setup_method(self):
        """Set up test instance."""
        self.action_space = ActionSpace(max_tools=3)
    
    def test_initialization(self):
        """Test ActionSpace initialization."""
        assert self.action_space.max_tools == 3
        assert isinstance(self.action_space.tool_combinations_cache, dict)
    
    def test_get_valid_actions_no_constraints(self):
        """Test getting valid actions without constraints."""
        tools = ['tool1', 'tool2', 'tool3']
        constraints = {}
        
        actions = self.action_space.get_valid_actions(tools, constraints)
        
        # Should have C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7 combinations
        assert len(actions) == 7
        assert ('tool1',) in actions
        assert ('tool1', 'tool2') in actions
        assert ('tool1', 'tool2', 'tool3') in actions
    
    def test_get_valid_actions_with_conflicts(self):
        """Test getting valid actions with conflicts."""
        tools = ['tool1', 'tool2', 'tool3']
        constraints = {
            'conflicts': {
                'tool1': ['tool2'],
                'tool2': ['tool1']
            }
        }
        
        actions = self.action_space.get_valid_actions(tools, constraints)
        
        # Should not have combinations with both tool1 and tool2
        for action in actions:
            assert not ('tool1' in action and 'tool2' in action)
    
    def test_get_valid_actions_with_requirements(self):
        """Test getting valid actions with requirements."""
        tools = ['tool1', 'tool2', 'tool3']
        constraints = {
            'requires': {
                'tool1': ['tool2']  # tool1 requires tool2
            }
        }
        
        actions = self.action_space.get_valid_actions(tools, constraints)
        
        # tool1 should only appear with tool2
        for action in actions:
            if 'tool1' in action:
                assert 'tool2' in action
    
    def test_action_encoding_decoding(self):
        """Test action encoding and decoding."""
        action = ('tool3', 'tool1', 'tool2')
        
        encoded = self.action_space.encode_action(action)
        decoded = self.action_space.decode_action(encoded)
        
        assert encoded == 'tool1|tool2|tool3'  # Should be sorted
        assert set(decoded) == set(action)  # Order may differ
    
    def test_caching(self):
        """Test action combinations caching."""
        tools = ['tool1', 'tool2']
        constraints = {}
        
        # First call should compute
        actions1 = self.action_space.get_valid_actions(tools, constraints)
        assert len(self.action_space.tool_combinations_cache) == 1
        
        # Second call should use cache
        actions2 = self.action_space.get_valid_actions(tools, constraints)
        assert actions1 == actions2
        assert len(self.action_space.tool_combinations_cache) == 1


class TestQTable:
    """Test cases for QTable class."""
    
    def setup_method(self):
        """Set up test instance."""
        self.q_table = QTable(learning_rate=0.1, discount_factor=0.9)
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test QTable initialization."""
        assert self.q_table.alpha == 0.1
        assert self.q_table.gamma == 0.9
        assert isinstance(self.q_table.q_values, dict)
        assert isinstance(self.q_table.update_count, dict)
    
    @pytest.mark.asyncio
    async def test_get_q_value(self):
        """Test getting Q-value."""
        state = np.random.rand(447)
        action = ('tool1', 'tool2')
        
        # Initially should return 0
        q_value = await self.q_table.get_q_value(state, action)
        assert q_value == 0.0
        
        # Set a value
        state_hash = self.q_table.state_encoder.encode_to_hash(state)
        action_str = self.q_table.action_space.encode_action(action)
        self.q_table.q_values[(state_hash, action_str)] = 0.5
        
        # Should return the set value
        q_value = await self.q_table.get_q_value(state, action)
        assert q_value == 0.5
    
    @pytest.mark.asyncio
    async def test_update(self):
        """Test Q-value update."""
        state = np.random.rand(447)
        action = ('tool1',)
        reward = 1.0
        next_state = np.random.rand(447)
        next_actions = [('tool1',), ('tool2',)]
        
        # Perform update
        await self.q_table.update(state, action, reward, next_state, next_actions)
        
        # Check that Q-value was updated
        q_value = await self.q_table.get_q_value(state, action)
        assert q_value > 0  # Should be positive after positive reward
        
        # Check update count
        state_hash = self.q_table.state_encoder.encode_to_hash(state)
        action_str = self.q_table.action_space.encode_action(action)
        assert self.q_table.update_count[(state_hash, action_str)] == 1
    
    def test_get_statistics(self):
        """Test getting Q-table statistics."""
        # Add some Q-values
        self.q_table.q_values[('state1', 'action1')] = 0.5
        self.q_table.q_values[('state2', 'action2')] = 0.8
        self.q_table.update_count[('state1', 'action1')] = 3
        
        stats = self.q_table.get_statistics()
        
        assert stats['total_entries'] == 2
        assert stats['total_updates'] == 3
        assert stats['avg_q_value'] == 0.65
        assert stats['max_q_value'] == 0.8
        assert stats['min_q_value'] == 0.5


class TestExperienceReplayBuffer:
    """Test cases for ExperienceReplayBuffer class."""
    
    def setup_method(self):
        """Set up test instance."""
        self.buffer = ExperienceReplayBuffer(capacity=10)
    
    def test_initialization(self):
        """Test buffer initialization."""
        assert self.buffer.capacity == 10
        assert len(self.buffer) == 0
    
    def test_add_experience(self):
        """Test adding experiences."""
        experience = {
            'state': np.random.rand(447),
            'action': ('tool1',),
            'reward': 1.0,
            'next_state': np.random.rand(447)
        }
        
        self.buffer.add(experience)
        assert len(self.buffer) == 1
        
        # Add more than capacity
        for i in range(15):
            self.buffer.add(experience)
        
        assert len(self.buffer) == 10  # Should not exceed capacity
    
    def test_sample_uniform(self):
        """Test uniform sampling."""
        # Add diverse experiences
        for i in range(5):
            experience = {
                'state': np.random.rand(447),
                'action': (f'tool{i}',),
                'reward': i * 0.2
            }
            self.buffer.add(experience)
        
        # Sample
        batch = self.buffer.sample(3, prioritized=False)
        
        assert len(batch) == 3
        assert all(isinstance(exp, dict) for exp in batch)
    
    def test_sample_prioritized(self):
        """Test prioritized sampling."""
        # Add experiences with different priorities
        for i in range(5):
            experience = {
                'state': np.random.rand(447),
                'action': (f'tool{i}',),
                'reward': i * 0.5,  # Higher reward = higher priority
                'success': i > 2
            }
            self.buffer.add(experience)
        
        # Sample many times to check bias
        sampled_rewards = []
        for _ in range(100):
            batch = self.buffer.sample(1, prioritized=True)
            sampled_rewards.append(batch[0]['reward'])
        
        # Higher reward experiences should be sampled more often
        avg_reward = np.mean(sampled_rewards)
        assert avg_reward > 1.0  # Should be biased towards higher rewards
    
    def test_clear(self):
        """Test clearing buffer."""
        for i in range(5):
            self.buffer.add({'reward': i})
        
        assert len(self.buffer) == 5
        
        self.buffer.clear()
        assert len(self.buffer) == 0


class TestQLearningEngine:
    """Test cases for QLearningEngine class."""
    
    def setup_method(self):
        """Set up test instance."""
        self.config = {
            'q_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.2,
                'exploration_decay': 0.995,
                'min_exploration_rate': 0.01,
                'max_tools': 3,
                'buffer_capacity': 100,
                'batch_size': 16,
                'update_frequency': 4
            }
        }
    
    @patch('src.learning.q_learning_engine.DatabaseManager')
    def test_initialization(self, mock_db):
        """Test QLearningEngine initialization."""
        engine = QLearningEngine(self.config)
        
        assert engine.learning_rate == 0.1
        assert engine.discount_factor == 0.9
        assert engine.exploration_rate == 0.2
        assert engine.batch_size == 16
        assert isinstance(engine.state_encoder, StateRepresentation)
        assert isinstance(engine.action_space, ActionSpace)
        assert isinstance(engine.q_table, QTable)
        assert isinstance(engine.experience_buffer, ExperienceReplayBuffer)
    
    @pytest.mark.asyncio
    @patch('src.learning.q_learning_engine.DatabaseManager')
    async def test_select_action_exploration(self, mock_db):
        """Test action selection in exploration mode."""
        engine = QLearningEngine(self.config)
        engine.exploration_rate = 1.0  # Always explore
        
        state = np.random.rand(447)
        tools = ['tool1', 'tool2', 'tool3']
        constraints = {}
        
        action = await engine.select_action(state, tools, constraints)
        
        assert isinstance(action, tuple)
        assert len(action) >= 1 and len(action) <= 3
        assert all(tool in tools for tool in action)
    
    @pytest.mark.asyncio
    @patch('src.learning.q_learning_engine.DatabaseManager')
    async def test_select_action_exploitation(self, mock_db):
        """Test action selection in exploitation mode."""
        engine = QLearningEngine(self.config)
        engine.exploration_rate = 0.0  # Always exploit
        
        state = np.random.rand(447)
        tools = ['tool1', 'tool2']
        constraints = {}
        
        # Set up Q-values to make tool1 preferred
        state_hash = engine.state_encoder.encode_to_hash(state)
        engine.q_table.q_values[(state_hash, 'tool1')] = 1.0
        engine.q_table.q_values[(state_hash, 'tool2')] = 0.1
        
        action = await engine.select_action(state, tools, constraints)
        
        assert action == ('tool1',)  # Should select highest Q-value
    
    @pytest.mark.asyncio
    @patch('src.learning.q_learning_engine.DatabaseManager')
    async def test_learn_from_experience(self, mock_db):
        """Test learning from experience."""
        engine = QLearningEngine(self.config)
        
        state = np.random.rand(447)
        action = ('tool1',)
        reward = 1.0
        next_state = np.random.rand(447)
        next_tools = ['tool1', 'tool2']
        constraints = {}
        
        # Initial metrics
        initial_reward = engine.total_reward
        
        await engine.learn_from_experience(
            state, action, reward, next_state, 
            next_tools, constraints, done=False
        )
        
        # Check updates
        assert engine.total_reward == initial_reward + reward
        assert engine.success_count == 1
        assert len(engine.experience_buffer) == 1
        assert engine.steps_since_update == 1
    
    @patch('src.learning.q_learning_engine.DatabaseManager')
    def test_decay_exploration(self, mock_db):
        """Test exploration rate decay."""
        engine = QLearningEngine(self.config)
        initial_rate = engine.exploration_rate
        
        engine.decay_exploration()
        
        assert engine.exploration_rate < initial_rate
        assert engine.exploration_rate == initial_rate * 0.995
        assert engine.episode_count == 1
        
        # Test minimum bound
        engine.exploration_rate = 0.01
        engine.decay_exploration()
        assert engine.exploration_rate == 0.01  # Should not go below minimum
    
    @patch('src.learning.q_learning_engine.DatabaseManager')
    def test_get_metrics(self, mock_db):
        """Test getting metrics."""
        engine = QLearningEngine(self.config)
        engine.episode_count = 10
        engine.total_reward = 5.0
        engine.success_count = 7
        
        metrics = engine.get_metrics()
        
        assert metrics['episode_count'] == 10
        assert metrics['total_reward'] == 5.0
        assert metrics['avg_reward'] == 0.5
        assert metrics['success_rate'] == 0.7
        assert 'q_table_stats' in metrics
        assert 'buffer_size' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])