"""Unit tests for Deep Q-Network implementation."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import tempfile
import os
from unittest.mock import MagicMock, patch
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.learning.deep_q_network import DQN, DuelingDQN, NoisyDQN, create_dqn, count_parameters
from src.learning.dqn_agent import DQNAgent
from src.learning.prioritized_replay_buffer import PrioritizedReplayBuffer, SumTree, UniformReplayBuffer


class TestDQN:
    """Test cases for DQN network architectures."""
    
    def test_dqn_initialization(self):
        """Test DQN initialization."""
        state_dim = 439
        action_dim = 10
        hidden_dims = [512, 256, 128]
        
        dqn = DQN(state_dim, action_dim, hidden_dims)
        
        assert dqn.state_dim == state_dim
        assert dqn.action_dim == action_dim
        assert dqn.hidden_dims == hidden_dims
        
        # Check network structure
        assert len(dqn.layers) == len(hidden_dims) * 3  # Linear + ReLU + Dropout per layer
        assert isinstance(dqn.output_layer, nn.Linear)
    
    def test_dqn_forward_pass(self):
        """Test DQN forward pass."""
        state_dim = 10
        action_dim = 4
        batch_size = 32
        
        dqn = DQN(state_dim, action_dim, [64, 32])
        
        # Create random input
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass
        q_values = dqn(state)
        
        assert q_values.shape == (batch_size, action_dim)
        assert not torch.isnan(q_values).any()
    
    def test_dqn_get_action_values(self):
        """Test getting action values for numpy input."""
        state_dim = 10
        action_dim = 4
        
        dqn = DQN(state_dim, action_dim)
        
        # Numpy state
        state = np.random.randn(state_dim)
        
        # Get action values
        q_values = dqn.get_action_values(state)
        
        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (action_dim,)
    
    def test_dueling_dqn(self):
        """Test Dueling DQN architecture."""
        state_dim = 10
        action_dim = 4
        batch_size = 16
        
        dueling_dqn = DuelingDQN(state_dim, action_dim, [64, 32])
        
        # Forward pass
        state = torch.randn(batch_size, state_dim)
        q_values = dueling_dqn(state)
        
        assert q_values.shape == (batch_size, action_dim)
        
        # Test that value and advantage streams work
        # The mean of advantages should be subtracted
        features = state
        for layer in dueling_dqn.feature_layers:
            features = layer(features)
        
        value = dueling_dqn.value_stream(features)
        advantages = dueling_dqn.advantage_stream(features)
        
        assert value.shape == (batch_size, 1)
        assert advantages.shape == (batch_size, action_dim)
    
    def test_noisy_dqn(self):
        """Test Noisy DQN."""
        state_dim = 10
        action_dim = 4
        
        noisy_dqn = NoisyDQN(state_dim, action_dim, [64, 32])
        
        # Check that noise can be reset
        noisy_dqn.reset_noise()
        
        # Forward pass
        state = torch.randn(1, state_dim)
        q_values = noisy_dqn(state)
        
        assert q_values.shape == (1, action_dim)
    
    def test_create_dqn_factory(self):
        """Test DQN factory function."""
        state_dim = 10
        action_dim = 4
        
        # Standard DQN
        config = {'network_type': 'standard'}
        dqn = create_dqn(config, state_dim, action_dim)
        assert isinstance(dqn, DQN)
        
        # Dueling DQN
        config = {'network_type': 'dueling'}
        dqn = create_dqn(config, state_dim, action_dim)
        assert isinstance(dqn, DuelingDQN)
        
        # Noisy DQN
        config = {'network_type': 'noisy'}
        dqn = create_dqn(config, state_dim, action_dim)
        assert isinstance(dqn, NoisyDQN)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        dqn = DQN(10, 4, [32, 16])
        param_count = count_parameters(dqn)
        
        # Manual calculation: 10*32 + 32 + 32*16 + 16 + 16*4 + 4 = 320 + 32 + 512 + 16 + 64 + 4 = 948
        assert param_count > 0
        assert param_count == sum(p.numel() for p in dqn.parameters())


class TestDQNAgent:
    """Test cases for DQN Agent."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'dqn': {
                'enabled': True,
                'learning_rate': 0.001,
                'batch_size': 32,
                'memory_size': 1000,
                'target_update_frequency': 100,
                'device': 'cpu'
            }
        }
    
    def test_agent_initialization(self, config):
        """Test DQN agent initialization."""
        state_dim = 10
        action_space_size = 4
        
        agent = DQNAgent(config, state_dim, action_space_size)
        
        assert agent.state_dim == state_dim
        assert agent.action_space_size == action_space_size
        assert isinstance(agent.q_network, nn.Module)
        assert isinstance(agent.target_network, nn.Module)
    
    def test_action_selection(self, config):
        """Test action selection."""
        agent = DQNAgent(config, 10, 4)
        
        state = np.random.randn(10)
        valid_actions = [('tool1',), ('tool2',), ('tool1', 'tool2')]
        
        # Test exploration
        agent.epsilon = 1.0  # Always explore
        action = agent.select_action(state, valid_actions, training=True)
        assert action in valid_actions
        
        # Test exploitation
        agent.epsilon = 0.0  # Never explore
        action = agent.select_action(state, valid_actions, training=True)
        assert action in valid_actions
    
    def test_store_transition(self, config):
        """Test storing transitions."""
        agent = DQNAgent(config, 10, 4)
        
        state = np.random.randn(10)
        action = ('tool1',)
        reward = 1.0
        next_state = np.random.randn(10)
        next_valid_actions = [('tool2',), ('tool3',)]
        done = False
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, 
                             next_valid_actions, done)
        
        # Check memory
        assert len(agent.memory) > 0
    
    def test_train_step(self, config):
        """Test training step."""
        agent = DQNAgent(config, 10, 4)
        
        # Add some transitions
        for _ in range(50):
            state = np.random.randn(10)
            action = ('tool1',)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            next_valid_actions = [('tool1',), ('tool2',)]
            done = False
            
            agent.store_transition(state, action, reward, next_state,
                                 next_valid_actions, done)
        
        # Train step
        loss = agent.train_step()
        assert loss is not None
        assert loss > 0
    
    def test_target_network_update(self, config):
        """Test target network update."""
        agent = DQNAgent(config, 10, 4)
        
        # Get initial target weights
        initial_weights = agent.target_network.state_dict()['output_layer.weight'].clone()
        
        # Update target network
        agent.update_target_network(soft=False)
        
        # Check weights are copied
        q_weights = agent.q_network.state_dict()['output_layer.weight']
        target_weights = agent.target_network.state_dict()['output_layer.weight']
        
        assert torch.allclose(q_weights, target_weights)
    
    def test_checkpoint_save_load(self, config):
        """Test checkpoint saving and loading."""
        agent = DQNAgent(config, 10, 4)
        
        # Modify some values
        agent.steps_done = 1000
        agent.epsilon = 0.5
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            agent.save_checkpoint(checkpoint_path)
            
            # Create new agent and load
            new_agent = DQNAgent(config, 10, 4)
            new_agent.load_checkpoint(checkpoint_path)
            
            assert new_agent.steps_done == 1000
            assert new_agent.epsilon == 0.5


class TestPrioritizedReplayBuffer:
    """Test cases for Prioritized Replay Buffer."""
    
    def test_sum_tree(self):
        """Test sum-tree data structure."""
        capacity = 8
        tree = SumTree(capacity)
        
        # Add elements
        for i in range(10):
            priority = i + 1
            data = f"data_{i}"
            tree.add(priority, data)
        
        # Check total
        assert tree.total() > 0
        
        # Sample based on priority
        idx, priority, data = tree.get(tree.total() / 2)
        assert data is not None
        assert priority > 0
    
    def test_prioritized_replay_buffer(self):
        """Test prioritized replay buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)
        
        # Add transitions
        for i in range(50):
            transition = {
                'state': np.random.randn(10),
                'action': i,
                'reward': np.random.randn(),
                'next_state': np.random.randn(10),
                'done': False
            }
            buffer.add(transition)
        
        # Sample batch
        batch, indices, weights = buffer.sample(32)
        
        assert len(batch) == 32
        assert len(indices) == 32
        assert len(weights) == 32
        assert all(w > 0 for w in weights)
        
        # Update priorities
        new_priorities = np.random.rand(32) + 0.01
        buffer.update_priorities(indices, new_priorities)
    
    def test_uniform_replay_buffer(self):
        """Test uniform replay buffer for comparison."""
        buffer = UniformReplayBuffer(capacity=100)
        
        # Add transitions
        for i in range(50):
            transition = {'data': i}
            buffer.add(transition)
        
        # Sample batch
        batch, indices, weights = buffer.sample(32)
        
        assert len(batch) == 32
        assert all(w == 1.0 for w in weights)


class TestIntegration:
    """Integration tests for DQN with Q-learning engine."""
    
    @pytest.mark.asyncio
    async def test_dqn_in_q_learning_engine(self):
        """Test DQN integration in Q-learning engine."""
        config = {
            'q_learning': {
                'max_tools': 3
            },
            'dqn': {
                'enabled': True,
                'batch_size': 32,
                'device': 'cpu'
            }
        }
        
        # Import here to avoid circular imports
        from src.learning.q_learning_engine import QLearningEngine
        
        engine = QLearningEngine(config)
        
        assert engine.use_dqn
        assert engine.dqn_agent is not None
        
        # Test action selection
        state = np.random.randn(439)  # Full state dimension
        available_tools = ['tool1', 'tool2', 'tool3']
        constraints = {}
        
        action = await engine.select_action(state, available_tools, constraints)
        assert isinstance(action, tuple)
        
        # Test learning
        next_state = np.random.randn(439)
        reward = 1.0
        
        await engine.learn_from_experience(
            state, action, reward, next_state,
            available_tools, constraints, done=False
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])