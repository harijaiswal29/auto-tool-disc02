"""Unit tests for baseline strategies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import asyncio

from src.evaluation.baseline_strategies import (
    BaselineStrategy,
    RandomSelectionBaseline,
    MostPopularToolsBaseline,
    FixedPolicyBaseline,
    GreedySingleToolBaseline,
    ContextAgnosticQLearningBaseline
)


class TestBaselineStrategy:
    """Test base strategy functionality."""
    
    def test_initialization(self):
        """Test baseline strategy initialization."""
        config = {'max_tools': 3}
        
        # Create a concrete implementation for testing
        class ConcreteStrategy(BaselineStrategy):
            async def select_tools(self, state, available_tools, constraints):
                return available_tools[:1]
        
        strategy = ConcreteStrategy("TestStrategy", config)
        
        assert strategy.name == "TestStrategy"
        assert strategy.config == config
        assert strategy.performance_history == []
        assert strategy.selection_history == []
    
    def test_update(self):
        """Test strategy update functionality."""
        class ConcreteStrategy(BaselineStrategy):
            async def select_tools(self, state, available_tools, constraints):
                return available_tools[:1]
        
        strategy = ConcreteStrategy("TestStrategy", {})
        
        state = np.random.randn(439)
        action = ['tool1', 'tool2']
        reward = 0.8
        next_state = np.random.randn(439)
        
        strategy.update(state, action, reward, next_state)
        
        assert len(strategy.performance_history) == 1
        assert strategy.performance_history[0]['action'] == action
        assert strategy.performance_history[0]['reward'] == reward
        assert strategy.selection_history == ['tool1', 'tool2']
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        class ConcreteStrategy(BaselineStrategy):
            async def select_tools(self, state, available_tools, constraints):
                return available_tools[:1]
        
        strategy = ConcreteStrategy("TestStrategy", {})
        
        # Add some performance data
        for i in range(5):
            strategy.update(None, ['tool1'], 0.5 + i * 0.1, None)
        
        stats = strategy.get_statistics()
        
        assert stats['avg_reward'] == pytest.approx(0.7, 0.01)
        assert stats['std_reward'] == pytest.approx(0.1414, 0.01)
        assert stats['max_reward'] == 0.9
        assert stats['min_reward'] == 0.5
        assert stats['total_episodes'] == 5
        assert stats['tool_frequency']['tool1'] == 5


class TestRandomSelectionBaseline:
    """Test random selection baseline."""
    
    @pytest.mark.asyncio
    async def test_random_selection(self):
        """Test random tool selection."""
        config = {'max_tools': 3}
        strategy = RandomSelectionBaseline(config)
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3', 'tool4']
        constraints = {'conflicts': {}}
        
        # Test multiple selections
        selections = []
        for _ in range(10):
            selected = await strategy.select_tools(state, available_tools, constraints)
            selections.append(selected)
            
            # Check constraints
            assert len(selected) >= 1
            assert len(selected) <= 3
            assert all(tool in available_tools for tool in selected)
        
        # Check randomness (different selections)
        unique_selections = set(tuple(sorted(s)) for s in selections)
        assert len(unique_selections) > 1
    
    @pytest.mark.asyncio
    async def test_constraint_handling(self):
        """Test constraint handling in random selection."""
        config = {'max_tools': 3}
        strategy = RandomSelectionBaseline(config)
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3']
        constraints = {
            'conflicts': {
                'tool1': ['tool2'],
                'tool2': ['tool1']
            }
        }
        
        # Test multiple times due to randomness
        for _ in range(20):
            selected = await strategy.select_tools(state, available_tools, constraints)
            
            # Check that conflicting tools are not selected together
            if 'tool1' in selected:
                assert 'tool2' not in selected
            if 'tool2' in selected:
                assert 'tool1' not in selected
    
    @pytest.mark.asyncio
    async def test_empty_tools(self):
        """Test behavior with no available tools."""
        strategy = RandomSelectionBaseline({})
        
        selected = await strategy.select_tools(
            np.zeros(439), [], {}
        )
        
        assert selected == []


class TestMostPopularToolsBaseline:
    """Test most popular tools baseline."""
    
    @pytest.mark.asyncio
    async def test_popularity_tracking(self):
        """Test that tool popularity is tracked correctly."""
        config = {'max_tools': 2, 'min_history': 3}
        strategy = MostPopularToolsBaseline(config)
        
        # Build up history
        strategy.update(None, ['tool1', 'tool2'], 0.8, None)
        strategy.update(None, ['tool1', 'tool3'], 0.7, None)
        strategy.update(None, ['tool1', 'tool2'], 0.9, None)
        
        assert strategy.tool_popularity['tool1'] == 3
        assert strategy.tool_popularity['tool2'] == 2
        assert strategy.tool_popularity['tool3'] == 1
    
    @pytest.mark.asyncio
    async def test_popular_selection(self):
        """Test selection based on popularity."""
        config = {'max_tools': 2, 'min_history': 0}
        strategy = MostPopularToolsBaseline(config)
        
        # Set up popularity
        strategy.tool_popularity = {
            'tool1': 10,
            'tool2': 5,
            'tool3': 3,
            'tool4': 1
        }
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3', 'tool4']
        constraints = {'conflicts': {}}
        
        selected = await strategy.select_tools(state, available_tools, constraints)
        
        # Should select most popular tools
        assert selected == ['tool1', 'tool2']
    
    @pytest.mark.asyncio
    async def test_insufficient_history(self):
        """Test behavior with insufficient history."""
        config = {'max_tools': 2, 'min_history': 10}
        strategy = MostPopularToolsBaseline(config)
        
        # Only add a little history
        strategy.update(None, ['tool1'], 0.5, None)
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3']
        
        selected = await strategy.select_tools(state, available_tools, {})
        
        # Should fall back to random selection
        assert len(selected) >= 1
        assert len(selected) <= 2


class TestFixedPolicyBaseline:
    """Test fixed policy baseline."""
    
    @pytest.mark.asyncio
    async def test_policy_selection(self):
        """Test fixed policy tool selection."""
        config = {}
        strategy = FixedPolicyBaseline(config)
        
        # Override intent classification for testing
        strategy._classify_intent = MagicMock(return_value='file_search')
        
        state = np.random.randn(439)
        available_tools = ['filesystem_mcp', 'search_mcp', 'github_mcp']
        constraints = {'conflicts': {}}
        
        selected = await strategy.select_tools(state, available_tools, constraints)
        
        # Should select tools based on file_search policy
        assert set(selected) == {'filesystem_mcp', 'search_mcp'}
    
    @pytest.mark.asyncio
    async def test_unavailable_tools_handling(self):
        """Test handling when policy tools are not available."""
        strategy = FixedPolicyBaseline({})
        strategy._classify_intent = MagicMock(return_value='weather_query')
        
        state = np.random.randn(439)
        available_tools = ['filesystem_mcp', 'search_mcp']  # weather_mcp not available
        
        selected = await strategy.select_tools(state, available_tools, {})
        
        # Should fall back to first available tool
        assert selected == ['filesystem_mcp']
    
    @pytest.mark.asyncio
    async def test_constraint_application(self):
        """Test constraint application in fixed policy."""
        strategy = FixedPolicyBaseline({})
        strategy._classify_intent = MagicMock(return_value='data_query')
        
        state = np.random.randn(439)
        available_tools = ['sqlite_mcp', 'postgres_mcp']
        constraints = {
            'conflicts': {
                'sqlite_mcp': ['postgres_mcp']
            }
        }
        
        selected = await strategy.select_tools(state, available_tools, constraints)
        
        # Should only select one due to conflict
        assert len(selected) == 1
        assert selected[0] in ['sqlite_mcp', 'postgres_mcp']


class TestGreedySingleToolBaseline:
    """Test greedy single tool baseline."""
    
    @pytest.mark.asyncio
    async def test_greedy_selection(self):
        """Test greedy selection of best tool."""
        config = {'min_history': 2}
        strategy = GreedySingleToolBaseline(config)
        
        # Set up tool rewards
        strategy.tool_rewards = {
            'tool1': [0.8, 0.9, 0.85],
            'tool2': [0.6, 0.7, 0.65],
            'tool3': [0.4, 0.5]
        }
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3']
        
        selected = await strategy.select_tools(state, available_tools, {})
        
        # Should select tool1 (highest average)
        assert selected == ['tool1']
    
    @pytest.mark.asyncio
    async def test_update_rewards(self):
        """Test reward update mechanism."""
        strategy = GreedySingleToolBaseline({})
        
        strategy.update(None, ['tool1', 'tool2'], 0.8, None)
        
        # Reward should be distributed
        assert len(strategy.tool_rewards['tool1']) == 1
        assert len(strategy.tool_rewards['tool2']) == 1
        assert strategy.tool_rewards['tool1'][0] == 0.4
        assert strategy.tool_rewards['tool2'][0] == 0.4
    
    @pytest.mark.asyncio
    async def test_insufficient_history_fallback(self):
        """Test fallback when insufficient history."""
        config = {'min_history': 5}
        strategy = GreedySingleToolBaseline(config)
        
        strategy.tool_rewards = {
            'tool1': [0.8, 0.9]  # Only 2 samples, need 5
        }
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3']
        
        selected = await strategy.select_tools(state, available_tools, {})
        
        # Should select randomly
        assert len(selected) == 1
        assert selected[0] in available_tools


class TestContextAgnosticQLearningBaseline:
    """Test context-agnostic Q-learning baseline."""
    
    @pytest.mark.asyncio
    async def test_q_learning_selection(self):
        """Test Q-learning based selection."""
        config = {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 0.0,  # No exploration for testing
            'max_tools': 2
        }
        strategy = ContextAgnosticQLearningBaseline(config)
        
        # Set up Q-values
        state_key = strategy._get_simple_state_key(['tool1', 'tool2'])
        strategy.q_table[state_key][str(('tool1',))] = 0.8
        strategy.q_table[state_key][str(('tool2',))] = 0.6
        strategy.q_table[state_key][str(('tool1', 'tool2'))] = 0.9
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2']
        
        selected = await strategy.select_tools(state, available_tools, {})
        
        # Should select highest Q-value action
        assert set(selected) == {'tool1', 'tool2'}
    
    @pytest.mark.asyncio
    async def test_exploration(self):
        """Test exploration behavior."""
        config = {
            'exploration_rate': 1.0,  # Always explore
            'max_tools': 2
        }
        strategy = ContextAgnosticQLearningBaseline(config)
        
        state = np.random.randn(439)
        available_tools = ['tool1', 'tool2', 'tool3']
        
        # Multiple selections should be different due to exploration
        selections = []
        for _ in range(10):
            selected = await strategy.select_tools(state, available_tools, {})
            selections.append(tuple(sorted(selected)))
        
        # Should have variety in selections
        unique_selections = set(selections)
        assert len(unique_selections) > 1
    
    def test_q_value_update(self):
        """Test Q-value update mechanism."""
        strategy = ContextAgnosticQLearningBaseline({
            'learning_rate': 0.1,
            'discount_factor': 0.9
        })
        
        state_key = 'state1'
        action_key = "('tool1',)"
        reward = 0.8
        next_state_key = 'state2'
        next_actions = [('tool1',), ('tool2',)]
        
        # Set initial Q-values
        strategy.q_table[next_state_key][str(('tool1',))] = 0.5
        strategy.q_table[next_state_key][str(('tool2',))] = 0.7
        
        strategy.update_q_value(state_key, action_key, reward, 
                              next_state_key, next_actions)
        
        # Check Q-value was updated correctly
        # new_q = 0 + 0.1 * (0.8 + 0.9 * 0.7 - 0) = 0.143
        expected_q = 0.1 * (0.8 + 0.9 * 0.7)
        assert strategy.q_table[state_key][action_key] == pytest.approx(expected_q, 0.001)
    
    def test_simple_state_key_generation(self):
        """Test state key generation."""
        strategy = ContextAgnosticQLearningBaseline({})
        
        tools1 = ['tool1', 'tool2', 'tool3']
        tools2 = ['tool3', 'tool1', 'tool2']  # Same tools, different order
        tools3 = ['tool1', 'tool2']  # Different tools
        
        key1 = strategy._get_simple_state_key(tools1)
        key2 = strategy._get_simple_state_key(tools2)
        key3 = strategy._get_simple_state_key(tools3)
        
        # Same tools should produce same key
        assert key1 == key2
        # Different tools should produce different key
        assert key1 != key3