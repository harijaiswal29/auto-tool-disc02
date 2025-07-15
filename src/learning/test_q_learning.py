"""Test script for Q-Learning Engine functionality."""

import asyncio
import json
import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.learning.q_learning_engine import QLearningEngine, StateRepresentation, ActionSpace
from src.agents.intent_models import Intent
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_state_representation():
    """Test state encoding functionality."""
    print("\n=== Testing State Representation ===")
    
    state_encoder = StateRepresentation()
    
    # Create mock intent with embedding
    class MockIntent:
        def __init__(self):
            self.embedding = np.random.rand(384)  # Sentence transformer dimension
    
    intent = MockIntent()
    
    # Create context
    context = {
        'domain': 'engineering',
        'query_count': 5,
        'session_duration': 1800,  # 30 minutes
        'total_queries': 50,
        'success_rate': 0.8,
        'metrics': {
            'avg_response_time': 500,
            'success_rate': 0.8,
            'error_rate': 0.1,
            'tools_invoked': 3,
            'cache_hit_rate': 0.6
        }
    }
    
    # Tool history
    history = ['filesystem_mcp', 'sqlite_mcp', 'filesystem_mcp', 'search_mcp']
    
    # Encode state
    state_vector = state_encoder.encode_state(intent, context, history)
    
    print(f"State vector shape: {state_vector.shape}")
    print(f"State vector dimensions: {len(state_vector)}")
    print(f"Expected dimensions: {state_encoder.total_dimensions}")
    
    # Test state hashing
    state_hash = state_encoder.encode_to_hash(state_vector)
    print(f"State hash: {state_hash}")
    
    assert state_vector.shape[0] == state_encoder.total_dimensions
    print("✓ State representation test passed!")


async def test_action_space():
    """Test action space and constraint validation."""
    print("\n=== Testing Action Space ===")
    
    action_space = ActionSpace(max_tools=3)
    
    # Available tools
    available_tools = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp', 'github_mcp']
    
    # Test without constraints
    constraints = {}
    valid_actions = action_space.get_valid_actions(available_tools, constraints)
    print(f"Without constraints: {len(valid_actions)} valid actions")
    
    # Test with conflicts
    constraints = {
        'conflicts': {
            'filesystem_mcp': ['github_mcp'],  # These tools conflict
            'github_mcp': ['filesystem_mcp']
        }
    }
    valid_actions_with_conflicts = action_space.get_valid_actions(available_tools, constraints)
    print(f"With conflicts: {len(valid_actions_with_conflicts)} valid actions")
    
    # Verify no conflicting combinations
    for action in valid_actions_with_conflicts:
        if 'filesystem_mcp' in action and 'github_mcp' in action:
            print("ERROR: Found conflicting combination!")
            assert False
    
    # Test with requirements
    constraints = {
        'requires': {
            'github_mcp': ['search_mcp']  # github_mcp requires search_mcp
        }
    }
    valid_actions_with_reqs = action_space.get_valid_actions(available_tools, constraints)
    print(f"With requirements: {len(valid_actions_with_reqs)} valid actions")
    
    # Verify requirements are met
    for action in valid_actions_with_reqs:
        if 'github_mcp' in action and 'search_mcp' not in action:
            print("ERROR: Found action violating requirements!")
            assert False
    
    print("✓ Action space test passed!")


async def test_q_learning_cycle():
    """Test complete Q-learning cycle."""
    print("\n=== Testing Q-Learning Cycle ===")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../../config/config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize Q-learning engine
    q_engine = QLearningEngine(config)
    await q_engine.db_manager.initialize()
    
    # Create mock intent
    class MockIntent:
        def __init__(self):
            self.embedding = np.random.rand(384)
    
    intent = MockIntent()
    
    # Initial context
    context = {
        'domain': 'engineering',
        'query_count': 0,
        'success_rate': 0.5,
        'metrics': {'avg_response_time': 1000, 'tools_invoked': 0}
    }
    
    # Simulate 5 learning episodes
    for episode in range(5):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Encode state
        state = q_engine.state_encoder.encode_state(intent, context, [])
        
        # Available tools
        available_tools = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp']
        constraints = {}
        
        # Select action
        action = await q_engine.select_action(state, available_tools, constraints)
        print(f"Selected action: {action}")
        print(f"Exploration rate: {q_engine.exploration_rate:.3f}")
        
        # Simulate execution and reward
        success = np.random.random() > 0.3  # 70% success rate
        reward = 1.0 if success else -0.5
        
        # Update context for next state
        context['query_count'] += 1
        context['success_rate'] = (context['success_rate'] * episode + (1 if success else 0)) / (episode + 1)
        
        # Encode next state
        next_state = q_engine.state_encoder.encode_state(intent, context, list(action))
        
        # Learn from experience
        await q_engine.learn_from_experience(
            state, action, reward, next_state, 
            available_tools, constraints, done=True
        )
        
        print(f"Reward: {reward}")
        print(f"Q-table entries: {len(q_engine.q_table.q_values)}")
        
        # Decay exploration
        q_engine.decay_exploration()
    
    # Get final metrics
    metrics = q_engine.get_metrics()
    print(f"\n=== Final Metrics ===")
    print(f"Episodes: {metrics['episode_count']}")
    print(f"Total reward: {metrics['total_reward']:.2f}")
    print(f"Average reward: {metrics['avg_reward']:.2f}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Q-table size: {metrics['q_table_stats']['total_entries']}")
    print(f"Buffer size: {metrics['buffer_size']}")
    
    # Save model
    await q_engine.save_model("test_model_v1")
    print("\n✓ Model saved successfully!")
    
    # Test loading model
    q_engine2 = QLearningEngine(config)
    await q_engine2.db_manager.initialize()
    await q_engine2.load_model("test_model_v1")
    
    print(f"Loaded Q-table size: {len(q_engine2.q_table.q_values)}")
    print(f"Loaded exploration rate: {q_engine2.exploration_rate:.3f}")
    
    print("\n✓ Q-learning cycle test passed!")


async def main():
    """Run all tests."""
    print("Starting Q-Learning Engine Tests...")
    
    await test_state_representation()
    await test_action_space()
    await test_q_learning_cycle()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())