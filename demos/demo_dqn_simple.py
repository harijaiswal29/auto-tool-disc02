#!/usr/bin/env python3
"""Simplified Demo of Deep Q-Learning with Neural Networks

This script provides a clean demonstration of the DQN implementation
with minimal logging and focused output.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime

# Suppress excessive logging
logging.getLogger('src.learning.reward_calculator').setLevel(logging.WARNING)
logging.getLogger('src.learning.q_learning_engine').setLevel(logging.WARNING)
logging.getLogger('PatternMiner').setLevel(logging.WARNING)

from src.learning.q_learning_engine import QLearningEngine
from src.learning.dqn_trainer import DQNTrainer, create_training_config
from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics
from src.agents.intent_models import Intent
from src.utils.logger import get_logger

# Setup logging
logger = get_logger("DQN_Simple_Demo")


class SimpleToolEnvironment:
    """Simplified simulated environment for demonstration."""
    
    def __init__(self):
        self.available_tools = [
            'filesystem_mcp', 'sqlite_mcp', 'search_mcp',
            'postgres_mcp', 'github_mcp', 'weather_mcp'
        ]
        self.query_types = ['search', 'retrieve', 'analyze']
        self.state_encoder = None
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.query_type = np.random.choice(self.query_types)
        self.optimal_tools = self._get_optimal_tools(self.query_type)
        self.current_state = self._generate_simple_state()
        self.constraints = {'conflicts': {}, 'requires': {}}
        return self.current_state, self.available_tools.copy(), self.constraints
    
    def _generate_simple_state(self) -> np.ndarray:
        """Generate a simplified state vector."""
        # Create a simple state vector (439 dimensions as expected)
        state = np.zeros(439)
        
        # Set some features based on query type
        if self.query_type == 'search':
            state[0:10] = np.random.randn(10) * 0.5 + 1.0
        elif self.query_type == 'retrieve':
            state[10:20] = np.random.randn(10) * 0.5 + 1.0
        else:  # analyze
            state[20:30] = np.random.randn(10) * 0.5 + 1.0
        
        # Add some random noise to other features
        state[30:] = np.random.randn(409) * 0.1
        
        return state
    
    def _get_optimal_tools(self, query_type: str) -> List[str]:
        """Get optimal tools for query type."""
        optimal_map = {
            'search': ['search_mcp', 'filesystem_mcp'],
            'retrieve': ['sqlite_mcp', 'postgres_mcp'],
            'analyze': ['github_mcp', 'filesystem_mcp']
        }
        return optimal_map.get(query_type, ['filesystem_mcp'])
    
    async def step(self, action: List[str]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        # Calculate simple reward based on tool selection
        selected_set = set(action)
        optimal_set = set(self.optimal_tools)
        
        # Reward calculation
        correct_tools = len(selected_set & optimal_set)
        incorrect_tools = len(selected_set - optimal_set)
        
        reward = correct_tools * 2.0 - incorrect_tools * 1.0
        
        # Episode is done after one step for simplicity
        done = True
        
        # Generate next state
        next_state = self._generate_simple_state()
        
        info = {
            'tools': self.available_tools,
            'constraints': self.constraints,
            'success': correct_tools > 0
        }
        
        return next_state, reward, done, info


async def demo_dqn():
    """Demonstrate DQN functionality."""
    print("\n" + "="*60)
    print("Deep Q-Learning Demonstration")
    print("="*60 + "\n")
    
    # Create configuration
    config = {
        'q_learning': {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 0.3,
            'exploration_decay': 0.99,
            'min_exploration_rate': 0.01,
            'max_tools': 3,
            'buffer_capacity': 1000,
            'batch_size': 32,
            'update_frequency': 4,
            'enable_learning': True
        },
        'dqn': {
            'enabled': True,
            'batch_size': 16,
            'memory_size': 1000,
            'target_update_frequency': 50,
            'device': 'cpu',
            'network_type': 'standard',
            'network_architecture': [256, 128],
            'learning_rate': 0.0001,
            'double_dqn': True,
            'prioritized_replay': True
        }
    }
    
    print("Configuration:")
    print(f"- Network: DQN with architecture [439 -> 256 -> 128 -> output]")
    print(f"- Learning rate: {config['dqn']['learning_rate']}")
    print(f"- Batch size: {config['dqn']['batch_size']}")
    print(f"- Device: CPU")
    print()
    
    # Create engine and environment
    engine = QLearningEngine(config)
    env = SimpleToolEnvironment()
    
    # Training parameters
    num_episodes = 30
    rewards = []
    successes = []
    
    print("Training DQN for {} episodes...".format(num_episodes))
    print("-" * 40)
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset environment
        state, tools, constraints = env.reset()
        
        # Select action using DQN
        action = await engine.select_action(state, tools, constraints)
        
        # Execute action
        next_state, reward, done, info = await env.step(action)
        
        # Learn from experience
        await engine.learn_from_experience(
            state, action, reward, next_state,
            info['tools'], info['constraints'], done
        )
        
        rewards.append(reward)
        successes.append(info['success'])
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            success_rate = np.mean(successes[-10:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Success Rate = {success_rate:.2%}, "
                  f"Epsilon = {engine.exploration_rate:.3f}")
    
    training_time = time.time() - start_time
    
    print("-" * 40)
    print(f"Training completed in {training_time:.2f} seconds")
    print()
    
    # Show final metrics
    metrics = engine.get_metrics()
    print("Final Metrics:")
    print(f"- Total episodes: {num_episodes}")
    print(f"- Average reward: {np.mean(rewards):.2f}")
    print(f"- Final reward: {rewards[-1]:.2f}")
    print(f"- Success rate: {np.mean(successes):.2%}")
    print(f"- Final epsilon: {engine.exploration_rate:.3f}")
    if hasattr(engine, 'dqn_agent') and engine.dqn_agent:
        print(f"- Replay buffer size: {len(engine.dqn_agent.memory)}")
    print()
    
    # Test learned policy
    print("Testing Learned Policy:")
    print("-" * 40)
    
    test_queries = ['search', 'retrieve', 'analyze']
    for query_type in test_queries:
        env.query_type = query_type
        env.optimal_tools = env._get_optimal_tools(query_type)
        state = env._generate_simple_state()
        
        # Get action without exploration (exploitation only)
        engine.exploration_rate = 0.0
        action = await engine.select_action(state, env.available_tools, env.constraints)
        
        print(f"Query Type: {query_type}")
        print(f"  Optimal Tools: {env.optimal_tools}")
        print(f"  Selected Tools: {action}")
        print(f"  Match: {'✓' if set(action) & set(env.optimal_tools) else '✗'}")
        print()
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    
    # Calculate moving average
    window = 5
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Rewards')
    plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Learning Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    success_moving_avg = np.convolve(successes, np.ones(window)/window, mode='valid')
    plt.plot(successes, 'o', alpha=0.3, label='Episode Success')
    plt.plot(range(window-1, len(successes)), success_moving_avg, label=f'{window}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_demo_results.png', dpi=150)
    print(f"Learning curves saved to 'dqn_demo_results.png'")
    
    # Show DQN architecture info
    if hasattr(engine, 'dqn_agent') and engine.dqn_agent:
        print("\nDQN Architecture Details:")
        print(f"- Network Type: {config['dqn']['network_type']}")
        print(f"- Double DQN: {config['dqn'].get('double_dqn', True)}")
        print(f"- Prioritized Replay: {config['dqn'].get('prioritized_replay', True)}")
        print(f"- Target Update Frequency: {config['dqn']['target_update_frequency']}")
        
        # Count parameters
        total_params = sum(p.numel() for p in engine.dqn_agent.q_network.parameters())
        print(f"- Total Parameters: {total_params:,}")


async def main():
    """Main entry point."""
    try:
        await demo_dqn()
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)