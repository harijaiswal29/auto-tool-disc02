#!/usr/bin/env python3
"""Fixed version of DQN comparison demo with improved logging and configuration."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress excessive logging
import logging
logging.getLogger('src.learning.reward_calculator').setLevel(logging.WARNING)
logging.getLogger('PatternMiner').setLevel(logging.WARNING)
logging.getLogger('src.learning.q_learning_engine').setLevel(logging.INFO)

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import torch

from src.learning.q_learning_engine import QLearningEngine
from src.learning.dqn_trainer import DQNTrainer, create_training_config
from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics
from src.agents.intent_models import Intent
from src.utils.logger import get_logger

# Setup logging
logger = get_logger("DQN_Demo_Fixed")


class SimulatedToolEnvironment:
    """Simulated environment for tool discovery and execution."""
    
    def __init__(self):
        self.available_tools = [
            'filesystem_mcp', 'sqlite_mcp', 'search_mcp',
            'postgres_mcp', 'github_mcp', 'weather_mcp'
        ]
        self.tool_patterns = {
            'search': ['search_mcp', 'filesystem_mcp'],
            'retrieve': ['sqlite_mcp', 'postgres_mcp'],
            'analyze': ['github_mcp', 'filesystem_mcp'],
            'create': ['filesystem_mcp', 'github_mcp'],
            'monitor': ['postgres_mcp', 'weather_mcp']
        }
        self.query_types = list(self.tool_patterns.keys())
        # Create a simple reward configuration
        reward_config = {
            'base_weights': {
                'success': 1.0,
                'failure': -0.5,
                'partial_success': 0.3
            }
        }
        self.reward_calculator = RewardCalculator(reward_config)
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.query_type = np.random.choice(self.query_types)
        self.optimal_tools = self.tool_patterns[self.query_type]
        self.steps_taken = 0
        self.current_state = self._generate_state()
        self.constraints = {'conflicts': {}, 'requires': {}}
        
        return self.current_state, self.available_tools.copy(), self.constraints
    
    def _generate_state(self) -> np.ndarray:
        """Generate state vector based on query type."""
        # Create a mock intent
        intent = Intent(
            type=f"query.{self.query_type}",
            confidence=0.8 + np.random.rand() * 0.2,
            keywords=[self.query_type],
            entities=[self.query_type]
        )
        
        # Generate random embedding
        intent.embedding = np.random.randn(384)
        
        # Context features with guaranteed non-zero failure counts
        context = {
            'domain': 'simulation',
            'query_count': np.random.randint(1, 10),
            'session_duration': np.random.randint(60, 3600),
            'metrics': {
                'avg_response_time': np.random.randint(100, 1000),
                'success_rate': np.random.rand(),
                'error_rate': np.random.rand() * 0.2
            },
            'failure_rates': {tool: np.random.rand() * 0.3 for tool in self.available_tools},
            'failure_types': {
                'network_timeout': np.random.randint(1, 5),
                'permission_error': np.random.randint(0, 2),
                'rate_limit': np.random.randint(0, 3),
                'connection_error': np.random.randint(1, 4),
                'other': np.random.randint(1, 3)
            },
            'retry_patterns': {
                'avg_retry_count': np.random.rand() * 3,
                'retry_success_rate': np.random.rand()
            }
        }
        
        # Tool history
        history = np.random.choice(self.available_tools, size=np.random.randint(0, 5)).tolist()
        
        # Use StateRepresentation to encode
        from src.learning.q_learning_engine import StateRepresentation
        encoder = StateRepresentation()
        state = encoder.encode_state(intent, context, history)
        
        return state
    
    def _create_execution_metrics(self, selected_tools: List[str]) -> ExecutionMetrics:
        """Create realistic execution metrics."""
        # Calculate success based on tool selection
        selected_set = set(selected_tools)
        optimal_set = set(self.optimal_tools)
        
        correct_tools = len(selected_set & optimal_set)
        incorrect_tools = len(selected_set - optimal_set)
        
        success = correct_tools > 0 and incorrect_tools == 0
        
        # Simulate execution metrics
        # For multiple tools, we'll use the first one as the primary tool
        primary_tool = selected_tools[0] if selected_tools else 'unknown'
        
        metrics = ExecutionMetrics(
            tool_id=primary_tool,
            success=success,
            partial_success=correct_tools > 0,
            completion_percentage=min(100, correct_tools / len(optimal_set) * 100) if optimal_set else 0,
            execution_time_ms=np.random.randint(100, 2000),
            error_type=None if success else 'tool_mismatch',
            retry_count=0,
            resource_usage={
                'memory_mb': np.random.randint(50, 500),
                'cpu_percent': np.random.randint(10, 80),
                'api_calls': len(selected_tools) * np.random.randint(1, 5)
            },
            result_quality=1.0 if success else 0.5
        )
        
        # Add failure info if not successful
        if not success:
            metrics.failure_type = np.random.choice([
                'tool_mismatch', 'execution_error', 'timeout'
            ])
        
        return metrics
    
    async def step(self, action: List[str]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        self.steps_taken += 1
        
        # Create execution metrics
        metrics = self._create_execution_metrics(action)
        
        # Calculate reward using reward calculator
        reward, _ = self.reward_calculator.calculate_reward(
            execution_results=[metrics],  # Pass as list
            context={
                'mode': 'exploration',
                'confidence': 0.8,
                'user_initiated': True
            }
        )
        
        # Episode ends after one action or max steps
        done = True  # Single step episodes for faster demo
        
        # Generate next state
        next_state = self._generate_state()
        
        info = {
            'tools': self.available_tools,
            'constraints': self.constraints,
            'metrics': metrics
        }
        
        return next_state, reward, done, info


async def train_approach(config: Dict[str, Any], num_episodes: int, use_dqn: bool = False) -> Dict[str, List]:
    """Train either tabular Q-learning or DQN."""
    approach_name = "DQN" if use_dqn else "Tabular Q-Learning"
    logger.info(f"Starting {approach_name} training...")
    
    # Configure for specific approach
    if use_dqn:
        # Force CPU usage to avoid CUDA issues
        device = 'cpu'
        config = create_training_config(config, {
            'dqn.enabled': True,
            'dqn.batch_size': 32,
            'dqn.memory_size': 10000,
            'dqn.target_update_frequency': 100,
            'dqn.device': device
        })
    else:
        config['dqn'] = {'enabled': False}
    
    # Create engine and environment
    engine = QLearningEngine(config)
    env = SimulatedToolEnvironment()
    
    # Training metrics
    episode_rewards = []
    episode_successes = []
    
    # Progress bar setup
    from tqdm import tqdm
    pbar = tqdm(total=num_episodes, desc=f"Training {approach_name}")
    
    for episode in range(num_episodes):
        # Reset environment
        state, tools, constraints = env.reset()
        
        # Select action
        action = await engine.select_action(state, tools, constraints)
        
        # Execute action
        next_state, reward, done, info = await env.step(action)
        
        # Learn from experience
        await engine.learn_from_experience(
            state, action, reward, next_state,
            info['tools'], info['constraints'], done
        )
        
        # Record metrics
        episode_rewards.append(reward)
        episode_successes.append(info['metrics'].success)
        
        # Update progress bar
        if episode > 0:
            avg_reward = np.mean(episode_rewards[-10:])
            success_rate = np.mean(episode_successes[-10:])
            pbar.set_postfix({
                'avg_reward': f"{avg_reward:.2f}",
                'success_rate': f"{success_rate:.2%}",
                'epsilon': f"{engine.exploration_rate:.3f}"
            })
        pbar.update(1)
    
    pbar.close()
    
    # Save model
    model_name = f"{approach_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await engine.save_model(model_name)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
        'final_metrics': engine.get_metrics()
    }


async def compare_approaches(num_episodes: int = 50):
    """Compare tabular Q-learning with DQN."""
    # Load base configuration
    config_path = 'config/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Use default configuration
        config = {
            'q_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.2,
                'exploration_decay': 0.995,
                'min_exploration_rate': 0.01,
                'max_tools': 3,
                'buffer_capacity': 10000,
                'batch_size': 32,
                'update_frequency': 4,
                'enable_learning': True
            }
        }
    
    print("\n" + "="*60)
    print("Comparing Tabular Q-Learning vs Deep Q-Network")
    print(f"Training episodes: {num_episodes}")
    print("="*60 + "\n")
    
    # Train tabular Q-learning
    print("\n[1/2] Training Tabular Q-Learning...")
    start_time = time.time()
    tabular_results = await train_approach(config.copy(), num_episodes, use_dqn=False)
    tabular_time = time.time() - start_time
    
    # Train DQN
    print("\n[2/2] Training Deep Q-Network...")
    start_time = time.time()
    dqn_results = await train_approach(config.copy(), num_episodes, use_dqn=True)
    dqn_time = time.time() - start_time
    
    # Print comparison results
    print("\n" + "="*60)
    print("Results Comparison")
    print("="*60)
    
    print("\nTabular Q-Learning:")
    print(f"  Training time: {tabular_time:.2f} seconds")
    print(f"  Final avg reward: {np.mean(tabular_results['episode_rewards'][-10:]):.3f}")
    print(f"  Final success rate: {np.mean(tabular_results['episode_successes'][-10:]):.2%}")
    
    print("\nDeep Q-Network:")
    print(f"  Training time: {dqn_time:.2f} seconds")
    print(f"  Final avg reward: {np.mean(dqn_results['episode_rewards'][-10:]):.3f}")
    print(f"  Final success rate: {np.mean(dqn_results['episode_successes'][-10:]):.2%}")
    
    # Plot comparison
    plot_comparison(tabular_results, dqn_results, num_episodes)
    
    return tabular_results, dqn_results


def plot_comparison(tabular_results: Dict, dqn_results: Dict, num_episodes: int):
    """Plot comparison between approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate moving averages
    window = min(10, num_episodes // 10)
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot rewards
    ax1.plot(tabular_results['episode_rewards'], alpha=0.3, color='blue')
    ax1.plot(range(window-1, len(tabular_results['episode_rewards'])), 
             moving_average(tabular_results['episode_rewards'], window),
             label='Tabular Q-Learning', color='blue', linewidth=2)
    
    ax1.plot(dqn_results['episode_rewards'], alpha=0.3, color='red')
    ax1.plot(range(window-1, len(dqn_results['episode_rewards'])),
             moving_average(dqn_results['episode_rewards'], window),
             label='DQN', color='red', linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot success rates
    ax2.plot(tabular_results['episode_successes'], alpha=0.3, color='blue')
    ax2.plot(range(window-1, len(tabular_results['episode_successes'])),
             moving_average(tabular_results['episode_successes'], window),
             label='Tabular Q-Learning', color='blue', linewidth=2)
    
    ax2.plot(dqn_results['episode_successes'], alpha=0.3, color='red')
    ax2.plot(range(window-1, len(dqn_results['episode_successes'])),
             moving_average(dqn_results['episode_successes'], window),
             label='DQN', color='red', linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('dqn_comparison_results.png', dpi=150)
    print(f"\nComparison plot saved to 'dqn_comparison_results.png'")


async def main():
    """Main entry point."""
    logger.info("Starting DQN vs Tabular Q-Learning Comparison Demo")
    logger.info("This demo compares traditional tabular Q-learning with Deep Q-Network")
    logger.info("on a simulated tool discovery task.\n")
    
    try:
        # Run comparison with reasonable episode count
        await compare_approaches(num_episodes=50)
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)