#!/usr/bin/env python3
"""Demo script comparing tabular Q-learning with Deep Q-Network (DQN).

This script demonstrates:
1. Training with tabular Q-learning
2. Training with DQN
3. Comparing performance between the two approaches
4. Visualizing learning curves and action distributions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime

from src.learning.q_learning_engine import QLearningEngine
from src.learning.dqn_trainer import DQNTrainer, create_training_config
from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics
from src.agents.intent_models import Intent
from src.utils.logger import get_logger

# Setup logging
logger = get_logger("DQN_Demo")


class SimulatedToolEnvironment:
    """Simulated environment for tool discovery and execution."""
    
    def __init__(self):
        self.available_tools = [
            'filesystem_mcp', 'sqlite_mcp', 'search_mcp',
            'postgres_mcp', 'github_mcp', 'weather_mcp'
        ]
        
        # Define tool success probabilities and relationships
        self.tool_success_rates = {
            'filesystem_mcp': 0.9,
            'sqlite_mcp': 0.85,
            'search_mcp': 0.8,
            'postgres_mcp': 0.85,
            'github_mcp': 0.75,
            'weather_mcp': 0.95
        }
        
        # Define synergistic combinations
        self.synergies = {
            ('filesystem_mcp', 'sqlite_mcp'): 0.1,  # Bonus for using together
            ('github_mcp', 'filesystem_mcp'): 0.15,
            ('search_mcp', 'weather_mcp'): 0.05
        }
        
        # Constraints
        self.constraints = {
            'conflicts': {
                'sqlite_mcp': ['postgres_mcp'],  # Can't use both databases
                'postgres_mcp': ['sqlite_mcp']
            },
            'max_tools': 3
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.steps = 0
        self.query_type = np.random.choice([
            'file_search', 'database_query', 'code_analysis', 
            'weather_check', 'general_search'
        ])
        
        # Generate state based on query type
        self.current_state = self._generate_state()
        
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
        
        # Context features
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
    
    async def step(self, action: Tuple[str, ...]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        self.steps += 1
        
        # Calculate success based on tools selected
        base_success_prob = np.mean([self.tool_success_rates[tool] for tool in action])
        
        # Apply synergy bonuses
        synergy_bonus = 0
        for tool1 in action:
            for tool2 in action:
                if tool1 < tool2:  # Avoid duplicates
                    key = (tool1, tool2)
                    if key in self.synergies:
                        synergy_bonus += self.synergies[key]
        
        # Final success probability
        success_prob = min(base_success_prob + synergy_bonus, 0.98)
        success = np.random.rand() < success_prob
        
        # Create execution metrics
        metrics = ExecutionMetrics(
            tool_id=action[0] if len(action) == 1 else 'multiple',
            success=success,
            partial_success=not success and np.random.rand() < 0.3,
            completion_percentage=1.0 if success else np.random.rand(),
            execution_time_ms=np.random.randint(100, 2000),
            error_type='network_timeout' if not success and np.random.rand() < 0.5 else None,
            retry_count=0 if success else np.random.randint(0, 3),
            resource_usage={
                'memory_mb': np.random.randint(50, 500),
                'cpu_percent': np.random.randint(10, 80),
                'api_calls': len(action)
            }
        )
        
        # Calculate reward
        reward_calc = RewardCalculator({})
        reward, _ = reward_calc.calculate_reward([metrics], {}, None)
        
        # Bonus for query-appropriate tool selection
        if self.query_type == 'file_search' and 'filesystem_mcp' in action:
            reward += 0.2
        elif self.query_type == 'database_query' and any(db in action for db in ['sqlite_mcp', 'postgres_mcp']):
            reward += 0.2
        elif self.query_type == 'code_analysis' and 'github_mcp' in action:
            reward += 0.2
        elif self.query_type == 'weather_check' and 'weather_mcp' in action:
            reward += 0.3
        
        # Generate next state
        next_state = self._generate_state()
        
        # Episode ends after 10 steps or great success
        done = self.steps >= 10 or (success and reward > 1.5)
        
        info = {
            'available_tools': self.available_tools.copy(),
            'constraints': self.constraints,
            'success': success,
            'query_type': self.query_type
        }
        
        return next_state, reward, done, info


async def train_tabular_q_learning(config: Dict[str, Any], num_episodes: int = 100) -> Dict[str, List]:
    """Train using tabular Q-learning."""
    logger.info("Starting tabular Q-learning training...")
    
    # Ensure tabular mode
    config = config.copy()
    config['dqn'] = {'enabled': False}
    
    # Create engine
    engine = QLearningEngine(config)
    env = SimulatedToolEnvironment()
    
    # Training metrics
    episode_rewards = []
    success_rates = []
    
    for episode in range(num_episodes):
        # Reset environment
        state, available_tools, constraints = env.reset()
        episode_reward = 0
        successes = 0
        steps = 0
        
        while True:
            # Select action
            action = await engine.select_action(state, available_tools, constraints)
            
            # Step environment
            next_state, reward, done, info = await env.step(action)
            
            # Learn from experience
            await engine.learn_from_experience(
                state, action, reward, next_state,
                info['available_tools'], info['constraints'], done
            )
            
            # Update metrics
            episode_reward += reward
            if info['success']:
                successes += 1
            steps += 1
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Decay exploration
        engine.decay_exploration()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        success_rates.append(successes / steps if steps > 0 else 0)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_success = np.mean(success_rates[-10:])
            logger.info(f"Episode {episode + 1}: avg_reward={avg_reward:.2f}, "
                       f"avg_success={avg_success:.2f}, epsilon={engine.exploration_rate:.3f}")
    
    # Save model
    await engine.save_model(f"tabular_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    return {
        'episode_rewards': episode_rewards,
        'success_rates': success_rates,
        'final_metrics': engine.get_metrics()
    }


async def train_dqn(config: Dict[str, Any], num_episodes: int = 100) -> Dict[str, List]:
    """Train using Deep Q-Network."""
    logger.info("Starting DQN training...")
    
    # Enable DQN
    config = create_training_config(config, {
        'dqn.enabled': True,
        'dqn.batch_size': 32,
        'dqn.memory_size': 10000,
        'dqn.target_update_frequency': 100,
        'dqn.device': 'cpu'  # Force CPU usage
    })
    
    # Create engine
    engine = QLearningEngine(config)
    env = SimulatedToolEnvironment()
    
    # Create trainer
    trainer = DQNTrainer(engine, eval_interval=50, checkpoint_interval=500)
    trainer.setup_lr_scheduler('step', step_size=1000, gamma=0.9)
    
    # Custom environment step function
    current_env_state = None
    
    async def env_step_func(action_or_reset):
        nonlocal current_env_state
        
        if action_or_reset == 'reset':
            current_env_state = env.reset()
            return current_env_state
        else:
            return await env.step(action_or_reset)
    
    # Train
    await trainer.train(env_step_func, num_episodes, max_steps_per_episode=10)
    
    # Save final model
    await engine.save_model(f"dqn_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    return {
        'episode_rewards': trainer.training_metrics['episode_rewards'],
        'eval_rewards': trainer.training_metrics['eval_rewards'],
        'final_metrics': engine.get_metrics()
    }


async def compare_approaches():
    """Compare tabular Q-learning with DQN."""
    # Load base configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Number of episodes
    num_episodes = 200
    
    # Train tabular Q-learning
    logger.info("\n" + "="*50)
    logger.info("Training Tabular Q-Learning")
    logger.info("="*50)
    
    start_time = time.time()
    tabular_results = await train_tabular_q_learning(config, num_episodes)
    tabular_time = time.time() - start_time
    
    # Train DQN
    logger.info("\n" + "="*50)
    logger.info("Training Deep Q-Network")
    logger.info("="*50)
    
    start_time = time.time()
    dqn_results = await train_dqn(config, num_episodes)
    dqn_time = time.time() - start_time
    
    # Plot comparison
    plot_comparison(tabular_results, dqn_results, tabular_time, dqn_time)
    
    # Print final metrics
    logger.info("\n" + "="*50)
    logger.info("Final Metrics Comparison")
    logger.info("="*50)
    
    logger.info("\nTabular Q-Learning:")
    logger.info(f"  Training time: {tabular_time:.2f} seconds")
    logger.info(f"  Final avg reward: {np.mean(tabular_results['episode_rewards'][-20:]):.3f}")
    logger.info(f"  Final success rate: {np.mean(tabular_results['success_rates'][-20:]):.3f}")
    logger.info(f"  Q-table entries: {tabular_results['final_metrics']['q_table_stats']['total_entries']}")
    
    logger.info("\nDeep Q-Network:")
    logger.info(f"  Training time: {dqn_time:.2f} seconds")
    logger.info(f"  Final avg reward: {np.mean(dqn_results['episode_rewards'][-20:]):.3f}")
    if dqn_results['eval_rewards']:
        logger.info(f"  Best eval reward: {max(r['mean'] for r in dqn_results['eval_rewards']):.3f}")
    logger.info(f"  Network parameters: ~50K")  # Approximate for our architecture


def plot_comparison(tabular_results: Dict, dqn_results: Dict, 
                   tabular_time: float, dqn_time: float):
    """Plot comparison between tabular and DQN results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards comparison
    ax = axes[0, 0]
    
    # Tabular rewards
    tabular_rewards = tabular_results['episode_rewards']
    ax.plot(tabular_rewards, alpha=0.3, color='blue', label='Tabular (raw)')
    
    # Tabular moving average
    window = 20
    if len(tabular_rewards) >= window:
        ma = np.convolve(tabular_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(tabular_rewards)), ma, 
               color='blue', linewidth=2, label='Tabular (MA)')
    
    # DQN rewards
    dqn_rewards = dqn_results['episode_rewards']
    ax.plot(dqn_rewards, alpha=0.3, color='red', label='DQN (raw)')
    
    # DQN moving average
    if len(dqn_rewards) >= window:
        ma = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(dqn_rewards)), ma,
               color='red', linewidth=2, label='DQN (MA)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rate comparison (for tabular only in this demo)
    ax = axes[0, 1]
    if 'success_rates' in tabular_results:
        success_rates = tabular_results['success_rates']
        ax.plot(success_rates, alpha=0.3, color='green')
        
        if len(success_rates) >= window:
            ma = np.convolve(success_rates, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(success_rates)), ma,
                   color='green', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (Tabular Q-Learning)')
    ax.grid(True, alpha=0.3)
    
    # Training time comparison
    ax = axes[1, 0]
    methods = ['Tabular\nQ-Learning', 'Deep\nQ-Network']
    times = [tabular_time, dqn_time]
    colors = ['blue', 'red']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time:.1f}s', ha='center', va='bottom')
    
    # Final performance summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate final metrics
    tabular_final_reward = np.mean(tabular_results['episode_rewards'][-20:])
    dqn_final_reward = np.mean(dqn_results['episode_rewards'][-20:])
    
    summary_text = f"""Performance Summary:
    
Tabular Q-Learning:
  • Final Avg Reward: {tabular_final_reward:.3f}
  • Training Time: {tabular_time:.1f}s
  • Memory: Q-table with {tabular_results['final_metrics']['q_table_stats']['total_entries']} entries
  
Deep Q-Network:
  • Final Avg Reward: {dqn_final_reward:.3f}
  • Training Time: {dqn_time:.1f}s
  • Memory: Neural network (~50K parameters)
  
Advantage: {'DQN' if dqn_final_reward > tabular_final_reward else 'Tabular'} 
({abs(dqn_final_reward - tabular_final_reward):.3f} better)"""
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'dqn_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    logger.info(f"\nComparison plot saved as: dqn_comparison_{timestamp}.png")
    
    plt.show()


async def main():
    """Main entry point."""
    logger.info("Starting DQN vs Tabular Q-Learning Demo")
    logger.info("This demo compares the performance of traditional tabular Q-learning")
    logger.info("with Deep Q-Network (DQN) on a simulated tool discovery task.\n")
    
    await compare_approaches()
    
    logger.info("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())