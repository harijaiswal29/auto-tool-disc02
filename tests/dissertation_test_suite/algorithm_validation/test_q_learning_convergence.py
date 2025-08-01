#!/usr/bin/env python3
"""Test Q-learning algorithm convergence and stability.

This test validates the core Q-learning implementation including:
- Value function convergence
- Policy stability
- Experience replay effectiveness
- Exploration-exploitation balance
"""

import pytest
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.learning.q_learning_engine import EnhancedQLearningEngine
from src.learning.state_representation import StateRepresentation
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.algorithm
@pytest.mark.asyncio
class TestQLearningConvergence:
    """Test suite for Q-learning algorithm validation."""
    
    @pytest.fixture
    async def q_learning_engine(self):
        """Create Q-learning engine with standard parameters."""
        config = {
            'state_dim': 447,
            'action_space_size': 10,
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.2,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'batch_size': 32,
            'memory_size': 10000,
            'update_frequency': 4
        }
        engine = EnhancedQLearningEngine(config)
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    async def test_bellman_equation_consistency(self, q_learning_engine):
        """Test that Q-value updates follow Bellman equation."""
        logger.info("Testing Bellman equation consistency")
        
        # Generate test experiences
        test_experiences = []
        for _ in range(100):
            state = np.random.randn(447)
            action = np.random.randint(0, 10)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.randn(447)
            done = np.random.random() < 0.1
            
            test_experiences.append((state, action, reward, next_state, done))
        
        # Track Bellman errors
        bellman_errors = []
        
        for state, action, reward, next_state, done in test_experiences:
            # Get current Q-value
            old_q = q_learning_engine.get_q_value(state, action)
            
            # Calculate target using Bellman equation
            if done:
                target = reward
            else:
                # Max Q-value for next state
                next_q_values = [
                    q_learning_engine.get_q_value(next_state, a) 
                    for a in range(10)
                ]
                max_next_q = max(next_q_values)
                target = reward + q_learning_engine.discount_factor * max_next_q
            
            # Update Q-value
            q_learning_engine.update_q_value(state, action, target)
            
            # Get new Q-value
            new_q = q_learning_engine.get_q_value(state, action)
            
            # Calculate Bellman error
            expected_new_q = old_q + q_learning_engine.learning_rate * (target - old_q)
            bellman_error = abs(new_q - expected_new_q)
            bellman_errors.append(bellman_error)
        
        # Verify Bellman consistency
        avg_error = np.mean(bellman_errors)
        max_error = np.max(bellman_errors)
        
        logger.info(f"Average Bellman error: {avg_error:.6f}")
        logger.info(f"Max Bellman error: {max_error:.6f}")
        
        # Errors should be minimal (floating point precision)
        assert avg_error < 1e-5, f"Average Bellman error too high: {avg_error}"
        assert max_error < 1e-4, f"Max Bellman error too high: {max_error}"
        
        # Save results
        self._save_results({
            'test': 'bellman_consistency',
            'avg_bellman_error': avg_error,
            'max_bellman_error': max_error,
            'num_updates': len(test_experiences)
        })
    
    async def test_experience_replay_distribution(self, q_learning_engine):
        """Test that experience replay maintains proper distribution."""
        logger.info("Testing experience replay distribution")
        
        # Fill replay buffer
        experience_types = {
            'high_reward': 0,
            'low_reward': 0,
            'terminal': 0,
            'non_terminal': 0
        }
        
        total_experiences = 1000
        for i in range(total_experiences):
            state = np.random.randn(447)
            action = np.random.randint(0, 10)
            
            # Create diverse experiences
            if i % 4 == 0:  # High reward
                reward = np.random.uniform(0.8, 1.0)
                experience_types['high_reward'] += 1
            else:  # Low reward
                reward = np.random.uniform(-0.2, 0.2)
                experience_types['low_reward'] += 1
            
            next_state = np.random.randn(447)
            done = i % 10 == 0  # 10% terminal states
            
            if done:
                experience_types['terminal'] += 1
            else:
                experience_types['non_terminal'] += 1
            
            q_learning_engine.remember(state, action, reward, next_state, done)
        
        # Sample from replay buffer multiple times
        sample_stats = {
            'high_reward': 0,
            'low_reward': 0,
            'terminal': 0,
            'non_terminal': 0
        }
        
        num_samples = 50
        batch_size = 32
        
        for _ in range(num_samples):
            batch = q_learning_engine.sample_batch(batch_size)
            
            for _, _, reward, _, done in batch:
                if reward > 0.5:
                    sample_stats['high_reward'] += 1
                else:
                    sample_stats['low_reward'] += 1
                
                if done:
                    sample_stats['terminal'] += 1
                else:
                    sample_stats['non_terminal'] += 1
        
        # Calculate sampling ratios
        total_sampled = num_samples * batch_size
        sampling_ratios = {
            k: sample_stats[k] / total_sampled 
            for k in sample_stats
        }
        
        expected_ratios = {
            k: experience_types[k] / total_experiences 
            for k in experience_types
        }
        
        # Verify uniform sampling (within statistical bounds)
        for exp_type in sampling_ratios:
            ratio_diff = abs(sampling_ratios[exp_type] - expected_ratios[exp_type])
            logger.info(f"{exp_type}: expected {expected_ratios[exp_type]:.3f}, "
                       f"got {sampling_ratios[exp_type]:.3f}")
            
            # Allow 5% deviation due to random sampling
            assert ratio_diff < 0.05, \
                f"{exp_type} sampling ratio off by {ratio_diff:.3f}"
        
        # Save results
        self._save_results({
            'test': 'experience_replay_distribution',
            'experience_types': experience_types,
            'sampling_ratios': sampling_ratios,
            'expected_ratios': expected_ratios
        })
    
    async def test_policy_improvement(self, q_learning_engine):
        """Test that policy improves over training."""
        logger.info("Testing policy improvement")
        
        # Define a simple MDP for testing
        class SimpleMDP:
            """Simple MDP with known optimal policy."""
            def __init__(self):
                self.states = [np.random.randn(447) for _ in range(5)]
                self.optimal_actions = [0, 1, 2, 1, 0]  # Known optimal policy
            
            def get_reward(self, state_idx, action):
                """Reward function favoring optimal actions."""
                if action == self.optimal_actions[state_idx]:
                    return 1.0
                else:
                    return -0.1
            
            def get_next_state(self, state_idx, action):
                """Deterministic transitions."""
                if action == self.optimal_actions[state_idx]:
                    return (state_idx + 1) % 5, state_idx == 4
                else:
                    return state_idx, False
        
        mdp = SimpleMDP()
        
        # Track policy accuracy over time
        policy_accuracy_history = []
        
        # Train for multiple episodes
        episodes = 500
        for episode in range(episodes):
            # Generate experience from MDP
            state_idx = 0
            
            for step in range(20):  # Max steps per episode
                state = mdp.states[state_idx]
                
                # Choose action (epsilon-greedy)
                if np.random.random() < q_learning_engine.epsilon:
                    action = np.random.randint(0, 10)
                else:
                    action = q_learning_engine.act(state)
                
                # Get reward and next state
                reward = mdp.get_reward(state_idx, action)
                next_state_idx, done = mdp.get_next_state(state_idx, action)
                next_state = mdp.states[next_state_idx]
                
                # Store experience
                q_learning_engine.remember(state, action, reward, next_state, done)
                
                # Learn from experience
                if len(q_learning_engine.memory) > q_learning_engine.batch_size:
                    await q_learning_engine.replay(q_learning_engine.batch_size)
                
                if done:
                    break
                state_idx = next_state_idx
            
            # Evaluate policy every 10 episodes
            if episode % 10 == 0:
                correct_actions = 0
                for i, state in enumerate(mdp.states):
                    predicted_action = q_learning_engine.act(state)
                    if predicted_action == mdp.optimal_actions[i]:
                        correct_actions += 1
                
                accuracy = correct_actions / len(mdp.states)
                policy_accuracy_history.append(accuracy)
                logger.info(f"Episode {episode}: Policy accuracy {accuracy:.1%}")
        
        # Verify policy improvement
        initial_accuracy = np.mean(policy_accuracy_history[:5])
        final_accuracy = np.mean(policy_accuracy_history[-5:])
        
        logger.info(f"Initial accuracy: {initial_accuracy:.1%}")
        logger.info(f"Final accuracy: {final_accuracy:.1%}")
        
        # Should show significant improvement
        improvement = final_accuracy - initial_accuracy
        assert improvement > 0.3, f"Insufficient improvement: {improvement:.1%}"
        assert final_accuracy > 0.8, f"Final accuracy too low: {final_accuracy:.1%}"
        
        # Save results
        self._save_results({
            'test': 'policy_improvement',
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'accuracy_history': policy_accuracy_history
        })
    
    async def test_memory_efficiency(self, q_learning_engine):
        """Test memory usage and efficiency of experience replay."""
        logger.info("Testing memory efficiency")
        
        # Track memory usage
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Fill memory to capacity
        memory_size = q_learning_engine.memory_size
        
        for i in range(memory_size + 1000):  # Overfill to test circular buffer
            state = np.random.randn(447)
            action = np.random.randint(0, 10)
            reward = np.random.random()
            next_state = np.random.randn(447)
            done = np.random.random() < 0.1
            
            q_learning_engine.remember(state, action, reward, next_state, done)
        
        # Check memory after filling
        gc.collect()
        filled_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = filled_memory - baseline_memory
        
        # Verify buffer size is maintained
        assert len(q_learning_engine.memory) == memory_size, \
            f"Memory size {len(q_learning_engine.memory)} != {memory_size}"
        
        # Calculate memory per experience
        memory_per_experience = memory_increase / memory_size * 1024  # KB
        
        logger.info(f"Memory increase: {memory_increase:.1f} MB")
        logger.info(f"Memory per experience: {memory_per_experience:.2f} KB")
        
        # Memory usage should be reasonable
        assert memory_per_experience < 5.0, \
            f"Memory per experience too high: {memory_per_experience:.2f} KB"
        
        # Test sampling efficiency
        import time
        
        sample_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            batch = q_learning_engine.sample_batch(32)
            sample_time = (time.perf_counter() - start_time) * 1000
            sample_times.append(sample_time)
        
        avg_sample_time = np.mean(sample_times)
        logger.info(f"Average batch sampling time: {avg_sample_time:.2f} ms")
        
        # Sampling should be fast
        assert avg_sample_time < 1.0, \
            f"Sampling too slow: {avg_sample_time:.2f} ms"
        
        # Save results
        self._save_results({
            'test': 'memory_efficiency',
            'memory_size': memory_size,
            'memory_increase_mb': memory_increase,
            'memory_per_experience_kb': memory_per_experience,
            'avg_sample_time_ms': avg_sample_time
        })
    
    async def test_convergence_metrics(self, q_learning_engine):
        """Test comprehensive convergence metrics."""
        logger.info("Testing convergence metrics")
        
        # Initialize metrics tracking
        metrics = {
            'q_value_changes': [],
            'policy_changes': [],
            'td_errors': [],
            'episode_rewards': []
        }
        
        # Previous Q-values for change tracking
        prev_q_values = {}
        prev_policy = {}
        
        # Sample states for consistent monitoring
        monitor_states = [np.random.randn(447) for _ in range(20)]
        
        # Training loop
        episodes = 200
        for episode in range(episodes):
            episode_reward = 0
            episode_td_errors = []
            
            # Run episode
            for step in range(50):
                state = np.random.randn(447)
                action = q_learning_engine.act(state)
                reward = np.random.uniform(-1, 1)
                next_state = np.random.randn(447)
                done = step == 49
                
                episode_reward += reward
                
                # Calculate TD error
                current_q = q_learning_engine.get_q_value(state, action)
                if done:
                    target = reward
                else:
                    next_q_values = [
                        q_learning_engine.get_q_value(next_state, a) 
                        for a in range(10)
                    ]
                    target = reward + q_learning_engine.discount_factor * max(next_q_values)
                
                td_error = abs(target - current_q)
                episode_td_errors.append(td_error)
                
                # Store and learn
                q_learning_engine.remember(state, action, reward, next_state, done)
                if len(q_learning_engine.memory) > q_learning_engine.batch_size:
                    await q_learning_engine.replay(q_learning_engine.batch_size)
            
            # Track metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['td_errors'].append(np.mean(episode_td_errors))
            
            # Track Q-value changes
            if episode % 5 == 0:
                q_value_change = 0
                policy_change = 0
                
                for i, state in enumerate(monitor_states):
                    # Q-value changes
                    state_key = f"state_{i}"
                    current_q_values = [
                        q_learning_engine.get_q_value(state, a) 
                        for a in range(10)
                    ]
                    
                    if state_key in prev_q_values:
                        q_change = np.mean(np.abs(
                            np.array(current_q_values) - np.array(prev_q_values[state_key])
                        ))
                        q_value_change += q_change
                    
                    prev_q_values[state_key] = current_q_values
                    
                    # Policy changes
                    current_action = np.argmax(current_q_values)
                    if state_key in prev_policy:
                        if current_action != prev_policy[state_key]:
                            policy_change += 1
                    prev_policy[state_key] = current_action
                
                metrics['q_value_changes'].append(q_value_change / len(monitor_states))
                metrics['policy_changes'].append(policy_change)
        
        # Analyze convergence
        # Q-values should stabilize
        late_q_changes = metrics['q_value_changes'][-10:]
        q_value_converged = np.mean(late_q_changes) < 0.01
        
        # Policy should stabilize
        late_policy_changes = metrics['policy_changes'][-10:]
        policy_converged = sum(late_policy_changes) < 5
        
        # TD errors should decrease
        early_td_errors = np.mean(metrics['td_errors'][:20])
        late_td_errors = np.mean(metrics['td_errors'][-20:])
        td_improvement = (early_td_errors - late_td_errors) / early_td_errors
        
        logger.info(f"Q-value converged: {q_value_converged}")
        logger.info(f"Policy converged: {policy_converged}")
        logger.info(f"TD error improvement: {td_improvement:.1%}")
        
        assert q_value_converged, "Q-values did not converge"
        assert policy_converged, "Policy did not converge"
        assert td_improvement > 0.5, f"TD errors did not improve enough: {td_improvement:.1%}"
        
        # Save results and generate plots
        self._save_results({
            'test': 'convergence_metrics',
            'q_value_converged': q_value_converged,
            'policy_converged': policy_converged,
            'td_improvement': td_improvement,
            'final_q_change': np.mean(late_q_changes),
            'final_policy_changes': sum(late_policy_changes)
        })
        
        self._plot_convergence_metrics(metrics)
    
    def _plot_convergence_metrics(self, metrics: Dict):
        """Generate convergence visualization plots."""
        output_dir = Path(__file__).parent.parent / "results" / "learning_curves"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0, 0].plot(metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # TD Errors
        axes[0, 1].plot(metrics['td_errors'])
        axes[0, 1].set_title('Mean TD Errors')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('TD Error')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-value changes
        if metrics['q_value_changes']:
            episodes = list(range(0, len(metrics['q_value_changes']) * 5, 5))
            axes[1, 0].plot(episodes, metrics['q_value_changes'])
            axes[1, 0].set_title('Q-Value Changes')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Mean |ΔQ|')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Policy changes
        if metrics['policy_changes']:
            episodes = list(range(0, len(metrics['policy_changes']) * 5, 5))
            axes[1, 1].plot(episodes, metrics['policy_changes'])
            axes[1, 1].set_title('Policy Changes')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Number of Changes')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Q-Learning Convergence Metrics')
        plt.tight_layout()
        plt.savefig(output_dir / 'algorithm_convergence_metrics.png', dpi=300)
        plt.close()
        
        logger.info(f"Convergence plots saved to {output_dir}")
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "algorithm_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"q_learning_{results.get('test', 'general')}.json"
        
        # Convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        results = convert_numpy(results)
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / filename}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])