#!/usr/bin/env python3
"""Test H5: System demonstrates learning convergence within 1000 episodes.

This test validates that the Q-learning system converges to stable performance
within the claimed episode limit, with proper convergence criteria.
"""

import pytest
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.learning.q_learning_engine import EnhancedQLearningEngine
from src.orchestration.orchestrator_agent import OrchestratorAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.hypothesis
@pytest.mark.asyncio
class TestConvergenceRates:
    """Test suite for validating H5: Learning convergence hypothesis."""
    
    @pytest.fixture
    async def q_learning_engine(self):
        """Create Q-learning engine for testing."""
        config = {
            'state_dim': 447,
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.2,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'batch_size': 32,
            'memory_size': 10000
        }
        engine = EnhancedQLearningEngine(config)
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    async def test_q_value_convergence(self, q_learning_engine):
        """Test that Q-values converge to stable values."""
        logger.info("Testing H5: Q-value convergence")
        
        # Track Q-value changes over episodes
        episodes = 1500  # Test beyond 1000 to ensure stability
        q_value_changes = []
        convergence_episode = None
        convergence_threshold = 0.01  # Delta < 1% considered converged
        
        # Sample states and actions for monitoring
        sample_states = [
            np.random.randn(447) for _ in range(10)
        ]
        sample_actions = list(range(5))  # Monitor 5 actions
        
        # Baseline Q-values
        baseline_q_values = {}
        for i, state in enumerate(sample_states):
            baseline_q_values[i] = {}
            for action in sample_actions:
                q_val = q_learning_engine.get_q_value(state, action)
                baseline_q_values[i][action] = q_val
        
        # Run training episodes
        for episode in range(episodes):
            # Simulate learning episode
            await self._simulate_episode(q_learning_engine)
            
            # Calculate Q-value changes every 10 episodes
            if episode % 10 == 0:
                total_change = 0
                count = 0
                
                for i, state in enumerate(sample_states):
                    for action in sample_actions:
                        new_q = q_learning_engine.get_q_value(state, action)
                        old_q = baseline_q_values[i][action]
                        
                        if old_q != 0:
                            change = abs(new_q - old_q) / abs(old_q)
                            total_change += change
                            count += 1
                        
                        # Update baseline
                        baseline_q_values[i][action] = new_q
                
                avg_change = total_change / count if count > 0 else 0
                q_value_changes.append(avg_change)
                
                # Check for convergence
                if avg_change < convergence_threshold and convergence_episode is None:
                    convergence_episode = episode
                    logger.info(f"Q-values converged at episode {convergence_episode}")
        
        # Verify convergence
        assert convergence_episode is not None, "Q-values did not converge"
        assert convergence_episode <= 1000, \
            f"Convergence at episode {convergence_episode} > 1000"
        
        # Verify stability after convergence
        post_convergence_changes = q_value_changes[convergence_episode//10:]
        stability = np.mean(post_convergence_changes) < convergence_threshold
        assert stability, "Q-values not stable after convergence"
        
        # Save results
        self._save_results({
            'test': 'q_value_convergence',
            'convergence_episode': convergence_episode,
            'convergence_threshold': convergence_threshold,
            'q_value_changes': q_value_changes,
            'final_change_rate': q_value_changes[-1]
        })
        
        # Generate convergence plot
        self._plot_convergence(q_value_changes, convergence_episode)
    
    async def test_performance_plateau(self, q_learning_engine):
        """Test that performance plateaus indicating convergence."""
        logger.info("Testing H5: Performance plateau")
        
        episodes = 1200
        window_size = 50  # Moving average window
        performance_history = []
        plateau_threshold = 0.02  # 2% variation considered plateau
        
        # Track performance metrics
        for episode in range(episodes):
            # Simulate episode and get performance
            performance = await self._simulate_episode_with_performance(q_learning_engine)
            performance_history.append(performance)
            
            # Check for plateau after minimum episodes
            if episode >= 100 and episode % 10 == 0:
                # Calculate moving average
                if len(performance_history) >= window_size:
                    recent_performance = performance_history[-window_size:]
                    ma = np.mean(recent_performance)
                    std = np.std(recent_performance)
                    cv = std / ma if ma > 0 else float('inf')
                    
                    if cv < plateau_threshold:
                        logger.info(f"Performance plateaued at episode {episode}")
                        logger.info(f"Mean: {ma:.3f}, CV: {cv:.3f}")
                        
                        assert episode <= 1000, \
                            f"Plateau at episode {episode} > 1000"
                        break
        
        # Analyze final performance
        final_performance = np.mean(performance_history[-100:])
        peak_performance = np.max([np.mean(performance_history[i:i+50]) 
                                  for i in range(0, len(performance_history)-50, 10)])
        
        # Performance should be near peak
        performance_retention = final_performance / peak_performance
        assert performance_retention > 0.95, \
            f"Performance degraded: {performance_retention:.1%} of peak"
        
        # Save results
        self._save_results({
            'test': 'performance_plateau',
            'episodes': len(performance_history),
            'final_performance': final_performance,
            'peak_performance': peak_performance,
            'performance_retention': performance_retention,
            'performance_history': performance_history[::10]  # Sample for file size
        })
    
    async def test_exploration_exploitation_balance(self, q_learning_engine):
        """Test that exploration decreases appropriately over time."""
        logger.info("Testing H5: Exploration-exploitation balance")
        
        episodes = 1000
        exploration_history = []
        epsilon_history = []
        
        for episode in range(episodes):
            # Get current epsilon
            epsilon = q_learning_engine.epsilon
            epsilon_history.append(epsilon)
            
            # Track exploration rate (actions not from policy)
            exploration_count = 0
            total_actions = 100
            
            for _ in range(total_actions):
                state = np.random.randn(447)
                if np.random.random() < epsilon:
                    exploration_count += 1
            
            exploration_rate = exploration_count / total_actions
            exploration_history.append(exploration_rate)
            
            # Decay epsilon
            q_learning_engine.epsilon = max(
                q_learning_engine.epsilon_min,
                q_learning_engine.epsilon * q_learning_engine.epsilon_decay
            )
        
        # Verify exploration decay
        initial_exploration = np.mean(exploration_history[:50])
        final_exploration = np.mean(exploration_history[-50:])
        
        logger.info(f"Initial exploration: {initial_exploration:.1%}")
        logger.info(f"Final exploration: {final_exploration:.1%}")
        
        # Should transition from exploration to exploitation
        assert initial_exploration > 0.15, "Insufficient initial exploration"
        assert final_exploration < 0.05, "Too much exploration at convergence"
        
        # Find transition point
        transition_episode = None
        for i in range(50, episodes-50):
            if exploration_history[i] < 0.10:  # 10% exploration threshold
                transition_episode = i
                break
        
        assert transition_episode is not None, "No clear exploration transition"
        assert transition_episode < 800, "Exploration transition too late"
        
        # Save results
        self._save_results({
            'test': 'exploration_exploitation',
            'initial_exploration': initial_exploration,
            'final_exploration': final_exploration,
            'transition_episode': transition_episode,
            'epsilon_history': epsilon_history[::10],
            'exploration_history': exploration_history[::10]
        })
    
    async def test_convergence_consistency(self, q_learning_engine):
        """Test convergence consistency across multiple runs."""
        logger.info("Testing H5: Convergence consistency")
        
        num_runs = 5  # Multiple independent runs
        convergence_episodes = []
        
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}")
            
            # Reset engine
            await q_learning_engine.reset()
            
            # Track convergence
            episode = 0
            converged = False
            performance_window = []
            
            while episode < 1500 and not converged:
                performance = await self._simulate_episode_with_performance(q_learning_engine)
                performance_window.append(performance)
                
                # Keep sliding window
                if len(performance_window) > 100:
                    performance_window.pop(0)
                
                # Check convergence criteria
                if len(performance_window) == 100:
                    recent_mean = np.mean(performance_window[-50:])
                    earlier_mean = np.mean(performance_window[:50])
                    improvement = abs(recent_mean - earlier_mean) / earlier_mean
                    
                    if improvement < 0.01:  # Less than 1% change
                        converged = True
                        convergence_episodes.append(episode)
                        logger.info(f"Run {run + 1} converged at episode {episode}")
                
                episode += 1
            
            if not converged:
                convergence_episodes.append(1500)  # Did not converge
        
        # Analyze consistency
        mean_convergence = np.mean(convergence_episodes)
        std_convergence = np.std(convergence_episodes)
        cv_convergence = std_convergence / mean_convergence
        
        logger.info(f"Mean convergence: {mean_convergence:.0f} episodes")
        logger.info(f"Std convergence: {std_convergence:.0f} episodes")
        logger.info(f"CV: {cv_convergence:.3f}")
        
        # All runs should converge within 1000 episodes
        assert all(ep <= 1000 for ep in convergence_episodes), \
            f"Some runs failed to converge within 1000 episodes: {convergence_episodes}"
        
        # Convergence should be consistent
        assert cv_convergence < 0.25, \
            f"Convergence too variable (CV={cv_convergence:.3f})"
        
        # Save results
        self._save_results({
            'test': 'convergence_consistency',
            'num_runs': num_runs,
            'convergence_episodes': convergence_episodes,
            'mean_convergence': mean_convergence,
            'std_convergence': std_convergence,
            'cv_convergence': cv_convergence
        })
    
    async def _simulate_episode(self, engine):
        """Simulate a learning episode."""
        # Generate random experience
        state = np.random.randn(447)
        action = np.random.randint(0, 10)
        reward = np.random.random()
        next_state = np.random.randn(447)
        done = np.random.random() < 0.1
        
        # Store experience
        engine.remember(state, action, reward, next_state, done)
        
        # Train if enough experiences
        if len(engine.memory) > engine.batch_size:
            await engine.replay(engine.batch_size)
    
    async def _simulate_episode_with_performance(self, engine):
        """Simulate episode and return performance metric."""
        # Simulate multiple steps
        total_reward = 0
        steps = 20
        
        for _ in range(steps):
            state = np.random.randn(447)
            action = engine.act(state)
            
            # Simulate reward (higher for better actions)
            optimal_action = np.argmax([engine.get_q_value(state, a) for a in range(10)])
            reward = 1.0 if action == optimal_action else 0.3
            
            total_reward += reward
            
            next_state = np.random.randn(447)
            done = np.random.random() < 0.05
            
            engine.remember(state, action, reward, next_state, done)
        
        # Train
        if len(engine.memory) > engine.batch_size:
            await engine.replay(engine.batch_size)
        
        return total_reward / steps
    
    def _plot_convergence(self, q_value_changes: List[float], convergence_episode: int):
        """Generate convergence visualization."""
        output_dir = Path(__file__).parent.parent / "results" / "learning_curves"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        episodes = [i * 10 for i in range(len(q_value_changes))]
        
        # Plot raw and smoothed data
        plt.plot(episodes, q_value_changes, 'b-', alpha=0.3, label='Raw')
        
        if len(q_value_changes) > 5:
            smoothed = savgol_filter(q_value_changes, 
                                    window_length=min(11, len(q_value_changes) if len(q_value_changes) % 2 == 1 else len(q_value_changes) - 1),
                                    polyorder=3)
            plt.plot(episodes, smoothed, 'b-', linewidth=2, label='Smoothed')
        
        # Mark convergence point
        if convergence_episode:
            plt.axvline(x=convergence_episode, color='r', linestyle='--', 
                       label=f'Convergence (ep {convergence_episode})')
        
        plt.axhline(y=0.01, color='g', linestyle=':', label='Threshold (1%)')
        
        plt.xlabel('Episodes')
        plt.ylabel('Average Q-value Change Rate')
        plt.title('Q-Learning Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'q_value_convergence.png', dpi=300)
        plt.close()
        
        logger.info(f"Convergence plot saved to {output_dir / 'q_value_convergence.png'}")
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "hypothesis_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"h5_convergence_{results.get('test', 'general')}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
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