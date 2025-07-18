"""DQN Training Utilities for systematic training and evaluation.

This module provides utilities for training DQN models including
learning curves, model evaluation, and hyperparameter scheduling.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque
import json
import os
from datetime import datetime
import asyncio
from tqdm import tqdm

from .q_learning_engine import QLearningEngine
from .dqn_agent import DQNAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class LearningRateScheduler:
    """Learning rate scheduler for DQN training."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 schedule_type: str = 'step',
                 **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            schedule_type: Type of schedule ('step', 'exponential', 'cosine')
            **kwargs: Schedule-specific parameters
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        
        if schedule_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=kwargs.get('step_size', 10000),
                gamma=kwargs.get('gamma', 0.9)
            )
        elif schedule_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.999)
            )
        elif schedule_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100000),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
    
    def step(self):
        """Update learning rate."""
        if self.scheduler:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class DQNTrainer:
    """Trainer for DQN models with monitoring and evaluation."""
    
    def __init__(self, q_learning_engine: QLearningEngine, 
                 eval_interval: int = 1000,
                 checkpoint_interval: int = 5000,
                 log_interval: int = 100):
        """
        Initialize DQN trainer.
        
        Args:
            q_learning_engine: Q-learning engine with DQN enabled
            eval_interval: Steps between evaluations
            checkpoint_interval: Steps between checkpoints
            log_interval: Steps between logging
        """
        if not q_learning_engine.use_dqn:
            raise ValueError("Q-learning engine must have DQN enabled")
        
        self.engine = q_learning_engine
        self.dqn_agent = q_learning_engine.dqn_agent
        
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        
        # Metrics tracking
        self.training_metrics = {
            'losses': deque(maxlen=1000),
            'rewards': deque(maxlen=1000),
            'episode_rewards': [],
            'eval_rewards': [],
            'learning_rates': [],
            'epsilon_values': []
        }
        
        # Learning rate scheduler
        self.lr_scheduler = None
        
        # Best model tracking
        self.best_eval_reward = float('-inf')
        self.best_model_path = None
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        
        logger.info("DQN Trainer initialized")
    
    def setup_lr_scheduler(self, schedule_type: str = 'step', **kwargs):
        """Setup learning rate scheduler."""
        self.lr_scheduler = LearningRateScheduler(
            self.dqn_agent.optimizer,
            schedule_type,
            **kwargs
        )
        logger.info(f"Learning rate scheduler setup: {schedule_type}")
    
    async def train_episode(self, env_step_func: Callable, 
                          max_steps: int = 1000) -> float:
        """
        Train for one episode.
        
        Args:
            env_step_func: Function that takes action and returns 
                         (state, reward, done, info)
            max_steps: Maximum steps per episode
            
        Returns:
            Total episode reward
        """
        episode_reward = 0
        step_count = 0
        
        # Get initial state from environment
        state, available_tools, constraints = await env_step_func('reset')
        
        while step_count < max_steps:
            # Select action
            action = await self.engine.select_action(
                state, available_tools, constraints
            )
            
            # Execute action in environment
            next_state, reward, done, info = await env_step_func(action)
            next_available_tools = info.get('available_tools', available_tools)
            next_constraints = info.get('constraints', constraints)
            
            # Learn from experience
            await self.engine.learn_from_experience(
                state, action, reward, next_state,
                next_available_tools, next_constraints, done
            )
            
            # Update metrics
            episode_reward += reward
            self.training_metrics['rewards'].append(reward)
            
            # Update state
            state = next_state
            available_tools = next_available_tools
            constraints = next_constraints
            
            step_count += 1
            self.total_steps += 1
            
            # Logging
            if self.total_steps % self.log_interval == 0:
                self._log_training_progress()
            
            # Evaluation
            if self.total_steps % self.eval_interval == 0:
                await self._evaluate()
            
            # Checkpoint
            if self.total_steps % self.checkpoint_interval == 0:
                await self._save_checkpoint()
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            if done:
                break
        
        # Episode complete
        self.engine.decay_exploration()
        self.total_episodes += 1
        self.training_metrics['episode_rewards'].append(episode_reward)
        
        return episode_reward
    
    async def train(self, env_step_func: Callable, 
                   num_episodes: int = 1000,
                   max_steps_per_episode: int = 1000):
        """
        Train for multiple episodes.
        
        Args:
            env_step_func: Environment step function
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
        """
        logger.info(f"Starting DQN training for {num_episodes} episodes")
        
        # Progress bar
        pbar = tqdm(total=num_episodes, desc="Training Episodes")
        
        for episode in range(num_episodes):
            # Train one episode
            episode_reward = await self.train_episode(
                env_step_func, max_steps_per_episode
            )
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{np.mean(self.training_metrics["episode_rewards"][-100:]):.2f}',
                'epsilon': f'{self.dqn_agent.epsilon:.3f}'
            })
        
        pbar.close()
        
        # Final evaluation and checkpoint
        await self._evaluate()
        await self._save_checkpoint(final=True)
        
        logger.info("Training complete")
    
    async def _evaluate(self, num_episodes: int = 10):
        """Evaluate current model performance."""
        logger.info("Evaluating model...")
        
        # Set to evaluation mode
        self.dqn_agent.set_eval_mode()
        eval_rewards = []
        
        # Run evaluation episodes
        for _ in range(num_episodes):
            # Evaluation uses a separate environment function
            # This is a placeholder - actual implementation would use eval_env_func
            eval_reward = np.mean(self.training_metrics['rewards'])
            eval_rewards.append(eval_reward)
        
        # Calculate statistics
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        self.training_metrics['eval_rewards'].append({
            'step': self.total_steps,
            'mean': mean_reward,
            'std': std_reward
        })
        
        logger.info(f"Evaluation: mean_reward={mean_reward:.2f}, "
                   f"std={std_reward:.2f}")
        
        # Save best model
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            self.best_model_path = f"models/best_dqn_model.pt"
            self.dqn_agent.save_checkpoint(self.best_model_path)
            logger.info(f"New best model saved: reward={mean_reward:.2f}")
        
        # Back to training mode
        self.dqn_agent.set_train_mode()
    
    async def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint_name = "final" if final else f"step_{self.total_steps}"
        checkpoint_path = f"models/dqn_checkpoint_{checkpoint_name}.pt"
        
        # Save model
        await self.engine.save_model(checkpoint_name)
        
        # Save training metrics
        metrics_path = f"models/training_metrics_{checkpoint_name}.json"
        self._save_metrics(metrics_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _log_training_progress(self):
        """Log current training progress."""
        recent_losses = list(self.training_metrics['losses'])[-100:]
        recent_rewards = list(self.training_metrics['rewards'])[-100:]
        
        if recent_losses:
            metrics = self.dqn_agent.get_metrics()
            logger.info(
                f"Step {self.total_steps}: "
                f"loss={np.mean(recent_losses):.4f}, "
                f"reward={np.mean(recent_rewards):.2f}, "
                f"epsilon={metrics['epsilon']:.3f}, "
                f"lr={self.lr_scheduler.get_lr() if self.lr_scheduler else self.dqn_agent.learning_rate:.6f}"
            )
            
            # Track metrics
            self.training_metrics['epsilon_values'].append({
                'step': self.total_steps,
                'value': metrics['epsilon']
            })
            self.training_metrics['learning_rates'].append({
                'step': self.total_steps,
                'value': self.lr_scheduler.get_lr() if self.lr_scheduler else self.dqn_agent.learning_rate
            })
    
    def _save_metrics(self, filepath: str):
        """Save training metrics to file."""
        metrics_data = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_eval_reward': self.best_eval_reward,
            'episode_rewards': list(self.training_metrics['episode_rewards']),
            'eval_rewards': self.training_metrics['eval_rewards'],
            'epsilon_values': self.training_metrics['epsilon_values'],
            'learning_rates': self.training_metrics['learning_rates']
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        ax = axes[0, 0]
        episode_rewards = self.training_metrics['episode_rewards']
        if episode_rewards:
            ax.plot(episode_rewards, alpha=0.5, label='Raw')
            # Moving average
            window = min(100, len(episode_rewards) // 4)
            if window > 1:
                ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(episode_rewards)), ma, 
                       label=f'{window}-episode MA')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title('Episode Rewards')
            ax.legend()
        
        # Evaluation rewards
        ax = axes[0, 1]
        eval_rewards = self.training_metrics['eval_rewards']
        if eval_rewards:
            steps = [r['step'] for r in eval_rewards]
            means = [r['mean'] for r in eval_rewards]
            stds = [r['std'] for r in eval_rewards]
            ax.plot(steps, means, 'b-', label='Mean')
            ax.fill_between(steps, 
                          [m-s for m,s in zip(means, stds)],
                          [m+s for m,s in zip(means, stds)],
                          alpha=0.3)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Evaluation Reward')
            ax.set_title('Evaluation Performance')
            ax.legend()
        
        # Epsilon decay
        ax = axes[1, 0]
        epsilon_values = self.training_metrics['epsilon_values']
        if epsilon_values:
            steps = [e['step'] for e in epsilon_values]
            values = [e['value'] for e in epsilon_values]
            ax.plot(steps, values)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Epsilon')
            ax.set_title('Exploration Rate Decay')
        
        # Learning rate
        ax = axes[1, 1]
        lr_values = self.training_metrics['learning_rates']
        if lr_values:
            steps = [lr['step'] for lr in lr_values]
            values = [lr['value'] for lr in lr_values]
            ax.plot(steps, values)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.show()


def create_training_config(base_config: Dict[str, Any], 
                         overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create training configuration with DQN enabled.
    
    Args:
        base_config: Base configuration
        overrides: Optional parameter overrides
        
    Returns:
        Training configuration
    """
    training_config = base_config.copy()
    
    # Enable DQN
    training_config['dqn']['enabled'] = True
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if '.' in key:
                # Nested key
                parts = key.split('.')
                current = training_config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                training_config[key] = value
    
    return training_config