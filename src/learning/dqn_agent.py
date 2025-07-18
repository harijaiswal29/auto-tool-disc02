"""Deep Q-Learning Agent for tool discovery and selection.

This module implements the DQN agent that uses neural networks for
value function approximation, including double DQN and target networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import asyncio
import json
from datetime import datetime
import os

from .deep_q_network import create_dqn, DQN
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from utils.logger import get_logger

logger = get_logger(__name__)


class DQNAgent:
    """Deep Q-Learning agent with target network and double DQN support."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int, action_space_size: int):
        """
        Initialize DQN Agent.
        
        Args:
            config: Configuration dictionary
            state_dim: Dimension of state vector
            action_space_size: Maximum number of possible actions
        """
        self.config = config
        self.dqn_config = config.get('dqn', {})
        
        # State and action dimensions
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        
        # Device selection
        device_config = self.dqn_config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        logger.info(f"Using device: {self.device}")
        
        # Create networks
        self.q_network = create_dqn(self.dqn_config, state_dim, action_space_size)
        self.target_network = create_dqn(self.dqn_config, state_dim, action_space_size)
        
        # Move networks to device
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.learning_rate = self.dqn_config.get('learning_rate', 0.0001)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Learning parameters
        self.gamma = self.dqn_config.get('discount_factor', 0.99)
        self.tau = self.dqn_config.get('tau', 0.001)  # Soft update parameter
        self.target_update_frequency = self.dqn_config.get('target_update_frequency', 1000)
        self.gradient_clip = self.dqn_config.get('gradient_clip', 1.0)
        
        # Double DQN
        self.double_dqn = self.dqn_config.get('double_dqn', True)
        
        # Exploration parameters
        self.epsilon = self.dqn_config.get('exploration_rate', 0.1)
        self.epsilon_decay = self.dqn_config.get('exploration_decay', 0.995)
        self.min_epsilon = self.dqn_config.get('min_exploration_rate', 0.01)
        
        # Experience replay
        self.batch_size = self.dqn_config.get('batch_size', 64)
        memory_size = self.dqn_config.get('memory_size', 100000)
        
        if self.dqn_config.get('prioritized_replay', True):
            alpha = self.dqn_config.get('prioritized_replay_alpha', 0.6)
            beta = self.dqn_config.get('prioritized_replay_beta', 0.4)
            beta_increment = self.dqn_config.get('prioritized_replay_beta_increment', 0.001)
            self.memory = PrioritizedReplayBuffer(
                memory_size, alpha, beta, beta_increment
            )
        else:
            # Use simple replay buffer if prioritized replay is disabled
            self.memory = deque(maxlen=memory_size)
            
        # Training metrics
        self.steps_done = 0
        self.episodes_done = 0
        self.losses = []
        
        # Action mapping (for variable action spaces)
        self.action_mapping = {}
        self.reverse_action_mapping = {}
        
        logger.info(f"DQN Agent initialized with {self.q_network.__class__.__name__}")
    
    def update_action_space(self, valid_actions: List[Tuple[str, ...]]):
        """
        Update the action mapping for current valid actions.
        
        Args:
            valid_actions: List of valid tool combinations
        """
        self.action_mapping = {i: action for i, action in enumerate(valid_actions)}
        self.reverse_action_mapping = {action: i for i, action in enumerate(valid_actions)}
    
    def select_action(self, state: np.ndarray, valid_actions: List[Tuple[str, ...]], 
                     training: bool = True) -> Tuple[str, ...]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            valid_actions: List of valid actions
            training: Whether in training mode
            
        Returns:
            Selected action (tool combination)
        """
        # Update action mapping
        self.update_action_space(valid_actions)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(len(valid_actions))
            return valid_actions[action_idx]
        
        # Exploit: choose best action based on Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions
            masked_q_values = torch.full_like(q_values, float('-inf'))
            for i in range(len(valid_actions)):
                masked_q_values[0, i] = q_values[0, i]
            
            action_idx = masked_q_values.argmax(dim=1).item()
            
        return valid_actions[action_idx]
    
    def store_transition(self, state: np.ndarray, action: Tuple[str, ...], 
                        reward: float, next_state: np.ndarray, 
                        next_valid_actions: List[Tuple[str, ...]], done: bool):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_valid_actions: Valid actions in next state
            done: Whether episode ended
        """
        # Get action index
        action_idx = self.reverse_action_mapping.get(action, 0)
        
        # Store valid action indices for next state
        next_valid_indices = list(range(len(next_valid_actions)))
        
        transition = {
            'state': state,
            'action': action_idx,
            'reward': reward,
            'next_state': next_state,
            'next_valid_indices': next_valid_indices,
            'done': done
        }
        
        if hasattr(self.memory, 'add'):
            # Prioritized replay buffer
            self.memory.add(transition)
        else:
            # Simple replay buffer
            self.memory.append(transition)
        
        self.steps_done += 1
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training happened, None otherwise
        """
        # Check if enough samples
        min_samples = self.batch_size if hasattr(self.memory, 'add') else len(self.memory)
        if min_samples < self.batch_size:
            return None
        
        # Sample batch
        if hasattr(self.memory, 'sample'):
            # Prioritized replay
            batch, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Simple replay
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Prepare batch tensors
        states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            
            # Compute targets
            targets = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss (weighted for prioritized replay)
        td_errors = (current_q_values - targets).squeeze()
        loss = (weights * td_errors.pow(2)).mean()
        
        # Update priorities for prioritized replay
        if indices is not None and hasattr(self.memory, 'update_priorities'):
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self, soft: bool = True):
        """
        Update target network weights.
        
        Args:
            soft: Whether to use soft update (True) or hard update (False)
        """
        if soft:
            # Soft update: θ_target = τ*θ_local + (1-τ)*θ_target
            for target_param, local_param in zip(
                self.target_network.parameters(), 
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            # Hard update: copy all weights
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        self.episodes_done += 1
    
    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'epsilon': self.epsilon,
            'losses': self.losses[-1000:],  # Keep last 1000 losses
            'config': self.dqn_config
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.losses = checkpoint.get('losses', [])
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        recent_losses = self.losses[-100:] if self.losses else []
        
        return {
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(recent_losses) if recent_losses else 0,
            'min_loss': np.min(recent_losses) if recent_losses else 0,
            'max_loss': np.max(recent_losses) if recent_losses else 0,
            'memory_size': len(self.memory) if hasattr(self.memory, '__len__') else self.memory.size(),
            'device': str(self.device)
        }
    
    def set_eval_mode(self):
        """Set network to evaluation mode."""
        self.q_network.eval()
    
    def set_train_mode(self):
        """Set network to training mode."""
        self.q_network.train()