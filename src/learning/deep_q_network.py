"""Deep Q-Network (DQN) implementation for tool discovery and selection.

This module implements neural network architectures for Deep Q-Learning,
including standard DQN and Dueling DQN variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from utils.logger import get_logger

logger = get_logger(__name__)


class DQN(nn.Module):
    """Standard Deep Q-Network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = None, dropout_rate: float = 0.2):
        """
        Initialize DQN.
        
        Args:
            state_dim: Dimension of input state vector
            action_dim: Number of possible actions (tool combinations)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(DQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_dim = state_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created DQN with architecture: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Special initialization for output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for all actions, shape (batch_size, action_dim)
        """
        x = state
        
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output Q-values
        q_values = self.output_layer(x)
        
        return q_values
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a single state (numpy interface).
        
        Args:
            state: State as numpy array
            
        Returns:
            Q-values as numpy array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.squeeze(0).numpy()


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture.
    
    Separates value and advantage estimation for better learning stability.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = None, dropout_rate: float = 0.2):
        """
        Initialize Dueling DQN.
        
        Args:
            state_dim: Dimension of input state vector
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(DuelingDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        
        prev_dim = state_dim
        for i, hidden_dim in enumerate(hidden_dims[:-1]):  # Use all but last hidden dim
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created Dueling DQN with architecture: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # Feature layers
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Value and advantage streams
        for module in [self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for all actions, shape (batch_size, action_dim)
        """
        # Feature extraction
        features = state
        for layer in self.feature_layers:
            features = layer(features)
        
        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using dueling formula: Q = V + (A - mean(A))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a single state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.squeeze(0).numpy()


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration via noisy networks."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation for noise
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class NoisyDQN(DQN):
    """DQN with noisy layers for exploration."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = None, std_init: float = 0.5):
        """Initialize Noisy DQN."""
        super().__init__(state_dim, action_dim, hidden_dims, dropout_rate=0)
        
        # Replace final linear layer with noisy layer
        prev_dim = self.hidden_dims[-1]
        self.output_layer = NoisyLinear(prev_dim, action_dim, std_init)
        
        logger.info(f"Created Noisy DQN with std_init={std_init}")
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        if hasattr(self.output_layer, 'reset_noise'):
            self.output_layer.reset_noise()


def create_dqn(config: Dict[str, Any], state_dim: int, action_dim: int) -> nn.Module:
    """
    Factory function to create DQN based on configuration.
    
    Args:
        config: DQN configuration dictionary
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        DQN model instance
    """
    network_type = config.get('network_type', 'standard')
    hidden_dims = config.get('network_architecture', [512, 256, 128])
    dropout_rate = config.get('dropout_rate', 0.2)
    
    if network_type == 'dueling':
        return DuelingDQN(state_dim, action_dim, hidden_dims, dropout_rate)
    elif network_type == 'noisy':
        std_init = config.get('noisy_std_init', 0.5)
        return NoisyDQN(state_dim, action_dim, hidden_dims, std_init)
    else:  # standard
        return DQN(state_dim, action_dim, hidden_dims, dropout_rate)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)