"""Prioritized Experience Replay Buffer for DQN.

This module implements a prioritized replay buffer using a sum-tree
data structure for efficient sampling based on TD-error priorities.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque

from utils.logger import get_logger

logger = get_logger(__name__)


class SumTree:
    """Sum-tree data structure for efficient prioritized sampling.
    
    Stores values in leaf nodes and maintains partial sums in internal nodes
    for O(log n) sampling based on priorities.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum-tree.
        
        Args:
            capacity: Maximum number of elements
        """
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1  # Number of nodes in binary tree
        self.tree = np.zeros(self.tree_size)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= self.tree_size:
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """Add new element with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority of element at given tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get element based on cumulative sum.
        
        Args:
            s: Cumulative sum value
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer using sum-tree for efficient sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.min_priority = 0.01
        
        logger.info(f"Created Prioritized Replay Buffer with capacity={capacity}, "
                   f"alpha={alpha}, beta={beta}")
    
    def add(self, transition: Dict[str, Any]):
        """
        Add transition with maximum priority.
        
        Args:
            transition: Experience dictionary
        """
        # New experiences get maximum priority
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], List[int], np.ndarray]:
        """
        Sample batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (batch, indices, importance_weights)
        """
        batch = []
        indices = []
        priorities = []
        
        # Divide total priority range into segments
        segment = self.tree.total() / batch_size
        
        # Increase beta for importance sampling
        self.beta = min(self.beta + self.beta_increment, self.max_beta)
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is not None:  # Check for valid data
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # Calculate importance sampling weights
        if priorities:
            priorities = np.array(priorities)
            # Normalize by max weight for stability
            min_priority = np.min(priorities) if len(priorities) > 0 else 1.0
            max_weight = (min_priority / self.tree.total()) ** self.beta
            
            weights = ((priorities / self.tree.total()) ** -self.beta) / max_weight
        else:
            weights = np.ones(batch_size)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Tree indices of experiences
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            # Clip priority to avoid zero
            priority = np.clip(priority, self.min_priority, None)
            
            # Apply alpha exponent
            priority_alpha = priority ** self.alpha
            
            # Update tree
            self.tree.update(idx, priority_alpha)
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.tree.n_entries
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size() >= self.capacity
    
    def get_beta(self) -> float:
        """Get current beta value."""
        return self.beta
    
    def reset_beta(self, beta: float = 0.4):
        """Reset beta to initial value."""
        self.beta = beta
    
    def clear(self):
        """Clear the buffer."""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        logger.info("Cleared prioritized replay buffer")


class UniformReplayBuffer:
    """Simple uniform replay buffer for comparison."""
    
    def __init__(self, capacity: int):
        """Initialize uniform replay buffer."""
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, transition: Dict[str, Any]):
        """Add transition to buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], List[int], np.ndarray]:
        """Sample batch uniformly."""
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size)  # Uniform weights
        
        return batch, indices.tolist(), weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """No-op for uniform buffer."""
        pass
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.capacity
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()