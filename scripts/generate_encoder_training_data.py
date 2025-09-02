#!/usr/bin/env python3
"""Generate synthetic training data for the supervised encoder.

This script creates training data by running a mock evaluation and capturing
state vectors with their corresponding success/failure labels.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse

from src.learning.q_learning_engine import StateRepresentation
from src.utils.logger import get_logger

# Mock intent class for synthetic data generation
class MockIntent:
    def __init__(self, query):
        self.query = query
        self.embedding = np.random.randn(384).astype(np.float32)  # Sentence transformer output
        self.confidence_scores = np.random.rand(10)

logger = get_logger(__name__)


def generate_synthetic_states(num_episodes: int = 100, 
                            episodes_per_type: int = 20) -> tuple:
    """Generate synthetic state vectors with labels.
    
    Args:
        num_episodes: Total number of episodes to generate
        episodes_per_type: Episodes per query type
        
    Returns:
        Tuple of (states, labels)
    """
    # Initialize state encoder
    state_encoder = StateRepresentation(use_pca=False)
    
    all_states = []
    all_labels = []
    
    # Query types with expected success rates
    query_types = [
        ("List all files in the current directory", 0.8),  # Simple
        ("Search for Python files containing 'test'", 0.7),  # Medium
        ("Create a database and insert data", 0.5),  # Complex
        ("Analyze code complexity and refactor", 0.3),  # Very complex
        ("Debug failing unit tests", 0.4),  # Complex
    ]
    
    for query, success_rate in query_types:
        for episode in range(episodes_per_type):
            # Generate states for this episode
            num_steps = np.random.randint(3, 8)  # 3-7 steps per episode
            
            # Determine episode outcome
            success = np.random.random() < success_rate
            
            for step in range(num_steps):
                # Create mock intent
                intent = MockIntent(query)
                
                # Create mock context
                context = {
                    'domain': np.random.choice(['engineering', 'data_science', 'general']),
                    'query_count': episode + 1,
                    'session_duration': (episode + 1) * 60,
                    'total_queries': (episode + 1) * 5,
                    'success_rate': success_rate + np.random.normal(0, 0.1),
                    'metrics': {
                        'avg_response_time': np.random.uniform(100, 2000),
                        'success_rate': success_rate,
                        'error_rate': 1 - success_rate,
                        'tools_invoked': np.random.randint(1, 4),
                        'cache_hit_rate': np.random.uniform(0, 0.5)
                    },
                    'failure_rates': {
                        'filesystem_mcp': np.random.uniform(0, 0.2),
                        'sqlite_mcp': np.random.uniform(0, 0.3),
                        'search_mcp': np.random.uniform(0, 0.1)
                    },
                    'failure_types': {
                        'network_timeout': np.random.randint(0, 3),
                        'permission_error': np.random.randint(0, 2),
                        'rate_limit': np.random.randint(0, 1)
                    },
                    'retry_patterns': {
                        'avg_retry_count': np.random.uniform(0, 2),
                        'retry_success_rate': np.random.uniform(0.3, 0.9),
                        'avg_retry_delay_ms': np.random.uniform(100, 5000)
                    },
                    'user_expertise': np.random.choice(['novice', 'intermediate', 'expert']),
                    'tool_categories': {
                        'search': np.random.uniform(0, 1),
                        'database': np.random.uniform(0, 1),
                        'filesystem': np.random.uniform(0, 1)
                    },
                    'query_complexity': {
                        'length': len(query.split()),
                        'tools_required': np.random.randint(1, 4),
                        'complexity_score': 1 - success_rate
                    },
                    'episode_progress': step / num_steps,
                    'learning_phase': 'exploration' if episode < 10 else 'exploitation'
                }
                
                # Create mock history
                history = [
                    np.random.choice(['filesystem_mcp', 'sqlite_mcp', 'search_mcp', 'github_mcp'])
                    for _ in range(min(step, 5))
                ]
                
                # Generate state vector
                try:
                    state_vector = state_encoder.encode_state(intent, context, history)
                    
                    # Add some noise to make states more diverse
                    state_vector += np.random.normal(0, 0.01, size=state_vector.shape)
                    
                    all_states.append(state_vector)
                    all_labels.append(1.0 if success else 0.0)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate state: {e}")
                    continue
    
    states = np.array(all_states, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    
    logger.info(f"Generated {len(states)} state vectors")
    logger.info(f"State shape: {states.shape}")
    logger.info(f"Positive ratio: {(labels > 0.5).mean():.3f}")
    
    return states, labels


def balance_dataset(states: np.ndarray, labels: np.ndarray, 
                   balance_ratio: float = 1.0) -> tuple:
    """Balance the dataset to have equal positive/negative samples.
    
    Args:
        states: State vectors
        labels: Binary labels
        balance_ratio: Ratio of negative to positive samples
        
    Returns:
        Balanced (states, labels)
    """
    positive_idx = np.where(labels > 0.5)[0]
    negative_idx = np.where(labels <= 0.5)[0]
    
    n_positive = len(positive_idx)
    n_negative = int(n_positive * balance_ratio)
    
    if n_negative > len(negative_idx):
        # Oversample negative
        negative_idx = np.random.choice(negative_idx, n_negative, replace=True)
    else:
        # Undersample negative
        negative_idx = np.random.choice(negative_idx, n_negative, replace=False)
    
    # Combine and shuffle
    all_idx = np.concatenate([positive_idx, negative_idx])
    np.random.shuffle(all_idx)
    
    balanced_states = states[all_idx]
    balanced_labels = labels[all_idx]
    
    logger.info(f"Balanced dataset: {len(balanced_states)} samples")
    logger.info(f"Positive: {(balanced_labels > 0.5).sum()}, "
               f"Negative: {(balanced_labels <= 0.5).sum()}")
    
    return balanced_states, balanced_labels


def main():
    parser = argparse.ArgumentParser(description="Generate training data for encoder")
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to generate')
    parser.add_argument('--episodes-per-type', type=int, default=20,
                       help='Episodes per query type')
    parser.add_argument('--balance', action='store_true',
                       help='Balance the dataset')
    parser.add_argument('--output', type=str, 
                       default='data/encoder_training_data_synthetic.npz',
                       help='Output file path')
    
    args = parser.parse_args()
    
    logger.info("Generating synthetic training data...")
    
    # Generate states
    states, labels = generate_synthetic_states(
        args.episodes, 
        args.episodes_per_type
    )
    
    # Balance if requested
    if args.balance:
        states, labels = balance_dataset(states, labels)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    np.savez_compressed(
        args.output,
        states=states,
        labels=labels,
        metadata={
            'num_episodes': args.episodes,
            'episodes_per_type': args.episodes_per_type,
            'balanced': args.balance,
            'timestamp': datetime.now().isoformat()
        }
    )
    
    logger.info(f"Saved training data to {args.output}")
    logger.info(f"Total samples: {len(states)}")
    logger.info(f"State dimensions: {states.shape[1]}")
    logger.info(f"Positive ratio: {(labels > 0.5).mean():.3f}")
    
    return args.output


if __name__ == "__main__":
    main()