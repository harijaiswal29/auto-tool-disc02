"""Extract labeled data from existing training runs for supervised encoder training.

This module provides utilities to extract state vectors and labels from
checkpoint files and JSON results for training the supervised state encoder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import glob
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class EncoderDataExtractor:
    """Extract labeled training data from existing checkpoint files."""
    
    def __init__(self, results_dir: str = None):
        """Initialize the data extractor.
        
        Args:
            results_dir: Directory containing training results and checkpoints
        """
        if results_dir is None:
            results_dir = "tests/dissertation_test_suite/results"
        self.results_dir = Path(results_dir)
        self.labeled_data = []
        self.statistics = defaultdict(int)
        
    def extract_from_checkpoint(self, checkpoint_path: str, 
                               labeling_strategy: str = "episode") -> List[Dict]:
        """Extract labeled data from a single checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint PKL file
            labeling_strategy: Strategy for labeling ('episode', 'trajectory', 'reward')
            
        Returns:
            List of dictionaries with 'state' and 'label' keys
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            extracted_data = []
            
            # Check for new format with state vectors
            if 'episode_states' in checkpoint:
                # New format with state vectors
                episode_states = checkpoint['episode_states']
                for state_data in episode_states:
                    label = self._get_label_for_state(
                        state_data, labeling_strategy
                    )
                    extracted_data.append({
                        'state': state_data['state_vector'],
                        'label': label,
                        'query': state_data.get('query', ''),
                        'success': state_data.get('success', False),
                        'reward': state_data.get('reward', 0.0),
                        'episode': state_data.get('episode', 0),
                        'tools_selected': state_data.get('tools_selected', []),
                        'optimal_tools': state_data.get('optimal_tools', [])
                    })
            
            # Extract episode data if available (old format)
            elif 'episodes' in checkpoint:
                for episode in checkpoint['episodes']:
                    episode_data = self._extract_episode_data(
                        episode, labeling_strategy
                    )
                    extracted_data.extend(episode_data)
            
            # Extract from metrics if episodes not available
            elif 'metrics' in checkpoint:
                metrics_data = self._extract_metrics_data(
                    checkpoint['metrics'], labeling_strategy
                )
                extracted_data.extend(metrics_data)
            
            logger.info(f"Extracted {len(extracted_data)} samples from {checkpoint_path}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting from checkpoint {checkpoint_path}: {e}")
            return []
    
    def _get_label_for_state(self, state_data: Dict, 
                            labeling_strategy: str) -> float:
        """Get label for a state based on labeling strategy.
        
        Args:
            state_data: State data dictionary
            labeling_strategy: Labeling strategy to use
            
        Returns:
            Label value (float)
        """
        if labeling_strategy == "episode":
            # Binary label based on success
            return 1.0 if state_data.get('success', False) else 0.0
        
        elif labeling_strategy == "reward":
            # Use reward directly (normalized)
            reward = state_data.get('reward', 0.0)
            return np.tanh(reward / 20.0)  # Normalize to [-1, 1]
        
        elif labeling_strategy == "trajectory":
            # Progressive labeling based on episode position
            episode = state_data.get('episode', 0)
            query_idx = state_data.get('query_idx', 0)
            success = state_data.get('success', False)
            
            # Calculate progress (0 to 1)
            progress = query_idx / 10.0  # Assuming ~10 queries per episode
            
            if success:
                # Successful: 0.5 to 1.0
                return 0.5 + 0.5 * progress
            else:
                # Failed: 0.0 to 0.5
                return 0.5 * progress
        
        else:
            # Default to binary
            return 1.0 if state_data.get('success', False) else 0.0
    
    def _extract_episode_data(self, episode: Dict, 
                             labeling_strategy: str) -> List[Dict]:
        """Extract data from a single episode.
        
        Args:
            episode: Episode dictionary
            labeling_strategy: Labeling strategy to use
            
        Returns:
            List of labeled data points
        """
        data = []
        
        # Get episode outcome
        completed = episode.get('completion_status', False)
        reward = episode.get('reward', 0.0)
        states = episode.get('states', [])
        
        if labeling_strategy == "episode":
            # Binary labels: 1 for successful episodes, 0 for failed
            label = 1.0 if completed and reward > 0 else 0.0
            for state in states:
                if isinstance(state, (list, np.ndarray)):
                    data.append({
                        'state': np.array(state, dtype=np.float32),
                        'label': label,
                        'episode_outcome': 'success' if label == 1.0 else 'failure'
                    })
                    
        elif labeling_strategy == "trajectory":
            # Graduated labels based on progress through episode
            num_states = len(states)
            for i, state in enumerate(states):
                if isinstance(state, (list, np.ndarray)):
                    progress = (i + 1) / num_states if num_states > 0 else 0
                    if completed and reward > 0:
                        label = 0.5 + 0.5 * progress  # 0.5 to 1.0 for success
                    else:
                        label = 0.5 * progress  # 0.0 to 0.5 for failure
                    
                    data.append({
                        'state': np.array(state, dtype=np.float32),
                        'label': label,
                        'progress': progress,
                        'episode_outcome': 'success' if completed else 'failure'
                    })
                    
        elif labeling_strategy == "reward":
            # Use actual reward values as continuous labels
            normalized_reward = np.tanh(reward / 20.0)  # Normalize rewards
            for state in states:
                if isinstance(state, (list, np.ndarray)):
                    data.append({
                        'state': np.array(state, dtype=np.float32),
                        'label': normalized_reward,
                        'raw_reward': reward,
                        'episode_outcome': 'success' if completed else 'failure'
                    })
        
        return data
    
    def _extract_metrics_data(self, metrics: Dict, 
                            labeling_strategy: str) -> List[Dict]:
        """Extract data from metrics when episode data not available.
        
        Args:
            metrics: Metrics dictionary
            labeling_strategy: Labeling strategy to use
            
        Returns:
            List of labeled data points
        """
        data = []
        
        # Try to extract from Q-values or state-action pairs
        if 'q_values' in metrics:
            q_values = metrics['q_values']
            for state_hash, actions in q_values.items():
                # Reconstruct state if possible (placeholder)
                # In practice, we'd need the actual state vectors
                pass
        
        return data
    
    def extract_from_json_results(self, json_path: str) -> List[Dict]:
        """Extract labeled data from JSON result files.
        
        Args:
            json_path: Path to JSON result file
            
        Returns:
            List of labeled data points
        """
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
            
            data = []
            
            # Extract from episode history if available
            if 'episode_history' in results:
                for episode in results['episode_history']:
                    if 'state_vectors' in episode:
                        states = episode['state_vectors']
                        completed = episode.get('completed', False)
                        reward = episode.get('reward', 0.0)
                        
                        label = 1.0 if completed and reward > 0 else 0.0
                        
                        for state in states:
                            if isinstance(state, list):
                                data.append({
                                    'state': np.array(state, dtype=np.float32),
                                    'label': label,
                                    'source': 'json_results'
                                })
            
            logger.info(f"Extracted {len(data)} samples from {json_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error extracting from JSON {json_path}: {e}")
            return []
    
    def extract_from_directory(self, directory: str = None,
                              labeling_strategy: str = "episode",
                              file_pattern: str = "*.pkl") -> Dict:
        """Extract labeled data from all files in a directory.
        
        Args:
            directory: Directory to search (uses latest if None)
            labeling_strategy: Strategy for labeling data
            file_pattern: Pattern for checkpoint files
            
        Returns:
            Dictionary with extracted data and statistics
        """
        if directory is None:
            # Find the latest curriculum optimization directory
            pattern = str(self.results_dir / "curriculum_opt_*")
            dirs = sorted(glob.glob(pattern))
            if dirs:
                directory = dirs[-1]
                logger.info(f"Using latest directory: {directory}")
            else:
                logger.error("No curriculum optimization directories found")
                return {'data': [], 'statistics': {}}
        
        directory_path = Path(directory)
        checkpoint_dir = directory_path / "checkpoints"
        
        all_data = []
        
        # Extract from checkpoint files
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob(file_pattern))
            logger.info(f"Found {len(checkpoint_files)} checkpoint files")
            
            for checkpoint_file in checkpoint_files:
                data = self.extract_from_checkpoint(
                    str(checkpoint_file), labeling_strategy
                )
                all_data.extend(data)
                
                # Update statistics
                for item in data:
                    outcome = item.get('episode_outcome', 'unknown')
                    self.statistics[f'samples_{outcome}'] += 1
        
        # Also extract from JSON results
        stage_results_dir = directory_path / "stage_results"
        if stage_results_dir.exists():
            for stage_dir in stage_results_dir.iterdir():
                if stage_dir.is_dir():
                    json_files = list(stage_dir.glob("*.json"))
                    for json_file in json_files:
                        data = self.extract_from_json_results(str(json_file))
                        all_data.extend(data)
        
        self.labeled_data = all_data
        self.statistics['total_samples'] = len(all_data)
        
        # Calculate label distribution
        if all_data:
            labels = [item['label'] for item in all_data]
            self.statistics['mean_label'] = np.mean(labels)
            self.statistics['std_label'] = np.std(labels)
            self.statistics['positive_ratio'] = np.mean([l > 0.5 for l in labels])
        
        logger.info(f"Total samples extracted: {len(all_data)}")
        logger.info(f"Statistics: {dict(self.statistics)}")
        
        return {
            'data': all_data,
            'statistics': dict(self.statistics)
        }
    
    def save_extracted_data(self, output_path: str = None):
        """Save extracted data to file for later use.
        
        Args:
            output_path: Path to save the data (NPZ format)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/encoder_training_data_{timestamp}.npz"
        
        if not self.labeled_data:
            logger.warning("No data to save")
            return
        
        # Prepare arrays
        states = []
        labels = []
        metadata = []
        
        for item in self.labeled_data:
            states.append(item['state'])
            labels.append(item['label'])
            
            # Store additional metadata
            meta = {k: v for k, v in item.items() 
                   if k not in ['state', 'label']}
            metadata.append(meta)
        
        states = np.array(states, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save data
        np.savez_compressed(
            output_path,
            states=states,
            labels=labels,
            metadata=metadata,
            statistics=dict(self.statistics)
        )
        
        logger.info(f"Saved {len(states)} samples to {output_path}")
        logger.info(f"State shape: {states.shape}, Label shape: {labels.shape}")
        
        return output_path
    
    def create_balanced_dataset(self, balance_ratio: float = 1.0) -> Dict:
        """Create a balanced dataset with equal positive/negative samples.
        
        Args:
            balance_ratio: Ratio of negative to positive samples (1.0 = equal)
            
        Returns:
            Balanced dataset dictionary
        """
        if not self.labeled_data:
            logger.warning("No data available for balancing")
            return {'data': [], 'statistics': {}}
        
        positive_samples = [d for d in self.labeled_data if d['label'] > 0.5]
        negative_samples = [d for d in self.labeled_data if d['label'] <= 0.5]
        
        logger.info(f"Original: {len(positive_samples)} positive, "
                   f"{len(negative_samples)} negative samples")
        
        # Balance the dataset
        if positive_samples and negative_samples:
            n_positive = len(positive_samples)
            n_negative = int(n_positive * balance_ratio)
            
            if n_negative > len(negative_samples):
                # Oversample negative samples
                import random
                negative_samples = negative_samples * (n_negative // len(negative_samples) + 1)
                negative_samples = random.sample(negative_samples, n_negative)
            elif n_negative < len(negative_samples):
                # Undersample negative samples
                import random
                negative_samples = random.sample(negative_samples, n_negative)
            
            balanced_data = positive_samples + negative_samples
            
            # Shuffle the data
            import random
            random.shuffle(balanced_data)
            
            logger.info(f"Balanced: {len(positive_samples)} positive, "
                       f"{len(negative_samples)} negative samples")
            
            return {
                'data': balanced_data,
                'statistics': {
                    'n_positive': len(positive_samples),
                    'n_negative': len(negative_samples),
                    'total': len(balanced_data),
                    'balance_ratio': balance_ratio
                }
            }
        
        return {'data': self.labeled_data, 'statistics': dict(self.statistics)}


def main():
    """Main function to demonstrate data extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract labeled data for encoder training")
    parser.add_argument('--results-dir', type=str, 
                       default="tests/dissertation_test_suite/results",
                       help="Directory containing training results")
    parser.add_argument('--directory', type=str, default=None,
                       help="Specific directory to extract from (uses latest if None)")
    parser.add_argument('--strategy', type=str, default="episode",
                       choices=['episode', 'trajectory', 'reward'],
                       help="Labeling strategy to use")
    parser.add_argument('--output', type=str, default=None,
                       help="Output file path for extracted data")
    parser.add_argument('--balance', action='store_true',
                       help="Balance the dataset")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = EncoderDataExtractor(args.results_dir)
    
    # Extract data
    result = extractor.extract_from_directory(
        directory=args.directory,
        labeling_strategy=args.strategy
    )
    
    print(f"\nExtraction complete:")
    print(f"Total samples: {result['statistics'].get('total_samples', 0)}")
    print(f"Statistics: {result['statistics']}")
    
    # Balance if requested
    if args.balance:
        balanced = extractor.create_balanced_dataset()
        print(f"\nBalanced dataset:")
        print(f"Statistics: {balanced['statistics']}")
    
    # Save data
    if result['data']:
        output_path = extractor.save_extracted_data(args.output)
        print(f"\nData saved to: {output_path}")
    else:
        print("\nNo data extracted")


if __name__ == "__main__":
    main()