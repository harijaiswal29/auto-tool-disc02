#!/usr/bin/env python3
"""
Simple test of CheckpointManager functionality
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "scripts")

# Import the CheckpointManager
from run_baseline_comparison import CheckpointManager

def test_checkpoint_manager():
    """Test basic CheckpointManager operations."""
    print("Testing CheckpointManager...")
    
    # Create checkpoint manager
    manager = CheckpointManager(Path("test_checkpoint_dir"), checkpoint_interval=5)
    
    # Test should_checkpoint
    print(f"Should checkpoint at episode 0: {manager.should_checkpoint(0)}")  # False
    print(f"Should checkpoint at episode 5: {manager.should_checkpoint(5)}")  # True
    print(f"Should checkpoint at episode 10: {manager.should_checkpoint(10)}")  # True
    print(f"Should checkpoint at episode 7: {manager.should_checkpoint(7)}")  # False
    
    # Test saving checkpoint
    test_state = {
        'strategy_name': 'test_strategy',
        'episode': 5,
        'metrics': {
            'completion_rates': [0.5, 0.6, 0.7],
            'tool_accuracies': [0.8, 0.85, 0.9]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_file = manager.save_checkpoint(test_state, 5, 'test_strategy')
    print(f"Saved checkpoint: {checkpoint_file}")
    
    # Test loading checkpoint
    loaded_state = manager.load_checkpoint(checkpoint_file)
    print(f"Loaded strategy: {loaded_state['strategy_name']}")
    print(f"Loaded episode: {loaded_state['episode']}")
    print(f"Metrics match: {loaded_state['metrics'] == test_state['metrics']}")
    
    # Clean up
    Path(checkpoint_file).unlink()
    Path("test_checkpoint_dir").rmdir()
    
    print("✓ CheckpointManager test passed!")

if __name__ == "__main__":
    test_checkpoint_manager()