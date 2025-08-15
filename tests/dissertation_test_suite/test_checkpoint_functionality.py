#!/usr/bin/env python3
"""
Test script to verify checkpoint functionality in run_baseline_comparison.py
"""

import asyncio
import sys
import json
from pathlib import Path
import subprocess
import time
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def test_checkpoint_save():
    """Test that checkpoints are being saved correctly."""
    print("\n" + "="*60)
    print("TEST 1: Checkpoint Saving")
    print("="*60)
    
    # Run a quick test with checkpoint interval of 5 episodes
    cmd = [
        "python", "scripts/run_baseline_comparison.py",
        "--query-set", "quick_test",
        "--episodes", "10",
        "--runs", "1",
        "--checkpoint-interval", "5",
        "--checkpoint-dir", "test_checkpoints"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return False
    
    # Check if checkpoint files were created
    checkpoint_dir = Path("test_checkpoints")
    if not checkpoint_dir.exists():
        print("ERROR: Checkpoint directory was not created")
        return False
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    if len(checkpoint_files) == 0:
        print("ERROR: No checkpoint files were created")
        return False
    
    # Verify checkpoint content
    for cp_file in checkpoint_files:
        print(f"\nVerifying checkpoint: {cp_file.name}")
        try:
            with open(cp_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            required_keys = ['strategy_name', 'episode', 'metrics', 'timestamp']
            for key in required_keys:
                if key not in checkpoint_data:
                    print(f"  ERROR: Missing key '{key}' in checkpoint")
                    return False
            
            print(f"  Strategy: {checkpoint_data['strategy_name']}")
            print(f"  Episode: {checkpoint_data['episode']}")
            print(f"  Metrics keys: {list(checkpoint_data['metrics'].keys())}")
            
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}")
            return False
    
    print("\n✓ Checkpoint saving test PASSED")
    return True


def test_checkpoint_resume():
    """Test that resuming from checkpoint works correctly."""
    print("\n" + "="*60)
    print("TEST 2: Checkpoint Resume")
    print("="*60)
    
    # First, create a checkpoint by running partial experiment
    print("\nStep 1: Creating initial checkpoint...")
    cmd1 = [
        "python", "scripts/run_baseline_comparison.py",
        "--query-set", "quick_test",
        "--episodes", "20",
        "--runs", "1",
        "--checkpoint-interval", "10",
        "--checkpoint-dir", "test_resume"
    ]
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    if result1.returncode != 0:
        print(f"Error in initial run: {result1.stderr}")
        return False
    
    # Find the checkpoint file
    checkpoint_dir = Path("test_resume")
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    
    if not checkpoint_files:
        print("ERROR: No checkpoint files created for resume test")
        return False
    
    latest_checkpoint = checkpoint_files[-1]
    print(f"Found checkpoint: {latest_checkpoint.name}")
    
    # Load checkpoint to verify episode number
    with open(latest_checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    saved_episode = checkpoint_data['episode']
    print(f"Checkpoint saved at episode: {saved_episode}")
    
    # Test resume command (dry run - just verify it starts)
    print("\nStep 2: Testing resume command...")
    cmd2 = [
        "python", "-c",
        f"""
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_baseline_comparison import CheckpointManager

# Test loading checkpoint
manager = CheckpointManager(Path('test_resume'))
state = manager.load_checkpoint('{latest_checkpoint}')
print(f"Successfully loaded checkpoint for {{state['strategy_name']}} at episode {{state['episode']}}")
"""
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result2.returncode != 0:
        print(f"Error testing resume: {result2.stderr}")
        return False
    
    print(result2.stdout)
    print("\n✓ Checkpoint resume test PASSED")
    return True


def cleanup_test_files():
    """Clean up test checkpoint directories."""
    print("\nCleaning up test files...")
    
    for dir_name in ["test_checkpoints", "test_resume"]:
        test_dir = Path(dir_name)
        if test_dir.exists():
            # Remove all files in directory
            for file in test_dir.glob("*"):
                file.unlink()
            # Remove directory
            test_dir.rmdir()
            print(f"  Removed {dir_name}/")


def main():
    """Run all checkpoint tests."""
    print("\n" + "="*80)
    print(" CHECKPOINT FUNCTIONALITY TEST SUITE")
    print("="*80)
    
    all_passed = True
    
    # Run tests
    try:
        # Test 1: Checkpoint saving
        if not test_checkpoint_save():
            all_passed = False
            print("✗ Checkpoint saving test FAILED")
        
        # Test 2: Checkpoint resume
        if not test_checkpoint_resume():
            all_passed = False
            print("✗ Checkpoint resume test FAILED")
        
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        all_passed = False
    
    finally:
        # Clean up test files
        cleanup_test_files()
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL CHECKPOINT TESTS PASSED")
    else:
        print("✗ SOME CHECKPOINT TESTS FAILED")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())