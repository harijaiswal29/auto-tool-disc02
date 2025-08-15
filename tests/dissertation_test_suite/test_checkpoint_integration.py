#!/usr/bin/env python3
"""
Integration test for checkpoint functionality
"""

import subprocess
import sys
from pathlib import Path
import time
import pickle

def test_checkpoint_integration():
    """Test checkpoint functionality with actual runner."""
    print("="*60)
    print("CHECKPOINT INTEGRATION TEST")
    print("="*60)
    
    checkpoint_dir = Path("test_checkpoint_int")
    
    # Run with checkpoints
    cmd = [
        sys.executable,
        "scripts/run_baseline_comparison.py",
        "--query-set", "quick_test",
        "--episodes", "4",
        "--runs", "1",
        "--checkpoint-interval", "2",
        "--checkpoint-dir", str(checkpoint_dir)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("This will take a moment...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.1f} seconds")
    
    # Check output for checkpoint messages
    if "Checkpointing enabled" in result.stdout:
        print("✓ Checkpoint system initialized")
    else:
        print("✗ No checkpoint initialization message found")
    
    # Check if checkpoint directory was created
    if checkpoint_dir.exists():
        print(f"✓ Checkpoint directory created: {checkpoint_dir}")
        
        # List checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pkl"))
        print(f"  Found {len(checkpoint_files)} checkpoint files:")
        
        for cp_file in checkpoint_files:
            print(f"    - {cp_file.name}")
            
            # Verify checkpoint content
            try:
                with open(cp_file, 'rb') as f:
                    cp_data = pickle.load(f)
                print(f"      Strategy: {cp_data.get('strategy_name', 'unknown')}")
                print(f"      Episode: {cp_data.get('episode', 0)}")
            except Exception as e:
                print(f"      Error loading: {e}")
    else:
        print("✗ Checkpoint directory not created")
    
    # Check for errors
    if result.returncode != 0:
        print(f"\n✗ Script exited with error code {result.returncode}")
        if result.stderr:
            print("Error output:")
            print(result.stderr[:500])
    else:
        print("\n✓ Script completed successfully")
    
    # Clean up
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*"):
            f.unlink()
        checkpoint_dir.rmdir()
        print("\nCleaned up test files")
    
    return result.returncode == 0

if __name__ == "__main__":
    success = test_checkpoint_integration()
    sys.exit(0 if success else 1)