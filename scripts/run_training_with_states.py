#!/usr/bin/env python3
"""Run training with state vector saving enabled for encoder training data collection."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

def update_config_for_state_saving():
    """Update config to enable state vector saving."""
    config_path = Path("config/config.json")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Enable state vector saving
    config['evaluation']['save_state_vectors'] = True
    config['evaluation']['state_sampling_rate'] = 1  # Save all states
    config['evaluation']['max_states_per_checkpoint'] = 10000
    
    # Create backup
    backup_path = config_path.with_suffix('.json.bak')
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config updated to enable state vector saving")
    print(f"   - save_state_vectors: True")
    print(f"   - state_sampling_rate: 1")
    print(f"   - max_states_per_checkpoint: 10000")
    
    return config_path, backup_path


def restore_config(config_path, backup_path):
    """Restore original config from backup."""
    if backup_path.exists():
        # Restore from backup
        with open(backup_path, 'r') as f:
            config = json.load(f)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        backup_path.unlink()
        print(f"✅ Config restored from backup")


def run_training(episodes=100, checkpoint_interval=10):
    """Run training with checkpoint saving."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tests/dissertation_test_suite/results/training_with_states_{timestamp}"
    checkpoint_dir = f"{output_dir}/checkpoints"
    
    cmd = [
        "python", "tests/dissertation_test_suite/scripts/run_baseline_comparison.py",
        "--episodes", str(episodes),
        "--checkpoint-interval", str(checkpoint_interval),
        "--checkpoint-dir", checkpoint_dir,
        "--output-dir", output_dir,
        "--query-set", "quick_test",  # Use quick test for faster training
        "--enable-retries"  # Enable for more robust execution
    ]
    
    print(f"\n📊 Running training with state saving...")
    print(f"   Episodes: {episodes}")
    print(f"   Checkpoint interval: {checkpoint_interval}")
    print(f"   Output directory: {output_dir}")
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            
            # Check for checkpoint files
            checkpoint_dir = Path(output_dir) / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pkl"))
                print(f"\n📁 Found {len(checkpoints)} checkpoint files:")
                for cp in checkpoints[:5]:  # Show first 5
                    print(f"   - {cp.name}")
                if len(checkpoints) > 5:
                    print(f"   ... and {len(checkpoints) - 5} more")
                
                return output_dir
            else:
                print("⚠️ No checkpoint directory found")
                return None
        else:
            print(f"❌ Training failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return None


def verify_state_vectors(output_dir):
    """Verify that state vectors were saved in checkpoints."""
    if not output_dir:
        return False
    
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        print("❌ No checkpoint directory found")
        return False
    
    import pickle
    
    # Check a checkpoint file
    checkpoints = list(checkpoint_dir.glob("*.pkl"))
    if not checkpoints:
        print("❌ No checkpoint files found")
        return False
    
    # Load first checkpoint
    checkpoint_path = checkpoints[0]
    print(f"\n🔍 Verifying checkpoint: {checkpoint_path.name}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Check for state vectors
        if 'episode_states' in checkpoint:
            episode_states = checkpoint['episode_states']
            print(f"✅ Found episode_states with {len(episode_states)} entries")
            
            if episode_states:
                # Check first state
                first_state = episode_states[0]
                state_vector = first_state.get('state_vector')
                if state_vector is not None:
                    import numpy as np
                    state_shape = np.array(state_vector).shape
                    print(f"✅ State vector shape: {state_shape}")
                    print(f"✅ State vector has required fields:")
                    for key in ['query', 'success', 'reward', 'tools_selected']:
                        if key in first_state:
                            print(f"   - {key}: ✓")
                    return True
                else:
                    print("❌ No state_vector field in episode_states")
                    return False
            else:
                print("⚠️ episode_states is empty")
                return False
        else:
            print("❌ No episode_states in checkpoint")
            print(f"   Available keys: {list(checkpoint.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run training with state vector saving")
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to run (default: 50)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N episodes (default: 10)')
    parser.add_argument('--no-restore', action='store_true',
                       help="Don't restore config after training")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training with State Vector Saving")
    print("=" * 60)
    
    # Update config
    config_path, backup_path = update_config_for_state_saving()
    
    try:
        # Run training
        output_dir = run_training(args.episodes, args.checkpoint_interval)
        
        # Verify state vectors were saved
        if output_dir:
            success = verify_state_vectors(output_dir)
            if success:
                print("\n✅ State vectors successfully saved in checkpoints!")
                print(f"📁 Training data location: {output_dir}")
                return 0
            else:
                print("\n❌ State vectors were not properly saved")
                return 1
        else:
            print("\n❌ Training did not complete successfully")
            return 1
            
    finally:
        # Restore config unless --no-restore flag is set
        if not args.no_restore:
            restore_config(config_path, backup_path)
        else:
            print("\n⚠️ Config not restored (--no-restore flag set)")


if __name__ == "__main__":
    exit(main())