#!/usr/bin/env python3
"""
Test Curriculum Learning with 50 Episodes
==========================================
Quick test to verify the evaluation pipeline works.
"""

import os
import sys
import subprocess
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)
for logger_name in ['src', 'learning', 'evaluation', 'PatternMiner', '__main__']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Configuration - VERY SMALL for testing
TOTAL_EPISODES = 50  # Just 50 episodes total
CHECKPOINT_INTERVAL = 25  # Checkpoint halfway

# Single stage test
CURRICULUM_STAGES = [
    {
        "name": "Test Stage",
        "episodes": (0, 50),
        "query_set": "quick_test",  # Use quick_test set
        "description": "Testing with 50 episodes"
    }
]

# Create timestamped directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"curriculum_test50_{TIMESTAMP}"
BASE_RESULTS_DIR = "tests/dissertation_test_suite/results"
OUTPUT_DIR = f"{BASE_RESULTS_DIR}/{RUN_ID}"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

def setup_directories():
    """Create necessary directories."""
    print(f"\n{'='*60}")
    print("Setting up test directories...")
    print(f"{'='*60}")
    
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/stage_results", exist_ok=True)
    
    print(f"✅ Created directory: {OUTPUT_DIR}")
    print(f"✅ Run ID: {RUN_ID}")
    
    # Save configuration
    config = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "total_episodes": TOTAL_EPISODES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "curriculum_stages": CURRICULUM_STAGES,
        "output_dir": OUTPUT_DIR,
        "test_mode": True
    }
    
    with open(f"{OUTPUT_DIR}/curriculum_config.json", "w") as f:
        json.dump(config, f, indent=2)

def run_test_stage():
    """Run the test stage with timeout protection."""
    stage_info = CURRICULUM_STAGES[0]
    stage_name = stage_info["name"]
    query_set = stage_info["query_set"]
    
    print(f"\n{'='*60}")
    print(f"Running {stage_name}")
    print(f"Episodes: 0 - {TOTAL_EPISODES}")
    print(f"Query Set: {query_set}")
    print(f"{'='*60}")
    
    stage_output = f"{OUTPUT_DIR}/stage_results/{query_set}_0_{TOTAL_EPISODES}"
    
    # Environment to suppress logs
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    env['LOG_LEVEL'] = 'WARNING'
    # Disable DQN to avoid initialization issues
    env['DISABLE_DQN'] = '1'
    
    # Build command
    cmd = [
        "python", "-u",
        "tests/dissertation_test_suite/scripts/run_baseline_comparison.py",
        "--query-set", query_set,
        "--episodes", str(TOTAL_EPISODES),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--output-dir", stage_output,
        "--runs", "1"  # Just 1 run per strategy for speed
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"\nStarting test execution (timeout: 5 minutes)...")
    
    try:
        # Run with strict timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Monitor with timeout
        start_time = time.time()
        timeout_seconds = 300  # 5 minutes max
        
        lines_printed = 0
        max_lines = 30
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"\n⚠️ Timeout reached ({timeout_seconds}s) - terminating")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                return False
            
            # Check if process finished
            if process.poll() is not None:
                break
                
            # Read output (non-blocking)
            try:
                line = process.stdout.readline()
                if line and lines_printed < max_lines:
                    # Only print key information
                    if any(word in line.lower() for word in 
                           ['strategy', 'episode', 'complete', 'error', 'dqn', 'tabular']):
                        print(f"  {line.strip()}")
                        lines_printed += 1
            except:
                pass
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"\n✅ Test completed successfully")
            return True
        else:
            stderr = process.stderr.read()
            print(f"\n❌ Test failed with return code {return_code}")
            if stderr:
                print(f"Error output: {stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception during test: {str(e)[:200]}")
        return False

def check_results():
    """Check what results were generated."""
    print(f"\n{'='*60}")
    print("Checking Results...")
    print(f"{'='*60}")
    
    # Count result files
    result_dir = f"{OUTPUT_DIR}/stage_results"
    if os.path.exists(result_dir):
        for subdir in os.listdir(result_dir):
            subdir_path = os.path.join(result_dir, subdir)
            if os.path.isdir(subdir_path):
                json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                
                # Count by strategy
                strategies = {}
                for f in json_files:
                    strategy = f.split('_run')[0]
                    strategies[strategy] = strategies.get(strategy, 0) + 1
                
                print(f"\nResults in {subdir}:")
                for strategy, count in sorted(strategies.items()):
                    print(f"  {strategy}: {count} runs")
                    
                # Check for DQN specifically
                if 'q_learning_dqn' not in strategies:
                    print("  ⚠️ Note: DQN strategy not found (may have been skipped)")
    
    # Check checkpoint files
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = os.listdir(CHECKPOINT_DIR)
        print(f"\nCheckpoints created: {len(checkpoints)}")

def main():
    """Main test execution."""
    print("="*70)
    print("CURRICULUM LEARNING TEST - 50 EPISODES")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Total Episodes: {TOTAL_EPISODES}")
    print(f"  - Query Set: quick_test")
    print(f"  - Timeout: 5 minutes")
    print(f"  - Output: {OUTPUT_DIR}")
    
    # Setup
    setup_directories()
    
    # Run test
    success = run_test_stage()
    
    # Check results
    check_results()
    
    # Summary
    print(f"\n{'='*70}")
    if success:
        print("✅ TEST COMPLETED SUCCESSFULLY")
    else:
        print("⚠️ TEST COMPLETED WITH ISSUES")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())