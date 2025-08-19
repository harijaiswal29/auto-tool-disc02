#!/usr/bin/env python3
"""
Optimized Curriculum Learning Evaluation Script
===============================================
Implements progressive training with reduced logging overhead.

Key optimizations:
- Reduced logging verbosity
- Progress reporting only at intervals
- Buffered output to prevent terminal overflow
- Timeout handling for long-running processes
"""

import os
import sys
import subprocess
import json
import time
import logging
import argparse
import pickle
import glob
import re
import shutil
from datetime import datetime
from pathlib import Path

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)
for logger_name in ['src', 'learning', 'evaluation', 'PatternMiner', '__main__']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Optimized Curriculum Learning Evaluation with Checkpoint Resume')
parser.add_argument('--episodes', type=int, default=50, 
                    help='Total episodes to run (default: 50)')
parser.add_argument('--resume', choices=['none', 'latest', 'checkpoint'], 
                    default='none', help='Resume mode: none, latest, or checkpoint')
parser.add_argument('--checkpoint-path', type=str, 
                    help='Specific checkpoint file to resume from (required with --resume checkpoint)')
parser.add_argument('--resume-dir', type=str, 
                    help='Directory containing checkpoints to resume from')
parser.add_argument('--checkpoint-interval', type=int, default=10,
                    help='Save checkpoint every N episodes (default: 10)')
parser.add_argument('--output-dir', type=str, 
                    help='Custom output directory (default: auto-generated)')
parser.add_argument('--log-interval', type=int, default=5,
                    help='Report progress every N episodes (default: 5)')

# Parse arguments
args = parser.parse_args()

# Configuration from arguments
TOTAL_EPISODES = args.episodes
CHECKPOINT_INTERVAL = args.checkpoint_interval
LOG_INTERVAL = args.log_interval

# Curriculum stages (adjusted for 50 episodes)
CURRICULUM_STAGES = [
    {
        "name": "Stage 1: Simple Queries",
        "episodes": (0, 15),
        "query_set": "simple_only",
        "description": "Foundation building with single tool operations"
    },
    {
        "name": "Stage 2: Mixed Complexity",
        "episodes": (15, 30),
        "query_set": "quick_test",  # Use quick_test as a proxy for mixed
        "description": "Gradual difficulty increase"
    },
    {
        "name": "Stage 3: Full Complexity",
        "episodes": (30, TOTAL_EPISODES),
        "query_set": "dissertation_core",
        "description": "All query types including challenging cases"
    }
]

# These will be set dynamically based on resume mode
TIMESTAMP = None
RUN_ID = None
BASE_RESULTS_DIR = "tests/dissertation_test_suite/results"
OUTPUT_DIR = None
CHECKPOINT_DIR = None
START_EPISODE = 0
RESUME_CHECKPOINT = None

# Resume logic functions
def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*.pkl")
    if not checkpoints:
        return None, 0
    
    # Extract episode numbers and find max
    episode_checkpoints = []
    for cp in checkpoints:
        # Match both regular checkpoints and final checkpoints
        match = re.search(r'ep(\d+)', cp)
        if match:
            episode_checkpoints.append((int(match.group(1)), cp))
    
    if not episode_checkpoints:
        return None, 0
    
    episode, checkpoint = max(episode_checkpoints, key=lambda x: x[0])
    return checkpoint, episode

def find_last_run_dir():
    """Find the most recent run directory."""
    pattern = f"{BASE_RESULTS_DIR}/curriculum_opt_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs)  # Most recent by timestamp in name

def load_checkpoint_state(checkpoint_path):
    """Load state from checkpoint file."""
    try:
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        return state
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        return None

def determine_resume_point(args):
    """Determine where to resume from based on arguments."""
    if args.resume == 'none':
        return 0, None, None
    
    elif args.resume == 'latest':
        # Find latest checkpoint in resume_dir or last run
        search_dir = args.resume_dir or find_last_run_dir()
        if not search_dir:
            print("⚠️ No previous runs found to resume from")
            return 0, None, None
        
        checkpoint, episode = find_latest_checkpoint(f"{search_dir}/checkpoints")
        if checkpoint:
            print(f"📂 Found checkpoint at episode {episode} in {search_dir}")
        return episode, checkpoint, search_dir
    
    elif args.resume == 'checkpoint':
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path required when using --resume checkpoint")
        
        if not os.path.exists(args.checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {args.checkpoint_path}")
        
        match = re.search(r'ep(\d+)', args.checkpoint_path)
        if not match:
            raise ValueError(f"Cannot extract episode number from checkpoint path: {args.checkpoint_path}")
        
        episode = int(match.group(1))
        resume_dir = str(Path(args.checkpoint_path).parent.parent)  # Go up to run directory
        return episode, args.checkpoint_path, resume_dir
    
    return 0, None, None

def copy_previous_results(source_dir, target_dir, up_to_episode):
    """Copy results from previous run up to resume point."""
    print(f"📋 Copying previous results up to episode {up_to_episode}")
    
    # Copy stage results for completed stages
    source_stage_dir = f"{source_dir}/stage_results"
    target_stage_dir = f"{target_dir}/stage_results"
    
    if os.path.exists(source_stage_dir):
        os.makedirs(target_stage_dir, exist_ok=True)
        
        for stage_result in os.listdir(source_stage_dir):
            # Parse stage episode range from directory name
            match = re.search(r'_(\d+)_(\d+)$', stage_result)
            if match:
                stage_start = int(match.group(1))
                stage_end = int(match.group(2))
                
                # Copy if stage is completely before resume point
                if stage_end <= up_to_episode:
                    source_path = os.path.join(source_stage_dir, stage_result)
                    target_path = os.path.join(target_stage_dir, stage_result)
                    print(f"  📁 Copying stage results: {stage_result}")
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    
    # Copy relevant checkpoints
    source_checkpoint_dir = f"{source_dir}/checkpoints"
    target_checkpoint_dir = f"{target_dir}/checkpoints"
    
    if os.path.exists(source_checkpoint_dir):
        os.makedirs(target_checkpoint_dir, exist_ok=True)
        
        for checkpoint in os.listdir(source_checkpoint_dir):
            if checkpoint.endswith('.pkl'):
                match = re.search(r'ep(\d+)', checkpoint)
                if match and int(match.group(1)) <= up_to_episode:
                    source_path = os.path.join(source_checkpoint_dir, checkpoint)
                    target_path = os.path.join(target_checkpoint_dir, checkpoint)
                    print(f"  💾 Copying checkpoint: {checkpoint}")
                    shutil.copy2(source_path, target_path)

def adjust_curriculum_stages(stages, start_episode, total_episodes):
    """Adjust stages based on resume point."""
    adjusted = []
    
    for stage in stages:
        stage_start, stage_end = stage["episodes"]
        
        # Skip completed stages
        if stage_end <= start_episode:
            print(f"  ⏭️ Skipping completed stage: {stage['name']} (episodes {stage_start}-{stage_end})")
            continue
        
        # Adjust stage to start from resume point
        adjusted_start = max(stage_start, start_episode)
        adjusted_end = min(stage_end, total_episodes)
        
        if adjusted_start < adjusted_end:
            adjusted_stage = stage.copy()
            adjusted_stage["episodes"] = (adjusted_start, adjusted_end)
            adjusted_stage["original_episodes"] = (stage_start, stage_end)
            adjusted.append(adjusted_stage)
            
            if adjusted_start > stage_start:
                print(f"  📝 Adjusting {stage['name']}: episodes {adjusted_start}-{adjusted_end} (was {stage_start}-{stage_end})")
            else:
                print(f"  ✅ Including {stage['name']}: episodes {adjusted_start}-{adjusted_end}")
    
    return adjusted

def setup_directories():
    """Create necessary directories."""
    print(f"\n{'='*60}")
    print("Setting up curriculum learning directories...")
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
        "optimizations": {
            "reduced_logging": True,
            "log_interval": LOG_INTERVAL,
            "buffered_output": True
        }
    }
    
    with open(f"{OUTPUT_DIR}/curriculum_config.json", "w") as f:
        json.dump(config, f, indent=2)

def create_optimized_runner():
    """Create an optimized baseline comparison runner with minimal logging."""
    runner_script = f"{OUTPUT_DIR}/optimized_runner.py"
    
    script_content = '''#!/usr/bin/env python3
"""Optimized runner with reduced logging."""

import sys
import logging
import os
from pathlib import Path

# Suppress all but critical logs
logging.basicConfig(level=logging.CRITICAL)
for module in ['src', 'learning', 'evaluation', 'agents', 'core', 'tools']:
    logging.getLogger(module).setLevel(logging.CRITICAL)

# Reduce TensorFlow/PyTorch logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and run with minimal output
from tests.dissertation_test_suite.scripts.run_baseline_comparison import main

if __name__ == "__main__":
    # Override logging in main module
    import tests.dissertation_test_suite.scripts.run_baseline_comparison as runner
    runner.logger.setLevel(logging.WARNING)
    
    # Run with progress reporting only
    sys.argv.extend(['--quiet', '--progress-interval', '100'])
    main()
'''
    
    with open(runner_script, "w") as f:
        f.write(script_content)
    
    os.chmod(runner_script, 0o755)
    return runner_script

def run_curriculum_stage(stage_info, start_episode, end_episode):
    """Run a single curriculum stage with optimized logging."""
    stage_name = stage_info["name"]
    query_set = stage_info["query_set"]
    
    print(f"\n{'='*60}")
    print(f"Running {stage_name}")
    print(f"Episodes: {start_episode} - {end_episode}")
    print(f"Query Set: {query_set}")
    print(f"{'='*60}")
    
    stage_output = f"{OUTPUT_DIR}/stage_results/{query_set}_{start_episode}_{end_episode}"
    
    # Use environment variables to control logging
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    env['LOG_LEVEL'] = 'WARNING'
    
    # Check if script exists
    script_path = "tests/dissertation_test_suite/scripts/run_baseline_comparison.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Build command without unsupported flags
    cmd = [
        sys.executable, "-u",  # Use same Python interpreter, unbuffered output
        script_path,
        "--query-set", query_set,
        "--episodes", str(end_episode - start_episode),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--output-dir", stage_output
    ]
    
    # Add resume flag if continuing
    # Check for global resume checkpoint first (from command line)
    if RESUME_CHECKPOINT and start_episode == START_EPISODE:
        cmd.extend(["--resume-from", RESUME_CHECKPOINT])
        print(f"📁 Resuming from specified checkpoint: {RESUME_CHECKPOINT}")
    elif start_episode > 0:
        # Check for checkpoint from previous stage in this run
        checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_episode_{start_episode}.pkl"
        if os.path.exists(checkpoint_file):
            cmd.extend(["--resume-from", checkpoint_file])
            print(f"📁 Resuming from checkpoint: {checkpoint_file}")
    
    print(f"Starting stage execution...")
    
    try:
        # Run with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,
            env=env
        )
        
        # Monitor progress without flooding output
        lines_printed = 0
        max_lines = 100  # Increase output lines for better debugging
        important_lines = []
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                # Always capture important lines for debugging
                if any(keyword in line.lower() for keyword in 
                       ['error', 'exception', 'traceback', 'failed']):
                    important_lines.append(line.strip())
                
                # Print progress lines
                if any(keyword in line.lower() for keyword in 
                       ['episode', 'checkpoint', 'completed', 'error', 'failed', '%', 'strategy']):
                    if lines_printed < max_lines:
                        print(f"  {line.strip()}")
                        lines_printed += 1
                    elif lines_printed == max_lines:
                        print("  [Output truncated for brevity...]")
                        lines_printed += 1
        
        # Wait for completion
        return_code = process.poll()
        
        if return_code == 0:
            print(f"✅ Completed {stage_name}")
            return True
        else:
            print(f"❌ Failed {stage_name} with return code: {return_code}")
            if important_lines:
                print("  Error details:")
                for line in important_lines[-10:]:  # Show last 10 error lines
                    print(f"    {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout for {stage_name} - killing process")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Error in {stage_name}: {str(e)[:200]}")
        return False

def generate_summary():
    """Generate a concise summary of results."""
    print(f"\n{'='*60}")
    print("CURRICULUM LEARNING SUMMARY")
    print(f"{'='*60}")
    print(f"Run ID: {RUN_ID}")
    print(f"Total Episodes: {TOTAL_EPISODES}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Check for result files
    from pathlib import Path
    result_files = list(Path(OUTPUT_DIR).glob("stage_results/*/*final*.json"))
    
    if result_files:
        print(f"\n✅ Found {len(result_files)} result files")
        
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                if "aggregated_results" in data:
                    print(f"\n📊 {result_file.parent.name}:")
                    for strategy, metrics in data["aggregated_results"].items():
                        if "completion_rate" in metrics:
                            print(f"  {strategy}: {metrics['completion_rate']:.1f}% completion")
            except:
                pass
    else:
        print("\n⚠️ No result files found yet")
    
    print(f"\n{'='*60}")

def main():
    """Main execution with optimized logging and resume support."""
    global TIMESTAMP, RUN_ID, OUTPUT_DIR, CHECKPOINT_DIR, START_EPISODE, RESUME_CHECKPOINT
    
    print("="*70)
    print("OPTIMIZED CURRICULUM LEARNING EVALUATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine resume point
    START_EPISODE, RESUME_CHECKPOINT, resume_dir = determine_resume_point(args)
    
    # Set up directories based on resume mode
    if START_EPISODE > 0 and resume_dir:
        # Resuming from checkpoint
        print(f"\n📂 RESUME MODE ACTIVATED")
        print(f"  - Resuming from episode: {START_EPISODE}")
        print(f"  - Checkpoint: {RESUME_CHECKPOINT}")
        print(f"  - Previous run: {resume_dir}")
        
        # Use new timestamp for resumed run
        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_ID = f"curriculum_opt_{TIMESTAMP}_resumed_ep{START_EPISODE}"
    else:
        # Fresh start
        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_ID = f"curriculum_opt_{TIMESTAMP}"
    
    # Set output directory (can be overridden by args)
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = f"{BASE_RESULTS_DIR}/{RUN_ID}"
    
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    
    print(f"\nConfiguration:")
    print(f"  - Total Episodes: {TOTAL_EPISODES}")
    print(f"  - Starting Episode: {START_EPISODE}")
    print(f"  - Episodes to Run: {TOTAL_EPISODES - START_EPISODE}")
    print(f"  - Checkpoint Interval: {CHECKPOINT_INTERVAL}")
    print(f"  - Output Directory: {OUTPUT_DIR}")
    print(f"  - Resume Mode: {args.resume}")
    
    # Setup directories
    setup_directories()
    
    # Copy previous results if resuming
    if START_EPISODE > 0 and resume_dir and resume_dir != OUTPUT_DIR:
        copy_previous_results(resume_dir, OUTPUT_DIR, START_EPISODE)
    
    # Adjust curriculum stages based on resume point
    adjusted_stages = adjust_curriculum_stages(CURRICULUM_STAGES, START_EPISODE, TOTAL_EPISODES)
    
    if not adjusted_stages:
        print(f"\n✅ All stages already completed up to episode {START_EPISODE}")
        print(f"Nothing to do for episodes {START_EPISODE}-{TOTAL_EPISODES}")
        return
    
    print(f"\nStages to run: {len(adjusted_stages)}")
    print(f"  - Reduced Logging: ENABLED")
    
    # Run curriculum stages
    print(f"\n{'='*70}")
    print("STARTING CURRICULUM LEARNING")
    print(f"{'='*70}")
    
    for stage in adjusted_stages:
        start, end = stage["episodes"]
        
        if start >= TOTAL_EPISODES:
            break
        
        actual_end = min(end, TOTAL_EPISODES)
        
        success = run_curriculum_stage(stage, start, actual_end)
        
        if not success:
            print(f"⚠️ Stage failed, continuing anyway...")
        
        # Brief pause
        time.sleep(2)
    
    # Generate summary
    generate_summary()
    
    print(f"\n{'='*70}")
    print("✅ CURRICULUM LEARNING COMPLETED")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()