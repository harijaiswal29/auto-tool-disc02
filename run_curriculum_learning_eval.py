#!/usr/bin/env python3
"""
Curriculum Learning Evaluation Script
=====================================
Implements progressive training from simple to complex queries for better convergence.

Training Phases:
1. Episodes 0-1000: Simple queries only (single tool, basic intents)
2. Episodes 1000-3000: Mix of simple and complex queries
3. Episodes 3000+: Full mixed queries including most challenging cases
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Configuration
TOTAL_EPISODES = 50000  # Extended training for full convergence
CHECKPOINT_INTERVAL = 1000  # Save every 1000 episodes

# Curriculum stages
CURRICULUM_STAGES = [
    {
        "name": "Stage 1: Simple Queries",
        "episodes": (0, 1000),
        "query_set": "simple_only",
        "description": "Single tool, basic intents - building foundation"
    },
    {
        "name": "Stage 2: Mixed Simple/Complex",
        "episodes": (1000, 3000),
        "query_set": "mixed_simple_complex",
        "description": "Gradual introduction of multi-tool queries"
    },
    {
        "name": "Stage 3: Full Complexity",
        "episodes": (3000, TOTAL_EPISODES),
        "query_set": "dissertation_core",
        "description": "Full range of queries including challenging cases"
    }
]

# Create unique timestamped directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"curriculum_{TIMESTAMP}"
BASE_RESULTS_DIR = "tests/dissertation_test_suite/results"
OUTPUT_DIR = f"{BASE_RESULTS_DIR}/{RUN_ID}"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

def setup_directories():
    """Create necessary directories for results."""
    print(f"\n{'='*60}")
    print("Setting up curriculum learning directories...")
    print(f"{'='*60}")
    
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/stage_results", exist_ok=True)
    
    print(f"✅ Created curriculum learning directory: {OUTPUT_DIR}")
    print(f"✅ Run ID: {RUN_ID}")
    
    # Save curriculum configuration
    config = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "total_episodes": TOTAL_EPISODES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "curriculum_stages": CURRICULUM_STAGES,
        "output_dir": OUTPUT_DIR
    }
    
    with open(f"{OUTPUT_DIR}/curriculum_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved curriculum configuration")

def get_stage_for_episode(episode):
    """Determine which curriculum stage an episode belongs to."""
    for stage in CURRICULUM_STAGES:
        start, end = stage["episodes"]
        if start <= episode < end:
            return stage
    return CURRICULUM_STAGES[-1]  # Default to last stage

def create_mixed_query_set():
    """Create a mixed query set file for Stage 2."""
    print(f"\n{'='*60}")
    print("Creating mixed query set for Stage 2...")
    print(f"{'='*60}")
    
    # Create a Python script to generate mixed queries
    mixed_queries_script = f"{OUTPUT_DIR}/generate_mixed_queries.py"
    
    script_content = '''#!/usr/bin/env python3
"""Generate mixed query set for curriculum learning Stage 2."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.dissertation_test_suite.data.test_queries import SIMPLE_QUERIES, COMPLEX_QUERIES

# Mix 70% simple, 30% complex for gradual difficulty increase
def create_mixed_queries():
    mixed = []
    
    # Add all simple queries
    for query in SIMPLE_QUERIES[:7]:  # 70% representation
        mixed.append(query)
    
    # Add some complex queries
    for query in COMPLEX_QUERIES[:3]:  # 30% representation
        mixed.append(query)
    
    return mixed

MIXED_SIMPLE_COMPLEX = create_mixed_queries()

if __name__ == "__main__":
    print(f"Created mixed query set with {len(MIXED_SIMPLE_COMPLEX)} queries")
    print(f"Simple: {sum(1 for q in MIXED_SIMPLE_COMPLEX if q.get('complexity', 'simple') == 'simple')}")
    print(f"Complex: {sum(1 for q in MIXED_SIMPLE_COMPLEX if q.get('complexity', 'simple') == 'complex')}")
'''
    
    with open(mixed_queries_script, "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(mixed_queries_script, 0o755)
    print(f"✅ Created mixed query set generator")

def run_curriculum_stage(stage_info, start_episode, end_episode):
    """Run a single curriculum stage."""
    stage_name = stage_info["name"]
    query_set = stage_info["query_set"]
    
    print(f"\n{'='*60}")
    print(f"Running {stage_name}")
    print(f"Episodes: {start_episode} - {end_episode}")
    print(f"Query Set: {query_set}")
    print(f"{'='*60}")
    
    # Prepare output directory for this stage
    stage_output = f"{OUTPUT_DIR}/stage_results/{query_set}_{start_episode}_{end_episode}"
    
    # Build command
    cmd = [
        "python",
        "tests/dissertation_test_suite/scripts/run_baseline_comparison.py",
        "--query-set", query_set,
        "--episodes", str(end_episode - start_episode),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--output-dir", stage_output
    ]
    
    # Add resume flag if continuing from checkpoint
    if start_episode > 0:
        checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_episode_{start_episode}.pkl"
        if os.path.exists(checkpoint_file):
            cmd.extend(["--resume-from", checkpoint_file])
            print(f"📁 Resuming from checkpoint: {checkpoint_file}")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run with real-time output
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=7200  # 2 hours timeout per stage
        )
        
        if result.returncode == 0:
            print(f"✅ Completed {stage_name}")
            return True
        else:
            print(f"❌ Failed {stage_name}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout for {stage_name}")
        return False
    except Exception as e:
        print(f"❌ Error in {stage_name}: {e}")
        return False

def analyze_curriculum_progress():
    """Analyze learning progress across curriculum stages."""
    print(f"\n{'='*60}")
    print("Analyzing Curriculum Learning Progress...")
    print(f"{'='*60}")
    
    analysis = {
        "timestamp": TIMESTAMP,
        "stages": [],
        "overall_metrics": {}
    }
    
    # Analyze each stage's results
    for stage in CURRICULUM_STAGES:
        stage_name = stage["name"]
        start, end = stage["episodes"]
        query_set = stage["query_set"]
        
        # Find result files for this stage
        stage_pattern = f"{OUTPUT_DIR}/stage_results/{query_set}_{start}_{end}"
        stage_results_file = f"{stage_pattern}/baseline_comparison_final*.json"
        
        from pathlib import Path
        result_files = list(Path(OUTPUT_DIR).glob(f"stage_results/{query_set}*/*final*.json"))
        
        if result_files:
            with open(result_files[0]) as f:
                data = json.load(f)
            
            # Extract key metrics
            if "aggregated_results" in data:
                q_learning_metrics = data["aggregated_results"].get("q_learning_tabular", {})
                dqn_metrics = data["aggregated_results"].get("q_learning_dqn", {})
                
                stage_analysis = {
                    "stage": stage_name,
                    "episodes": f"{start}-{end}",
                    "q_learning_completion": q_learning_metrics.get("completion_rate", 0),
                    "dqn_completion": dqn_metrics.get("completion_rate", 0),
                    "q_learning_reward": q_learning_metrics.get("average_reward", 0),
                    "dqn_reward": dqn_metrics.get("average_reward", 0)
                }
                
                analysis["stages"].append(stage_analysis)
                
                print(f"\n{stage_name}:")
                print(f"  Q-Learning: {stage_analysis['q_learning_completion']:.1f}% completion")
                print(f"  DQN: {stage_analysis['dqn_completion']:.1f}% completion")
    
    # Calculate overall improvement
    if len(analysis["stages"]) > 0:
        first_stage = analysis["stages"][0]
        last_stage = analysis["stages"][-1]
        
        q_improvement = last_stage["q_learning_completion"] - first_stage["q_learning_completion"]
        dqn_improvement = last_stage["dqn_completion"] - first_stage["dqn_completion"]
        
        analysis["overall_metrics"] = {
            "q_learning_improvement": q_improvement,
            "dqn_improvement": dqn_improvement,
            "curriculum_effective": q_improvement > 20 or dqn_improvement > 20
        }
        
        print(f"\n📊 Overall Curriculum Impact:")
        print(f"  Q-Learning improvement: {q_improvement:.1f}%")
        print(f"  DQN improvement: {dqn_improvement:.1f}%")
        print(f"  Curriculum Effective: {'✅ Yes' if analysis['overall_metrics']['curriculum_effective'] else '❌ No'}")
    
    # Save analysis
    with open(f"{OUTPUT_DIR}/curriculum_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    return analysis

def generate_summary_report():
    """Generate comprehensive curriculum learning report."""
    print(f"\n{'='*60}")
    print("Generating Curriculum Learning Report...")
    print(f"{'='*60}")
    
    report = f"""======================================================================
CURRICULUM LEARNING EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
======================================================================

RUN INFORMATION
----------------------------------------
Run ID: {RUN_ID}
Total Episodes: {TOTAL_EPISODES}
Output Directory: {OUTPUT_DIR}

CURRICULUM DESIGN
----------------------------------------
Stage 1 (0-1000): Simple queries only
  - Single tool operations
  - Basic intent recognition
  - Foundation building

Stage 2 (1000-3000): Mixed complexity
  - 70% simple, 30% complex queries
  - Gradual difficulty increase
  - Multi-tool introduction

Stage 3 (3000+): Full complexity
  - All query types
  - Complex multi-tool scenarios
  - Real-world challenges

EXPECTED BENEFITS
----------------------------------------
1. Faster initial learning on simple tasks
2. Better foundation for complex tasks
3. Reduced catastrophic forgetting
4. Higher final performance
5. More stable convergence

HYPOTHESIS
----------------------------------------
Curriculum learning will achieve:
- >70% completion on simple queries by episode 1000
- >80% completion on mixed queries by episode 3000
- >85% completion on full queries by episode 10000
- 40%+ improvement over random baseline

======================================================================
"""
    
    # Save report
    report_file = f"{OUTPUT_DIR}/curriculum_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Report saved to: {report_file}")

def main():
    """Main execution function for curriculum learning."""
    print("="*70)
    print("CURRICULUM LEARNING EVALUATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Total Episodes: {TOTAL_EPISODES}")
    print(f"  - Curriculum Stages: {len(CURRICULUM_STAGES)}")
    print(f"  - Output Directory: {OUTPUT_DIR}")
    
    # Setup
    setup_directories()
    create_mixed_query_set()
    
    # Run curriculum stages
    print(f"\n{'='*70}")
    print("STARTING CURRICULUM LEARNING")
    print(f"{'='*70}")
    
    for stage in CURRICULUM_STAGES:
        start, end = stage["episodes"]
        
        # Skip if beyond total episodes
        if start >= TOTAL_EPISODES:
            break
        
        # Adjust end if needed
        actual_end = min(end, TOTAL_EPISODES)
        
        success = run_curriculum_stage(stage, start, actual_end)
        
        if not success:
            print(f"⚠️ Stage failed, but continuing...")
        
        # Brief pause between stages
        time.sleep(5)
    
    # Analyze results
    analysis = analyze_curriculum_progress()
    
    # Generate report
    generate_summary_report()
    
    print(f"\n{'='*70}")
    print("✅ CURRICULUM LEARNING COMPLETED")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()