#!/usr/bin/env python3
"""
Test Curriculum Learning without DQN
=====================================
Test run that explicitly skips DQN to avoid hanging.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Monkey patch to skip DQN strategy
import src.evaluation.evaluation_engine as eval_engine

original_init = eval_engine.EvaluationEngine._initialize_strategies

def patched_init(self):
    """Initialize strategies but skip DQN."""
    original_init(self)
    # Remove DQN strategy if it exists
    if 'q_learning_dqn' in self.strategies:
        print("⚠️ Removing q_learning_dqn strategy to prevent hanging")
        del self.strategies['q_learning_dqn']
    print(f"✅ Active strategies: {list(self.strategies.keys())}")

eval_engine.EvaluationEngine._initialize_strategies = patched_init

# Now import and run the baseline comparison
from tests.dissertation_test_suite.scripts.run_baseline_comparison import main as run_baseline

def run_test():
    """Run a quick 50-episode test without DQN."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tests/dissertation_test_suite/results/test_no_dqn_{timestamp}"
    
    print("="*70)
    print("TESTING WITHOUT DQN - 50 EPISODES")
    print("="*70)
    print(f"Output: {output_dir}")
    print()
    
    # Prepare arguments
    sys.argv = [
        "run_baseline_comparison.py",
        "--query-set", "quick_test",
        "--episodes", "50",
        "--runs", "1",
        "--output-dir", output_dir,
        "--checkpoint-interval", "25"
    ]
    
    try:
        # Run with timeout monitoring
        start_time = time.time()
        
        # This will run the baseline comparison directly
        run_baseline()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        
        # Check results
        if os.path.exists(output_dir):
            result_files = []
            for root, dirs, files in os.walk(output_dir):
                result_files.extend([f for f in files if f.endswith('.json')])
            
            print(f"\n📊 Results generated: {len(result_files)} files")
            
            # Count by strategy
            strategies = {}
            for f in result_files:
                if '_run' in f:
                    strategy = f.split('_run')[0]
                    strategies[strategy] = strategies.get(strategy, 0) + 1
            
            print("\nStrategies tested:")
            for strategy, count in sorted(strategies.items()):
                print(f"  - {strategy}: {count} runs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_test())