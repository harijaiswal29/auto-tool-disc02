#!/bin/bash
# Quick test of checkpoint functionality

echo "Testing checkpoint functionality with 4 episodes, checkpoint every 2..."
python scripts/run_baseline_comparison.py \
    --query-set quick_test \
    --episodes 4 \
    --runs 1 \
    --checkpoint-interval 2 \
    --checkpoint-dir test_cp_integration 2>&1 | \
    grep -E "(Checkpointing enabled|Checkpoint saved|Episode|strategy:)" | head -30

echo ""
echo "Checking created checkpoint files..."
ls -la test_cp_integration/*.pkl 2>/dev/null || echo "No checkpoint files found"

# Clean up
rm -rf test_cp_integration