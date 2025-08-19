#!/bin/bash
# Monitor curriculum learning progress

RESULTS_DIR="tests/dissertation_test_suite/results/curriculum_opt_20250813_044755"

echo "=================================="
echo "CURRICULUM LEARNING MONITOR"
echo "=================================="
echo "Run ID: curriculum_opt_20250813_044755"
echo "Start Time: 04:47:55"
echo ""

# Check process status
if pgrep -f "curriculum_opt" > /dev/null; then
    echo "✅ Evaluation is RUNNING"
    echo ""
    
    # Show current stage
    CURRENT_STAGE=$(ps aux | grep baseline_comparison | grep -v grep | grep -oE "query-set [a-z_]+" | cut -d' ' -f2)
    if [ ! -z "$CURRENT_STAGE" ]; then
        echo "📊 Current Stage: $CURRENT_STAGE"
    fi
    
    # Check for checkpoints
    if [ -d "$RESULTS_DIR/checkpoints" ]; then
        CHECKPOINTS=$(ls -1 $RESULTS_DIR/checkpoints 2>/dev/null | wc -l)
        echo "💾 Checkpoints saved: $CHECKPOINTS"
    fi
    
    # Check for result files
    if [ -d "$RESULTS_DIR/stage_results" ]; then
        STAGES_COMPLETE=$(find $RESULTS_DIR/stage_results -name "*final*.json" 2>/dev/null | wc -l)
        echo "✅ Stages completed: $STAGES_COMPLETE / 3"
    fi
    
    # Show resource usage
    echo ""
    echo "📈 Resource Usage:"
    ps aux | grep -E "curriculum|baseline" | grep -v grep | awk '{printf "  CPU: %.1f%% MEM: %.1f%%\n", $3, $4}'
    
else
    echo "⚠️ Evaluation is NOT running"
    
    # Check if completed
    if [ -f "$RESULTS_DIR/curriculum_report.txt" ]; then
        echo "✅ Evaluation COMPLETED"
        echo ""
        echo "📄 Summary Report:"
        tail -20 "$RESULTS_DIR/curriculum_report.txt"
    fi
fi

echo ""
echo "Output Directory: $RESULTS_DIR"
echo "=================================="