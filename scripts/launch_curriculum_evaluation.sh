#!/bin/bash
# Launch Curriculum Learning Evaluation
# ======================================
# This script launches the full curriculum learning evaluation with mock servers
# Expected runtime: 6-8 hours for 50,000 episodes

echo "========================================================================"
echo "LAUNCHING CURRICULUM LEARNING EVALUATION"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Total Episodes: 50,000"
echo "  - Stages: 3 (Simple → Mixed → Complex)"
echo "  - Checkpoint Interval: 1,000 episodes"
echo "  - Mock Servers: ENABLED"
echo "  - All 7 Improvements: ACTIVE"
echo ""
echo "Hypotheses Being Tested:"
echo "  - H1:  Q-learning achieves >30% improvement over baselines"
echo "  - H1a: DQN outperforms Tabular Q-learning after 1000+ episodes"
echo "  - H1b: Both converge to >85% task completion rate"
echo "  - H2:  Intent recognition <100ms (p95)"
echo "  - H3:  Pattern mining discovers >50 patterns within 500 episodes"
echo "  - H5:  Convergence within 1000 episodes"
echo ""
echo "Starting in 5 seconds..."
echo "Press Ctrl+C to cancel"
sleep 5

# Start mock servers in background
echo ""
echo "Starting mock MCP servers..."
python start_mock_servers.py > mock_servers.log 2>&1 &
MOCK_PID=$!
echo "Mock servers started (PID: $MOCK_PID)"
sleep 2

# Launch curriculum learning evaluation
echo ""
echo "Starting curriculum learning evaluation..."
echo "Output will be saved to: tests/dissertation_test_suite/results/curriculum_[timestamp]"
echo ""
echo "========================================================================"
echo ""

# Run the evaluation
python run_curriculum_learning_eval.py

# Clean up
echo ""
echo "========================================================================"
echo "Evaluation completed!"
echo ""
echo "Cleaning up mock servers..."
kill $MOCK_PID 2>/dev/null
echo "Done!"
echo ""
echo "Check results in: tests/dissertation_test_suite/results/"
echo "========================================================================"