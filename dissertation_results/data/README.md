# Dissertation Raw Experimental Data

This directory contains the raw experimental data that generated the results presented in the dissertation "Autonomous Tool Discovery Through Reinforcement Learning with Model Context Protocol (MCP)".

## Directory Structure

```
data/
├── training_200ep/     # 200 episode training results
├── training_400ep/     # 400 episode training results  
└── training_600ep/     # 600 episode training results (final)
```

## Data Files

Each training directory contains:
- **36 JSON files** per training run (5 runs × 7 strategies + 1 summary)
- **Total**: 108 files across all three training milestones

### File Naming Convention

Individual run files:
- `{strategy}_run{n}_{timestamp}.json` 
- Example: `q_learning_dqn_run0_20250814_204610.json`

Summary files:
- `baseline_comparison_final_{timestamp}.json`
- Contains aggregated metrics across all strategies and runs

### Strategies Evaluated

1. **random** - Uniform random tool selection
2. **popular** - Frequency-based selection
3. **fixed_policy** - Rule-based keyword mapping
4. **greedy** - First-match by keywords
5. **context_agnostic** - Static preference ordering
6. **q_learning_tabular** - Q-learning with state discretization
7. **q_learning_dqn** - Deep Q-learning with neural network

## Data Format

### Individual Run Files
```json
{
  "strategy": "q_learning_dqn",
  "run_id": 0,
  "episodes": 600,
  "results": [
    {
      "episode": 1,
      "query": "Find weather information",
      "success": true,
      "tools_selected": ["weather"],
      "reward": 20.0,
      "execution_time": 0.234
    }
  ],
  "metrics": {
    "task_completion_rate": 0.5033,
    "tool_selection_accuracy": 0.111,
    "average_reward": 11.43
  }
}
```

### Summary Files
```json
{
  "experiment_config": {
    "episodes": 600,
    "query_set": "dissertation_core",
    "runs_per_strategy": 5
  },
  "strategy_summaries": {
    "q_learning_dqn": {
      "task_completion_rate": {
        "mean": 0.5033,
        "std": 0.0029,
        "values": [0.500, 0.503, 0.506, 0.502, 0.505]
      },
      "tool_selection_accuracy": {
        "mean": 0.111,
        "std": 0.015
      },
      "average_reward": {
        "mean": 11.43,
        "std": 0.21
      }
    }
  },
  "statistical_tests": {
    "hypothesis_1": {
      "p_value": 0.00012,
      "significant": true
    }
  }
}
```

## Key Metrics

### Task Completion Rate
- Percentage of queries successfully completed
- Primary performance metric for dissertation

### Tool Selection Accuracy  
- Percentage of optimal tool selections
- Measures learning quality

### Average Reward
- Cumulative reward per episode
- Indicates overall performance

## Usage

These files were used to:
1. Generate visualizations in `../figures/`
2. Create summary statistics in `../DISSERTATION_RESULTS.md`
3. Validate research hypotheses (H1-H5)
4. Compare Q-learning against baseline strategies

## Reproducibility

To reproduce the visualizations:
```bash
cd ../
python generate_visualizations.py
```

The script reads these data files and generates:
- Learning curves
- Performance comparisons
- Statistical distributions
- Hypothesis validation plots

## File Sizes

- **training_200ep/**: ~1.3MB (36 files)
- **training_400ep/**: ~1.6MB (36 files)
- **training_600ep/**: ~2.4MB (36 files)
- **Total**: ~5.3MB (108 files)

## Source

These files are copies from:
- `tests/dissertation_test_suite/results/training_200ep_fixed/`
- `tests/dissertation_test_suite/results/training_400ep_resumed/`
- `tests/dissertation_test_suite/results/training_600ep_continued/`

Generated on: August 14, 2025
Experiment duration: ~45 minutes total