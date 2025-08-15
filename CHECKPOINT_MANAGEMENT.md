# Checkpoint Management Guide

## Current Issue
Checkpoints from different evaluation runs are being saved in the same directory (`tests/dissertation_test_suite/results/dissert-result-v4/checkpoints/`), which can cause:
- Confusion between runs
- Potential checkpoint loading issues
- Difficulty tracking which checkpoints belong to which evaluation

## Solution Implemented

### 1. **New Evaluation Script (V2)**
`run_comprehensive_dissertation_eval_v2.py`

**Key Features:**
- Creates **unique timestamped directory** for each run
- Format: `dissert-result-run_YYYYMMDD_HHMMSS/`
- Each run has its own isolated checkpoint directory
- No mixing of checkpoints between runs

**Example Structure:**
```
results/
├── dissert-result-run_20250812_160000/
│   ├── checkpoints/           # Checkpoints for this run only
│   ├── simple_queries/
│   ├── complex_queries/
│   ├── mixed_queries/
│   ├── visualizations/
│   └── run_config.json        # Run configuration
├── dissert-result-run_20250812_170000/
│   ├── checkpoints/           # Different run, separate checkpoints
│   └── ...
```

### 2. **Checkpoint Management Utility**
`manage_checkpoints.py`

**Commands:**

```bash
# List all checkpoint directories
python manage_checkpoints.py list

# Archive old runs (keeps latest 2 by default)
python manage_checkpoints.py archive --keep 2

# Clean duplicate checkpoints within directories
python manage_checkpoints.py clean

# Get info about specific checkpoint directory
python manage_checkpoints.py info --path tests/dissertation_test_suite/results/dissert-result-v4/checkpoints
```

## Checkpoint Naming Convention

Checkpoints are named with timestamp to ensure uniqueness:
```
checkpoint_{strategy}_ep{episode}_{YYYYMMDD_HHMMSS}.pkl
```

Examples:
- `checkpoint_q_learning_tabular_ep200_20250812_151633.pkl`
- `checkpoint_q_learning_dqn_ep1000_20250812_152815.pkl`

## Best Practices

### For New Evaluations
1. **Use the V2 script** for new runs:
   ```bash
   python run_comprehensive_dissertation_eval_v2.py
   ```
   This automatically creates isolated directories.

2. **Check disk space** before starting:
   ```bash
   python manage_checkpoints.py list
   ```

3. **Archive old runs** if needed:
   ```bash
   python manage_checkpoints.py archive --keep 3
   ```

### For Resuming from Checkpoints
When resuming, specify the exact checkpoint file:
```bash
python tests/dissertation_test_suite/scripts/run_baseline_comparison.py \
    --resume-from tests/dissertation_test_suite/results/dissert-result-run_20250812_160000/checkpoints/checkpoint_q_learning_dqn_ep800_20250812_160800.pkl
```

## Benefits of This Approach

1. **No Checkpoint Mixing**: Each run is completely isolated
2. **Easy Comparison**: Can compare multiple runs side-by-side
3. **Clean History**: Previous runs are preserved for reference
4. **Reproducibility**: Each run's configuration is saved
5. **Easy Cleanup**: Can archive/delete entire run directories

## Migration from Old Structure

If you have existing checkpoints in the old structure:

1. **Keep them for reference**: The old checkpoints in `dissert-result-v4/` are still valid
2. **Use new script for future runs**: All new evaluations should use V2 script
3. **Archive if needed**: Use the management utility to archive old runs

## Disk Space Considerations

Each full evaluation (1000 episodes, all strategies) generates approximately:
- Checkpoints: ~50-100 MB
- Results: ~10-20 MB  
- Visualizations: ~5-10 MB
- **Total per run: ~65-130 MB**

With checkpoint intervals of 200 episodes:
- 5 checkpoints per strategy
- 7 strategies
- ~35 checkpoint files per full run

## Summary

✅ **Problem Solved**: Checkpoints from different runs no longer mix
✅ **New Structure**: Each run gets unique timestamped directory  
✅ **Management Tools**: Utilities for listing, archiving, and cleaning
✅ **Best Practice**: Use `run_comprehensive_dissertation_eval_v2.py` for all new evaluations

This ensures clean, organized, and reproducible evaluation runs without checkpoint confusion.