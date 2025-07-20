# Demo Scripts Documentation

This directory contains demonstration scripts showcasing various features of the Auto Tool Discovery system.

## Available Demos

1. **A/B Testing Framework** (`demo_ab_testing_framework.py`)
2. **Advanced Reward Strategies** (`demo_advanced_rewards.py`)
3. **Baseline Evaluation** (`demo_baseline_evaluation.py`)
4. **DQN Learning** (`demo_dqn_learning.py`, `demo_dqn_simple.py`)
5. **Pattern Mining** (`demo_pattern_mining.py`)
6. **Q-Learning Orchestration** (`demo_q_learning_orchestration.py`)
7. **MCP Tool Demos** (GitHub, Notion, etc.)

## A/B Testing Framework Demo

**File**: `demo_ab_testing_framework.py`

Demonstrates comprehensive A/B testing capabilities across 6 different scenarios:

### Demo 1: Basic A/B Test
- Simple conversion rate testing between two variants
- Random assignment strategy
- Frequentist statistical analysis
- Example: Testing button colors (blue vs green)

### Demo 2: Multi-Variant Experiment
- Tests multiple variants with weighted assignment
- Tracks multiple metrics (engagement, time on page, bounce rate)
- Shows how to handle more than 2 variants
- Example: Testing 4 different page layouts

### Demo 3: Bayesian A/B Testing
- Uses Bayesian statistical methods
- Provides probability estimates and credible intervals
- Better for smaller sample sizes
- Example: Feature adoption rate comparison

### Demo 4: Multi-Armed Bandit
- Adaptive allocation using Thompson sampling
- Automatically shifts traffic to better-performing variants
- Optimizes for performance over learning
- Example: Finding best recommendation algorithm

### Demo 5: Reward Strategy A/B Test
- **Integration with StrategyManager**
- Tests different reward calculation strategies (temporal, hierarchical, adaptive)
- Uses actual strategy calculations (not simulated)
- Demonstrates proper instance sharing between components
- Shows real reward values with proper sample sizes

### Demo 6: Full Lifecycle Management
- Complete experiment lifecycle with persistence
- Database storage and retrieval
- Monitoring and status updates
- Shows how to list all experiments

### Running the Demo
```bash
# Run all 6 demos
python demos/demo_ab_testing_framework.py

# Run specific demo only (example for Demo 5)
python -c "import asyncio; from demos.demo_ab_testing_framework import ABTestingDemo; asyncio.run(ABTestingDemo().demo_strategy_comparison())"
```

### Key Implementation Notes
- The demo ensures ABTestManager instance is shared between components
- Rewards are calculated using actual strategies, not simulated values
- All metrics are properly recorded with non-zero sample sizes
- Uses timestamp suffixes to ensure unique experiment names

## Advanced Reward Strategies Demo

**File**: `demo_advanced_rewards.py`

Demonstrates the four advanced reward calculation strategies:

### Strategies Shown
1. **Temporal Difference (TD)** - Credit assignment across time
2. **Hierarchical Goal-Based** - Multi-level goal tracking
3. **Adaptive Reward Shaping** - Dynamic adjustment based on progress
4. **Information-Theoretic** - Curiosity-driven exploration

### Running the Demo
```bash
python demos/demo_advanced_rewards.py
```

## Pattern Mining Demo

**File**: `demo_pattern_mining.py`

Shows how the system discovers patterns in tool usage:

### Features Demonstrated
- Sequential pattern mining (tool A → tool B → tool C)
- Combination pattern mining (tools that work well together)
- Pattern metrics (support, confidence, lift)
- Pattern-based tool suggestions

### Running the Demo
```bash
python demos/demo_pattern_mining.py
```

## Q-Learning Demos

### Basic Q-Learning Orchestration
**File**: `demo_q_learning_orchestration.py`

Shows Q-learning integration with the orchestrator:
- State representation
- Action selection
- Reward calculation
- Learning updates

### DQN Comparison
**File**: `demo_dqn_learning.py`

Compares tabular Q-learning with Deep Q-Learning:
- Performance comparison
- Learning curves
- Convergence analysis

### Simple DQN Demo
**File**: `demo_dqn_simple.py`

Basic DQN implementation demonstration.

## Baseline Evaluation Demo

**File**: `demo_baseline_evaluation.py`

Compares the learning system against various baselines:

### Baselines Included
1. Random selection
2. Most popular tools
3. Fixed policy
4. Greedy single-tool
5. Context-agnostic Q-learning

### Running Options
```bash
# Quick evaluation (500 episodes)
python demos/demo_baseline_evaluation.py --mode quick

# Full evaluation (2000 episodes)
python demos/demo_baseline_evaluation.py --mode full
```

## MCP Tool Demos

### GitHub MCP Demo
**Files**: `demo_github_mcp.py`, `demo_github_real.py`

Demonstrates GitHub integration capabilities.

### Notion MCP Demo
**File**: `demo_notion_mcp.py`

Shows Notion workspace integration.

## Other Demo Scripts

### Pipeline Refactor Demo
**File**: `demo_pipeline_refactor.py`

Demonstrates the modular pipeline architecture.

### Retry Logic Demo
**File**: `demo_retry_logic.py`

Shows retry mechanisms and circuit breakers in action.

### Integration Demo
**File**: `test_integration_demo.py`

End-to-end system integration demonstration.

## Common Patterns

### Async Execution
All demos use asyncio for asynchronous execution:
```python
async def main():
    demo = DemoClass()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration
Most demos load configuration from `config/config.json` or create custom configs:
```python
config = {
    'key': 'value',
    # ... configuration options
}
```

### Error Handling
Demos include error handling and logging:
```python
try:
    await demo_function()
except Exception as e:
    logger.error(f"Demo failed: {e}")
    import traceback
    traceback.print_exc()
```

## Tips for Running Demos

1. **Environment Setup**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Working Directory**: Run demos from the project root:
   ```bash
   cd /path/to/auto-tool-disc
   python demos/demo_name.py
   ```

3. **Database Files**: Some demos create database files in the `data/` directory

4. **API Keys**: Some MCP demos require API keys (set as environment variables)

5. **Logging**: Most demos include detailed logging output

## Creating New Demos

When creating new demos:
1. Use clear section separators
2. Include descriptive print statements
3. Handle errors gracefully
4. Clean up resources (databases, connections)
5. Document any special requirements
6. Use timestamp suffixes for unique identifiers