# Advanced Reward Calculation Strategies

## Overview

The advanced reward calculation strategies module extends the basic reward system with sophisticated techniques from reinforcement learning literature. These strategies enable more nuanced credit assignment, better exploration-exploitation balance, and adaptive learning based on task complexity and system performance.

## Architecture

The advanced reward system consists of four main strategies coordinated by a strategy manager:

1. **Temporal Difference (TD) Rewards** - Credit assignment across time
2. **Hierarchical Goal-Based Rewards** - Multi-level objective tracking
3. **Adaptive Reward Shaping** - Dynamic weight adjustment
4. **Information-Theoretic Rewards** - Curiosity and exploration bonuses

## Strategy Components

### 1. Temporal Difference Reward Calculator

Located in `src/learning/advanced_rewards/temporal_rewards.py`

**Key Features:**
- **TD(λ) with Eligibility Traces**: Propagates rewards backward through time
- **N-Step Returns**: Considers future rewards up to n steps ahead
- **Experience Buffer**: Maintains history for temporal calculations
- **Generalized Advantage Estimation (GAE)**: Optional advanced credit assignment

**Configuration:**
```json
{
  "temporal_difference": {
    "enabled": true,
    "lambda": 0.9,        // TD(λ) trace decay
    "n_steps": 5,         // Steps for n-step returns
    "gamma": 0.9,         // Discount factor
    "use_gae": false      // Use GAE
  }
}
```

**How it Works:**
1. Maintains eligibility traces for state-action pairs
2. Calculates TD error: δ = r + γV(s') - V(s)
3. Updates traces with decay: e(s,a) = γλe(s,a) + 1
4. Combines immediate rewards with bootstrapped future values

### 2. Hierarchical Goal-Based Rewards

Located in `src/learning/advanced_rewards/hierarchical_rewards.py`

**Key Features:**
- **Multi-Level Goal Hierarchy**: Primary, secondary, tertiary goals
- **Progress Tracking**: Partial completion rewards
- **Milestone Bonuses**: Extra rewards for significant achievements
- **Goal Cascading**: Parent goals influenced by child completion

**Goal Types:**
- `PRIMARY`: Main user task completion
- `SECONDARY`: Important sub-objectives
- `TERTIARY`: Supporting goals
- `MILESTONE`: Significant checkpoints
- `SUBTASK`: Granular task components

**Configuration:**
```json
{
  "hierarchical": {
    "enabled": true,
    "goal_weights": {
      "primary": 1.0,
      "secondary": 0.5,
      "tertiary": 0.25,
      "milestone": 0.4,
      "subtask": 0.1
    },
    "milestone_bonus": 0.5,
    "progress_reward": true,
    "subtask_completion_threshold": 0.8
  }
}
```

### 3. Adaptive Reward Shaping

Located in `src/learning/advanced_rewards/adaptive_shaping.py`

**Key Features:**
- **Dynamic Weight Adjustment**: Adapts component weights based on performance
- **Curriculum Learning**: Progressive difficulty stages
- **Meta-Learning**: Learns how to learn better
- **Performance Tracking**: Monitors and adjusts based on metrics

**Component Weights:**
- `success`: Task completion weight
- `efficiency`: Resource usage weight
- `exploration`: Novelty seeking weight
- `complexity`: Complex task handling weight
- `consistency`: Smooth behavior weight

**Curriculum Stages:**
1. **Stage 0**: Focus on basic success (simplified rewards)
2. **Stage 1**: Balance all components
3. **Stage 2**: Optimize for efficiency and complexity

**Configuration:**
```json
{
  "adaptive_shaping": {
    "enabled": true,
    "adaptation_rate": 0.01,
    "curriculum_stages": 3,
    "performance_window": 100,
    "meta_learning_rate": 0.001
  }
}
```

### 4. Information-Theoretic Rewards

Located in `src/learning/advanced_rewards/information_theoretic.py`

**Key Features:**
- **Curiosity-Driven Exploration**: Rewards for visiting novel states
- **Entropy Bonuses**: Encourages diverse action selection
- **Information Gain**: Rewards that reduce uncertainty
- **Novelty Detection**: Identifies and rewards new experiences

**Reward Components:**
- **State Novelty**: Similarity-based novelty detection
- **Visit Counts**: UCB-style exploration bonuses
- **Entropy Rewards**: Action distribution entropy
- **Mutual Information**: Information gained from actions
- **Surprise**: Prediction error as intrinsic motivation

**Configuration:**
```json
{
  "information_theoretic": {
    "enabled": true,
    "curiosity_weight": 0.1,
    "entropy_bonus": 0.05,
    "novelty_threshold": 0.7,
    "state_visit_decay": 0.99,
    "mutual_info_weight": 0.15
  }
}
```

## Strategy Manager

Located in `src/learning/advanced_rewards/strategy_manager.py`

The Strategy Manager coordinates multiple reward strategies and provides:

### Combination Methods

1. **Weighted Average** (default): Combines strategies with configurable weights
2. **Max**: Takes the maximum reward across strategies
3. **Voting**: Majority vote with weighted strength

### A/B Testing Framework

Enables comparing strategies:
```python
manager.enable_ab_testing(['control', 'temporal', 'hierarchical'])
```

### Performance Tracking

Monitors each strategy's:
- Average reward contribution
- Computation time
- Success correlation
- Execution count

## Integration with Q-Learning

The advanced strategies integrate seamlessly with the existing reward calculator:

```python
# In RewardCalculator
if self.use_advanced_strategies and state is not None:
    advanced_reward, breakdown = self.strategy_manager.calculate_reward(
        state, action, next_state, execution_results, context
    )
```

## Database Schema

New tables for tracking advanced rewards:

- `reward_strategy_metrics`: Performance metrics per strategy
- `goal_hierarchy`: Hierarchical goal definitions
- `goal_progress`: Goal achievement tracking
- `novelty_metrics`: State-action novelty scores
- `adaptive_shaping_metrics`: Curriculum and weight tracking
- `temporal_reward_states`: TD states and traces
- `reward_ab_test_results`: A/B testing data

## Usage Example

```python
# Configure advanced strategies
config = {
    "advanced_reward_strategies": {
        "enabled": true,
        "combination_method": "weighted_average",
        "strategies": {
            "temporal_difference": {"enabled": true},
            "hierarchical": {"enabled": true},
            "adaptive_shaping": {"enabled": true},
            "information_theoretic": {"enabled": true}
        }
    }
}

# Create reward calculator with advanced strategies
calculator = RewardCalculator(config, use_advanced_strategies=True)

# Calculate reward with state information
reward, breakdown = calculator.calculate_reward(
    execution_results=results,
    context=context,
    state=current_state,
    action=tools_used,
    next_state=next_state
)
```

## Performance Considerations

1. **Computation Overhead**: Advanced strategies add ~10-50ms per calculation
2. **Memory Usage**: Eligibility traces and state history require additional memory
3. **Caching**: State hashes and action combinations are cached
4. **Async Operations**: Database operations are asynchronous

## Best Practices

1. **Start Simple**: Enable one strategy at a time to understand impact
2. **Monitor Performance**: Use the performance tracking to identify bottlenecks
3. **Tune Weights**: Adjust strategy weights based on domain requirements
4. **Use A/B Testing**: Compare strategies before full deployment
5. **Database Maintenance**: Regularly clean old novelty metrics and traces

## Future Enhancements

1. **Risk-Aware Strategies**: VaR and CVaR calculations
2. **Meta-Learning Rewards**: Transfer learning bonuses
3. **Human-in-the-Loop**: Learn rewards from human feedback
4. **Multi-Objective Optimization**: Pareto-optimal reward balancing
5. **Causal Reasoning**: Counterfactual reward analysis