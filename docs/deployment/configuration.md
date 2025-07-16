# Configuration Guide

## Overview

This document provides detailed information about configuring the Auto Tool Discovery system, with a focus on the enhanced learning system parameters.

## Learning System Configuration

### Q-Learning Configuration

```json
{
  "q_learning": {
    "learning_rate": 0.1,
    "discount_factor": 0.9,
    "exploration_rate": 0.2,
    "exploration_decay": 0.995,
    "min_exploration_rate": 0.01,
    "max_tools": 3,
    "buffer_capacity": 10000,
    "batch_size": 32,
    "update_frequency": 4,
    "enable_learning": true
  }
}
```

### Reward Calculation Configuration

The enhanced reward calculator uses sophisticated parameters to evaluate tool execution outcomes:

```json
{
  "reward_calculation": {
    "base_weights": {
      "success": 1.0,       // Base reward for successful execution
      "failure": -0.5,      // Base penalty for failed execution
      "partial_success": 0.3 // Base reward for partial completion
    },
    "failure_penalties": {
      "network_timeout": -0.2,    // Network connectivity issues
      "permission_error": -0.8,   // Access denied errors (severe)
      "rate_limit": -0.3,         // API rate limit exceeded
      "connection_error": -0.25,  // Connection failures
      "retryable": -0.1,          // Generic retryable errors
      "non_retryable": -0.7,      // Permanent failures
      "unknown": -0.5             // Unclassified errors
    },
    "resource_penalties": {
      "memory_weight": 0.05,      // Penalty per GB of memory used
      "cpu_weight": 0.05,         // Penalty per CPU core utilized
      "api_calls_weight": 0.1,    // Penalty per API call made
      "time_weight": 0.1          // Logarithmic time penalty
    },
    "synergy_bonuses": {
      "known_good_combo": 0.2,    // Bonus for proven tool combinations
      "discovered_combo": 0.3,    // Bonus for newly discovered synergies
      "complementary_tools": 0.15 // Bonus for complementary tool usage
    },
    "context_multipliers": {
      "exploration": 1.2,         // Encourage trying new combinations
      "production": 0.8,          // Conservative in production mode
      "high_confidence": 1.0,     // Standard rewards when confident
      "low_confidence": 1.1,      // Slight boost when uncertain
      "user_initiated": 1.0,      // Full rewards for user queries
      "system_initiated": 0.9     // Slightly discounted for system queries
    }
  }
}
```

## Configuration Parameters Explained

### Base Weights
- **success**: Reward given when a tool execution completes successfully
- **failure**: Penalty applied when a tool execution fails completely
- **partial_success**: Reward for partial completion (scaled by completion percentage)

### Failure Penalties
Different failure types receive different penalties based on severity and recoverability:
- **network_timeout** (-0.2): Transient network issues, likely retryable
- **permission_error** (-0.8): Severe penalty as retry unlikely to succeed
- **rate_limit** (-0.3): May succeed after delay
- **connection_error** (-0.25): Could be temporary
- **retryable** (-0.1): Light penalty for errors that can be retried
- **non_retryable** (-0.7): Heavy penalty for permanent failures
- **unknown** (-0.5): Default for unclassified errors

### Resource Penalties
Penalties applied based on resource consumption:
- **memory_weight**: Penalty multiplier for memory usage (MB)
- **cpu_weight**: Penalty multiplier for CPU usage (percentage)
- **api_calls_weight**: Penalty per external API call
- **time_weight**: Logarithmic penalty based on execution time

### Synergy Bonuses
Rewards for effective tool combinations:
- **known_good_combo**: Bonus for using proven successful combinations
- **discovered_combo**: Higher bonus for discovering new synergies
- **complementary_tools**: Bonus when tools enhance each other

### Context Multipliers
Adjust rewards based on execution context:
- **exploration**: Higher multiplier encourages trying new approaches
- **production**: Lower multiplier for conservative behavior
- **high_confidence**: Standard rewards when system is confident
- **low_confidence**: Slight boost to encourage learning
- **user_initiated**: Full rewards for user-triggered actions
- **system_initiated**: Slightly reduced for automated actions

## Tuning Guidelines

### For Risk-Averse Environments
```json
{
  "base_weights": {
    "success": 1.2,
    "failure": -0.8,
    "partial_success": 0.2
  },
  "context_multipliers": {
    "exploration": 1.0,
    "production": 0.6
  }
}
```

### For Exploration-Heavy Development
```json
{
  "base_weights": {
    "success": 1.0,
    "failure": -0.3,
    "partial_success": 0.5
  },
  "context_multipliers": {
    "exploration": 1.5,
    "production": 1.0
  }
}
```

### For Resource-Constrained Environments
```json
{
  "resource_penalties": {
    "memory_weight": 0.1,
    "cpu_weight": 0.1,
    "api_calls_weight": 0.2,
    "time_weight": 0.15
  }
}
```

## Monitoring Configuration Impact

After adjusting configuration:
1. Monitor the learning metrics to ensure convergence
2. Check reward distributions are balanced
3. Verify exploration/exploitation ratio is appropriate
4. Ensure failure rates are decreasing over time

## Best Practices

1. **Start with defaults**: The provided defaults are well-balanced
2. **Adjust incrementally**: Make small changes and observe impact
3. **Environment-specific**: Tailor to your specific use case
4. **Monitor continuously**: Track metrics after configuration changes
5. **Document changes**: Keep a log of configuration adjustments