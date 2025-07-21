# A/B Testing Framework Documentation

## Overview

The A/B Testing Framework is a comprehensive system for running controlled experiments to compare different strategies, tools, and approaches in the Auto Tool Discovery system. It supports multiple assignment strategies, statistical methods, and includes features like multi-armed bandits and early stopping.

## Architecture

The A/B testing system consists of three main components:

1. **ABTestingFramework** - Core experiment management and statistical analysis
2. **ABTestManager** - Lifecycle management, persistence, and monitoring
3. **StrategyManager Integration** - Seamless A/B testing for reward strategies

## Key Features

### 1. Multiple Assignment Strategies
- **Random Assignment**: Equal or weighted probability assignment
- **Deterministic Assignment**: Hash-based consistent assignment
- **Multi-Armed Bandit**: Thompson sampling for adaptive allocation

### 2. Statistical Methods
- **Frequentist Analysis**: T-tests, chi-square tests
- **Bayesian Analysis**: Posterior distributions, credible intervals
- **Sequential Testing**: Early stopping with statistical guarantees

### 3. Experiment Management
- **Lifecycle Control**: Draft → Running → Completed states
- **Database Persistence**: Full experiment history and results
- **Real-time Monitoring**: Automatic completion conditions
- **Result Analysis**: Comprehensive statistical reports

## Core Components

### ABTestingFramework

The main framework class that handles experiment logic:

```python
from src.evaluation.ab_testing_framework import (
    ABTestingFramework, ExperimentConfig, 
    AssignmentStrategy, StatisticalMethod
)

# Create framework
framework = ABTestingFramework()

# Configure experiment
config = ExperimentConfig(
    name="button_color_test",
    description="Testing button color impact on conversion",
    variants=["control_blue", "treatment_green"],
    primary_metric="conversion_rate",
    assignment_strategy=AssignmentStrategy.RANDOM,
    statistical_method=StatisticalMethod.FREQUENTIST,
    min_sample_size=100,
    confidence_level=0.95,
    enable_early_stopping=True
)

# Create and start experiment
experiment = await framework.create_experiment(config)
await experiment.start()
```

### ABTestManager

Manages experiment lifecycle with database persistence:

```python
from src.evaluation.ab_test_manager import ABTestManager

# Initialize manager
manager = ABTestManager(config)
await manager.initialize()

# Create experiment through manager
experiment_id = await manager.create_experiment(config)
await manager.start_experiment(experiment_id)

# Get variant for user
variant = await manager.get_variant_for_user(experiment_id, user_id)

# Record metrics
await manager.record_metric(
    experiment_id, user_id, "conversion_rate", 
    1.0, success=True
)

# Analyze results
results = await manager.stop_experiment(experiment_id)
```

### StrategyManager Integration

The reward strategy manager includes built-in A/B testing support:

```python
# Create strategy experiment
experiment_id = await strategy_manager.create_strategy_experiment(
    name="reward_strategy_comparison",
    description="Comparing reward calculation strategies",
    strategies_to_test=["temporal", "hierarchical", "adaptive"],
    min_sample_size=100,
    max_duration_days=7
)

# Get variant for execution
variant = await strategy_manager.get_experiment_variant(experiment_id, user_id)

# Record results
await strategy_manager.record_experiment_result(
    experiment_id, user_id, reward, success_rate, computation_time
)
```

## Integration Considerations

### Sharing ABTestManager Instances

When integrating the A/B testing framework with other components (like StrategyManager), ensure that both components use the same ABTestManager instance to maintain consistent experiment state:

```python
# Correct approach - share the same instance
ab_test_manager = ABTestManager(config)
await ab_test_manager.initialize()

# Pass the shared instance to StrategyManager
config['_shared_ab_test_manager'] = ab_test_manager
strategy_manager = StrategyManager(config)

# Both components now share the same active_experiments state
```

**Important**: If different components create their own ABTestManager instances, they won't share the same `active_experiments` set, which can lead to variant assignments returning `None`.

### Why This Matters

The ABTestManager maintains an `active_experiments` set that tracks which experiments are currently running. When `get_variant_for_user()` is called, it first checks if the experiment is in this set. If different instances are used:

1. Component A creates and starts an experiment (adds to its `active_experiments`)
2. Component B tries to get a variant but has its own empty `active_experiments`
3. The variant assignment returns `None` because the experiment appears inactive

### Example: StrategyManager Integration

```python
# In your demo or application
demo = ABTestingDemo()
# demo.ab_test_manager is created

# Pass it to strategy manager
demo.config['_shared_ab_test_manager'] = demo.ab_test_manager
demo.strategy_manager = StrategyManager(demo.config)

# Now both use the same instance and share state
```

## Experiment Configuration

### Basic Configuration

```python
config = ExperimentConfig(
    name="experiment_name",
    description="Experiment description",
    variants=["control", "treatment_a", "treatment_b"],
    primary_metric="conversion_rate",
    secondary_metrics=["revenue", "engagement"],
    min_sample_size=100,
    confidence_level=0.95
)
```

### Advanced Configuration

```python
config = ExperimentConfig(
    # ... basic config ...
    
    # Assignment strategy
    assignment_strategy=AssignmentStrategy.WEIGHTED,
    assignment_weights={
        "control": 0.5,
        "treatment_a": 0.25,
        "treatment_b": 0.25
    },
    
    # Statistical method
    statistical_method=StatisticalMethod.BAYESIAN,
    
    # Sample size and power
    target_sample_size=1000,
    power=0.8,
    minimum_detectable_effect=0.02,
    
    # Duration and stopping
    max_duration_days=14,
    enable_early_stopping=True,
    early_stopping_threshold=0.001,
    
    # Multi-armed bandit
    enable_multi_armed_bandit=True,
    mab_exploration_rate=0.1
)
```

## Assignment Strategies

### Random Assignment
- Assigns users randomly to variants
- Supports weighted assignment for unequal splits
- Best for: Traditional A/B tests with fixed allocation

### Deterministic Assignment
- Uses MD5 hash of user ID for consistent assignment
- Same user always gets same variant
- Best for: Tests where user consistency is critical

### Multi-Armed Bandit (MAB)
- Uses Thompson sampling for adaptive allocation
- Allocates more traffic to better-performing variants
- Updates allocation based on observed outcomes
- Best for: Optimization-focused experiments

## Statistical Analysis

### Frequentist Analysis

For binary metrics (conversion rates):
- Chi-square test for independence
- Reports p-value and statistical significance

For continuous metrics:
- T-test for difference in means
- Assumes normal distribution

### Bayesian Analysis

For binary metrics:
- Beta-Binomial conjugate model
- Reports probability that treatment is better
- Provides credible intervals

For continuous metrics:
- Normal-Normal conjugate model (when applicable)
- Monte Carlo simulation for complex cases

### Early Stopping

- Sequential testing to stop experiments early
- Maintains statistical validity
- Reduces time to decision
- Configurable significance threshold

## API Endpoints

The framework includes RESTful API endpoints for experiment management:

### Create Experiment
```http
POST /api/v1/ab-testing/experiments
Content-Type: application/json

{
  "name": "homepage_redesign",
  "description": "Testing new homepage layout",
  "variants": ["control", "new_design"],
  "primary_metric": "engagement_rate",
  "min_sample_size": 500
}
```

### Assign User
```http
POST /api/v1/ab-testing/experiments/{experiment_name}/assign
Content-Type: application/json

{
  "user_id": "user_123",
  "context": {"source": "mobile"}
}
```

### Record Metric
```http
POST /api/v1/ab-testing/experiments/{experiment_name}/metrics
Content-Type: application/json

{
  "user_id": "user_123",
  "metric_name": "engagement_rate",
  "value": 0.75,
  "success": true
}
```

### Get Results
```http
GET /api/v1/ab-testing/experiments/{experiment_name}/results
```

## Database Schema

The A/B testing framework uses the following database tables:

### ab_experiments
- Stores experiment configuration and status
- Tracks start/end times
- JSON configuration storage

### ab_assignments
- Records user-variant assignments
- Ensures consistent assignment
- Indexed for fast lookups

### ab_metrics
- Stores all metric events
- Supports metadata for detailed analysis
- Time-series data for trend analysis

### ab_results
- Caches analysis results
- Stores winner determination
- Historical record of experiments

## Usage Examples

### Example 1: Simple Conversion Test

```python
# Test button colors
config = ExperimentConfig(
    name="button_color_test",
    description="Green vs Blue CTA buttons",
    variants=["blue_button", "green_button"],
    primary_metric="click_rate",
    min_sample_size=1000
)

experiment_id = await manager.create_experiment(config)
await manager.start_experiment(experiment_id)

# In your application
user_id = get_current_user_id()
variant = await manager.get_variant_for_user(experiment_id, user_id)

# Show appropriate button
if variant == "blue_button":
    show_blue_button()
else:
    show_green_button()

# Record click
if user_clicked:
    await manager.record_metric(
        experiment_id, user_id, "click_rate", 1.0, True
    )
```

### Example 2: Multi-Variant with MAB

```python
# Test multiple algorithms with adaptive allocation
config = ExperimentConfig(
    name="algorithm_optimization",
    description="Finding best recommendation algorithm",
    variants=["baseline", "collaborative", "content", "hybrid"],
    primary_metric="engagement_score",
    assignment_strategy=AssignmentStrategy.MULTI_ARMED_BANDIT,
    enable_multi_armed_bandit=True,
    min_sample_size=200
)

# MAB will automatically allocate more traffic to better algorithms
```

### Example 3: Bayesian Analysis

```python
# Use Bayesian statistics for faster decisions
config = ExperimentConfig(
    name="feature_adoption",
    description="New feature adoption rate",
    variants=["without_feature", "with_feature"],
    primary_metric="adoption_rate",
    statistical_method=StatisticalMethod.BAYESIAN,
    min_sample_size=500
)

# Results will include probability estimates
results = await manager.analyze_experiment(experiment_id)
print(f"P(feature better): {results['statistical_significance']['probability_treatment_better']}")
```

## Best Practices

### 1. Experiment Design
- Define clear success metrics before starting
- Calculate required sample size using power analysis
- Consider practical significance, not just statistical
- Document experiment hypothesis and expected outcomes

### 2. Implementation
- Always check for existing assignments before creating new ones
- Record all relevant metrics, not just primary
- Use consistent user IDs across sessions
- Handle edge cases (new users, missing data)

### 3. Analysis
- Wait for minimum sample size before drawing conclusions
- Consider multiple testing corrections for many variants
- Look at segment analysis for insights
- Document learnings for future experiments

### 4. Multi-Armed Bandits
- Use when optimization is more important than learning
- Monitor exploration/exploitation balance
- Consider regret vs statistical power tradeoff
- Not suitable for precise effect size estimation

## Monitoring and Debugging

### Experiment Status
```python
# Get current experiment status
status = await manager.get_experiment_status(experiment_id)
print(f"Total users: {status['total_users']}")
print(f"Variant distribution: {status['variant_metrics']}")
```

### List All Experiments
```python
# List all running experiments
running = await manager.list_experiments(status="running")

# List completed experiments
completed = await manager.list_experiments(status="completed")
```

### Debug Assignment Issues
```python
# Check user assignment
variant = await manager.get_variant_for_user(experiment_id, user_id)

# Verify in database
async with manager.db_manager.get_connection() as conn:
    cursor = await conn.execute(
        "SELECT * FROM ab_assignments WHERE user_id = ?",
        (user_id,)
    )
    assignments = await cursor.fetchall()
```

## Integration with Q-Learning

The A/B testing framework integrates with the Q-learning system for testing different learning strategies:

```python
# Test different exploration rates
variants = ["epsilon_0.1", "epsilon_0.2", "epsilon_0.3"]

# Assign users to different Q-learning configurations
variant = await manager.get_variant_for_user(experiment_id, session_id)
epsilon = float(variant.split("_")[1])

# Configure Q-learning with variant-specific parameters
q_config = {
    "exploration_rate": epsilon,
    # ... other config ...
}
```

## Performance Considerations

### Scalability
- Assignments are O(1) lookup after initial assignment
- Metric recording is asynchronous
- Analysis can be cached for repeated queries
- Database indexes on critical paths

### Resource Usage
- Minimal memory overhead per experiment
- Database storage grows with metrics
- Analysis computation scales with data size
- Consider archiving old experiments

## Troubleshooting

### Common Issues

1. **No assignments happening**
   - Check experiment status (must be "running")
   - Verify experiment exists in database
   - Check user ID format consistency

2. **Statistical tests failing**
   - Ensure minimum sample size reached
   - Check for data quality issues
   - Verify metric values are reasonable

3. **MAB not adapting**
   - Confirm MAB is enabled in config
   - Check that success/failure signals are recorded
   - Verify update_priors is being called

4. **Database errors**
   - Check database file permissions
   - Verify schema is up to date
   - Look for connection pool exhaustion

5. **Variant assignments returning None**
   - Ensure all components share the same ABTestManager instance
   - Check that the experiment is in the active_experiments set
   - Verify that start_experiment() was called before get_variant_for_user()
   - If using StrategyManager, confirm it's using the shared instance

## Future Enhancements

### Planned Features
- Real-time performance regression alerts
- Advanced segmentation analysis
- Automated insight generation
- Integration with external analytics platforms

### Experimental Features
- Contextual bandits for personalization
- Causal inference methods
- Network effects handling
- Time-series analysis for temporal patterns

## References

1. Kohavi et al. (2020). "Trustworthy Online Controlled Experiments"
2. Chapelle & Li (2011). "An Empirical Evaluation of Thompson Sampling"
3. Johari et al. (2017). "Peeking at A/B Tests"
4. Deng et al. (2016). "Continuous Monitoring of A/B Tests without Pain"