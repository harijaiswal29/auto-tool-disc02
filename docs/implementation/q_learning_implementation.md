# Q-Learning Engine Implementation

## Overview

The Q-Learning Engine is a core component of Phase 4 that enables the system to learn optimal tool selection strategies through reinforcement learning. It implements a complete Q-learning system with state representation, action space management, experience replay, and model persistence.

## Architecture

### Core Components

1. **StateRepresentation**: Encodes system state from intent vectors, context, and tool history
2. **ActionSpace**: Defines and validates tool combinations as actions
3. **QTable**: Manages Q-values with sparse storage and update logic
4. **ExperienceReplayBuffer**: Stores and samples past experiences for batch learning
5. **QLearningEngine**: Orchestrates all components and manages the learning process

## Implementation Details

### State Representation

The state vector consists of 419 dimensions:
- **Intent Vector** (384 dims): Sentence transformer embeddings from intent recognition
- **Context Features** (10 dims): Domain, session info, user history
- **Tool History** (20 dims): Recent tool usage patterns
- **Performance Metrics** (5 dims): Success rates, response times, cache hits

```python
state_dimensions = {
    'intent_vector': 384,      # Sentence transformer output
    'context_features': 10,    # Domain, user history, etc.
    'tool_history': 20,        # Recent tool usage
    'performance_metrics': 5   # Success rate, response time, etc.
}
```

### Action Space

Actions are tool combinations (1-3 tools) with constraint validation:
- **Conflicts**: Tools that cannot be used together
- **Requirements**: Tool dependencies that must be satisfied
- **Max Tools**: Configurable limit on simultaneous tool execution

### Q-Learning Algorithm

```python
# Q-learning update rule
Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))

# Where:
# α = learning rate (0.1)
# γ = discount factor (0.9)
# r = reward
# s = current state
# a = action taken
# s' = next state
# a' = possible next actions
```

### Reward Function

The reward calculation considers multiple factors:

```python
reward = base_reward + time_penalty + efficiency_bonus

# Base reward: +1.0 for success, -0.5 for failure
# Time penalty: -0.1 * log(execution_time)
# Efficiency bonus: +0.1 * (1 - tools_used/max_tools)
```

## Integration with Orchestrator

The Q-learning engine integrates seamlessly with the orchestrator agent:

1. **Tool Selection**: When Q-learning is enabled, the orchestrator uses the engine to select optimal tool combinations
2. **Learning Updates**: After each execution, results are used to update Q-values
3. **Model Persistence**: The learned model is periodically saved to database
4. **Exploration vs Exploitation**: Epsilon-greedy strategy balances trying new combinations vs using known good ones

## Configuration

Add to `config/config.json`:

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

## Usage

### Basic Usage

```python
# Initialize Q-learning engine
config = load_config()
q_engine = QLearningEngine(config)

# Select action (tool combination)
state = encode_current_state()
available_tools = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp']
constraints = {'conflicts': {}, 'requires': {}}

action = await q_engine.select_action(state, available_tools, constraints)

# Learn from experience
reward = calculate_reward(execution_results)
await q_engine.learn_from_experience(
    state, action, reward, next_state, 
    next_available_tools, constraints
)
```

### With Orchestrator

```python
# Enable Q-learning in orchestrator
config['q_learning']['enable_learning'] = True
orchestrator = OrchestratorAgent(config)

# Process queries - Q-learning happens automatically
result = await orchestrator.process_user_query("Find Python files")
```

## Database Schema

The Q-learning system uses several tables for persistence:

```sql
-- Model snapshots
CREATE TABLE model_snapshots (
    id INTEGER PRIMARY KEY,
    version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_data TEXT NOT NULL,
    created_at TIMESTAMP
);

-- Q-learning states
CREATE TABLE q_learning_states (
    id INTEGER PRIMARY KEY,
    state_hash TEXT UNIQUE,
    state_vector JSON,
    created_at TIMESTAMP
);

-- Q-values
CREATE TABLE q_values (
    state_id INTEGER,
    action_hash TEXT,
    action_tools JSON,
    q_value REAL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP
);
```

## Monitoring and Metrics

The Q-learning engine provides comprehensive metrics:

```python
metrics = q_engine.get_metrics()
# Returns:
{
    'episode_count': 100,
    'total_reward': 75.5,
    'avg_reward': 0.755,
    'success_rate': 0.82,
    'exploration_rate': 0.135,
    'q_table_stats': {
        'total_entries': 250,
        'avg_q_value': 0.42,
        'max_q_value': 0.95
    }
}
```

## Testing

Run unit tests:
```bash
pytest tests/unit/test_q_learning_engine.py -v
```

Run integration demo:
```bash
python demos/demo_q_learning_orchestration.py
```

## Performance Considerations

1. **State Hashing**: MD5 hashing ensures consistent state identification
2. **Sparse Storage**: Q-table uses dictionary for memory efficiency
3. **Batch Learning**: Experience replay reduces sample correlation
4. **Async Operations**: All database operations are asynchronous
5. **Caching**: Action space combinations are cached for performance

## Future Enhancements

1. **Deep Q-Learning**: Replace tabular Q-learning with neural networks
2. **Multi-Agent Learning**: Coordinate multiple agents
3. **Transfer Learning**: Apply learned knowledge to new domains
4. **Online Learning**: Continuous adaptation in production
5. **Explainability**: Visualize learned policies and decisions