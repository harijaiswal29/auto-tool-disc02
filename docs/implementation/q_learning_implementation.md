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

The state vector consists of 476 dimensions:
- **Intent Vector** (384 dims): Sentence transformer embeddings from intent recognition
- **Context Features** (10 dims): Domain, session info, user history
- **Tool History** (20 dims): Recent tool usage patterns
- **Performance Metrics** (5 dims): Success rates, response times, cache hits
- **Failure Rates** (10 dims): Per-tool failure rate tracking
- **Failure Types** (5 dims): Network, permission, timeout, rate_limit, other
- **Retry Patterns** (5 dims): Retry statistics and patterns
- **User Expertise** (3 dims): One-hot encoding for novice, intermediate, expert
- **Domain Context** (5 dims): One-hot encoding for general, engineering, data_science, web_dev, devops
- **Tool Categories** (10 dims): Tool category features for semantic matching
- **Query Complexity** (5 dims): Query complexity indicators
- **Temporal Features** (4 dims): Episode progress, learning phase
- **Attention Weights** (10 dims): Attention mechanism for relevant features

```python
state_dimensions = {
    'intent_vector': 384,      # Sentence transformer output
    'context_features': 10,    # Domain, user history, etc.
    'tool_history': 20,        # Recent tool usage
    'performance_metrics': 5,  # Success rate, response time, etc.
    'failure_rates': 10,       # Per-tool failure rates
    'failure_types': 5,        # Network, permission, timeout, rate_limit, other
    'retry_patterns': 5,       # Retry statistics and patterns
    'user_expertise': 3,       # One-hot: novice, intermediate, expert
    'domain_context': 5        # One-hot: general, engineering, data_science, web_dev, devops
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

The enhanced reward calculation considers multiple sophisticated factors:

```python
# Enhanced reward calculation with failure differentiation
reward = (base_reward + failure_adjustment + partial_success_bonus + 
          synergy_bonus + user_satisfaction) * context_multiplier * 
          uncertainty_factor - resource_penalty

# Components:
# Base reward: +1.0 for success, -0.5 for failure
# Failure adjustment: Type-specific penalties (e.g., -0.2 for timeout, -0.8 for permission)
# Partial success bonus: Based on completion percentage (0-1.0)
# Resource penalty: CPU, memory, API calls, execution time (logarithmic)
# Tool synergy bonus: +0.2 for known combos, +0.3 for discovered combos
# User satisfaction: Based on explicit ratings and implicit signals
# Context multiplier: 1.2x for exploration, 0.8x for production
```

The reward calculator (`src/learning/reward_calculator.py`) implements:
- **Failure Type Differentiation**: Network timeout (-0.2), permission error (-0.8), rate limit (-0.3)
- **Partial Success Handling**: Rewards based on completion percentage with quality bonuses
- **Resource Efficiency Tracking**: Using psutil for CPU/memory monitoring
- **Tool Synergy Recognition**: Bonuses for complementary tool combinations
- **User Satisfaction Signals**: Explicit ratings (1-5) and implicit feedback

## Integration with Orchestrator

The Q-learning engine integrates seamlessly with the orchestrator agent:

1. **Tool Selection**: When Q-learning is enabled, the orchestrator uses the engine to select optimal tool combinations
2. **Learning Updates**: After each execution, results are used to update Q-values
3. **Model Persistence**: The learned model is periodically saved to database
4. **Exploration vs Exploitation**: Epsilon-greedy strategy balances trying new combinations vs using known good ones

## Context-Aware Tool Selection

The Q-learning engine now incorporates user expertise and domain context for personalized tool selection:

### Context Extraction
```python
# Extract context from query
context_extractor = ContextExtractor()
user_context = context_extractor.extract_context(
    query=user_query,
    user_stats={'success_rate': 0.8, 'query_count': 50},
    intent_type='query.search'
)
# Returns: UserContext(user_expertise='intermediate', domain='engineering')
```

### State Encoding with Context
The context is encoded as part of the state vector:
- User expertise: 3-dimensional one-hot encoding (novice, intermediate, expert)
- Domain: 5-dimensional one-hot encoding (general, engineering, data_science, web_dev, devops)

### Pattern-Based Selection
```python
# Get context-aware patterns
patterns = await pattern_miner.get_context_matching_patterns(
    current_tools=['filesystem_mcp'],
    context=user_context
)
# Returns patterns relevant to intermediate engineering users
```

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
    "enable_learning": true,
    "use_context_aware_patterns": true,
    "min_sequences_per_context": 10,
    "context_relevance_threshold": 0.7
  }
}
```

## Usage

### Basic Usage

```python
# Initialize Q-learning engine
config = load_config()
q_engine = QLearningEngine(config)

# Extract context from query
context_extractor = ContextExtractor()
user_context = context_extractor.extract_context(
    query="Find and analyze Python files",
    user_stats={'success_rate': 0.8, 'query_count': 50}
)

# Select action (tool combination) with context
state = encode_current_state()
available_tools = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp']
constraints = {'conflicts': {}, 'requires': {}}

action = await q_engine.select_action(state, available_tools, constraints, context=user_context)

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

The Q-learning system uses several tables for persistence, including enhanced tables for failure tracking:

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

-- Failure history tracking
CREATE TABLE failure_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    recovery_successful BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resource metrics tracking
CREATE TABLE resource_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    memory_mb REAL,
    cpu_percent REAL,
    api_calls INTEGER,
    execution_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback tracking
CREATE TABLE user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    rating INTEGER,
    query_reformulated BOOLEAN DEFAULT FALSE,
    result_used BOOLEAN,
    follow_up_time_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tool synergies tracking
CREATE TABLE tool_synergies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_combination TEXT NOT NULL UNIQUE,
    success_rate REAL,
    occurrences INTEGER,
    synergy_score REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Enhanced Integration Features

### Partial Success Tracking
The Q-learning engine now tracks partial success scenarios:
- **Completion Percentage**: Tracks how much of a task was completed (0-100%)
- **Result Quality**: Quality score (0-1) for partial results
- **Learning Impact**: Partial successes receive proportional rewards

### Resource Efficiency Monitoring
Using psutil integration:
- **CPU Usage**: Tracked per tool execution
- **Memory Usage**: Peak memory consumption recorded
- **API Calls**: Count of external API calls made
- **Execution Time**: Millisecond precision timing

### User Satisfaction Integration
The system captures:
- **Explicit Feedback**: 1-5 star ratings from users
- **Implicit Signals**: Query reformulation, follow-up timing, result usage
- **Feedback Impact**: Directly influences reward calculation

### Tool Synergy Recognition
- **Complementary Tools**: Identified through success patterns
- **Synergy Scoring**: Quantifies how well tools work together
- **Dynamic Discovery**: New synergies discovered through exploration

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