# Learning System Architecture

## Implementation Status

**✅ Implemented Components:**
- Q-Learning Engine (`src/learning/q_learning_engine.py`)
- Enhanced State Representation with 439-dimensional vectors (includes failure tracking)
- Action Space with constraint validation
- Experience Replay Buffer
- Model persistence to database
- Integration with Orchestrator Agent
- Enhanced Reward Calculator (`src/learning/reward_calculator.py`)
- Failure Learning System
- Resource Efficiency Tracking
- User Satisfaction Signals
- Tool Synergy Recognition

**⏳ Not Yet Implemented:**
- Pattern Miner (PatternMiner class)
- Deep Q-Learning with neural networks

## Overview

The learning system uses Q-learning with pattern mining to continuously improve tool selection and execution strategies. The core Q-learning engine has been implemented in `src/learning/q_learning_engine.py`.

## Q-Learning Implementation

### Configuration
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.9
- **Exploration Rate (ε)**: 0.2

### State Representation
```python
class StateRepresentation:
    def __init__(self):
        self.state_dimensions = {
            'intent_vector': 384,      # Sentence transformer output
            'context_features': 10,    # Domain, user history, etc.
            'tool_history': 20,        # Recent tool usage
            'performance_metrics': 5,   # Success rate, response time, etc.
            'failure_rates': 10,       # Per-tool failure rates
            'failure_types': 5,        # Network, permission, timeout, rate_limit, other
            'retry_patterns': 5        # Retry statistics and patterns
        }
    
    def encode_state(self, intent, context, history):
        # Combine all features into state vector
        state_vector = np.concatenate([
            intent.embedding,
            self.encode_context(context),
            self.encode_history(history),
            self.encode_metrics(context.metrics),
            self.encode_failure_rates(context.failure_rates),
            self.encode_failure_types(context.failure_types),
            self.encode_retry_patterns(context.retry_patterns)
        ])
        return state_vector
```

### Action Space Definition
```python
class ActionSpace:
    def __init__(self, max_tools=3):
        self.max_tools = max_tools
        self.tool_combinations = {}  # Cache valid combinations
    
    def get_valid_actions(self, available_tools, constraints):
        # Generate all valid tool combinations
        actions = []
        for r in range(1, min(len(available_tools), self.max_tools) + 1):
            for combo in itertools.combinations(available_tools, r):
                if self.validate_combination(combo, constraints):
                    actions.append(combo)
        return actions
```

### Q-Table Structure
```python
class QTable:
    def __init__(self, state_dim, learning_rate=0.1, discount_factor=0.9):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.q_values = {}  # Sparse representation
        self.state_encoder = StateEncoder(state_dim)
        self.update_count = defaultdict(int)
    
    def get_q_value(self, state, action):
        state_key = self.state_encoder.encode(state)
        action_key = self.encode_action(action)
        return self.q_values.get((state_key, action_key), 0.0)
    
    def update(self, state, action, reward, next_state):
        state_key = self.state_encoder.encode(state)
        action_key = self.encode_action(action)
        next_state_key = self.state_encoder.encode(next_state)
        
        # Q-learning update rule
        current_q = self.get_q_value(state, action)
        max_next_q = max(
            self.get_q_value(next_state, a) 
            for a in self.get_possible_actions(next_state)
        )
        
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        
        self.q_values[(state_key, action_key)] = new_q
        self.update_count[(state_key, action_key)] += 1
```

## Enhanced Reward Calculation

The enhanced reward calculator (`src/learning/reward_calculator.py`) provides sophisticated reward calculation with multiple factors:

### Reward Function Components

```python
class RewardCalculator:
    def calculate_reward(self, execution_results, context, user_feedback=None):
        # Calculate individual components
        base_reward = self._calculate_base_reward(execution_results)
        failure_adjustment = self._failure_type_adjustment(execution_results)
        partial_success_bonus = self._partial_success_bonus(execution_results)
        resource_penalty = self._resource_efficiency_penalty(execution_results)
        synergy_bonus = self._tool_synergy_bonus(execution_results)
        user_satisfaction = self._user_satisfaction_adjustment(user_feedback)
        
        # Apply context sensitivity
        context_multiplier = self._get_context_multiplier(context)
        
        # Apply uncertainty factor
        uncertainty_factor = self._uncertainty_adjustment(execution_results, context)
        
        # Combine all components
        total_reward = (base_reward + failure_adjustment + partial_success_bonus + 
                       synergy_bonus + user_satisfaction) * context_multiplier * 
                       uncertainty_factor - resource_penalty
        
        return np.clip(total_reward, -1.0, 2.0), breakdown
```

### Key Features

1. **Failure Type Differentiation**
   - Network timeouts: -0.2
   - Permission errors: -0.8 (severe)
   - Rate limits: -0.3
   - Retryable errors: -0.1 (light penalty)
   - Non-retryable errors: -0.7

2. **Partial Success Handling**
   - Rewards partial completion based on percentage
   - Quality-based bonuses for high-quality partial results

3. **Resource Efficiency**
   - Memory usage penalties
   - CPU usage penalties
   - API call tracking
   - Execution time penalties (logarithmic)

4. **Tool Synergy Recognition**
   - Known good combinations: +0.2
   - Discovered combinations: +0.3
   - Redundant tool penalties: -0.1

5. **Context Sensitivity**
   - Exploration mode: 1.2x multiplier
   - Production mode: 0.8x multiplier
   - Confidence-based adjustments
   - User vs system initiated adjustments

6. **User Satisfaction Signals**
   - Explicit ratings (1-5 scale)
   - Query reformulation detection
   - Follow-up timing analysis
   - Result usage tracking

## Experience Replay Buffer

```python
class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, experience, priority=None):
        self.buffer.append(experience)
        self.priorities.append(priority or self.calculate_priority(experience))
    
    def sample(self, batch_size, prioritized=True):
        if prioritized:
            # Prioritized experience replay
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
        
        return [self.buffer[i] for i in indices]
```

## Pattern Mining Architecture

The PatternMiner class (`src/learning/pattern_miner.py`) has been fully implemented to discover and analyze tool usage patterns.

### Core Components

```python
class PatternMiner:
    def __init__(self, config: Config, min_support: float = 0.1, min_confidence: float = 0.8):
        self.config = config
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = 1.0  # Only consider patterns with positive correlation
        self.discovered_patterns = {}
        self.pattern_cache = {}
    
    async def extract_sequences(self, time_window: Optional[timedelta] = None) -> List[ExecutionSequence]:
        """Extract execution sequences from database."""
        # Queries execution_history table
        # Returns ExecutionSequence objects with tools, success, reward, context
        pass
    
    async def mine_sequential_patterns(self, sequences: List[ExecutionSequence]) -> List[Pattern]:
        """Mine sequential patterns using simplified PrefixSpan algorithm."""
        # Finds patterns where order matters (A -> B -> C)
        # Calculates support, confidence, and lift
        # Filters by minimum thresholds
        pass
    
    async def mine_combination_patterns(self, sequences: List[ExecutionSequence]) -> List[Pattern]:
        """Mine combination patterns where order doesn't matter."""
        # Finds patterns of tools that work well together
        # Useful for discovering tool synergies
        pass
```

### Pattern Metrics

1. **Support**: Frequency of pattern occurrence in all sequences
   - Formula: `count(pattern) / total_sequences`
   - Minimum threshold: 0.1 (10%)

2. **Confidence**: Success rate when pattern is used
   - Formula: `successful_with_pattern / total_with_pattern`
   - Minimum threshold: 0.8 (80%)

3. **Lift**: How much more likely the pattern is compared to random
   - Formula: `P(pattern) / (P(prefix) * P(suffix))`
   - Values > 1.0 indicate positive correlation

### Integration with Q-Learning

The PatternMiner is integrated with QLearningEngine to enhance tool selection:

```python
class QLearningEngine:
    def __init__(self, config: Dict[str, Any]):
        # ... other initialization ...
        self.pattern_miner = PatternMiner(
            config,
            min_support=q_config.get('pattern_min_support', 0.1),
            min_confidence=q_config.get('pattern_min_confidence', 0.8)
        )
        self.use_patterns = q_config.get('use_patterns', True)
        self.pattern_weight = q_config.get('pattern_weight', 0.3)
    
    async def _select_best_action_with_patterns(self, state, valid_actions, current_tools):
        """Combines Q-values with pattern scores for action selection."""
        # Get Q-values for all valid actions
        q_values = await self.q_table.get_all_q_values(state, valid_actions)
        
        # Get pattern-based scores
        pattern_scores = self._calculate_pattern_scores(valid_actions, current_tools)
        
        # Weighted combination
        combined_score = (1 - self.pattern_weight) * q_value + self.pattern_weight * pattern_score
```

### Key Features

1. **Sequential Pattern Mining**: Discovers ordered tool sequences that lead to success
2. **Combination Pattern Mining**: Finds tool combinations that work well together
3. **Pattern-Based Suggestions**: Suggests next tools based on current sequence
4. **Database Persistence**: Patterns are stored and loaded from the database
5. **Performance Optimization**: Caching and efficient algorithms for real-time use

## Model Persistence

**Note**: The ModelPersistence class shown below is not implemented as a separate class. Model persistence is integrated directly into the QLearningEngine with `save_model` and `load_model` methods.

```python
class ModelPersistence:  # EXAMPLE ONLY - Integrated into QLearningEngine
    def __init__(self, base_path='./models'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save_q_table(self, q_table, version):
        filepath = self.base_path / f'q_table_v{version}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_values': q_table.q_values,
                'update_counts': q_table.update_count,
                'metadata': {
                    'version': version,
                    'timestamp': datetime.now(),
                    'alpha': q_table.alpha,
                    'gamma': q_table.gamma
                }
            }, f)
    
    def save_patterns(self, patterns, version):
        filepath = self.base_path / f'patterns_v{version}.json'
        with open(filepath, 'w') as f:
            json.dump({
                'patterns': [p.to_dict() for p in patterns],
                'metadata': {
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'count': len(patterns)
                }
            }, f)
```

## Learning Metrics

```python
class LearningMetrics:
    def __init__(self):
        self.metrics = {
            'cumulative_reward': [],
            'episode_lengths': [],
            'exploration_rate': [],
            'q_value_stats': [],
            'pattern_discovery_rate': []
        }
    
    def update(self, episode_result):
        self.metrics['cumulative_reward'].append(
            sum(episode_result.rewards)
        )
        self.metrics['episode_lengths'].append(
            len(episode_result.actions)
        )
        self.metrics['exploration_rate'].append(
            episode_result.exploration_ratio
        )
```

## Tool Selection Algorithm

```python
def select_tools(discovered_tools, context, epsilon=0.2):
    if random.random() < epsilon:
        # Exploration: try new combinations
        return random.sample(discovered_tools, k=min(3, len(discovered_tools)))
    else:
        # Exploitation: use best known tools
        scores = calculate_contextual_scores(discovered_tools, context)
        return top_k_tools(scores, k=3)
```

## Q-Learning Update Process

```python
def update_learning(execution_result):
    # Calculate reward based on execution success, time, and feedback
    reward = calculate_reward(execution_result)
    
    # Update Q-table
    q_learning.update(
        state=execution_result.context,
        action=execution_result.tools_used,
        reward=reward,
        next_state=execution_result.new_context
    )
    
    # Mine patterns if successful
    if reward > threshold:
        pattern_miner.add_successful_sequence(execution_result.tools_used)
```

## Performance Metrics

### Tracked Metrics
- Task Completion Rate
- Tool Selection Accuracy
- Learning Efficiency
- Response Time
- Resource Usage
- Exploration Rate
- Pattern Discovery Rate

### Evaluation Methods
1. **Learning Curve Analysis**: Plot cumulative reward over episodes
2. **Pattern Quality**: Measure support, confidence, and lift
3. **Convergence Rate**: Track Q-value stability
4. **Generalization**: Test on unseen queries

## Database Schema for Failure Tracking

The learning system persists failure metrics and patterns in the database:

### Failure History Table
```sql
CREATE TABLE IF NOT EXISTS failure_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    recovery_successful BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
)
```

### Resource Metrics Table
```sql
CREATE TABLE IF NOT EXISTS resource_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    memory_mb REAL,
    cpu_percent REAL,
    api_calls INTEGER,
    execution_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
)
```

### User Feedback Table
```sql
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    rating INTEGER,
    query_reformulated BOOLEAN DEFAULT FALSE,
    result_used BOOLEAN,
    follow_up_time_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
)
```

### Tool Synergies Table
```sql
CREATE TABLE IF NOT EXISTS tool_synergies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_combination TEXT NOT NULL UNIQUE,
    success_rate REAL,
    occurrences INTEGER,
    synergy_score REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Integration with Orchestrator Agent

The orchestrator agent has been enhanced to integrate with the failure learning system:

### Enhanced Tool Execution Result
```python
@dataclass
class ToolExecutionResult:
    """Result from executing a tool with enhanced metrics."""
    tool_id: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    # Enhanced fields for partial success and resource tracking
    partial_success: bool = False
    completion_percentage: float = 0.0
    error_type: Optional[str] = None
    retry_count: int = 0
    resource_usage: Optional[Dict[str, float]] = None
    result_quality: float = 1.0  # Quality score 0-1
```

### Key Integration Points

1. **Error Classification**: The orchestrator classifies errors into types (network_timeout, permission_error, rate_limit, etc.)
2. **Partial Success Detection**: Checks for partial results in failed executions
3. **Resource Tracking**: Uses psutil to monitor CPU, memory, and tracks API calls
4. **Failure Metrics Update**: Updates failure rates using exponential moving average
5. **Reward Calculation**: Uses the enhanced reward calculator instead of simple calculation

## User Feedback System

### Recording User Feedback
```python
# Example: Recording user feedback after execution
await orchestrator.record_user_feedback(
    execution_id="exec-123",
    feedback_type="positive",
    rating=5,
    result_used=True
)

# Negative feedback with query reformulation
await orchestrator.record_user_feedback(
    execution_id="exec-456", 
    feedback_type="negative",
    rating=2,
    query_reformulated=True,
    follow_up_time_seconds=3.5
)
```

### Feedback Signals
- **Explicit Feedback**: User ratings (1-5 scale)
- **Implicit Signals**:
  - Query reformulation detection (Jaccard similarity < 0.3)
  - Follow-up query timing
  - Result usage tracking

## Performance Considerations

### Failure Rate Tracking
- Uses exponential moving average (α=0.2) for smooth updates
- Per-tool failure rates tracked individually
- Global failure type distribution maintained

### Resource Monitoring
- CPU and memory tracked via psutil
- API call counting integrated into tool execution
- Resource penalties calculated logarithmically to avoid harsh penalties

### State Encoding Efficiency
- 439-dimensional state vector efficiently encoded
- MD5 hashing for state identification
- Sparse Q-table representation for memory efficiency

## Configuration Tuning

For detailed configuration information, see [Configuration Guide](../deployment/configuration.md).

### Reward Weights
Adjust these based on your priorities:
```json
"base_weights": {
    "success": 1.0,       # Increase for success-focused learning
    "failure": -0.5,      # Make more negative for risk-averse behavior
    "partial_success": 0.3 # Increase to encourage partial results
}
```

### Failure Penalties
Customize based on error severity in your environment:
```json
"failure_penalties": {
    "network_timeout": -0.2,    # Transient, likely retryable
    "permission_error": -0.8,   # Severe, unlikely to succeed on retry
    "rate_limit": -0.3,        # May succeed after delay
    "connection_error": -0.25,  # Could be temporary
    "retryable": -0.1,         # Light penalty for retryable errors
    "non_retryable": -0.7,     # Heavy penalty for permanent failures
    "unknown": -0.5            # Default for unclassified errors
}
```

### Context Multipliers
Adjust learning behavior in different modes:
```json
"context_multipliers": {
    "exploration": 1.2,     # Encourage trying new things
    "production": 0.8,      # Conservative in production
    "high_confidence": 1.0, # Standard rewards when confident
    "low_confidence": 1.1,  # Slightly boost learning when uncertain
    "user_initiated": 1.0,  # User queries get full rewards
    "system_initiated": 0.9 # System queries slightly discounted
}
```

## Hyperparameter Tuning

### Grid Search Parameters
- Learning rate: [0.01, 0.05, 0.1, 0.2]
- Discount factor: [0.8, 0.9, 0.95, 0.99]
- Exploration rate: [0.1, 0.2, 0.3]
- Experience replay batch size: [16, 32, 64]
- Failure rate smoothing (α): [0.1, 0.2, 0.3]

### Optimization Strategy
1. Start with default parameters
2. Run grid search on validation set
3. Select best performing combination
4. Fine-tune with Bayesian optimization
5. Monitor failure learning convergence