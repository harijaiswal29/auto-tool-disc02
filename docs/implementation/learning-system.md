# Learning System Architecture

## Implementation Status

**✅ Implemented Components:**
- Q-Learning Engine (`src/learning/q_learning_engine.py`)
- State Representation with 419-dimensional vectors
- Action Space with constraint validation
- Experience Replay Buffer
- Model persistence to database
- Integration with Orchestrator Agent

**⏳ Not Yet Implemented:**
- Pattern Miner (PatternMiner class)
- Advanced Reward Calculator
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
            'intent_vector': 384,  # Sentence transformer output
            'context_features': 10,  # Domain, user history, etc.
            'tool_history': 20,    # Recent tool usage
            'performance_metrics': 5  # Success rate, response time, etc.
        }
    
    def encode_state(self, intent, context, history):
        # Combine all features into state vector
        state_vector = np.concatenate([
            intent.embedding,
            self.encode_context(context),
            self.encode_history(history),
            self.encode_metrics(context.metrics)
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

## Reward Calculation

### Reward Function Components

**Note**: The RewardCalculator class shown below is not yet implemented. Currently, reward calculation is handled directly in the Orchestrator Agent's `_calculate_reward` method.

```python
class RewardCalculator:  # NOT YET IMPLEMENTED
    def __init__(self):
        self.weights = {
            'task_completion': 0.5,
            'execution_time': 0.2,
            'resource_usage': 0.1,
            'user_feedback': 0.2
        }
    
    def calculate_reward(self, execution_result):
        components = {
            'task_completion': self.task_completion_reward(execution_result),
            'execution_time': self.time_penalty(execution_result.duration),
            'resource_usage': self.resource_penalty(execution_result.resources),
            'user_feedback': self.feedback_reward(execution_result.feedback)
        }
        
        # Weighted sum
        total_reward = sum(
            self.weights[key] * value 
            for key, value in components.items()
        )
        
        return total_reward, components
```

### Reward Formulas
1. **Task Completion**: +1.0 for success, -0.5 for failure
2. **Time Penalty**: -0.1 * log(execution_time_seconds)
3. **Resource Usage**: -0.05 * (cpu_usage + memory_usage)
4. **User Feedback**: +1.0 for positive, -1.0 for negative

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

**Note**: The PatternMiner class is not yet implemented. This is a planned feature for discovering common tool usage patterns.

```python
class PatternMiner:  # NOT YET IMPLEMENTED
    def __init__(self, min_support=0.1, min_confidence=0.8):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.pattern_db = PatternDatabase()
    
    def mine_sequential_patterns(self, execution_logs):
        # Extract tool sequences
        sequences = self.extract_sequences(execution_logs)
        
        # Apply PrefixSpan algorithm
        patterns = self.prefixspan(sequences, self.min_support)
        
        # Calculate pattern metrics
        for pattern in patterns:
            pattern.support = self.calculate_support(pattern, sequences)
            pattern.confidence = self.calculate_confidence(pattern, sequences)
            pattern.lift = self.calculate_lift(pattern, sequences)
        
        # Store high-value patterns
        valuable_patterns = [
            p for p in patterns 
            if p.confidence >= self.min_confidence
        ]
        
        self.pattern_db.store(valuable_patterns)
        return valuable_patterns
```

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

## Hyperparameter Tuning

### Grid Search Parameters
- Learning rate: [0.01, 0.05, 0.1, 0.2]
- Discount factor: [0.8, 0.9, 0.95, 0.99]
- Exploration rate: [0.1, 0.2, 0.3]
- Experience replay batch size: [16, 32, 64]

### Optimization Strategy
1. Start with default parameters
2. Run grid search on validation set
3. Select best performing combination
4. Fine-tune with Bayesian optimization