# Evaluation Strategy Details

## Overview
The evaluation framework implements 7 distinct strategies for tool selection, ranging from simple baselines to sophisticated learning approaches. Each strategy represents a different approach to the autonomous tool discovery problem.

## Strategy Comparison Table

| Strategy | Type | Learning | State-Aware | Context-Aware | Complexity |
|----------|------|----------|-------------|---------------|------------|
| Random | Baseline | No | No | No | O(1) |
| Popular | Heuristic | Passive | No | No | O(n) |
| Fixed Policy | Rule-based | No | Partial | No | O(1) |
| Greedy | Heuristic | Passive | No | No | O(n) |
| Context-Agnostic Q-Learning | RL | Active | Limited | No | O(n²) |
| Q-Learning Tabular | RL | Active | Full | Yes | O(n²) |
| Q-Learning DQN | Deep RL | Active | Full | Yes | O(n) |

## Detailed Strategy Descriptions

### 1. Random Selection Baseline
**File**: `src/evaluation/baseline_strategies.py:84-125`

#### Description
The simplest baseline that randomly selects tools without any learning or heuristics.

#### Key Features
- Randomly selects 1-3 tools from available options
- Applies basic constraint checking (conflicts)
- No learning or adaptation
- Serves as the absolute baseline for comparison

#### Algorithm
```python
1. Randomly determine number of tools to select (1 to max_tools)
2. Randomly sample that many tools from available
3. Apply conflict constraints
4. Return selected tools
```

#### Performance Characteristics
- **Tool Accuracy**: ~10% (chance level for 10 tools)
- **Task Completion**: 19.7% (low due to random selection)
- **Selection Time**: <1ms (no computation)
- **Memory**: O(1) (no state storage)

#### Use Case
Baseline for statistical comparison; demonstrates improvement from learning.

---

### 2. Most Popular Tools Baseline
**File**: `src/evaluation/baseline_strategies.py:127-185`

#### Description
Selects tools based on historical usage frequency across all contexts.

#### Key Features
- Tracks tool usage frequency globally
- Falls back to random when insufficient history
- Applies constraint checking
- Simple passive learning through frequency counting

#### Algorithm
```python
1. If history < min_history: use random selection
2. Sort available tools by usage frequency
3. Select top N tools (up to max_tools)
4. Apply conflict constraints
5. Update popularity counts after each use
```

#### Performance Characteristics
- **Tool Accuracy**: 20% (2x random)
- **Task Completion**: 30.9% (highest among all)
- **Selection Time**: <5ms (sorting overhead)
- **Memory**: O(t) where t = number of unique tools

#### Use Case
Simple production baseline; works well when certain tools are universally useful.

---

### 3. Fixed Policy Baseline
**File**: `src/evaluation/baseline_strategies.py:187-248`

#### Description
Pre-defined rule-based mappings between intent types and tool selections.

#### Key Features
- Hard-coded intent-to-tool mappings
- No learning or adaptation
- Fast deterministic selection
- Falls back to default policy

#### Policy Mappings
```python
intent_policies = {
    'file_search': ['filesystem_mcp', 'search_mcp'],
    'data_query': ['sqlite_mcp', 'postgres_mcp'],
    'code_analysis': ['github_mcp', 'filesystem_mcp'],
    'web_search': ['search_mcp'],
    'weather_query': ['weather_mcp'],
    'default': ['filesystem_mcp']
}
```

#### Performance Characteristics
- **Tool Accuracy**: 10% (poor due to rigid rules)
- **Task Completion**: 13.1% (worst performance)
- **Selection Time**: <1ms (lookup only)
- **Memory**: O(1) (static rules)

#### Use Case
Legacy systems; when domain is well-understood and static.

---

### 4. Greedy Single Tool Baseline
**File**: `src/evaluation/baseline_strategies.py:250-289`

#### Description
Always selects the single tool with highest average historical reward.

#### Key Features
- Tracks per-tool average rewards
- Selects single best-performing tool
- Simple exploitation-only strategy
- No exploration after initial learning

#### Algorithm
```python
1. Calculate average reward for each available tool
2. If insufficient history: random selection
3. Select tool with highest average reward
4. Return single tool only
```

#### Performance Characteristics
- **Tool Accuracy**: 10% (limited by single-tool constraint)
- **Task Completion**: 18.6% (moderate)
- **Selection Time**: <2ms (average calculation)
- **Memory**: O(t×h) where h = history length

#### Use Case
Simple tasks requiring single tool; risk-averse scenarios.

---

### 5. Context-Agnostic Q-Learning Baseline
**File**: `src/evaluation/baseline_strategies.py:291-370`

#### Description
Q-Learning implementation that ignores contextual features, using only tool availability as state.

#### Key Features
- Simplified state representation (hash of available tools)
- Standard Q-learning with epsilon-greedy exploration
- Limited generalization capability
- Faster learning for simple patterns

#### State Representation
```python
# Ignores intent and context features
state_key = hash(sorted(available_tools))
```

#### Performance Characteristics
- **Tool Accuracy**: ~15% (limited by state representation)
- **Task Completion**: ~20% (moderate)
- **Selection Time**: 5-10ms (Q-table lookup)
- **Memory**: O(s×a) where s = unique tool sets

#### Use Case
Environments with limited context; fast prototyping.

---

### 6. Q-Learning Tabular Strategy
**File**: `src/evaluation/dqn_strategy.py:126-198`

#### Description
Full Q-learning implementation with complete 476-dimensional state representation.

#### Key Features
- Full state encoding (intent, context, history)
- Tabular Q-learning with state discretization
- Epsilon-greedy exploration
- Constraint-aware action space

#### State Components
```python
State Vector (476 dimensions):
- Intent embedding: 384 dims (Sentence-BERT)
- Tool availability: 32 dims (binary)
- Context features: 20 dims
- Historical performance: 20 dims
- Constraint encoding: 20 dims
```

#### Learning Parameters
- **Learning Rate (α)**: 0.2 (tool-optimized) / 0.1 (standard)
- **Discount Factor (γ)**: 0.95 / 0.9
- **Exploration (ε)**: 0.3 → 0.05 (decaying)

#### Performance Characteristics
- **Tool Accuracy**: 30% (3x random) ✅
- **Task Completion**: 18.5% (with tool-optimized rewards)
- **Selection Time**: 10-20ms (state encoding + lookup)
- **Memory**: O(s×a) potentially large

#### Use Case
Primary learning strategy; best for discrete state spaces.

---

### 7. Q-Learning DQN Strategy
**File**: `src/evaluation/dqn_strategy.py:22-123`

#### Description
Deep Q-Network implementation using neural networks for value approximation.

#### Key Features
- Neural network function approximation
- Experience replay buffer
- Target network for stability
- Optional: Dueling architecture, prioritized replay

#### Network Architecture
```python
DQN Architecture:
Input Layer: 476 neurons (state vector)
Hidden Layer 1: 512 neurons + ReLU + Dropout(0.2)
Hidden Layer 2: 512 neurons + ReLU + Dropout(0.16)
Hidden Layer 3: 256 neurons + ReLU + Dropout(0.13)
Hidden Layer 4: 128 neurons + ReLU + Dropout(0.1)
Output Layer: |A| neurons (Q-values per action)
```

#### Advanced Features
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage
- **Prioritized Replay**: Samples important experiences more
- **Noisy Networks**: Alternative exploration mechanism

#### Performance Characteristics
- **Tool Accuracy**: 25-35% (with sufficient training)
- **Task Completion**: Variable (depends on training)
- **Selection Time**: 20-50ms (forward pass)
- **Memory**: O(B) where B = replay buffer size (100K)

#### Use Case
Complex environments; continuous learning; large state spaces.

---

## Strategy Selection Guide

### When to Use Each Strategy

| Scenario | Recommended Strategy | Reasoning |
|----------|---------------------|-----------|
| Cold Start (No Data) | Random → Popular | Start random, transition to frequency-based |
| Simple Domain | Fixed Policy | When mappings are well-known |
| Limited Context | Context-Agnostic Q-Learning | Faster learning with less state |
| Rich Context Available | Q-Learning Tabular | Full state utilization |
| Large State Space | Q-Learning DQN | Neural approximation scales better |
| Production Baseline | Popular Tools | Simple, effective, explainable |
| A/B Testing Control | Random | Unbiased baseline |

## Performance Analysis

### Experiment Results (10 Episodes, Tool-Optimized Rewards)

```
Strategy               Tool Accuracy    Task Completion    Learning
-----------------------------------------------------------------
Q-Learning Tabular          30%             18.5%            Yes
Popular Tools               20%             30.9%            Passive
Random                      10%             19.7%            No
Greedy Single              10%             18.6%            Passive
Fixed Policy               10%             13.1%            No
```

### Key Insights

1. **Learning Superiority**: Q-learning achieves 3x tool accuracy over random
2. **Heuristic Value**: Popular strategy achieves 2x improvement with simple counting
3. **Context Importance**: Full state representation critical for 30% accuracy
4. **Trade-offs**: Tool accuracy vs task completion (inverse relationship observed)

## Implementation Notes

### Common Interfaces
All strategies implement the `BaselineStrategy` abstract class:
```python
async def select_tools(state, available_tools, constraints) -> List[str]
def update(state, action, reward, next_state) -> None
def get_statistics() -> Dict[str, Any]
```

### Constraint Handling
All strategies respect:
- Tool conflicts (mutually exclusive tools)
- Tool requirements (dependencies)
- Maximum tool limit (typically 3)

### Evaluation Integration
Strategies are initialized in `EvaluationEngine._initialize_strategies()` based on configuration:
```python
config['evaluation']['baselines'] = [
    'random', 'popular', 'fixed_policy', 
    'greedy', 'context_agnostic', 
    'q_learning_tabular', 'q_learning_dqn'
]
```

## Future Enhancements

### Proposed Strategies
1. **Ensemble Methods**: Combine multiple strategies with weighted voting
2. **Meta-Learning**: Learn which strategy to use based on context
3. **Transfer Learning**: Pre-train on similar domains
4. **Hierarchical RL**: Decompose into sub-policies
5. **Imitation Learning**: Learn from expert demonstrations

### Optimization Opportunities
1. **State Representation**: Dimensionality reduction, feature engineering
2. **Action Space**: Hierarchical actions, macro-actions
3. **Reward Shaping**: Multi-objective optimization
4. **Exploration**: Curiosity-driven, information gain
5. **Memory**: Episodic memory, attention mechanisms

---

*Last Updated: 2025-08-17*
*Version: 1.0*