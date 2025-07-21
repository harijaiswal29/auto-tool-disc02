# Deep Q-Learning Implementation

## Overview

This document describes the Deep Q-Learning (DQN) implementation that enhances the autonomous tool discovery system by replacing tabular Q-learning with neural network-based value function approximation.

## Architecture

### Key Components

1. **Deep Q-Network Module** (`src/learning/deep_q_network.py`)
   - Standard DQN architecture
   - Dueling DQN variant
   - Noisy DQN for exploration
   - Factory pattern for network creation

2. **DQN Agent** (`src/learning/dqn_agent.py`)
   - Target network for stability
   - Double DQN to reduce overestimation
   - Experience replay management
   - Epsilon-greedy exploration

3. **Prioritized Experience Replay** (`src/learning/prioritized_replay_buffer.py`)
   - Sum-tree data structure
   - TD-error based prioritization
   - Importance sampling weights

4. **Training Utilities** (`src/learning/dqn_trainer.py`)
   - Training loop management
   - Learning rate scheduling
   - Model evaluation and checkpointing
   - Performance visualization

## Network Architectures

### Standard DQN
```python
Input (439 dims) → FC(512) → ReLU → Dropout(0.2) → 
FC(256) → ReLU → Dropout(0.2) → 
FC(128) → ReLU → Dropout(0.2) → 
Output (action_dim)
```

### Dueling DQN
```python
Input (439 dims) → Shared Features → 
├─ Value Stream → FC(128) → ReLU → FC(1)
└─ Advantage Stream → FC(128) → ReLU → FC(action_dim)
Output = V(s) + A(s,a) - mean(A(s,·))
```

### Noisy DQN
- Replaces epsilon-greedy exploration
- Adds learnable noise to network weights
- Automatically adjusts exploration during training

## Configuration

### Enabling DQN

Set in `config/config.json`:
```json
{
  "dqn": {
    "enabled": true,
    "network_type": "standard",  // or "dueling", "noisy"
    "network_architecture": [512, 256, 128],
    "learning_rate": 0.0001,
    "batch_size": 64,
    "memory_size": 100000,
    "target_update_frequency": 1000,
    "double_dqn": true,
    "prioritized_replay": true,
    "device": "auto"  // or "cpu", "cuda"
  }
}
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0001 | Neural network learning rate |
| `discount_factor` | 0.99 | Future reward discount |
| `tau` | 0.001 | Soft target update rate |
| `batch_size` | 64 | Training batch size |
| `memory_size` | 100000 | Experience replay capacity |
| `target_update_frequency` | 1000 | Steps between target updates |
| `gradient_clip` | 1.0 | Gradient clipping threshold |

## Usage

### Basic Training

```python
# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Enable DQN
config['dqn']['enabled'] = True

# Create Q-learning engine
engine = QLearningEngine(config)

# Training loop
for episode in range(num_episodes):
    state, tools, constraints = env.reset()
    
    while not done:
        # Select action using DQN
        action = await engine.select_action(state, tools, constraints)
        
        # Execute in environment
        next_state, reward, done, info = env.step(action)
        
        # Learn from experience
        await engine.learn_from_experience(
            state, action, reward, next_state,
            info['tools'], info['constraints'], done
        )
        
        state = next_state
```

### Using DQN Trainer

```python
from src.learning.dqn_trainer import DQNTrainer

# Create trainer
trainer = DQNTrainer(engine)

# Setup learning rate scheduler
trainer.setup_lr_scheduler('cosine', T_max=100000)

# Train
await trainer.train(env_step_func, num_episodes=1000)

# Plot results
trainer.plot_training_curves('training_curves.png')
```

## Advantages over Tabular Q-Learning

### 1. Generalization
- **Tabular**: Stores value for each state-action pair separately
- **DQN**: Generalizes across similar states using neural networks

### 2. State Space Handling
- **Tabular**: Limited to discrete, small state spaces
- **DQN**: Handles continuous, high-dimensional states (439 dims)

### 3. Memory Efficiency
- **Tabular**: O(|S| × |A|) memory requirement
- **DQN**: O(network_parameters) - constant size

### 4. Learning Speed
- **Tabular**: Slow to propagate values across states
- **DQN**: Faster learning through generalization

### 5. Transfer Learning
- **Tabular**: No transfer between similar tasks
- **DQN**: Pre-trained networks can be fine-tuned

## Implementation Details

### State Encoding
The 439-dimensional state vector includes:
- Intent embeddings from sentence transformers
- Context features (domain, session info)
- Tool usage history
- Performance metrics
- Failure tracking information

### Action Space
- Variable action space handled through action mapping
- Invalid actions masked during selection
- Supports tool combinations up to `max_tools`

### Experience Replay
- Stores transitions in circular buffer
- Prioritized sampling based on TD-error
- Importance sampling to correct bias

### Target Network
- Separate network for stable Q-targets
- Updated every `target_update_frequency` steps
- Soft updates with parameter τ (default 0.001)

### Double DQN
- Uses online network to select actions
- Uses target network to evaluate Q-values
- Reduces overestimation bias

## Performance Optimizations

### GPU Acceleration
- Automatic device selection
- Batch processing on GPU
- Efficient tensor operations

### Memory Management
- Experience replay with fixed capacity
- Efficient state encoding
- Gradient accumulation for large batches

### Training Efficiency
- Parallel environment interaction
- Asynchronous model updates
- Cached action space computations

## Monitoring and Debugging

### Metrics Tracked
- Training loss
- Episode rewards
- Exploration rate (epsilon)
- Learning rate
- Q-value statistics
- Memory usage

### Visualization
- Learning curves
- Q-value distributions
- Action selection frequencies
- Gradient norms

### Checkpointing
- Periodic model saves
- Best model tracking
- Training state preservation

## Best Practices

### 1. Hyperparameter Tuning
- Start with default values
- Use grid search or Bayesian optimization
- Monitor validation performance

### 2. Network Architecture
- Deeper networks for complex tasks
- Dueling DQN for value-based problems
- Noisy networks for better exploration

### 3. Training Stability
- Gradient clipping to prevent explosions
- Learning rate scheduling
- Proper weight initialization

### 4. Memory Management
- Appropriate replay buffer size
- Regular memory cleanup
- Efficient state representations

## Troubleshooting

### Common Issues

1. **Diverging Q-values**
   - Reduce learning rate
   - Increase target update frequency
   - Enable gradient clipping

2. **Poor Performance**
   - Increase network capacity
   - Tune exploration parameters
   - Check reward function

3. **Memory Issues**
   - Reduce replay buffer size
   - Use smaller batch sizes
   - Enable memory profiling

4. **Slow Training**
   - Enable GPU acceleration
   - Reduce network size
   - Use parallel environments

## Future Enhancements

1. **Rainbow DQN**: Combine all DQN improvements
2. **Continuous Actions**: Support for continuous action spaces
3. **Meta-Learning**: Quick adaptation to new tools
4. **Explainability**: Visualize decision process

## References

1. Mnih et al. "Human-level control through deep reinforcement learning" (2015)
2. Van Hasselt et al. "Deep reinforcement learning with double Q-learning" (2016)
3. Wang et al. "Dueling network architectures for deep reinforcement learning" (2016)
4. Schaul et al. "Prioritized experience replay" (2016)
5. Fortunato et al. "Noisy networks for exploration" (2018)