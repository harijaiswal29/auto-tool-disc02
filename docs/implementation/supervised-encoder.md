# Supervised State Encoder Implementation

## Overview

The Supervised State Encoder is a neural network-based dimensionality reduction technique that addresses the sparse 476-dimensional state space issue in the Q-learning system. It reduces the state space to 50 dimensions using supervised learning with contrastive loss, enabling more efficient learning and better generalization.

## Architecture

### Neural Network Structure

```
Input Layer (476) → Hidden Layer 1 (256) → Hidden Layer 2 (128) → Latent Space (50)
```

- **Input**: 476-dimensional state vector (from StateRepresentation)
- **Hidden Layers**: Two fully connected layers with ReLU activation
- **Batch Normalization**: Applied after each hidden layer
- **Dropout**: 0.3 rate for regularization (0.15 before output)
- **Output**: 50-dimensional encoded representation

### Projection Head (Training Only)

During training, an additional projection head is used for contrastive learning:
```
Latent Space (50) → Linear (50) → ReLU → Linear (50) → Projected Space (50)
```

## Training Process

### 1. Data Extraction

The encoder uses labeled data from existing training runs:

```python
# Extract data from latest training run
python src/learning/encoder_data_extractor.py \
    --strategy episode \
    --balance \
    --output data/encoder_training_data.npz
```

#### Labeling Strategies

1. **Episode-level** (Default):
   - Success episodes → label = 1.0
   - Failed episodes → label = 0.0
   - All states in episode get same label

2. **Trajectory-based**:
   - Progressive labels based on position in episode
   - Success: 0.5 → 1.0
   - Failure: 0.0 → 0.5

3. **Reward-based**:
   - Continuous labels using tanh(reward/20)
   - Range: [-1, 1]

### 2. Training Script

```bash
# Train the encoder
python scripts/train_supervised_encoder.py \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --loss-type contrastive \
    --patience 20 \
    --save-dir models/supervised_encoder
```

#### Key Parameters

- **Epochs**: 100 (with early stopping)
- **Batch Size**: 256
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Loss Function**: Contrastive or Triplet
- **Early Stopping**: Patience of 20 epochs

### 3. Loss Functions

#### Contrastive Loss
- Pulls together embeddings from same class (success/failure)
- Pushes apart embeddings from different classes
- Temperature parameter: 0.07

#### Triplet Loss
- Ensures anchor-positive distance < anchor-negative distance
- Margin: 1.0

## Integration with Q-Learning

### Configuration

Add to `config/config.json`:

```json
{
  "state_encoder": {
    "enabled": false,
    "model_path": "models/supervised_encoder/best_encoder.pth",
    "encoding_dim": 50,
    "use_for_training": true,
    "use_for_inference": true,
    "fallback_to_raw": true,
    "batch_encoding": true,
    "batch_size": 256
  }
}
```

### Q-Learning Engine Integration

The encoder is integrated into the `StateRepresentation` class:

```python
class StateRepresentation:
    def __init__(self, use_encoder=False, encoder_path=None):
        # Load encoder if specified
        if use_encoder and encoder_path:
            self._load_encoder(encoder_path)
        
    def encode_state(self, intent, context, history):
        # Build 476-dim state vector
        state_vector = self._build_raw_state(...)
        
        # Apply encoder if available
        if self.use_encoder and self.encoder:
            state_vector = self._encode_with_supervised_encoder(state_vector)
        
        return state_vector
```

### DQN Compatibility

The DQN automatically adjusts input dimension based on encoder usage:

```python
if self.state_encoder.use_encoder and self.state_encoder.encoder:
    state_dim = self.state_encoder.encoder_dim  # 50
else:
    state_dim = self.state_encoder.total_dimensions  # 476

self.dqn_agent = DQNAgent(config, state_dim, max_actions)
```

## Usage Examples

### 1. Training with Encoder

```bash
# Using curriculum learning script
python scripts/run_curriculum_learning_eval_optimized.py \
    --episodes 500 \
    --use-encoder \
    --encoder-path models/supervised_encoder/best_encoder.pth
```

### 2. Evaluating Encoder Impact

```bash
# Compare performance with and without encoder
python scripts/evaluate_encoder_impact.py \
    --episodes 100 \
    --encoder-path models/supervised_encoder/best_encoder.pth \
    --plot
```

### 3. Extracting Training Data

```python
from src.learning.encoder_data_extractor import EncoderDataExtractor

# Extract from latest run
extractor = EncoderDataExtractor()
result = extractor.extract_from_directory(
    labeling_strategy="episode"
)

# Balance dataset
balanced = extractor.create_balanced_dataset(balance_ratio=1.0)

# Save for training
extractor.save_extracted_data("data/training_data.npz")
```

### 4. Using Trained Encoder

```python
from src.learning.state_encoder import load_encoder, encode_states

# Load encoder
encoder = load_encoder("models/supervised_encoder/best_encoder.pth")

# Encode states
states = np.random.randn(100, 476)  # Raw states
encoded = encode_states(encoder, states)  # Returns (100, 50)
```

## Performance Analysis

### Expected Improvements

1. **Convergence Speed**: 30-50% faster convergence
2. **Final Performance**: 10-20% improvement in completion rate
3. **Memory Efficiency**: 9.5x reduction in state size
4. **Generalization**: Better performance on unseen queries

### Metrics to Monitor

- **Separation Ratio**: Inter-class distance / Intra-class distance (higher is better)
- **Embedding Statistics**: Mean, std of encoded representations
- **Training Loss**: Contrastive loss convergence
- **Validation Loss**: Early stopping indicator

## Troubleshooting

### Common Issues

1. **Encoder Not Loading**
   ```
   Warning: Encoder enabled but model not found
   ```
   - Ensure model file exists at specified path
   - Check PyTorch version compatibility

2. **Poor Encoding Quality**
   ```
   Low separation ratio < 1.5
   ```
   - Increase training epochs
   - Try different labeling strategy
   - Balance dataset

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - Reduce batch size
   - Use CPU instead: `--cpu` flag

4. **No Improvement with Encoder**
   - Check if encoder is properly trained
   - Verify data quality and labeling
   - Try different hidden dimensions

## Advanced Configuration

### Custom Architecture

Modify hidden dimensions:
```python
encoder = SupervisedStateEncoder(
    input_dim=476,
    hidden_dims=[512, 256, 128],  # Deeper network
    latent_dim=30,  # Smaller encoding
    dropout_rate=0.4  # More regularization
)
```

### Alternative Training Strategies

1. **Curriculum Training**: Train on simple queries first
2. **Transfer Learning**: Pre-train on related task
3. **Multi-task Learning**: Add auxiliary prediction heads

## Testing

### Unit Tests

```bash
# Run encoder-specific tests
python -m pytest tests/unit/test_supervised_encoder.py -v
```

### Integration Tests

```bash
# Test full pipeline with encoder
python -m pytest tests/integration/test_encoder_integration.py -v
```

## Files and Modules

- **`src/learning/state_encoder.py`**: Core encoder implementation
- **`src/learning/encoder_data_extractor.py`**: Data extraction utilities
- **`scripts/train_supervised_encoder.py`**: Training script
- **`scripts/evaluate_encoder_impact.py`**: Evaluation script
- **`tests/unit/test_supervised_encoder.py`**: Unit tests
- **`config/config.json`**: Configuration with encoder settings

## Theoretical Background

### Why Supervised Encoding?

1. **Task-Specific Compression**: Learns representations relevant to success/failure
2. **Non-linear Dimensionality Reduction**: Captures complex patterns better than PCA
3. **Contrastive Learning**: Creates well-separated clusters in latent space

### Comparison with Alternatives

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Supervised Encoder** | Task-specific, non-linear, good separation | Needs labeled data, training time | When labeled data available |
| **VAE** | Generative, probabilistic | Complex, slower inference | Uncertainty quantification |
| **iPCA** | Simple, fast, no training | Linear only, not task-specific | Quick baseline |
| **Raw States** | No preprocessing | High dimensionality, sparse | Small state spaces |

## Future Enhancements

1. **Online Learning**: Update encoder during training
2. **Attention Mechanisms**: Focus on important features
3. **Hierarchical Encoding**: Multi-scale representations
4. **Ensemble Encoders**: Multiple encoders for robustness