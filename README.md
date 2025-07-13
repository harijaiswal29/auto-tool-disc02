# Autonomous Tool Discovery and Integration System

An AI dissertation project implementing autonomous tool discovery and integration through Model Context Protocol (MCP). The system uses Q-learning and sentence transformers to enable agents to discover, learn, and optimize tool usage autonomously.

## Overview

This system provides an intelligent framework for:
- **Intent Recognition**: Understanding user queries through NLP and semantic analysis
- **Tool Discovery**: Finding relevant tools based on capabilities and relationships
- **Learning & Optimization**: Improving tool selection through Q-learning
- **Execution & Monitoring**: Managing tool execution with performance tracking
- **Continuous Adaptation**: Learning from patterns and user feedback

## Features

- **Modular Pipeline Architecture**: 7-stage processing pipeline with pluggable components
- **Real-time Performance Monitoring**: Comprehensive metrics collection and analysis
- **Conversation State Management**: Intelligent state machine for user interactions
- **Multi-Intent Query Support**: Handle complex queries with multiple intents
- **Context Persistence**: User profiles and session management
- **Semantic Understanding**: Using sentence-transformers (all-MiniLM-L6-v2)
- **Comprehensive Testing**: >90% test coverage with unit and integration tests
- **Asynchronous Execution**: High-performance async/await architecture
- **Retry and Resilience System**: Exponential backoff, circuit breakers, and connection pooling
- **Fault Tolerance**: Automatic retry with jitter and intelligent failure handling
- **Connection Pool Management**: Health checks and efficient resource utilization
- **Retry Metrics & Alerting**: Comprehensive monitoring of retry patterns and failures
- **Extensible Design**: Easy to add new tools and capabilities

## Architecture

The system consists of 5 core layers:

1. **Intent Recognition Layer** - 7-stage NLP pipeline with semantic understanding
   - Text preprocessing and normalization
   - Tokenization with question detection
   - Feature extraction using sentence-transformers
   - Intent classification with confidence scoring
   - Context enrichment and state management
   
2. **Tool Discovery Layer** - Graph-based tool exploration and matching
   - Semantic search with embedding similarity
   - Capability-based matching
   - Tool relationship graph traversal
   
3. **Tool Selection & Learning Layer** - Q-learning optimization
   - Epsilon-greedy exploration strategy
   - Experience replay buffer
   - Pattern mining for tool combinations
   
4. **Execution & Monitoring Layer** - Parallel execution with resilience
   - Asynchronous tool execution with connection pooling
   - Real-time performance tracking and metrics
   - Exponential backoff retry with configurable policies
   - Circuit breakers for fault tolerance
   - Connection health checking and reuse
   - Comprehensive retry metrics and alerting
   
5. **Learning & Adaptation Layer** - Pattern mining and model adaptation
   - Continuous learning from feedback
   - Performance-based reward calculation
   - Model persistence and versioning

## Project Status

**Current Phase**: Phase 4 - Learning System (Weeks 9-11)

### Completed Phases:
- ✅ **Phase 1**: Foundation (Core infrastructure, logging, configuration)
- ✅ **Phase 2**: Tool Ecosystem (MCP integrations, tool registry)
- ✅ **Phase 3**: Core Intelligence (Intent recognition, tool discovery, orchestration)

### Recent Accomplishments:
- ✅ **Retry and Resilience System**: Exponential backoff, circuit breakers, connection pooling
- ✅ **Comprehensive Retry Metrics**: Monitoring, alerting, and failure pattern analysis
- ✅ **Fault Tolerance**: Automatic recovery with configurable policies per service

### Performance Achievements:
- Intent Recognition: <50ms average, <100ms p95
- Test Coverage: >90%
- Cache Hit Rate: >70%
- Classification Accuracy: >90%
- Retry Success Rate: >90% (with exponential backoff)
- Circuit Breaker Effectiveness: Prevents cascading failures

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- SQLite (included with Python)

### Key Dependencies

- `sentence-transformers` - For semantic understanding (all-MiniLM-L6-v2 model)
- `asyncio` - For asynchronous execution
- `aiosqlite` - For async database operations
- `networkx` - For tool relationship graphs
- `scikit-learn` - For machine learning utilities
- `pytest` - For testing framework
- `black`, `flake8`, `mypy` - For code quality

### Installation

```bash
# Clone the repository
git clone https://github.com/harijaiswal29/auto-tool-disc
cd auto-tool-disc

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

### Running the System

```bash
# Run the main application
python src/main.py

# Run integrated demo
python test_integration_demo.py

# Run all tests
pytest tests/ -v

# Test retry and resilience system
python demo_retry_logic.py
pytest tests/test_retry_logic.py -v

# Monitor retry metrics
python -c "from src.monitoring.retry_metrics import RetryMetricsCollector; from src.core.tool_registry import ToolRegistry; collector = RetryMetricsCollector(ToolRegistry()); print(collector.get_retry_statistics())"
```

## Testing

The project includes comprehensive test suites:

### Unit Tests
```bash
# Test individual pipeline stages
pytest tests/unit/test_intent_pipeline_stages.py -v

# Test state machine components
pytest tests/unit/test_conversation_state_machine.py -v
```

### Integration Tests
```bash
# Test Intent Recognition integration
pytest tests/integration/test_intent_recognition_integration.py -v

# Test complete system integration
pytest tests/test_integration.py -v
```

### Coverage Report
```bash
pytest --cov=src tests/ --cov-report=html
# Open htmlcov/index.html in browser
```

## Performance Monitoring

The system includes comprehensive performance monitoring:

### Intent Recognition Metrics
- Processing time (avg, p50, p95, p99)
- Classification accuracy
- Cache hit rates
- Pipeline stage performance
- Error rates and types

### Retry and Resilience Metrics
- Total retry attempts and success rates
- Circuit breaker states and transitions
- Retry delay statistics (avg, p95, p99)
- Failure pattern analysis
- Per-service retry performance

### Accessing Metrics
```python
from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.monitoring.retry_metrics import RetryMetricsCollector
from src.core.tool_registry import ToolRegistry

# Intent Recognition metrics
agent = IntentRecognitionAgent()
# Process queries...
metrics = agent.get_metrics_summary()
agent.export_metrics("metrics_report.json")

# Retry metrics
collector = RetryMetricsCollector(ToolRegistry())
retry_stats = collector.get_retry_statistics()
alerts = collector.check_alerts()
```

## Performance Benchmarks

### Achieved Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Intent Recognition (p95) | <100ms | <100ms | ✅ |
| Intent Recognition (avg) | <50ms | ~45ms | ✅ |
| Classification Accuracy | >90% | >92% | ✅ |
| Cache Hit Rate | >70% | ~78% | ✅ |
| Test Coverage | >90% | >90% | ✅ |
| Pipeline Stage Overhead | <25ms | <20ms | ✅ |

### Performance by Pipeline Stage

- Text Preprocessing: <5ms
- Tokenization: <2ms
- Feature Extraction: <50ms (with caching)
- Intent Classification: <20ms
- Context Enrichment: <5ms
- Confidence Scoring: <3ms

## Configuration

Edit `config/config.json` to customize:
- Intent recognition thresholds
- Learning parameters
- Tool discovery weights
- Performance limits
- Monitoring settings

Key configuration options:
```json
{
  "intent_recognition": {
    "model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "confidence_threshold": 0.7,
    "cache_size": 1000,
    "enable_state_tracking": true,
    "enable_persistence": true,
    "collect_metrics": true,
    "text_preprocessor": {
      "expand_contractions": true,
      "remove_special_chars": true
    },
    "confidence_scorer": {
      "feature_weights": {
        "semantic_similarity": 0.4,
        "keyword_match": 0.3,
        "context_relevance": 0.2,
        "historical_accuracy": 0.1
      }
    }
  },
  "learning": {
    "algorithm": "q_learning",
    "parameters": {
      "learning_rate": 0.1,
      "discount_factor": 0.9,
      "epsilon": 0.2
    }
  },
  "retry_config": {
    "default": {
      "retry_policy": {
        "type": "exponential_backoff",
        "max_attempts": 5,
        "base_delay": 1.0,
        "max_delay": 16.0,
        "jitter_factor": 0.2
      },
      "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout": 30.0,
        "half_open_test_requests": 3
      }
    },
    "services": {
      "filesystem_mcp": {
        "retry_policy": {
          "type": "fixed_delay",
          "max_attempts": 3,
          "delay": 0.5
        }
      },
      "external_api": {
        "retry_policy": {
          "type": "exponential_backoff",
          "max_attempts": 10,
          "base_delay": 2.0,
          "max_delay": 60.0
        }
      }
    }
  }
}
```

## Project Structure

```
auto-tool-disc/
├── src/
│   ├── agents/           # AI agents (intent, discovery, orchestrator)
│   ├── core/             # Core MCP integration
│   ├── learning/         # Q-learning implementation
│   ├── monitoring/       # Performance monitoring
│   ├── pipeline/         # Modular pipeline architecture
│   ├── services/         # Context persistence, etc.
│   └── tools/            # Tool implementations
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── config/              # Configuration files
├── data/                # Logs, metrics, registry
└── docs/                # Detailed documentation
```

## Documentation

For detailed documentation, see:
- [Architecture Overview](docs/architecture/system-architecture.md)
- [Retry and Resilience Architecture](docs/architecture/retry-architecture.md)
- [Intent Recognition](docs/implementation/intent-recognition.md)
- [Tool Discovery](docs/implementation/tool-discovery.md)
- [Learning System](docs/implementation/learning-system.md)
- [API Reference](docs/api/rest-api.md)

## Development

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features
1. Write tests first (TDD approach)
2. Implement feature following existing patterns
3. Update documentation
4. Run full test suite
5. Check code quality

## Contributing

We welcome contributions to improve the Autonomous Tool Discovery system!

### Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use `black` for code formatting
   - Run `flake8` for linting
   - Use type hints where appropriate

2. **Testing Requirements**
   - Write tests for all new features
   - Maintain >90% test coverage
   - Include both unit and integration tests
   - Test edge cases and error conditions

3. **Pull Request Process**
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/amazing-feature`)
   - Commit your changes (`git commit -m 'Add amazing feature'`)
   - Push to the branch (`git push origin feature/amazing-feature`)
   - Open a Pull Request with detailed description

4. **Commit Message Convention**
   - Use clear, descriptive commit messages
   - Start with a verb (Add, Update, Fix, Remove)
   - Reference issue numbers when applicable

### Areas for Contribution

- Implement additional MCP tool integrations
- Enhance the Q-learning algorithm
- Add new pipeline stages
- Improve documentation
- Add more comprehensive tests
- Optimize performance
- Create visualization tools

## License

[Your License Here]

## Author

[Your Name] - AI Dissertation Project