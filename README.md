# Autonomous Tool Discovery and Integration System

An AI dissertation project implementing autonomous tool discovery and integration through Model Context Protocol (MCP). The system uses Q-learning and sentence transformers to enable agents to discover, learn, and optimize tool usage autonomously.

## Overview

This system provides an intelligent framework for:
- **Intent Recognition**: Understanding user queries through NLP and semantic analysis
- **Tool Discovery**: Finding relevant tools based on capabilities and relationships
- **Learning & Optimization**: Improving tool selection through Q-learning
- **Execution & Monitoring**: Managing tool execution with performance tracking
- **Continuous Adaptation**: Learning from patterns and user feedback

## Architecture

The system consists of 5 core layers:

1. **Intent Recognition Layer** - NLP pipeline with semantic understanding
2. **Tool Discovery Layer** - Graph-based tool exploration and matching
3. **Tool Selection & Learning Layer** - Q-learning optimization
4. **Execution & Monitoring Layer** - Parallel execution with monitoring
5. **Learning & Adaptation Layer** - Pattern mining and model adaptation

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- SQLite (included with Python)

### Installation

```bash
# Clone the repository
git clone <repository-url>
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

### Accessing Metrics
```python
from src.agents.intent_recognition_agent import IntentRecognitionAgent

agent = IntentRecognitionAgent()
# Process queries...

# Get metrics summary
metrics = agent.get_metrics_summary()

# Export metrics to file
agent.export_metrics("metrics_report.json")
```

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
    "similarity_threshold": 0.7,
    "confidence_threshold": 0.7,
    "enable_monitoring": true
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

## License

[Your License Here]

## Author

[Your Name] - AI Dissertation Project