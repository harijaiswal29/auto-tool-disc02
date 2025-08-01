# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI dissertation project implementing autonomous tool discovery and integration through Model Context Protocol (MCP). The system uses Q-learning and sentence transformers to enable agents to discover, learn, and optimize tool usage autonomously.

## High-Level Architecture

The system consists of 5 core layers:

1. **Intent Recognition Layer**: Understands user goals using keyword matching, semantic analysis (sentence-transformers), context tracking, and intent classification
2. **Tool Discovery Layer**: Finds relevant tools via registry interface, capability matching, graph exploration (NetworkX), and discovery patterns
3. **Tool Selection & Learning Layer**: Optimizes tool choice using epsilon-greedy multi-armed bandit, Q-learning with 447-dimensional state vectors, combination scoring, and contextual analysis
4. **Execution & Monitoring Layer**: Manages MCP connections, executes tools in parallel (asyncio), monitors performance, handles errors, and tracks resource usage
5. **Learning & Adaptation Layer**: Improves over time using enhanced Q-learning engine with failure differentiation, pattern mining, feedback processing, and model adaptation

## System Behavior

For detailed examples of how the system processes queries and workflows, see `docs/architecture/workflows.md`

## Technology Stack

- **Language**: Python 3.8+ (tested with 3.12.3)
- **Database**: SQLite (with aiosqlite for async operations)
- **ML Libraries**: scikit-learn, sentence-transformers (all-MiniLM-L6-v2), numpy/pandas
- **Async**: asyncio
- **Graph**: NetworkX
- **Web Framework**: FastAPI
- **MCP Integration**: Official Model Context Protocol SDK, JSON-RPC 2.0

### Version Requirements

- **Python**: 3.8 or higher (tested with 3.12.3)
- **Dependencies**: See `requirements.txt` for all required packages and versions

## Implementation Status

For detailed implementation status and progress tracking, see `docs/implementation/implementation-status.md`

## Quick Start Commands

```bash
# Setup
pip install -r requirements.txt
python verify_setup.py

# Run main system
python src/main.py

# Run tests (see tests/README.md for comprehensive test commands)
pytest tests/ -v
pytest --cov=src tests/ --cov-report=html

# Code quality
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Environment Setup

```bash
# 1. Copy the example environment file
cp .env.example .env

# 2. Edit .env with your API keys

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py
```

See `.env.example` for required API keys and configuration options.

### Testing Without API Keys

The system includes mock servers for all MCP tools, allowing full functionality testing without API keys:

```bash
# Tests will automatically use mock servers if API keys are not set
pytest tests/unit/ -v
```

## Important Conventions

- **MCP Communication**: Follow JSON-RPC 2.0 spec strictly
- **Logging**: All modules must integrate with existing logging system
- **Imports**: Python scripts often modify sys.path - maintain this pattern
- **Mock Servers**: Temporary until official MCP servers available
  - Located in `src/tools/mock_*.py` files
  - Used by default when real servers unavailable or API keys missing
  - System automatically falls back to mock if real server connection fails
  - Mock servers provide full functionality for testing without external dependencies
- **Configuration**: Q-learning parameters (α=0.1, γ=0.9, ε=0.2) in config.json
- **Pipeline Architecture**: All stages must implement PipelineStage interface
- **Testing**: Follow test organization in tests/README.md, maintain >80% overall coverage (>90% for core components)
- **Performance**: Intent recognition must complete within 100ms (p95)
- **Security**: Zero Trust principles, validate all inputs, sandbox executions

## Troubleshooting

For troubleshooting common issues and solutions, see `docs/troubleshooting.md`

## Documentation Map

### Setup Guides
- `docs/setup/zerodha-mcp-setup.md` - Zerodha trading platform MCP setup
- `docs/setup/notion-mcp-setup.md` - Notion integration MCP setup
- `docs/setup/postgresql-setup-guide.md` - PostgreSQL database setup for MCP testing

### Architecture & Design
- `docs/architecture/system-architecture.md` - Component architecture and design principles
- `docs/architecture/mcp-communication.md` - MCP protocol details and message formats
- `docs/architecture/workflows.md` - Key system workflows and processes
- `docs/architecture/database-schema.md` - Complete database schema and tables
- `docs/architecture/retry-architecture.md` - Retry and resilience patterns

### Implementation Details
- `docs/implementation/implementation-status.md` - Detailed implementation tracking
- `docs/implementation/learning-system.md` - Q-learning, rewards, pattern mining
- `docs/implementation/q_learning_implementation.md` - Q-learning engine implementation
- `docs/implementation/deep-q-learning.md` - Deep Q-Learning with neural networks
- `docs/implementation/intent-recognition.md` - NLP pipeline and classification
- `docs/implementation/tool-discovery.md` - Discovery algorithms and caching
- `docs/implementation/execution-engine.md` - Task management and monitoring
- `docs/implementation/learning-system-updates.md` - Summary of learning system enhancements
- `docs/implementation/advanced-reward-strategies.md` - Advanced reward calculation strategies

### API & Data Models
- `docs/api/rest-api.md` - RESTful endpoints and specifications
- `docs/api/websocket-api.md` - Real-time communication protocols
- `docs/api/data-models.md` - Core data models and schemas

### Project Management
- `docs/project/phase-completion.md` - Development phases and accomplishments
- `docs/project/project-structure.md` - Complete directory structure

### Development
- `docs/development/commands-reference.md` - Development commands and scripts

### Evaluation
- `docs/evaluation/evaluation-targets.md` - Performance targets and baselines
- `docs/evaluation/baseline-comparisons.md` - Baseline strategy comparisons
- `docs/evaluation/ab-testing-framework.md` - A/B testing documentation

### Testing & Deployment
- `tests/README.md` - **Comprehensive test suite documentation and ALL test commands**
- `tests/dissertation_test_suite/` - **Dissertation-focused test suite for validating research hypotheses**
  - `dissertation-testing-strategy.md` - Pragmatic testing strategy for dissertation goals
  - Hypothesis validation tests (H1-H5)
  - Performance benchmarks aligned with evaluation targets
  - Statistical validation and reproducibility tests
- `docs/testing/test-summary.md` - Test coverage summary and metrics
- `docs/testing/coverage_summary.md` - Detailed test coverage report
- `docs/deployment/requirements.md` - Non-functional requirements and SLOs
- `docs/deployment/infrastructure.md` - Container specs and CI/CD pipelines
- `docs/deployment/security.md` - Security architecture and best practices
- `docs/deployment/configuration.md` - Configuration guide for learning system

## Development Timeline

For project phases and timeline, see `docs/project/phase-completion.md`

## Performance Targets

For detailed performance targets and evaluation metrics, see `docs/evaluation/evaluation-targets.md`

## Demo Scripts

The project includes comprehensive demonstration scripts in the `demos/` directory:
- **A/B Testing Framework**: `demos/demo_ab_testing_framework.py` - Complete A/B testing with 6 scenarios
- **Advanced Rewards**: `demos/demo_advanced_rewards.py` - Advanced reward strategies demonstration
- **Pattern Mining**: `demos/demo_pattern_mining.py` - Pattern discovery and mining
- **Q-Learning**: `demos/demo_q_learning_orchestration.py` - Q-learning with orchestrator
- **Deep Q-Learning**: `demos/demo_dqn_learning.py` - Neural network-based learning
- **Baseline Evaluation**: `demos/demo_baseline_evaluation.py` - Strategy comparisons
- **Real-time Monitoring**: `demos/demo_realtime_monitoring.py` - Live performance tracking

See `demos/README.md` for detailed documentation of all available demos.