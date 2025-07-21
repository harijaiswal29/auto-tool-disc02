# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI dissertation project implementing autonomous tool discovery and integration through Model Context Protocol (MCP). The system uses Q-learning and sentence transformers to enable agents to discover, learn, and optimize tool usage autonomously.

## High-Level Architecture

The system consists of 5 core layers:

1. **Intent Recognition Layer**: Understands user goals using keyword matching, semantic analysis (sentence-transformers), context tracking, and intent classification
2. **Tool Discovery Layer**: Finds relevant tools via registry interface, capability matching, graph exploration (NetworkX), and discovery patterns
3. **Tool Selection & Learning Layer**: Optimizes tool choice using epsilon-greedy multi-armed bandit, Q-learning with 439-dimensional state vectors, combination scoring, and contextual analysis
4. **Execution & Monitoring Layer**: Manages MCP connections, executes tools in parallel (asyncio), monitors performance, handles errors, and tracks resource usage
5. **Learning & Adaptation Layer**: Improves over time using enhanced Q-learning engine with failure differentiation, pattern mining, feedback processing, and model adaptation

## Technology Stack

- **Language**: Python 3.8+
- **Database**: SQLite (with aiosqlite for async operations)
- **ML Libraries**: scikit-learn, sentence-transformers (all-MiniLM-L6-v2), numpy/pandas
- **Async**: asyncio
- **Graph**: NetworkX
- **Web Framework**: FastAPI
- **MCP Integration**: Official Model Context Protocol SDK, JSON-RPC 2.0

## Current Implementation Status

**Phase Status**: Phase 4 - Learning System (Weeks 9-11)

### ✅ Implemented

#### Recent Accomplishments (Phase 3 Completion)
- **Intent Recognition Pipeline**:
  - Modular 7-stage pipeline architecture
  - Comprehensive test suite with >90% coverage
  - Performance monitoring and metrics collection
  - Enhanced configuration system
  
- **Testing Infrastructure**:
  - Unit tests for all pipeline stages
  - Integration tests for end-to-end flows
  - Performance benchmarking tests
  - Edge case and error handling tests

- **Documentation**:
  - Complete API documentation
  - Architecture diagrams updated
  - Performance requirements defined
  - Testing strategy documented

#### Core Components
- **Core Infrastructure**:
  - Core MCP Integration (`src/core/mcp_integration.py`) with intent-based discovery
  - Tool Registry with relationship tracking and search capabilities
  - Configuration System (`config/config.json`) with orchestration settings
  
- **MCP Client Implementations**:
  - SQLite MCP (`src/tools/sqlite_mcp.py`)
  - Search MCP (`src/tools/search_mcp.py`)
  - Weather MCP (`src/tools/custom_wrappers/weather_mcp.py`)
  - Filesystem MCP (`src/tools/filesystem_mcp.py`)
  - PostgreSQL MCP (`src/tools/postgres_mcp.py`)
  - GitHub MCP (`src/tools/github_mcp.py`)
  - Financial Datasets MCP (`src/tools/financial_datasets_mcp.py`) - Remote API for financial data
  - Zerodha MCP (`src/tools/zerodha_mcp.py`) - Trading and portfolio management API
  - Notion MCP (`src/tools/notion_mcp.py`) - Notion workspace integration with pages, databases, and blocks
  - Mock MCP Server Infrastructure for all services
  
- **AI Agent Implementations**:
  - Intent Recognition Agent with NLP pipeline (`src/agents/intent_recognition_agent.py`)
    - Modular pipeline architecture with 7 stages
    - Semantic embeddings using sentence-transformers
    - Multi-intent handling support
    - Conversation state management
    - Context persistence service
  - Tool Discovery Agent with semantic search (`src/agents/tool_discovery_agent.py`)
  - Orchestrator Agent for pipeline coordination (`src/agents/orchestrator_agent.py`)
  
- **Integration Features**:
  - Complete pipeline from natural language query to tool execution
  - Intent-based tool discovery and selection
  - Parallel tool execution support
  - Main application entry point (`src/main.py`)
  - Integration tests (`tests/test_integration.py`)

- **Testing Infrastructure**:
  - Comprehensive unit tests for pipeline stages (`tests/unit/test_intent_pipeline_stages.py`)
  - Intent Recognition integration tests (`tests/integration/test_intent_recognition_integration.py`)
  - State machine tests (`tests/unit/test_conversation_state_machine.py`)
  - Performance benchmarking tests
  - Test organization: unit tests, integration tests, e2e tests
  - Target coverage: >80% overall, >90% for core components

- **Monitoring & Metrics**:
  - Performance monitoring system (`src/monitoring/intent_recognition_metrics.py`)
  - Real-time metrics collection (processing time, accuracy, cache hits)
  - Pipeline stage performance tracking
  - Exportable metrics reports
  - Retry metrics monitoring (`src/monitoring/retry_metrics.py`)

- **Retry and Resilience System**:
  - Exponential backoff retry policies with jitter
  - Circuit breaker pattern implementation
  - Connection pooling with health checks
  - Configurable retry policies per service
  - Comprehensive retry metrics collection
  - Failure pattern analysis and alerting
  - See [Retry Architecture](docs/architecture/retry-architecture.md)

#### Learning System (Phase 4 - In Progress)
- **Q-Learning Engine** (`src/learning/q_learning_engine.py`):
  - ✅ Core Q-learning implementation with experience replay
  - ✅ Enhanced state representation (447 dimensions) with failure tracking and context
  - ✅ Action space management with constraint validation
  - ✅ Epsilon-greedy exploration with decay
  - ✅ Integration with orchestrator for automatic learning
  - ✅ Model persistence to database
  - ✅ Enhanced reward calculator with sophisticated failure differentiation
  - ✅ Partial success handling with completion percentage tracking
  - ✅ Resource efficiency tracking using psutil
  - ✅ User satisfaction signals (explicit and implicit feedback)
  - ✅ Tool synergy recognition with configurable bonuses
  - ✅ Context-aware tool selection with user expertise and domain tracking

- **Pattern Miner** (`src/learning/pattern_miner.py`):
  - ✅ Sequential pattern mining using simplified PrefixSpan algorithm
  - ✅ Combination pattern mining for tool synergies
  - ✅ Temporal pattern mining for time-based patterns
    - Hourly patterns (tools used at specific times)
    - Periodic patterns (daily/weekly cycles)
    - Duration patterns (consistent execution times)
    - Time-clustered patterns (tools used together in time windows)
  - ✅ Context-aware pattern mining with user expertise and domain grouping
  - ✅ Pattern metrics calculation (support, confidence, lift)
  - ✅ Database persistence for discovered patterns with temporal metadata
  - ✅ Pattern-based tool suggestions with temporal and context awareness
  - ✅ Integration with Q-Learning engine for enhanced action selection
  - ✅ Comprehensive unit tests including temporal patterns (`tests/unit/test_pattern_miner.py`)
  - ✅ Demo script (`demos/demo_pattern_mining.py`)

- **Context-Aware Patterns** (`src/learning/context_extractor.py`):
  - ✅ User expertise extraction (novice, intermediate, expert)
  - ✅ Domain detection (engineering, data_science, web_dev, devops, general)
  - ✅ Context-based pattern grouping and mining
  - ✅ Personalized tool recommendations based on user context
  - ✅ Database schema updates with context columns and indexes

- **Deep Q-Learning** (`src/learning/deep_q_network.py`, `src/learning/dqn_agent.py`):
  - ✅ Neural network architectures (Standard DQN, Dueling DQN, Noisy DQN)
  - ✅ Target network for stable learning
  - ✅ Double DQN to reduce overestimation bias
  - ✅ Prioritized Experience Replay with sum-tree implementation
  - ✅ GPU acceleration support
  - ✅ Model checkpointing and persistence
  - ✅ Integration with Q-Learning engine (configurable via `dqn.enabled`)
  - ✅ Training utilities with learning rate scheduling
  - ✅ Comprehensive unit tests (`tests/unit/test_dqn.py`)
  - ✅ Comparison demo script (`demos/demo_dqn_learning.py`)

- **Advanced Reward Strategies** (`src/learning/advanced_rewards/`):
  - ✅ Temporal Difference (TD) rewards with eligibility traces for credit assignment
  - ✅ Hierarchical goal-based rewards with multi-level progress tracking
  - ✅ Adaptive reward shaping with curriculum learning and dynamic adjustment
  - ✅ Information-theoretic rewards for curiosity-driven exploration
  - ✅ Strategy manager for ensemble coordination (weighted average, max, voting)
  - ✅ A/B testing framework for strategy comparison
  - ✅ Integration with existing reward calculator
  - ✅ Comprehensive unit tests (`tests/unit/test_advanced_rewards.py`)
  - ✅ Demo script (`demos/demo_advanced_rewards.py`)
  - ✅ See [Learning System Documentation](docs/implementation/learning-system.md#advanced-reward-strategies) for details

### ✅ Recently Implemented (Phase 4 - Evaluation Framework)
  
- **Evaluation Framework** (`src/evaluation/`):
  - ✅ Automated baseline comparisons with 5 strategies
  - ✅ Statistical significance testing (t-test, Mann-Whitney U)
  - ✅ Effect size calculation (Cohen's d)
  - ✅ Comprehensive metrics collection
  - ✅ Visualization and report generation
  - ✅ Performance regression detection (basic)
  - ✅ Demo script and full test coverage
  
### ✅ Recently Completed - Advanced Evaluation Features
  
- **A/B Testing Framework** (`src/evaluation/ab_testing_framework.py`, `src/evaluation/ab_test_manager.py`)
  - ✅ Comprehensive A/B testing with multiple assignment strategies
  - ✅ Statistical analysis (frequentist and Bayesian)
  - ✅ Multi-armed bandit support with Thompson sampling
  - ✅ Database persistence and experiment lifecycle management
  - ✅ API endpoints for experiment management
  - ✅ Integration with reward strategy manager
  - ✅ Full documentation and demo scripts
  - ✅ Unit test coverage

- **Real-time Performance Regression Alerts** (`src/evaluation/`)
  - ✅ Performance Regression Detector (`performance_regression_detector.py`)
    - Multiple statistical algorithms (CUSUM, EWMA, Z-score)
    - Adaptive baseline tracking
    - Configurable sensitivity thresholds
  - ✅ Alert Manager (`alert_manager.py`)
    - Multi-channel alert routing (log, file, webhook)
    - Alert suppression and deduplication
    - Severity-based handling
  - ✅ Enhanced Metrics Collector with real-time tracking
  - ✅ Online evaluation in EvaluationEngine
  - ✅ Real-time Monitoring Service (`realtime_monitor.py`)
    - WebSocket support for live dashboards
    - Continuous performance analysis
  - ✅ API endpoints for monitoring (`src/api/monitoring_api.py`)
  - ✅ Demo script (`demos/demo_realtime_monitoring.py`)

### ⏳ Not Yet Implemented
  
- **Production Features**:
  - Real-time monitoring dashboard UI
  - Production deployment configurations
  - Advanced tool relationship graph visualization
  - API rate limiting and throttling
  - Multi-tenant support
  - Distributed execution support

## Key Commands

```bash
# Setup
pip install -r requirements.txt
python verify_setup.py

# Code Quality
black src/ tests/
flake8 src/ tests/
mypy src/

# Testing
pytest tests/ -v
pytest --cov=src tests/ --cov-report=html
pytest --cov=src --cov-fail-under=80  # Enforce coverage threshold

# Run tests by category
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests only
pytest tests/e2e/ -v          # End-to-end tests only

# Run tests matching a pattern
pytest -k "test_retry" -v

# Test Intent Recognition
pytest tests/unit/test_intent_pipeline_stages.py -v
pytest tests/integration/test_intent_recognition_integration.py -v
pytest tests/test_intent_recognition.py -v

# Run Components
cd src && python utils/logger.py
python src/hello_mcp.py
python src/tools/mock_mcp_servers.py

# Run Integrated System
python src/main.py

# Test Integration
python test_integration_demo.py
python -m pytest tests/test_integration.py -v

# Test Retry Logic
python demo_retry_logic.py
pytest tests/test_retry_logic.py -v

# Test Q-Learning
pytest tests/unit/test_q_learning_engine.py -v
python src/learning/test_q_learning.py  # Run Q-learning demo
python demos/demo_q_learning_orchestration.py  # Demo with orchestrator

# Test Pattern Mining
pytest tests/unit/test_pattern_miner.py -v
python demos/demo_pattern_mining.py  # Run pattern mining demo

# Test Deep Q-Learning
pytest tests/unit/test_dqn.py -v
python demos/demo_dqn_learning.py  # Compare DQN vs tabular Q-learning

# Test Advanced Reward Strategies
pytest tests/unit/test_advanced_rewards.py -v
python demos/demo_advanced_rewards.py  # Demo advanced reward strategies

# Test A/B Testing Framework
pytest tests/unit/test_ab_testing_framework.py -v
pytest tests/unit/test_ab_test_manager.py -v
python demos/demo_ab_testing_framework.py  # Run full A/B testing demo (6 scenarios)
python run_demo5_only.py  # Run only Demo 5 (reward strategy comparison)

# Test Evaluation Framework
pytest tests/unit/test_baseline_strategies.py -v
pytest tests/unit/test_evaluation_engine.py -v
pytest tests/unit/test_metrics_collector.py -v
pytest tests/integration/test_baseline_evaluation.py -v
python demos/demo_baseline_evaluation.py --mode quick  # Run baseline evaluation demo

# Test Real-time Performance Monitoring
python demos/demo_realtime_monitoring.py  # Run real-time monitoring demo
# Note: The demo simulates performance degradation and shows:
# - Statistical regression detection (CUSUM, EWMA, Z-score)
# - Alert generation and routing
# - WebSocket-based real-time updates
# - Performance baseline tracking

# Test Financial Datasets MCP
pytest tests/integration/test_financial_datasets_mcp.py -v
python src/tools/financial_datasets_mcp.py  # Run Financial Datasets demo

# Test Zerodha MCP
pytest tests/integration/test_zerodha_mcp.py -v
python src/tools/zerodha_mcp.py  # Run Zerodha demo

# Test Notion MCP
pytest tests/integration/test_notion_mcp.py -v
pytest tests/unit/test_notion_mcp.py -v
python demos/demo_notion_mcp.py  # Run Notion demo
python src/tools/notion_mcp.py  # Run basic Notion test

# Monitor Performance
python -c "from src.agents.intent_recognition_agent import IntentRecognitionAgent; agent = IntentRecognitionAgent(); print(agent.get_metrics_summary())"

# Monitor Retry Metrics
python -c "from src.monitoring.retry_metrics import RetryMetricsCollector; from src.core.tool_registry import ToolRegistry; collector = RetryMetricsCollector(ToolRegistry()); print(collector.get_retry_statistics())"
```

## Project Structure

```
auto-tool-disc/
├── src/
│   ├── agents/             # AI agent implementations
│   │   ├── intent_recognition_agent.py
│   │   ├── intent_models.py
│   │   ├── tool_discovery_agent.py
│   │   └── orchestrator_agent.py
│   ├── core/               # Core MCP integration
│   │   └── mcp_integration.py
│   ├── database/           # Data models and persistence
│   │   ├── context_models.py
│   │   └── tool_registry.py
│   ├── evaluation/         # Evaluation framework
│   │   ├── __init__.py
│   │   ├── baseline_strategies.py  # Baseline comparison strategies
│   │   ├── evaluation_engine.py   # Main evaluation orchestrator
│   │   ├── metrics_collector.py   # Performance metrics collection
│   │   ├── comparison_visualizer.py  # Visualization and reports
│   │   └── reports/               # Generated evaluation reports
│   ├── learning/           # Q-learning algorithms
│   │   ├── __init__.py
│   │   ├── q_learning_engine.py  # Core Q-learning implementation
│   │   ├── pattern_miner.py      # Pattern mining for tool synergies
│   │   ├── reward_calculator.py  # Enhanced reward calculation
│   │   ├── deep_q_network.py     # DQN architectures
│   │   ├── dqn_agent.py          # DQN agent implementation
│   │   ├── dqn_trainer.py        # DQN training utilities
│   │   ├── prioritized_replay_buffer.py  # Experience replay
│   │   ├── advanced_rewards/     # Advanced reward strategies
│   │   │   ├── __init__.py
│   │   │   ├── base_strategy.py
│   │   │   ├── temporal_rewards.py
│   │   │   ├── hierarchical_rewards.py
│   │   │   ├── adaptive_shaping.py
│   │   │   ├── information_theoretic.py
│   │   │   └── strategy_manager.py
│   │   └── test_q_learning.py    # Q-learning test script
│   ├── monitoring/         # Performance monitoring
│   │   └── intent_recognition_metrics.py
│   ├── pipeline/           # Modular pipeline architecture
│   │   ├── base.py
│   │   └── stages/
│   │       ├── text_preprocessor.py
│   │       ├── tokenizer_module.py
│   │       ├── feature_extractor.py
│   │       ├── intent_classifier.py
│   │       ├── context_enricher.py
│   │       ├── confidence_scorer.py
│   │       └── state_manager.py
│   ├── services/           # Service layer
│   │   └── context_persistence_service.py
│   ├── state_machine/      # Conversation state management
│   │   ├── base.py
│   │   └── conversation_state_machine.py
│   ├── tools/              # Tool implementations and wrappers
│   │   ├── sqlite_mcp.py
│   │   ├── filesystem_mcp.py
│   │   ├── postgres_mcp.py
│   │   ├── github_mcp.py
│   │   ├── search_mcp.py
│   │   └── mock_mcp_servers.py
│   └── utils/              # Utilities
│       └── logger.py
├── tests/
│   ├── unit/               # Unit tests
│   │   ├── test_mcp_integration.py
│   │   ├── test_connection_pool.py
│   │   ├── test_orchestrator_agent.py
│   │   ├── test_tool_discovery_agent.py
│   │   ├── test_intent_pipeline_stages.py
│   │   ├── test_conversation_state_machine.py
│   │   ├── test_search_mcp.py
│   │   ├── test_state_machine_base.py
│   │   └── test_retry.py
│   ├── integration/        # Integration tests
│   │   ├── test_filesystem_mcp.py
│   │   ├── test_github_mcp.py
│   │   ├── test_intent_recognition_integration.py
│   │   ├── test_postgres_mcp.py
│   │   ├── test_sqlite_mcp.py
│   │   ├── test_state_machine_integration.py
│   │   └── test_weather_mcp.py
│   ├── e2e/               # End-to-end tests
│   │   └── test_filesystem_e2e.py
│   ├── demos/             # Demonstration scripts
│   │   ├── demo_pipeline_refactor.py
│   │   ├── demo_retry_logic.py
│   │   ├── test_integration_demo.py
│   │   ├── demo_github_mcp.py
│   │   └── demo_github_real.py
│   ├── utilities/         # Test utilities
│   │   └── check_encoding.py
│   ├── data/              # Test data and fixtures
│   ├── conftest.py        # Pytest configuration
│   ├── test_context_persistence.py
│   ├── test_integration.py
│   ├── test_intent_recognition.py
│   ├── test_pipeline_architecture.py
│   └── test_retry_logic.py
├── data/                   # Runtime data
│   ├── logs/              # Application logs
│   ├── metrics/           # Performance metrics
│   ├── patterns/          # Discovered patterns
│   └── registry/          # Tool registry database
├── config/                 # Configuration files
│   └── config.json        # Main configuration
├── docs/                   # Documentation
│   ├── architecture/      # System design docs
│   ├── implementation/    # Implementation details
│   ├── api/              # API documentation
│   └── deployment/       # Deployment guides
├── notebooks/             # Jupyter notebooks
├── experiments/           # Experimental code
├── requirements.txt       # Python dependencies
├── README.md             # Project overview
└── CLAUDE.md            # This file
```

## Important Conventions

- **MCP Communication**: Follow JSON-RPC 2.0 spec strictly
- **Logging**: All modules must integrate with existing logging system
- **Imports**: Python scripts often modify sys.path - maintain this pattern
- **Mock Servers**: Temporary until official MCP servers available
- **Experiments**: Must be reproducible with logged parameters
- **Pattern Mining**: Results must be persisted for analysis
- **Configuration**: Q-learning parameters (α=0.1, γ=0.9, ε=0.2) in config.json
- **Pipeline Architecture**: All stages must implement PipelineStage interface
- **Testing**: Follow test organization in tests/README.md, maintain >80% overall coverage (>90% for core components)
- **Monitoring**: All agents must integrate with metrics collection system
- **State Management**: Use conversation state machine for user interactions
- **Performance**: Intent recognition must complete within 100ms (p95)

## Security Notes

- Implement Zero Trust principles for tool access
- Validate all MCP server responses
- Sandbox untrusted tool executions
- Log all tool invocations for audit
- Use temporary tokens for external API access
- Input validation: Sanitize all user queries in Intent Recognition
- Embedding cache security: No sensitive data in cached embeddings
- Context persistence: Implement user privacy controls
- Rate limiting: Prevent abuse of Intent Recognition API

## Documentation References

For detailed information on specific topics, refer to these documentation files:

### Architecture Documentation
- @docs/architecture/system-architecture.md - Component architecture and design principles
- @docs/architecture/mcp-communication.md - MCP protocol details and message formats
- @docs/architecture/workflows.md - Key system workflows and processes
- @docs/architecture/database-schema.md - Complete database schema and tables

### Implementation Details
- @docs/implementation/learning-system.md - Q-learning, rewards, pattern mining
- @docs/implementation/q_learning_implementation.md - Q-learning engine implementation details
- @docs/implementation/deep-q-learning.md - Deep Q-Learning with neural networks
- @docs/implementation/intent-recognition.md - NLP pipeline and classification
- @docs/implementation/tool-discovery.md - Discovery algorithms and caching
- @docs/implementation/execution-engine.md - Task management and monitoring

### API Documentation
- @docs/api/rest-api.md - RESTful endpoints and specifications
- @docs/api/websocket-api.md - Real-time communication protocols
- @docs/api/data-models.md - Core data models and schemas

### Deployment & Operations
- @docs/deployment/requirements.md - Non-functional requirements and SLOs
- @docs/deployment/infrastructure.md - Container specs and CI/CD pipelines
- @docs/deployment/security.md - Security architecture and best practices
- @docs/deployment/configuration.md - Configuration guide for learning system parameters

### Testing Documentation
- @tests/README.md - Comprehensive test suite documentation, organization, and commands
- @docs/testing/test-summary.md - Test coverage summary, metrics, and remaining work

## Development Timeline

- **Phase 1**: Foundation (Weeks 1-3) ✅
- **Phase 2**: Tool Ecosystem (Weeks 4-5) ✅
- **Phase 3**: Core Intelligence (Weeks 6-8) ✅
- **Phase 4**: Learning System (Weeks 9-11) - Current
- **Phase 5**: Optimization & Testing (Weeks 12-13)
- **Phase 6**: Documentation & Submission (Weeks 14-16)

## Evaluation Target

Demonstrate measurable improvement in tool selection accuracy and task completion rate over 16-week development period compared to random selection baseline.

### Performance Targets
- **Intent Recognition Accuracy**: >90%
- **Processing Time**: <100ms (p95) for intent recognition
- **Cache Hit Rate**: >70% for embedding cache
- **Tool Selection Accuracy**: >80% (improvement from baseline)
- **Task Completion Rate**: >85%
- **Learning Convergence**: Within 1000 episodes
- **System Availability**: 99.9% uptime