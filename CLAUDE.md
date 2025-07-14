# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI dissertation project implementing autonomous tool discovery and integration through Model Context Protocol (MCP). The system uses Q-learning and sentence transformers to enable agents to discover, learn, and optimize tool usage autonomously.

## High-Level Architecture

The system consists of 5 core layers:

1. **Intent Recognition Layer**: Understands user goals using keyword matching, semantic analysis (sentence-transformers), context tracking, and intent classification
2. **Tool Discovery Layer**: Finds relevant tools via registry interface, capability matching, graph exploration (NetworkX), and discovery patterns
3. **Tool Selection & Learning Layer**: Optimizes tool choice using epsilon-greedy multi-armed bandit, Q-learning, combination scoring, and contextual analysis
4. **Execution & Monitoring Layer**: Manages MCP connections, executes tools in parallel (asyncio), monitors performance, and handles errors
5. **Learning & Adaptation Layer**: Improves over time using Q-learning engine, pattern mining, feedback processing, and model adaptation

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

### ⏳ Not Yet Implemented
- **Learning System** (In Progress):
  - Q-Learning Engine with experience replay
  - Pattern Miner for tool combinations
  - Reward Calculator for reinforcement learning
  - Model persistence and adaptation
  
- **Evaluation Framework**:
  - Automated baseline comparisons
  - A/B testing framework
  - Performance regression detection
  
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
│   ├── learning/           # Q-learning algorithms
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