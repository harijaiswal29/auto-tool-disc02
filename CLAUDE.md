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

## Technology Stack

- **Language**: Python 3.8+
- **Database**: SQLite (with aiosqlite for async operations)
- **ML Libraries**: scikit-learn, sentence-transformers (all-MiniLM-L6-v2), numpy/pandas
- **Async**: asyncio
- **Graph**: NetworkX
- **Web Framework**: FastAPI
- **MCP Integration**: Official Model Context Protocol SDK, JSON-RPC 2.0

## Current Implementation Status

**Phase**: Phase 4 - Learning System (Weeks 9-11)

### Key Implementations
- ✅ **Core Infrastructure**: MCP Integration, Tool Registry, Configuration System
- ✅ **MCP Clients**: SQLite, Search, Weather, Filesystem, PostgreSQL, GitHub, Financial Datasets, Zerodha, Notion
- ✅ **AI Agents**: Intent Recognition (7-stage pipeline), Tool Discovery, Orchestrator
- ✅ **Learning System**: Q-Learning Engine, Pattern Miner, Context-Aware Patterns, Deep Q-Learning, Advanced Rewards
- ✅ **Evaluation**: Baseline Comparisons, A/B Testing, Real-time Performance Monitoring
- ✅ **Testing**: >80% coverage target, comprehensive test suite across unit/integration/e2e/performance

### Not Yet Implemented
- Real-time monitoring dashboard UI
- Production deployment configurations
- Advanced tool relationship graph visualization
- API rate limiting and throttling
- Multi-tenant support
- Distributed execution support

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

## Important Conventions

- **MCP Communication**: Follow JSON-RPC 2.0 spec strictly
- **Logging**: All modules must integrate with existing logging system
- **Imports**: Python scripts often modify sys.path - maintain this pattern
- **Mock Servers**: Temporary until official MCP servers available
- **Configuration**: Q-learning parameters (α=0.1, γ=0.9, ε=0.2) in config.json
- **Pipeline Architecture**: All stages must implement PipelineStage interface
- **Testing**: Follow test organization in tests/README.md, maintain >80% overall coverage (>90% for core components)
- **Performance**: Intent recognition must complete within 100ms (p95)
- **Security**: Zero Trust principles, validate all inputs, sandbox executions

## Documentation Map

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
- `docs/testing/test-summary.md` - Test coverage summary and metrics
- `docs/deployment/requirements.md` - Non-functional requirements and SLOs
- `docs/deployment/infrastructure.md` - Container specs and CI/CD pipelines
- `docs/deployment/security.md` - Security architecture and best practices
- `docs/deployment/configuration.md` - Configuration guide for learning system

## Development Timeline

- **Phase 1**: Foundation (Weeks 1-3) ✅
- **Phase 2**: Tool Ecosystem (Weeks 4-5) ✅
- **Phase 3**: Core Intelligence (Weeks 6-8) ✅
- **Phase 4**: Learning System (Weeks 9-11) - Current
- **Phase 5**: Optimization & Testing (Weeks 12-13)
- **Phase 6**: Documentation & Submission (Weeks 14-16)

## Performance Targets

- **Intent Recognition Accuracy**: >90%
- **Processing Time**: <100ms (p95) for intent recognition
- **Cache Hit Rate**: >70% for embedding cache
- **Tool Selection Accuracy**: >80% (improvement from baseline)
- **Task Completion Rate**: >85%
- **Learning Convergence**: Within 1000 episodes
- **System Availability**: 99.9% uptime