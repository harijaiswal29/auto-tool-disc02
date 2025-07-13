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

**Phase Status**: Phase 3 - Core Intelligence (Weeks 6-8)

### ✅ Implemented
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
  - Tool Discovery Agent with semantic search (`src/agents/tool_discovery_agent.py`)
  - Orchestrator Agent for pipeline coordination (`src/agents/orchestrator_agent.py`)
  
- **Integration Features**:
  - Complete pipeline from natural language query to tool execution
  - Intent-based tool discovery and selection
  - Parallel tool execution support
  - Main application entry point (`src/main.py`)
  - Integration tests (`tests/test_integration.py`)

### ⏳ Not Yet Implemented
- **Learning System**:
  - Q-Learning Engine with experience replay
  - Pattern Miner for tool combinations
  - Reward Calculator for reinforcement learning
  - Model persistence and adaptation
  
- **Evaluation Framework**:
  - Metrics Collector for performance tracking
  - Baseline Comparisons
  - Performance Evaluator
  
- **Additional Components**:
  - Real-time monitoring dashboard
  - Production deployment configurations
  - Advanced tool relationship graph visualization

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
pytest --cov=src tests/

# Run Components
cd src && python utils/logger.py
python src/hello_mcp.py
python src/tools/mock_mcp_servers.py

# Run Integrated System
python src/main.py

# Test Integration
python test_integration_demo.py
python -m pytest tests/test_integration.py -v
```

## Project Structure

```
auto-tool-disc/
├── src/
│   ├── core/           # MCP integration core
│   ├── agents/         # AI agent implementations
│   ├── tools/          # Tool management and mock servers
│   ├── learning/       # Q-learning algorithms
│   ├── utils/          # Utilities and logging
│   └── evaluation/     # Evaluation framework
├── data/               # Logs, metrics, patterns, registry
├── config/             # Configuration files
├── tests/              # Test suites
├── docs/               # Detailed documentation
└── requirements.txt
```

## Important Conventions

- **MCP Communication**: Follow JSON-RPC 2.0 spec strictly
- **Logging**: All modules must integrate with existing logging system
- **Imports**: Python scripts often modify sys.path - maintain this pattern
- **Mock Servers**: Temporary until official MCP servers available
- **Experiments**: Must be reproducible with logged parameters
- **Pattern Mining**: Results must be persisted for analysis
- **Configuration**: Q-learning parameters (α=0.1, γ=0.9, ε=0.2) in config.json

## Security Notes

- Implement Zero Trust principles for tool access
- Validate all MCP server responses
- Sandbox untrusted tool executions
- Log all tool invocations for audit
- Use temporary tokens for external API access

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

## Development Timeline

- **Phase 1**: Foundation (Weeks 1-3) ✅
- **Phase 2**: Tool Ecosystem (Weeks 4-5) - Current
- **Phase 3**: Core Intelligence (Weeks 6-8)
- **Phase 4**: Learning System (Weeks 9-11)
- **Phase 5**: Optimization & Testing (Weeks 12-13)
- **Phase 6**: Documentation & Submission (Weeks 14-16)

## Evaluation Target

Demonstrate measurable improvement in tool selection accuracy and task completion rate over 16-week development period compared to random selection baseline.