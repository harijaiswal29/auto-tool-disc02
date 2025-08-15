# System Architecture

## Overview

The Autonomous Tool Discovery and Integration System consists of 5 core layers working together to understand user intent, discover appropriate tools, optimize selection through learning, and execute tasks efficiently.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                       │
│         (CLI, REST API, WebSocket, Web Demonstration UI)         │
└─────────────────┬───────────────────────────┬──────────────────┘
                  │                           │
┌─────────────────▼───────────────────────────▼──────────────────┐
│                    Intent Recognition Layer                      │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │   NLP       │ │   Context    │ │  Intent              │   │
│  │ Processor   │ │  Manager     │ │  Classifier          │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                    Tool Discovery Layer                         │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │   Registry  │ │  Capability  │ │   Graph              │   │
│  │  Interface  │ │   Matcher    │ │  Explorer            │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│              Tool Selection & Learning Layer                    │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │ Q-Learning  │ │   Epsilon    │ │  Combination         │   │
│  │   Engine    │ │   Greedy     │ │   Scorer            │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│               Execution & Monitoring Layer                      │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │    MCP      │ │   Parallel   │ │   Performance        │   │
│  │ Connection  │ │  Executor    │ │    Monitor           │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │   Retry     │ │   Circuit    │ │   Connection         │   │
│  │   Logic     │ │   Breakers   │ │      Pool            │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│               Learning & Adaptation Layer                       │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐   │
│  │   Pattern   │ │   Feedback   │ │    Model             │   │
│  │    Miner    │ │  Processor   │ │   Adapter            │   │
│  └─────────────┘ └──────────────┘ └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. Intent Recognition Layer
Understands user goals using:
- Keyword matching
- Semantic analysis (sentence-transformers)
- Context tracking
- Intent classification
- Performance monitoring and metrics collection
- Multi-intent handling
- Conversation state management

### 2. Tool Discovery Layer
Finds relevant tools via:
- Registry interface
- Capability matching
- Graph exploration (NetworkX)
- Discovery patterns

### 3. Tool Selection & Learning Layer
Optimizes tool choice using:
- Epsilon-greedy multi-armed bandit
- Q-learning with 476-dimensional state representation
- Deep Q-Networks (DQN) with multiple architectures
- Combination scoring
- Contextual analysis
- Pattern mining for tool synergies

### 4. Execution & Monitoring Layer
Manages:
- MCP connections with connection pooling
- Parallel tool execution (asyncio)
- Performance monitoring and metrics
- Error handling with retry logic
- Exponential backoff retry policies
- Circuit breakers for fault tolerance
- Connection health checking and reuse

### 5. Learning & Adaptation Layer
Improves over time using:
- Q-learning engine (✅ Implemented in `src/learning/q_learning_engine.py`)
- Pattern mining (✅ Implemented in `src/learning/pattern_miner.py`)
- Deep Q-Learning (✅ Implemented with DQN architectures)
- Advanced reward strategies (✅ TD, Hierarchical, Adaptive, Information-theoretic)
- Feedback processing (✅ Integrated in orchestrator)
- Model adaptation (✅ Model persistence and checkpointing)

## Component Interactions

### Synchronous Operations
- Intent Recognition → Tool Discovery
- Tool Selection → Execution Planning
- Result Processing → Learning Update

### Asynchronous Operations
- Parallel tool execution
- Background pattern mining
- Metric collection
- Model adaptation

### Event-Driven Operations
- Tool registry updates
- Performance threshold alerts
- Learning milestone notifications

## Data Flow

1. User Query → Intent Recognition
2. Intent Vector → Tool Discovery
3. Candidate Tools → Tool Selection (Q-learning)
4. Selected Tools → MCP Execution
5. Results → Learning Update
6. Metrics → Performance Monitoring

## Key Design Principles

1. **Modularity**: Each layer operates independently with well-defined interfaces
2. **Scalability**: Asynchronous operations enable parallel processing
3. **Learning**: Continuous improvement through reinforcement learning
4. **Extensibility**: Easy to add new tools and capabilities
5. **Resilience**: Fallback mechanisms and error recovery

## Integration Points

### MCP Integration
- Central integration point for all MCP servers
- Unified tool execution interface
- Server lifecycle management
- Performance tracking integration

### Learning Integration
- Q-learning updates after each execution
- Pattern mining for successful tool combinations
- Performance metrics feed into reward calculation

### Monitoring Integration
- Real-time performance tracking
- Resource usage monitoring
- Error rate tracking
- Learning progress visualization
- Intent recognition metrics collection
- Pipeline stage performance analysis
- Cache hit rate monitoring
- Classification accuracy tracking
- Retry attempt tracking and analysis
- Circuit breaker state monitoring
- Connection pool health metrics

### Retry and Resilience Integration
- Exponential backoff with configurable policies
- Circuit breaker pattern implementation
- Connection pooling for resource efficiency
- Comprehensive retry metrics collection
- Failure pattern analysis and alerting
- Per-service retry configuration
- Detailed documentation: [Retry Architecture](./retry-architecture.md)

### Web Demonstration Interface
- FastAPI-based real-time demonstration (`src/web/demo_app.py`)
- 5-stage pipeline visualization with live updates
- WebSocket support for real-time monitoring
- Q-learning metrics and decision visualization
- Performance dashboard with cache and system metrics
- Professional UI design for dissertation presentations
- Quick launch via `launch_demo.py`

## Available MCP Tools

The system integrates with 10+ MCP tools, each with mock server fallback:

### Database Tools
- **SQLite MCP**: Database query execution and schema introspection
- **PostgreSQL MCP**: Advanced database operations with connection pooling

### File System Tools
- **Filesystem MCP**: File operations, directory navigation, search capabilities

### External Service Tools
- **Search MCP**: Web search via Brave Search API
- **Weather MCP**: Weather data from OpenWeatherMap
- **GitHub MCP**: Repository operations, issue/PR management, code search
- **Notion MCP**: Workspace integration for pages, databases, blocks

### Specialized Tools
- **Financial Datasets MCP**: Remote API for financial data access
- **Zerodha MCP**: Trading platform integration for market data and orders

All tools support:
- Automatic fallback to mock servers when APIs unavailable
- Connection pooling and retry logic
- Performance monitoring and metrics collection
- Parallel execution where applicable