# Project Structure

This document provides a comprehensive overview of the directory structure and organization of the Autonomous Tool Discovery and Integration System.

## Directory Tree

```
auto-tool-disc02/
├── src/                        # Source code
│   ├── agents/                 # AI agent implementations
│   │   ├── __init__.py
│   │   ├── intent_recognition_agent.py    # NLP-based intent recognition
│   │   ├── intent_models.py               # Intent data models
│   │   ├── tool_discovery_agent.py        # Tool discovery algorithms
│   │   └── orchestrator_agent.py          # Query orchestration
│   │
│   ├── core/                   # Core MCP integration
│   │   ├── __init__.py
│   │   ├── mcp_integration.py             # MCP protocol implementation
│   │   ├── connection_pool.py             # Connection management
│   │   ├── tool_registry.py               # Tool registry implementation
│   │   └── data/                          # Core data directory
│   │
│   ├── database/               # Data models and persistence
│   │   ├── __init__.py
│   │   ├── context_models.py              # Context storage models
│   │   ├── database.py                    # Database module
│   │   └── migrations/                    # Database migrations
│   │       └── add_context_columns.py     # Context columns migration
│   │
│   ├── evaluation/             # Evaluation framework
│   │   ├── __init__.py
│   │   ├── baseline_strategies.py         # Baseline comparison strategies
│   │   ├── evaluation_engine.py           # Main evaluation orchestrator
│   │   ├── metrics_collector.py           # Performance metrics collection
│   │   ├── comparison_visualizer.py       # Visualization and reports
│   │   ├── ab_testing_framework.py        # A/B testing implementation
│   │   ├── ab_test_manager.py             # A/B test management
│   │   ├── performance_regression_detector.py  # Regression detection
│   │   ├── alert_manager.py               # Alert routing and management
│   │   ├── realtime_monitor.py            # Real-time monitoring service
│   │   └── reports/                       # Generated evaluation reports
│   │
│   ├── learning/               # Q-learning algorithms
│   │   ├── __init__.py
│   │   ├── q_learning_engine.py           # Core Q-learning implementation
│   │   ├── pattern_miner.py               # Pattern mining for tool synergies
│   │   ├── reward_calculator.py           # Enhanced reward calculation
│   │   ├── context_extractor.py           # Context extraction for patterns
│   │   ├── deep_q_network.py              # DQN architectures
│   │   ├── dqn_agent.py                   # DQN agent implementation
│   │   ├── dqn_trainer.py                 # DQN training utilities
│   │   ├── prioritized_replay_buffer.py   # Experience replay
│   │   ├── advanced_rewards/              # Advanced reward strategies
│   │   │   ├── __init__.py
│   │   │   ├── base_strategy.py
│   │   │   ├── temporal_rewards.py
│   │   │   ├── hierarchical_rewards.py
│   │   │   ├── adaptive_shaping.py
│   │   │   ├── information_theoretic.py
│   │   │   └── strategy_manager.py
│   │   └── test_q_learning.py             # Q-learning test script
│   │
│   ├── monitoring/             # Performance monitoring
│   │   ├── __init__.py
│   │   ├── intent_recognition_metrics.py  # Intent recognition metrics
│   │   └── retry_metrics.py               # Retry and resilience metrics
│   │
│   ├── pipeline/               # Modular pipeline architecture
│   │   ├── __init__.py
│   │   ├── base.py                        # Base pipeline classes
│   │   └── stages/                        # Pipeline stages
│   │       ├── __init__.py
│   │       ├── text_preprocessor.py       # Text preprocessing
│   │       ├── tokenizer_module.py        # Tokenization
│   │       ├── feature_extractor.py       # Feature extraction
│   │       ├── intent_classifier.py       # Intent classification
│   │       ├── context_enricher.py        # Context enrichment
│   │       ├── confidence_scorer.py       # Confidence scoring
│   │       └── state_manager.py           # State management
│   │
│   ├── services/               # Service layer
│   │   ├── __init__.py
│   │   └── context_persistence_service.py # Context persistence
│   │
│   ├── state_machine/          # Conversation state management
│   │   ├── __init__.py
│   │   ├── base.py                        # Base state machine
│   │   └── conversation_state_machine.py  # Conversation states
│   │
│   ├── tools/                  # Tool implementations and wrappers
│   │   ├── __init__.py
│   │   ├── sqlite_mcp.py                  # SQLite MCP client
│   │   ├── filesystem_mcp.py              # Filesystem MCP client
│   │   ├── postgres_mcp.py                # PostgreSQL MCP client
│   │   ├── github_mcp.py                  # GitHub MCP client
│   │   ├── search_mcp.py                  # Search MCP client
│   │   ├── financial_datasets_mcp.py      # Financial Datasets MCP
│   │   ├── zerodha_mcp.py                 # Zerodha trading MCP
│   │   ├── notion_mcp.py                  # Notion MCP client
│   │   ├── custom_wrappers/               # Custom tool wrappers
│   │   │   └── weather_mcp.py             # Weather MCP wrapper
│   │   └── mock_mcp_servers.py            # Mock MCP servers
│   │
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py                      # Logging configuration
│   │   └── retry.py                       # Retry logic utilities
│   │
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   └── intent.py                      # Intent model
│   │
│   ├── data/                   # Application data
│   │   └── test_combined/                 # Test data
│   │       └── products.csv               # Sample products data
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── monitoring_api.py              # Monitoring API endpoints
│   │   └── ab_testing_api.py              # A/B testing API endpoints
│   │
│   ├── __init__.py
│   └── main.py                 # Main application entry point
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   │   ├── __init__.py
│   │   ├── test_mcp_integration.py
│   │   ├── test_connection_pool.py
│   │   ├── test_orchestrator_agent.py
│   │   ├── test_tool_discovery_agent.py
│   │   ├── test_intent_pipeline_stages.py
│   │   ├── test_conversation_state_machine.py
│   │   ├── test_search_mcp.py
│   │   ├── test_github_mcp.py
│   │   ├── test_notion_mcp.py
│   │   ├── test_weather_mcp.py
│   │   ├── test_financial_datasets_mcp.py
│   │   ├── test_zerodha_mcp.py
│   │   ├── test_postgres_mcp.py
│   │   ├── test_postgres_real_server_unit.py
│   │   ├── test_sqlite_mcp.py
│   │   ├── test_state_machine_base.py
│   │   ├── test_retry.py
│   │   ├── test_retry_extended.py
│   │   ├── test_intent_recognition.py
│   │   ├── test_intent_recognition_metrics.py
│   │   ├── test_retry_metrics.py
│   │   ├── test_q_learning_engine.py
│   │   ├── test_pattern_miner.py
│   │   ├── test_context_extractor.py
│   │   ├── test_reward_calculator.py
│   │   ├── test_enhanced_state_representation.py
│   │   ├── test_incremental_pattern_mining.py
│   │   ├── test_dqn.py
│   │   ├── test_advanced_rewards.py
│   │   ├── test_baseline_strategies.py
│   │   ├── test_evaluation_engine.py
│   │   ├── test_metrics_collector.py
│   │   ├── test_ab_testing_framework.py
│   │   ├── test_ab_test_manager.py
│   │   └── WEATHER_MCP_TEST_SUMMARY.md
│   │
│   ├── integration/            # Integration tests
│   │   ├── __init__.py
│   │   ├── test_filesystem_mcp.py
│   │   ├── test_github_mcp.py
│   │   ├── test_github_direct.py
│   │   ├── test_github_real_direct.py
│   │   ├── test_github_simple.py
│   │   ├── test_notion_mcp.py
│   │   ├── test_intent_recognition_integration.py
│   │   ├── test_postgres_mcp.py
│   │   ├── test_postgres_real_server.py
│   │   ├── test_brave_search_direct.py
│   │   ├── test_sqlite_mcp.py
│   │   ├── test_state_machine_integration.py
│   │   ├── test_weather_mcp.py
│   │   ├── test_financial_datasets_mcp.py
│   │   ├── test_financial_datasets_mcp_backup.py
│   │   ├── test_zerodha_mcp.py
│   │   ├── test_all_mcp_tools.py
│   │   ├── test_pipeline_workflow.py
│   │   ├── test_retry_integration.py
│   │   ├── test_context_persistence.py
│   │   ├── test_context_aware_pattern_mining.py
│   │   ├── test_failure_learning.py
│   │   ├── test_q_learning_integration.py
│   │   ├── test_baseline_evaluation.py
│   │   ├── test_pipeline_architecture.py
│   │   ├── test_integration.py
│   │   ├── test_real_mcp.py
│   │   └── test_real_tools.py
│   │
│   ├── performance/            # Performance tests
│   │   ├── __init__.py
│   │   ├── test_intent_recognition_performance.py
│   │   └── test_tool_discovery_performance.py
│   │
│   ├── e2e/                    # End-to-end tests
│   │   ├── __init__.py
│   │   └── test_filesystem_e2e.py
│   │
│   ├── demos/                  # Test demonstration scripts
│   │   ├── demo_pipeline_refactor.py
│   │   ├── demo_retry_logic.py
│   │   ├── test_integration_demo.py
│   │   ├── demo_github_mcp.py
│   │   ├── demo_github_real.py
│   │   ├── demo_financial_datasets.py
│   │   ├── demo_financial_datasets_output.md
│   │   ├── demo_postgres_mcp.py
│   │   └── README.md
│   │
│   ├── utilities/              # Test utilities
│   │   ├── __init__.py
│   │   ├── check_encoding.py
│   │   ├── verify_setup.py
│   │   └── verify_setup_windows.py
│   │
│   ├── data/                   # Test data and fixtures
│   │   ├── fixtures/           # Reusable test data
│   │   │   ├── tools.json
│   │   │   ├── intents.json
│   │   │   └── queries.json
│   │   ├── expected/           # Expected output files
│   │   ├── logs/               # Test execution logs
│   │   ├── results/            # Test results
│   │   └── temp/               # Temporary test files
│   │
│   ├── __init__.py
│   ├── conftest.py             # Pytest configuration
│   └── README.md               # Test suite documentation
│
├── data/                       # Runtime data
│   ├── logs/                   # Application logs (contains numerous log files)
│   ├── databases/              # Database files
│   ├── context.db              # Context database
│   ├── learning.db             # Learning database
│   ├── sqlite_mcp_verification_report.json  # SQLite verification report
│   ├── test_combined_registry.db            # Combined test registry
│   ├── test_fs_integration_registry.db      # Filesystem integration test registry
│   ├── test_integration_registry.db         # Integration test registry
│   └── test_search_registry.db              # Search test registry
│
├── config/                     # Configuration files
│   └── config.json             # Main configuration
│
├── demos/                      # Main demonstration scripts
│   ├── README.md
│   ├── demo_ab_testing_framework.py
│   ├── demo_ab_testing_rewards.py
│   ├── demo_advanced_rewards.py
│   ├── demo_baseline_evaluation.py
│   ├── demo_dqn_learning.py
│   ├── demo_dqn_learning_fixed.py
│   ├── demo_dqn_simple.py
│   ├── demo_incremental_pattern_mining.py
│   ├── demo_notion_mcp.py
│   ├── demo_pattern_mining.py
│   ├── demo_q_learning_orchestration.py
│   ├── demo_realtime_monitoring.py
│   ├── hello_mcp.py
│   ├── pattern_qlearning_integration.py
│   ├── run_demo5_only.py
│   └── simple_pattern_demo.py
│
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   │   ├── data-models.md
│   │   ├── rest-api.md
│   │   └── websocket-api.md
│   │
│   ├── architecture/           # System design docs
│   │   ├── database-schema.md
│   │   ├── mcp-communication.md
│   │   ├── retry-architecture.md
│   │   ├── system-architecture.md
│   │   └── workflows.md
│   │
│   ├── deployment/             # Deployment guides
│   │   ├── configuration.md
│   │   ├── infrastructure.md
│   │   ├── requirements.md
│   │   └── security.md
│   │
│   ├── design/                 # Design documents
│   │   └── diagrams/           # Architecture diagrams
│   │       ├── README.md
│   │       └── *.puml          # PlantUML diagrams
│   │
│   ├── evaluation/             # Evaluation documentation
│   │   ├── ab-testing-framework.md
│   │   ├── baseline-comparisons.md
│   │   └── evaluation-targets.md
│   │
│   ├── implementation/         # Implementation details
│   │   ├── advanced-reward-strategies.md
│   │   ├── deep-q-learning.md
│   │   ├── execution-engine.md
│   │   ├── implementation-status.md
│   │   ├── intent-recognition.md
│   │   ├── learning-system-updates.md
│   │   ├── learning-system.md
│   │   ├── q_learning_implementation.md
│   │   └── tool-discovery.md
│   │
│   ├── migration/              # Migration guides
│   │   └── github-mcp-tool-names.md
│   │
│   ├── project/                # Project management
│   │   ├── phase-completion.md
│   │   └── project-structure.md
│   │
│   ├── setup/                  # Setup guides
│   │   └── postgresql-setup-guide.md
│   │
│   ├── testing/                # Testing documentation
│   │   ├── coverage_summary.md
│   │   ├── postgres-real-server-test-results.md
│   │   └── test-summary.md
│   │
│   ├── development/            # Development guides
│   │   └── commands-reference.md
│   │
│   ├── setup/                  # Setup guides
│   │   ├── notion-mcp-setup.md
│   │   ├── postgresql-setup-guide.md
│   │   └── zerodha-mcp-setup.md
│
├── experiments/                # Experimental code (currently empty)
│
├── scripts/                    # Utility scripts
│   ├── check_postgres_status.sh
│   ├── init-db.sql
│   ├── quick_postgres_setup.sh
│   └── setup_postgres.sh
│
├── infrastructure/             # Infrastructure configuration
│   ├── README.md
│   └── docker-compose.yml
│
├── setup/                      # Setup related files
│
├── htmlcov/                    # HTML coverage reports (generated)
│
├── node_modules/               # Node.js dependencies (generated)
│
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Project overview
├── CLAUDE.md                   # AI assistant guidance
├── setup.md                    # Setup documentation
├── coverage.xml                # Test coverage report
├── docker-compose.postgres.yml # PostgreSQL Docker compose configuration
├── financial_datasets_integration_test_results.md  # Integration test results
├── package.json                # Node.js package configuration
└── package-lock.json           # Node.js package lock file

```

## Directory Descriptions

### `/src` - Source Code
Contains all production code organized by functionality:
- **agents/**: AI agents for intent recognition, tool discovery, and orchestration
- **core/**: Core MCP integration and connection management
- **database/**: Data models and persistence layer
- **evaluation/**: Evaluation framework with baselines and metrics
- **learning/**: Q-learning and pattern mining implementations
- **monitoring/**: Performance and metrics monitoring
- **pipeline/**: Modular pipeline architecture for intent recognition
- **services/**: Service layer for business logic
- **state_machine/**: Conversation state management
- **tools/**: MCP tool implementations and wrappers
- **utils/**: Utility functions and helpers
- **api/**: REST and WebSocket API endpoints

### `/tests` - Test Suite
Comprehensive test suite organized by test type:
- **unit/**: Isolated unit tests for individual components
- **integration/**: Tests for component interactions
- **performance/**: Performance benchmarking tests
- **e2e/**: End-to-end workflow tests
- **demos/**: Demonstration scripts
- **utilities/**: Test helpers and verification scripts
- **data/**: Test fixtures and data

### `/data` - Runtime Data
Stores runtime data:
- **logs/**: Application and error logs
- **metrics/**: Performance metrics data
- **patterns/**: Discovered patterns from learning
- **registry/**: Tool registry database

### `/config` - Configuration
Configuration files for the system:
- **config.json**: Main configuration file with all settings

### `/docs` - Documentation
Comprehensive documentation:
- **api/**: API specifications and data models
- **architecture/**: System design and architecture
- **deployment/**: Deployment and infrastructure guides
- **design/**: Design documents and diagrams
- **evaluation/**: Evaluation framework documentation
- **implementation/**: Detailed implementation guides
- **project/**: Project management documentation
- **testing/**: Test documentation and coverage

### `/notebooks` - Jupyter Notebooks
Interactive notebooks for experimentation and analysis

### `/experiments` - Experimental Code
Prototype implementations and experimental features

### `/scripts` - Utility Scripts
Helper scripts for setup, deployment, and maintenance

## Key Files

### Root Directory
- **README.md**: Project overview and quick start guide
- **CLAUDE.md**: Guidance for AI assistants
- **requirements.txt**: Python package dependencies
- **pytest.ini**: Pytest configuration
- **setup.md**: Setup documentation and instructions
- **package.json**: Node.js dependencies for MCP tools
- **docker-compose.postgres.yml**: PostgreSQL Docker setup

### Entry Points
- **src/main.py**: Main application entry point
- **demos/hello_mcp.py**: Simple MCP test script

### Configuration
- **config/config.json**: Central configuration file
- **.gitignore**: Git ignore patterns
- **pytest.ini**: Test configuration

## Naming Conventions

### Python Files
- Snake_case for module names: `tool_discovery_agent.py`
- PascalCase for class names: `ToolDiscoveryAgent`
- snake_case for functions: `discover_tools()`

### Test Files
- Prefix with `test_`: `test_mcp_integration.py`
- Test classes: `TestMCPIntegration`
- Test methods: `test_tool_discovery()`

### Documentation
- Kebab-case for doc files: `system-architecture.md`
- Descriptive names for clarity

## Module Organization

### Import Structure
- Absolute imports from src: `from src.agents.intent_recognition_agent import IntentRecognitionAgent`
- Relative imports within modules: `from .base import BaseAgent`

### Package Structure
- `__init__.py` files in all packages
- Public API exposed through `__init__.py`
- Private implementations prefixed with underscore