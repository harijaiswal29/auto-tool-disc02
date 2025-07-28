# Test Suite Documentation

This directory contains the comprehensive test suite for the Auto Tool Discovery system, organized by test type and following best practices for Python testing.

## Directory Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_mcp_integration.py
│   ├── test_connection_pool.py
│   ├── test_orchestrator_agent.py
│   ├── test_tool_discovery_agent.py
│   ├── test_intent_pipeline_stages.py
│   ├── test_conversation_state_machine.py
│   ├── test_search_mcp.py
│   ├── test_github_mcp.py                  # GitHub MCP unit tests
│   ├── test_notion_mcp.py                  # Notion MCP unit tests
│   ├── test_weather_mcp.py                 # Weather MCP unit tests
│   ├── test_financial_datasets_mcp.py      # Financial Datasets MCP unit tests
│   ├── test_zerodha_mcp.py                 # Zerodha MCP unit tests
│   ├── test_postgres_mcp.py                # PostgreSQL MCP unit tests
│   ├── test_postgres_real_server_unit.py   # PostgreSQL real server unit tests
│   ├── test_sqlite_mcp.py                  # SQLite MCP unit tests
│   ├── test_state_machine_base.py
│   ├── test_retry.py
│   ├── test_retry_extended.py              # Extended retry tests (connection pool, registry)
│   ├── test_intent_recognition.py          # Intent recognition unit tests
│   ├── test_intent_recognition_metrics.py  # Intent recognition metrics
│   ├── test_retry_metrics.py               # Retry metrics monitoring
│   ├── test_q_learning_engine.py           # Q-Learning engine tests
│   ├── test_pattern_miner.py               # Pattern mining tests
│   ├── test_context_extractor.py           # Context extraction tests
│   ├── test_reward_calculator.py           # Reward calculation tests
│   ├── test_enhanced_state_representation.py  # Enhanced state representation tests
│   ├── test_incremental_pattern_mining.py    # Incremental pattern mining tests
│   ├── test_dqn.py                         # Deep Q-Network tests
│   ├── test_advanced_rewards.py            # Advanced reward strategies tests
│   ├── test_baseline_strategies.py         # Baseline strategy tests
│   ├── test_evaluation_engine.py           # Evaluation engine tests
│   ├── test_metrics_collector.py           # Metrics collection tests
│   ├── test_ab_testing_framework.py        # A/B testing framework tests
│   ├── test_ab_test_manager.py             # A/B test manager tests
│   └── WEATHER_MCP_TEST_SUMMARY.md         # Weather MCP test documentation
├── integration/              # Integration tests
│   ├── test_filesystem_mcp.py
│   ├── test_github_mcp.py
│   ├── test_github_direct.py               # GitHub direct protocol testing
│   ├── test_github_real_direct.py          # GitHub real server testing
│   ├── test_github_simple.py               # GitHub basic server startup test
│   ├── test_notion_mcp.py                  # Notion MCP integration tests
│   ├── test_intent_recognition_integration.py
│   ├── test_postgres_mcp.py
│   ├── test_postgres_real_server.py        # PostgreSQL real server tests
│   ├── test_brave_search_direct.py         # Brave Search API integration
│   ├── test_sqlite_mcp.py
│   ├── test_state_machine_integration.py
│   ├── test_weather_mcp.py                 # Weather MCP integration tests
│   ├── test_financial_datasets_mcp.py      # Financial Datasets MCP integration tests
│   ├── test_financial_datasets_mcp_backup.py  # Financial Datasets backup tests
│   ├── test_zerodha_mcp.py                 # Zerodha MCP integration tests
│   ├── test_all_mcp_tools.py               # All MCP tools integration
│   ├── test_pipeline_workflow.py           # Full pipeline workflow
│   ├── test_retry_integration.py           # Retry scenarios with MCP
│   ├── test_context_persistence.py         # Context persistence integration
│   ├── test_context_aware_pattern_mining.py  # Context-aware pattern mining
│   ├── test_failure_learning.py            # Failure learning mechanisms
│   ├── test_q_learning_integration.py      # Q-Learning integration tests
│   ├── test_baseline_evaluation.py         # Baseline evaluation tests
│   ├── test_pipeline_architecture.py       # Pipeline architecture tests
│   ├── test_integration.py                 # End-to-end integration tests
│   ├── test_real_mcp.py                    # Real MCP server testing
│   └── test_real_tools.py                  # Real tools integration
├── performance/             # Performance tests
│   ├── test_intent_recognition_performance.py
│   └── test_tool_discovery_performance.py
├── e2e/                     # End-to-end tests
│   └── test_filesystem_e2e.py
├── demos/                   # Test-specific demonstration scripts
│   ├── demo_pipeline_refactor.py
│   ├── demo_retry_logic.py
│   ├── test_integration_demo.py
│   ├── demo_github_mcp.py
│   ├── demo_github_real.py
│   ├── demo_financial_datasets.py
│   ├── demo_financial_datasets_output.md
│   ├── demo_postgres_mcp.py
│   └── README.md
├── utilities/              # Test utilities and helpers
│   ├── check_encoding.py
│   ├── verify_setup.py     # Verify test environment setup
│   └── verify_setup_windows.py  # Windows-specific setup verification
├── data/                   # Test data and fixtures
│   ├── fixtures/           # Reusable test data
│   │   ├── tools.json     # Sample tool definitions
│   │   ├── intents.json   # Sample intent data
│   │   └── queries.json   # Sample user queries
│   ├── expected/          # Expected output files
│   ├── logs/             # Test execution logs
│   ├── results/          # Test results (JUnit XML, summaries, reports)
│   └── temp/             # Temporary test files
├── conftest.py            # Pytest configuration
```

**Note**: The project also has a main `demos/` directory at the project root containing comprehensive demonstration scripts for various features:
- `demo_ab_testing_framework.py` - A/B testing demonstrations
- `demo_advanced_rewards.py` - Advanced reward strategies
- `demo_baseline_evaluation.py` - Baseline comparison demos
- `demo_dqn_learning.py` - Deep Q-Learning demonstrations
- `demo_pattern_mining.py` - Pattern mining features
- `demo_q_learning_orchestration.py` - Q-learning with orchestrator
- `demo_realtime_monitoring.py` - Real-time monitoring
- And several others (see `/demos/README.md` for complete list)

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Characteristics**:
  - Use extensive mocking for external dependencies
  - Fast execution (< 1 second per test)
  - High coverage of edge cases and error conditions
  - No external service dependencies

**Key Unit Tests:**
- `test_mcp_integration.py` - Core MCP integration functionality
- `test_connection_pool.py` - Connection pooling and management
- `test_orchestrator_agent.py` - Query orchestration logic
- `test_tool_discovery_agent.py` - Tool discovery algorithms
- `test_retry.py` - Core retry logic and circuit breakers
- `test_retry_extended.py` - Extended retry tests including connection pool and tool registry
- `test_intent_recognition.py` - Intent recognition unit tests
- `test_intent_recognition_metrics.py` - Intent recognition performance monitoring
- `test_retry_metrics.py` - Retry attempt and circuit breaker metrics
- `test_q_learning_engine.py` - Q-Learning engine with state representation, action space, and experience replay
- `test_pattern_miner.py` - Pattern mining algorithms for discovering tool synergies
- `test_context_extractor.py` - Context extraction for pattern discovery
- `test_reward_calculator.py` - Reward calculation strategies and implementations
- `test_enhanced_state_representation.py` - Enhanced state representations for Q-learning
- `test_incremental_pattern_mining.py` - Incremental pattern mining algorithms
- `test_dqn.py` - Deep Q-Network implementations and architectures
- `test_advanced_rewards.py` - Advanced reward shaping strategies
- `test_baseline_strategies.py` - Baseline strategy implementations for comparison
- `test_evaluation_engine.py` - Evaluation framework engine tests
- `test_metrics_collector.py` - Performance metrics collection and aggregation
- `test_ab_testing_framework.py` - A/B testing framework implementation
- `test_ab_test_manager.py` - A/B test management and orchestration
- `test_sqlite_mcp.py` - SQLite MCP client unit tests with mocking
- `test_postgres_mcp.py` - PostgreSQL MCP client unit tests
- `test_postgres_real_server_unit.py` - PostgreSQL real server unit tests
- `test_notion_mcp.py` - Notion MCP client unit tests (covering all operations)
- `test_weather_mcp.py` - Weather MCP client unit tests (covering all operations)
- `test_financial_datasets_mcp.py` - Financial Datasets MCP client unit tests (covering all operations)
- `test_zerodha_mcp.py` - Zerodha trading platform MCP client unit tests

### Integration Tests (`tests/integration/`)
- **Purpose**: Test multiple components working together
- **Characteristics**:
  - May use real or mock MCP servers
  - Test actual I/O operations (file, database, network)
  - Verify component interactions
  - May require external dependencies

**Key Integration Tests:**
- MCP tool integrations:
  - **SQLite MCP** (`test_sqlite_mcp.py`)
  - **PostgreSQL MCP** (`test_postgres_mcp.py`)
  - **PostgreSQL Real Server** (`test_postgres_real_server.py` - tests with actual PostgreSQL)
  - **GitHub MCP** (`test_github_mcp.py`)
  - **GitHub Direct** (`test_github_direct.py`, `test_github_real_direct.py`, `test_github_simple.py`)
  - **Search MCP** - unit tests with mock (see `test_search_mcp.py` in unit tests)
  - **Brave Search Direct** (`test_brave_search_direct.py` - real API integration)
  - **Filesystem MCP** (`test_filesystem_mcp.py`)
  - **Weather MCP** (`test_weather_mcp.py`)
  - **Notion MCP** (`test_notion_mcp.py` - comprehensive integration testing)
  - **Financial Datasets MCP** (`test_financial_datasets_mcp.py` - Comprehensive integration testing with 27 test cases)
  - **Financial Datasets Backup** (`test_financial_datasets_mcp_backup.py` - backup/recovery testing)
  - **Zerodha MCP** (`test_zerodha_mcp.py` - trading platform integration)
  - **All MCP Tools** (`test_all_mcp_tools.py` - comprehensive multi-tool testing)
  - **Real MCP** (`test_real_mcp.py` - real MCP server testing)
  - **Real Tools** (`test_real_tools.py` - real tool implementations)
    - Mock server testing (default mode)
    - Real API simulation with mocked HTTP responses
    - Concurrent operations and load testing
    - Caching and performance optimization testing
    - Error recovery and reconnection scenarios
    - Integration with MCP Integration framework
    - Multi-tool workflow integration (with Filesystem MCP)
- Learning and pattern discovery:
  - `test_context_aware_pattern_mining.py` - Context-aware pattern discovery
  - `test_failure_learning.py` - Learning from failures and errors
  - `test_q_learning_integration.py` - Q-learning integration with real tools
  - `test_baseline_evaluation.py` - Baseline strategy evaluation
- System integration:
  - `test_intent_recognition_integration.py` - Intent recognition pipeline integration
  - `test_state_machine_integration.py` - State machine workflow integration
  - `test_pipeline_workflow.py` - Complete end-to-end pipeline testing
  - `test_retry_integration.py` - Retry mechanisms with real MCP connections
  - `test_context_persistence.py` - Context and conversation persistence
  - `test_integration.py` - Full system integration tests
  - `test_pipeline_architecture.py` - Pipeline architecture and stage interactions

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Characteristics**:
  - Simulate real user scenarios
  - Test full system behavior
  - Longer execution time acceptable
  - Verify business requirements

### Performance Tests (`tests/performance/`)
- **Purpose**: Benchmark and validate system performance
- **Characteristics**:
  - Measure processing times and throughput
  - Test scalability under load
  - Monitor resource usage
  - Identify performance bottlenecks

**Key Performance Tests:**
- `test_intent_recognition_performance.py` - Intent recognition speed and scalability
- `test_tool_discovery_performance.py` - Tool discovery and pipeline performance

### Demo Scripts (`tests/demos/`)
- **Purpose**: Demonstrate system capabilities
- **Usage**: Educational and testing purposes
- See `tests/demos/README.md` for details

### Utilities (`tests/utilities/`)
- Helper scripts for testing
- `check_encoding.py` - Verify Unicode/emoji support

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run all tests with coverage report
pytest --cov=src --cov-report=html --cov-report=term
```

### Run by Category

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# End-to-end tests only
pytest tests/e2e/ -v

# Performance tests only
pytest tests/performance/ -v

# Using markers
pytest -m unit
pytest -m integration
pytest -m asyncio  # All async tests
```

### Run Specific Tests

```bash
# Run a specific test file
pytest tests/unit/test_mcp_integration.py -v

# Run a specific test class
pytest tests/unit/test_mcp_integration.py::TestMCPIntegration -v

# Run a specific test method
pytest tests/unit/test_mcp_integration.py::TestMCPIntegration::test_initialization -v

# Run tests matching a pattern
pytest -k "test_retry" -v

# Run Q-Learning tests
pytest tests/unit/test_q_learning_engine.py -v

# Run SQLite MCP tests
pytest tests/unit/test_sqlite_mcp.py -v  # Unit tests with mock server
pytest tests/integration/test_sqlite_mcp.py -v  # Integration tests
# Both test files support testing with real and mock SQLite MCP servers
# The tests automatically fall back to mock server if real server is not available

# Run Search MCP tests
pytest tests/unit/test_search_mcp.py -v  # Unit tests with mock server
# For real API testing:
export BRAVE_API_KEY='your-api-key'
python tests/integration/test_brave_search_direct.py  # Direct Brave Search API test

# Run GitHub MCP tests
pytest tests/unit/test_github_mcp.py -v  # Unit tests with mock server
pytest tests/integration/test_github_mcp.py -v  # Integration tests
# For real GitHub API testing:
export GITHUB_TOKEN='your-github-token'
python tests/integration/test_github_mcp.py  # Tests with real GitHub API

# Run Notion MCP tests
pytest tests/unit/test_notion_mcp.py -v  # Unit tests with mock server
python demos/demo_notion_mcp.py  # Comprehensive demo with mock/real server
# For real Notion API testing:
export NOTION_INTEGRATION_TOKEN='your-notion-integration-token'
python tests/integration/test_notion_mcp.py  # Integration tests

# Run Weather MCP tests
pytest tests/unit/test_weather_mcp.py -v  # Unit tests with mock server
pytest tests/integration/test_weather_mcp.py -v  # Integration tests
# For real OpenWeather API testing:
export OPENWEATHER_API_KEY='your-openweather-api-key'
python src/tools/custom_wrappers/weather_mcp.py  # Run basic Weather test

# Run Financial Datasets MCP tests
pytest tests/unit/test_financial_datasets_mcp.py -v  # Unit tests with mock server
pytest tests/integration/test_financial_datasets_mcp.py -v  # Integration tests

# Run Zerodha MCP tests
pytest tests/unit/test_zerodha_mcp.py -v  # Unit tests with mock server
pytest tests/integration/test_zerodha_mcp.py -v  # Integration tests

# Zerodha MCP Integration test scenarios include:
# - Connection testing with mock Zerodha MCP server
# - Tool discovery and registration (12 trading tools)
# - Trading operations (holdings, positions, orders, quotes)
# - Order management (place, modify, cancel orders)
# - Market data operations (quotes, LTP, historical data)
# - Account operations (margins, instruments)
# - Caching functionality for market data
# - Error handling and order validation
# - Concurrent operations testing
# - Load testing (50+ concurrent requests)
# - Reconnection scenarios
# - Real API simulation with delays
# - Integration with MCP Integration framework
# - Multi-tool workflow with Filesystem MCP
# - Error recovery testing

# For real Zerodha API testing:
# IMPORTANT: Real Zerodha MCP server requires valid API credentials
# The mock server provides full trading functionality for testing
export ZERODHA_API_KEY='your-api-key'
export ZERODHA_API_SECRET='your-api-secret'
export ZERODHA_ACCESS_TOKEN='your-access-token'
python tests/integration/test_zerodha_mcp.py  # Tests with real API (requires valid credentials)

# Run specific integration tests:
pytest tests/integration/test_financial_datasets_mcp.py -k "concurrent" -v  # Concurrent operations
pytest tests/integration/test_financial_datasets_mcp.py -k "performance" -v  # Performance tests
pytest tests/integration/test_financial_datasets_mcp.py -k "load" -v  # Load testing

# For real Financial Datasets API testing:
# IMPORTANT: Real Financial Datasets MCP server requires OAuth 2.1 authentication
# Current implementation uses API keys which is not supported by the real server
# To use real server, OAuth 2.1 flow must be implemented (see https://docs.financialdatasets.ai/mcp-server)
# The mock server provides full functionality for testing all features
export FINANCIAL_DATASETS_API_KEY='your-api-key'  # Won't work - OAuth required
python tests/integration/test_financial_datasets_mcp.py  # Tests with real API (will fail)
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate terminal coverage report with missing lines
pytest --cov=src --cov-report=term-missing

# Coverage for specific modules
pytest --cov=src.core --cov=src.agents --cov-report=html

# Set coverage failure threshold
pytest --cov=src --cov-report=term --cov-fail-under=80
```

### Performance and Benchmarking

```bash
# Run with performance profiling
pytest --profile

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run only fast tests (custom marker)
pytest -m "not slow"
```

### Debugging Tests

```bash
# Stop on first failure
pytest -x

# Drop into debugger on failures
pytest --pdb

# Show local variables for failures
pytest -l

# Increase verbosity for debugging
pytest -vv

# Show print statements
pytest -s
```

## Test Configuration

See `pytest.ini` in the project root for pytest configuration including:
- Test discovery patterns
- Coverage settings
- Asyncio configuration
- Custom markers
- Logging configuration

## Writing New Tests

### Best Practices

1. **Test Organization**
   - Place unit tests in `tests/unit/`
   - Place integration tests in `tests/integration/`
   - Place e2e tests in `tests/e2e/`
   - Use descriptive test names: `test_<functionality>_<scenario>`

2. **Test Structure**
   ```python
   class TestComponentName:
       """Test cases for ComponentName."""
       
       @pytest.fixture
       def setup_data(self):
           """Setup test data."""
           return {...}
       
       def test_normal_operation(self, setup_data):
           """Test normal operation scenario."""
           # Arrange
           component = Component(setup_data)
           
           # Act
           result = component.operation()
           
           # Assert
           assert result.success is True
       
       def test_error_scenario(self):
           """Test error handling."""
           with pytest.raises(ExpectedError):
               component.failing_operation()
   ```

3. **Mocking Guidelines**
   - Mock external dependencies in unit tests
   - Use `unittest.mock` for synchronous code
   - Use `AsyncMock` for async functions
   - Keep mocks simple and focused

4. **Async Testing**
   ```python
   @pytest.mark.asyncio
   async def test_async_operation():
       result = await async_function()
       assert result is not None
   ```

5. **Test Markers**
   ```python
   @pytest.mark.unit
   @pytest.mark.asyncio
   @pytest.mark.slow
   def test_example():
       pass
   ```

## Common Fixtures (from conftest.py)

### Core Fixtures
- `event_loop`: Async event loop for test session
- `cleanup_async_tasks`: Ensures async tasks are cleaned up

### Data Fixtures
- `temp_dir`: Temporary directory for test files
- `sample_test_db`: SQLite database with sample data
- `sample_config`: Sample configuration dictionary

### Mock Fixtures
- `mock_mcp_client`: Mock MCP client for testing
- `mock_tool_registry`: Mock tool registry
- `mock_intent_result`: Mock intent recognition result

### Utility Fixtures
- `capture_logs`: Capture log output for assertions
- `benchmark`: Performance benchmarking fixture

## Test Data Organization

```
tests/data/
├── fixtures/          # Reusable test data
│   ├── tools.json    # Sample tool definitions
│   ├── intents.json  # Sample intent data
│   └── queries.json  # Sample user queries
├── expected/         # Expected output files
├── logs/            # Test execution logs
├── results/          # Test results (JUnit XML, summaries, reports)
└── temp/            # Temporary test files
```

## Coverage Guidelines

### Target Coverage
- **Overall**: 80% minimum
- **Core modules** (`src/core/`): 90% minimum
- **Agent modules** (`src/agents/`): 90% minimum
- **Utilities** (`src/utils/`): 70% minimum

### Coverage Exclusions
- Demo scripts
- Main entry points (`if __name__ == "__main__"`)
- Abstract base classes
- Type checking blocks (`if TYPE_CHECKING:`)

## GitHub MCP Testing

### Overview
The GitHub MCP tests are organized into unit tests and integration tests:

**Unit Tests** (`tests/unit/test_github_mcp.py`):
- Comprehensive test cases covering GitHubMCPClient functionality
- Tests connection handling (mock/real server fallback)
- Tests tool discovery and execution
- Tests error handling and edge cases
- Mock server tests (MockGitHubMCPServer)
- All tests use mock server by default (no GitHub token required)

**Integration Tests** (`tests/integration/`):
- `test_github_mcp.py` - Main comprehensive integration test
- `test_github_direct.py` - Direct protocol testing
- `test_github_simple.py` - Basic server startup test
- `test_github_real_direct.py` - Real server testing with debugging
- `test_all_mcp_tools.py` - Tests GitHub MCP with other tools in the system

### Mock vs Real Server Testing
- **Mock Server**: Default for all tests, provides comprehensive GitHub API simulation
- **Real Server**: Requires `GITHUB_TOKEN` environment variable
- The system automatically falls back to mock server if real server connection fails

### Integration Test Results

**Real GitHub Server Test Results:**
- Successfully connects to real GitHub MCP server with valid token
- Discovers 26 tools from the real server (vs 8 in mock)
- Real server tools include: `create_or_update_file`, `search_repositories`, `get_file_contents`, `push_files`, `create_issue`, `create_pull_request`, etc.
- Tool names differ between real and mock servers (e.g., `create_pull_request` vs `create_pull`)

**Mock Server Test Results:**
- All 8 mock tools execute successfully
- Provides fast, reliable testing without external dependencies
- Simulates: `list_repos`, `search_repos`, `get_repo`, `create_issue`, `list_issues`, `create_pull`, `list_pulls`, `search_code`

### Running GitHub MCP Integration Tests

```bash
# Unit tests (always uses mock server)
pytest tests/unit/test_github_mcp.py -v

# Integration tests with mock server (default)
python tests/integration/test_github_mcp.py

# Integration tests with real GitHub server
export GITHUB_TOKEN='your-github-token'
python tests/integration/test_github_mcp.py

# Test GitHub MCP with all other tools
python tests/integration/test_all_mcp_tools.py

# Run all GitHub-related tests
pytest tests -k "github" -v

# Run PostgreSQL tests
pytest tests/unit/test_postgres_mcp.py -v  # Unit tests
pytest tests/unit/test_postgres_real_server_unit.py -v  # Real server unit tests
pytest tests/integration/test_postgres_mcp.py -v  # Integration tests
pytest tests/integration/test_postgres_real_server.py -v  # Real server integration
# For real PostgreSQL testing:
export POSTGRES_TEST_URL='postgresql://user:pass@localhost/testdb'

# Run learning and pattern mining tests
pytest tests/unit/test_pattern_miner.py -v  # Pattern mining algorithms
pytest tests/unit/test_context_extractor.py -v  # Context extraction
pytest tests/unit/test_incremental_pattern_mining.py -v  # Incremental mining
pytest tests/integration/test_context_aware_pattern_mining.py -v  # Context-aware patterns
pytest tests/integration/test_failure_learning.py -v  # Failure learning

# Run reward and evaluation tests
pytest tests/unit/test_reward_calculator.py -v  # Reward calculation
pytest tests/unit/test_enhanced_state_representation.py -v  # Enhanced states
pytest tests/unit/test_advanced_rewards.py -v  # Advanced reward strategies
pytest tests/unit/test_baseline_strategies.py -v  # Baseline strategies
pytest tests/unit/test_evaluation_engine.py -v  # Evaluation engine
pytest tests/integration/test_baseline_evaluation.py -v  # Baseline evaluation

# Run A/B testing framework tests
pytest tests/unit/test_ab_testing_framework.py -v  # A/B testing framework
pytest tests/unit/test_ab_test_manager.py -v  # A/B test management

# Run Deep Q-Learning tests
pytest tests/unit/test_dqn.py -v  # Deep Q-Network tests
pytest tests/integration/test_q_learning_integration.py -v  # Q-learning integration

# Run real MCP server tests
python tests/integration/test_real_mcp.py  # Test with real MCP servers
python tests/integration/test_real_tools.py  # Test with real tool implementations
```

### Manual Testing Instructions

1. **Test with Mock Server (No Token Required)**:
   ```python
   from src.tools.github_mcp import GitHubMCPClient
   
   client = GitHubMCPClient()
   await client.connect(use_mock=True)
   repos = await client.execute_tool("list_repos", {"username": "test"})
   await client.disconnect()
   ```

2. **Test with Real Server**:
   ```python
   import os
   from src.tools.github_mcp import GitHubMCPClient
   
   # Set token
   os.environ['GITHUB_TOKEN'] = 'your-github-token'
   
   client = GitHubMCPClient()
   await client.connect(use_mock=False)
   
   # Use real tool names from server
   result = await client.execute_tool("search_repositories", {
       "query": "language:python",
       "max_results": 5
   })
   await client.disconnect()
   ```

### Mock Server Capabilities
The mock server (`src/tools/mock_github_mcp.py`) simulates:
- Repository operations (list, search, get details, get content)
- Issue management (create, list)
- Pull request operations (create, list)
- Code search functionality
- File operations (create/update, get contents, push multiple files)
- User information retrieval
- Proper JSON-RPC protocol handling

### Mock Server Tools (13 tools)
1. **list_repositories** - List repositories for a user or organization
2. **search_repositories** - Search for repositories
3. **get_repository** - Get repository details
4. **create_issue** - Create a new issue
5. **list_issues** - List issues for a repository
6. **create_pull_request** - Create a pull request
7. **list_pull_requests** - List pull requests
8. **search_code** - Search for code across GitHub
9. **create_or_update_file** - Create or update a file in a repository
10. **get_file_contents** - Get the contents of a file
11. **push_files** - Push multiple files to a repository
12. **get_user** - Get user information
13. **get_repository_content** - Get repository content at a specific path

### Key Differences: Mock vs Real Server
| Feature | Mock Server | Real Server |
|---------|-------------|-------------|
| Tools Count | 13 (core tools) | 26 (full feature set) |
| Authentication | Not required | GitHub token required |
| Tool Names | Matches real server | Full names |
| Response Time | Instant | Network dependent |
| Data | Simulated | Real GitHub data |
| Rate Limits | None | GitHub API limits apply |
| Backward Compatibility | Yes (old names work) | No |

### Mock-Only Tools
The mock server includes 4 essential tools not available in the real server:

| Tool | Purpose | Why Included |
|------|---------|--------------|
| `list_repositories` | List user/org repos | Fundamental GitHub operation |
| `get_repository` | Get repo details | Essential for testing |
| `get_user` | Get user info | Common requirement |
| `get_repository_content` | Browse repo contents | Basic navigation need |

These tools exist to make the mock server more useful for testing and development. See [Migration Guide](../docs/migration/github-mcp-tool-names.md#mock-only-tools) for real server alternatives.

### Tool Name Mapping (Backward Compatibility)
The mock server supports both old and new tool names:
- `list_repos` → `list_repositories`
- `search_repos` → `search_repositories`
- `get_repo` → `get_repository`
- `create_pull` → `create_pull_request`
- `list_pulls` → `list_pull_requests`

### Unimplemented Tools in Mock Server
The following tools are available only in the real GitHub MCP server:
- `create_branch`, `merge_pull_request`, `create_release`
- `list_commits`, `get_commit`, `list_branches`, `delete_branch`
- `list_releases`, `get_release`, `update_issue`, `close_issue`
- `add_labels`, `remove_labels`, `create_comment`, `update_comment`
- And several others

When attempting to use these tools with the mock server, you'll receive an error message:
```
Tool '{tool_name}' is not implemented in mock server. Available in real GitHub MCP server only.
```

## Environment Variables

### Required for Integration Tests
```bash
# GitHub integration
export GITHUB_TOKEN=your_github_token

# PostgreSQL tests
export POSTGRES_TEST_URL=postgresql://user:pass@localhost/testdb

# Weather API tests
export WEATHER_API_KEY=your_api_key

# Brave Search API tests (for real API testing)
export BRAVE_API_KEY=your_brave_api_key

# Financial Datasets API tests (for real API testing)
export FINANCIAL_DATASETS_API_KEY=your_api_key
```

### Optional Configuration
```bash
# Test execution
export PYTEST_TIMEOUT=300  # Test timeout in seconds
export PYTEST_WORKERS=4    # Parallel test workers

# Coverage
export COVERAGE_THRESHOLD=80  # Minimum coverage percentage
```

## Continuous Integration

### Pre-commit Checks
1. Run unit tests: `pytest tests/unit/ -x`
2. Check coverage: `pytest --cov=src --cov-fail-under=80`
3. Verify formatting: `black --check src/ tests/`
4. Run linting: `flake8 src/ tests/`

### CI Pipeline Stages
1. **Lint & Format**: Code quality checks
2. **Unit Tests**: Fast, isolated tests
3. **Integration Tests**: Component interaction tests
4. **Coverage Report**: Generate and upload coverage
5. **E2E Tests**: Full workflow validation

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in PYTHONPATH
   - Check for circular imports
   - Verify `__init__.py` files exist

2. **Async Test Failures**
   - Use `pytest-asyncio` markers
   - Properly await async calls
   - Clean up async resources

3. **Mock Issues**
   - Verify mock spec matches interface
   - Check mock call counts and arguments
   - Use `assert_called_with` for verification

4. **Flaky Tests**
   - Add proper delays for async operations
   - Use deterministic test data
   - Mock time-dependent functions

## Test Data Organization

### Fixtures (`tests/data/fixtures/`)
The test suite includes comprehensive test data fixtures:

- **tools.json**: Complete tool definitions with capabilities, operations, and relationships
- **intents.json**: Intent taxonomy with keywords, entities, and confidence mappings
- **queries.json**: Sample queries for testing various scenarios including:
  - Test queries with expected outcomes
  - Edge cases for robustness testing
  - Performance benchmarking queries
  - Validation queries for accuracy testing

### Using Test Fixtures
```python
# Example: Loading test data in your tests
import json

def load_test_tools():
    with open('tests/data/fixtures/tools.json', 'r') as f:
        return json.load(f)

def load_test_queries():
    with open('tests/data/fixtures/queries.json', 'r') as f:
        return json.load(f)
```

## Notes

- Integration tests may require external dependencies (npm packages, database servers)
- Some tests require specific environment variables (see above)
- Use `verify_setup.py` to check prerequisites
- Run `pytest --markers` to see available test markers
- See `docs/testing/test-summary.md` for comprehensive testing documentation
- New monitoring and performance tests have been added - see directory structure above