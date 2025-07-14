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
│   ├── test_state_machine_base.py
│   ├── test_retry.py
│   ├── test_intent_recognition_metrics.py  # NEW: Intent recognition metrics
│   └── test_retry_metrics.py               # NEW: Retry metrics monitoring
├── integration/              # Integration tests
│   ├── test_filesystem_mcp.py
│   ├── test_github_mcp.py
│   ├── test_intent_recognition_integration.py
│   ├── test_postgres_mcp.py
│   ├── test_search_mcp_integration.py
│   ├── test_sqlite_mcp.py
│   ├── test_state_machine_integration.py
│   ├── test_weather_mcp.py
│   ├── test_pipeline_workflow.py           # NEW: Full pipeline workflow
│   └── test_retry_integration.py           # NEW: Retry scenarios with MCP
├── performance/             # Performance tests (NEW)
│   ├── test_intent_recognition_performance.py
│   └── test_tool_discovery_performance.py
├── e2e/                     # End-to-end tests
│   └── test_filesystem_e2e.py
├── demos/                   # Demonstration scripts
│   ├── demo_pipeline_refactor.py
│   ├── demo_retry_logic.py
│   ├── test_integration_demo.py
│   ├── demo_github_mcp.py
│   ├── demo_github_real.py
│   └── README.md
├── utilities/              # Test utilities and helpers
│   └── check_encoding.py
├── data/                   # Test data and fixtures
│   ├── fixtures/           # Reusable test data (NEW)
│   │   ├── tools.json     # Sample tool definitions
│   │   ├── intents.json   # Sample intent data
│   │   └── queries.json   # Sample user queries
│   ├── expected/          # Expected output files
│   ├── logs/             # Test execution logs
│   └── temp/             # Temporary test files
├── conftest.py            # Pytest configuration
├── test_context_persistence.py
├── test_intent_recognition.py
├── test_integration.py
├── test_pipeline_architecture.py
└── test_retry_logic.py
```

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
- `test_retry.py` - Retry logic and circuit breakers
- `test_intent_recognition_metrics.py` - Intent recognition performance monitoring
- `test_retry_metrics.py` - Retry attempt and circuit breaker metrics

### Integration Tests (`tests/integration/`)
- **Purpose**: Test multiple components working together
- **Characteristics**:
  - May use real or mock MCP servers
  - Test actual I/O operations (file, database, network)
  - Verify component interactions
  - May require external dependencies

**Key Integration Tests:**
- MCP tool integrations (SQLite, PostgreSQL, GitHub, etc.)
- Intent recognition pipeline integration
- State machine workflow integration
- `test_pipeline_workflow.py` - Complete end-to-end pipeline testing
- `test_retry_integration.py` - Retry mechanisms with real MCP connections

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

## Environment Variables

### Required for Integration Tests
```bash
# GitHub integration
export GITHUB_TOKEN=your_github_token

# PostgreSQL tests
export POSTGRES_TEST_URL=postgresql://user:pass@localhost/testdb

# Weather API tests
export WEATHER_API_KEY=your_api_key
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