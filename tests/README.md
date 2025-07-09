# Test Suite Organization

This directory contains all tests for the Auto Tool Discovery project, organized by test type.

## Directory Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for multiple components
├── e2e/              # End-to-end tests (to be added)
├── utilities/        # Test utilities and helper scripts
├── data/             # Test data and fixtures
└── conftest.py       # Pytest configuration and shared fixtures
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Use mocking for external dependencies
- Fast execution
- Example: `test_search_mcp.py` - Tests SearchMCPClient methods

### Integration Tests (`tests/integration/`)
- Test multiple components working together
- May spawn real processes or use external services
- Test actual file I/O, database operations, etc.
- Examples:
  - `test_sqlite_*.py` - SQLite MCP integration tests
  - `test_filesystem_mcp.py` - Filesystem operations
  - `test_github_*.py` - GitHub API integration
  - `verify_setup*.py` - System setup verification

### E2E Tests (`tests/e2e/`)
- Test complete workflows from start to finish
- Simulate real user scenarios
- (To be implemented)

### Utilities (`tests/utilities/`)
- Helper scripts for testing
- `check_encoding.py` - Verify Unicode/emoji support

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test categories:
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With markers
pytest -m unit
pytest -m integration
```

### Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file:
```bash
pytest tests/unit/test_search_mcp.py
```

### Run in verbose mode:
```bash
pytest -v
```

## Test Configuration

See `pytest.ini` in the project root for pytest configuration including:
- Test discovery patterns
- Coverage settings
- Asyncio configuration
- Custom markers
- Logging configuration

## Writing New Tests

1. **Unit Tests**: Place in `tests/unit/`, use mocking, test single components
2. **Integration Tests**: Place in `tests/integration/`, test real interactions
3. **Use Fixtures**: Common fixtures are in `conftest.py`
4. **Follow Naming**: Test files must start with `test_`
5. **Use Markers**: Apply appropriate markers (@pytest.mark.unit, etc.)

## Common Fixtures (from conftest.py)

- `event_loop`: Async event loop for test session
- `temp_dir`: Temporary directory for test files
- `sample_test_db`: SQLite database with sample data
- `mock_mcp_client`: Mock MCP client for testing
- `mock_tool_registry`: Mock tool registry
- `sample_config`: Sample configuration dict
- `cleanup_async_tasks`: Ensures async tasks are cleaned up

## Notes

- Integration tests may require external dependencies (npm packages, etc.)
- Some tests require environment variables (e.g., GITHUB_TOKEN)
- Use `verify_setup.py` to check if all prerequisites are installed