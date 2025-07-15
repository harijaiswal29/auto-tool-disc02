# End-to-End Test Suite

This directory contains comprehensive end-to-end tests for the Autonomous Tool Discovery system, covering all major workflows and tool integrations.

## Test Structure

The E2E test suite is organized into the following categories:

### Core Workflows (`core_workflows/`)
Tests for fundamental system workflows:

1. **test_query_to_execution_e2e.py**
   - Complete pipeline from natural language query to execution
   - Intent recognition → Tool discovery → Selection → Execution
   - Performance metrics validation
   - Error handling workflows

2. **test_multi_intent_e2e.py**
   - Multi-intent query processing
   - Sequential and parallel intent execution
   - Dependency resolution between intents
   - Complex real-world scenarios

3. **test_context_persistence_e2e.py**
   - Session continuity across queries
   - User preference learning
   - Context-aware tool selection
   - Conversation state tracking

### Tool Integrations (`tool_integrations/`)
Tests for specific MCP tool integrations:

1. **test_sqlite_e2e.py**
   - Database query workflows
   - Data modification operations
   - Schema exploration
   - Complex queries with joins and aggregations

2. **test_search_e2e.py**
   - Web search functionality
   - Research and fact-checking workflows
   - Multi-source search
   - Search with filters and constraints

3. **test_filesystem_e2e.py** (existing)
   - File operations (read, write, exists)
   - Directory operations
   - Path navigation
   - File search and filtering

### Existing Tests
1. **test_filesystem_simple_e2e.py** - Basic filesystem operations
2. **test_filesystem_standalone.py** - Standalone filesystem tests

## Running the Tests

### Prerequisites
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure MCP server binaries are available:
   ```bash
   ls -la node_modules/.bin/mcp-server-filesystem
   ```

### Running Tests

#### Run All E2E Tests
```bash
# Run all E2E tests with pytest
pytest tests/e2e/ -v

# Run with coverage report
pytest tests/e2e/ --cov=src --cov-report=html
```

#### Run by Category
```bash
# Core workflow tests
pytest tests/e2e/core_workflows/ -v

# Tool integration tests
pytest tests/e2e/tool_integrations/ -v

# Specific workflow
pytest tests/e2e/core_workflows/test_query_to_execution_e2e.py -v
```

#### Run Specific Test Methods
```bash
# Run a specific test
pytest tests/e2e/core_workflows/test_multi_intent_e2e.py::TestMultiIntentE2E::test_parallel_multi_intent -v

# Run tests matching a pattern
pytest tests/e2e/ -k "workflow" -v
```

#### Run Legacy Tests
```bash
# Filesystem tests
python tests/e2e/test_filesystem_simple_e2e.py

# Use the test runner script
python tests/e2e/run_e2e_tests.py --type simple
```

## Test Coverage

### Core System Components Tested

1. **Intent Recognition Pipeline**
   - Natural language understanding
   - Intent classification accuracy
   - Multi-intent detection
   - Context awareness
   - Performance metrics (<100ms p95)

2. **Tool Discovery & Selection**
   - Semantic search for tools
   - Capability matching
   - Context-based tool ranking
   - Tool relationship graph navigation
   - Q-learning optimization

3. **Execution & Orchestration**
   - Sequential execution workflows
   - Parallel execution optimization
   - Error handling and recovery
   - Retry logic with exponential backoff
   - Circuit breaker protection

4. **Context & Learning**
   - Session persistence
   - User preference learning
   - Pattern mining from successful executions
   - Performance improvement over time
   - Feedback integration

5. **MCP Tool Integrations**
   - Filesystem operations
   - Database queries (SQLite)
   - Web search functionality
   - Multi-tool orchestration
   - Tool-specific error handling

### Test Scenarios by Category

#### Query Processing (core_workflows/)
- Simple single-intent queries
- Complex multi-intent queries
- Ambiguous query handling
- Context-dependent queries
- Error recovery scenarios

#### Data Operations (tool_integrations/)
- CRUD operations on databases
- File system navigation
- Search and information retrieval
- Complex data transformations
- Schema exploration

#### System Capabilities
- Performance under load
- Concurrent session handling
- Learning convergence
- Metric collection accuracy
- Resilience to failures

## Expected Output

### Successful Test Run
```
Starting Simple Filesystem MCP End-to-End Tests
==================================================

=== Test: File Operations Flow ===
Step 1: Writing test file...
✓ File written successfully
Step 2: Checking file existence...
✓ File existence confirmed
Step 3: Reading file content...
✓ File content verified
Step 4: Listing directory...
✓ Directory listing successful, found 1 items
✅ File operations flow test passed!

[... more tests ...]

==================================================
✅ All Simple E2E tests passed successfully!
==================================================
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'aiohttp'**
   - Solution: `pip install -r requirements.txt`

2. **MCP server not found**
   - The tests will automatically fall back to mock implementation
   - To use real server: `npm install` in project root

3. **Permission denied errors**
   - Ensure you have write permissions in the test directory
   - Tests create temporary directories for isolation

4. **Agent implementation missing**
   - Comprehensive tests require full agent implementation
   - Use simple tests for current functionality

## Planned E2E Tests (To Be Implemented)

### Complex Workflows (`complex_workflows/`)
1. **test_multi_tool_orchestration_e2e.py**
   - GitHub + Filesystem workflows
   - Database + Search combinations
   - Multi-step data pipelines

2. **test_data_pipeline_e2e.py**
   - Extract data from files
   - Transform and analyze
   - Store results in database
   - Generate reports

### Resilience Tests (`resilience/`)
1. **test_retry_scenarios_e2e.py**
   - Network timeout recovery
   - Service unavailability handling
   - Rate limit management

2. **test_circuit_breaker_e2e.py**
   - Circuit breaker activation
   - Graceful degradation
   - Recovery monitoring

### Performance Tests (`performance/`)
1. **test_load_scenarios_e2e.py**
   - Concurrent user simulations
   - Resource usage monitoring
   - Scalability testing

2. **test_optimization_e2e.py**
   - Learning convergence rates
   - Cache effectiveness
   - Query optimization

### Additional Tool Integrations
1. **test_postgres_e2e.py** - PostgreSQL operations
2. **test_github_e2e.py** - Repository operations
3. **test_weather_e2e.py** - External API integration

## Adding New Tests

To add new E2E tests:

1. Choose appropriate category directory
2. Follow the existing test structure
3. Use descriptive test names: `test_<feature>_<aspect>`
4. Include comprehensive logging
5. Test both success and failure paths
6. Add performance assertions
7. Clean up test artifacts

Example structure:
```python
class TestNewFeatureE2E:
    """E2E tests for new feature."""
    
    @pytest.fixture
    async def setup_system(self):
        """Set up test environment."""
        # Initialize components
        # Return test context
        # Handle cleanup
        pass
    
    @pytest.mark.asyncio
    async def test_feature_workflow(self, setup_system):
        """Test complete feature workflow."""
        logger.info("\n=== E2E Test: Feature Workflow ===")
        
        # Arrange
        orchestrator = setup_system["orchestrator"]
        
        # Act
        result = await orchestrator.process_user_query("...")
        
        # Assert
        assert result.success
        assert result.intent is not None
        
        logger.info("✅ Feature workflow test passed!")
```