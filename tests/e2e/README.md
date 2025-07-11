# Filesystem MCP End-to-End Tests

This directory contains end-to-end tests for the Filesystem MCP integration in the Autonomous Tool Discovery system.

## Test Structure

### 1. Simple E2E Tests (`test_filesystem_simple_e2e.py`)
These tests work with the current implementation and test:
- Basic file operations (read, write, exists)
- Directory operations (create, list)
- Tool registry integration
- Error handling scenarios
- Performance tracking

### 2. Comprehensive E2E Tests (`test_filesystem_e2e.py`)
These tests demonstrate full workflow integration (requires complete agent implementation):
- User query → Intent recognition → Tool discovery → Execution
- Multi-tool workflows
- Learning system integration
- Complex error scenarios

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

#### Option 1: Use the test runner script
```bash
# Run simple tests (works with current implementation)
python tests/e2e/run_e2e_tests.py --type simple

# Run comprehensive tests (requires full agent implementation)
python tests/e2e/run_e2e_tests.py --type comprehensive

# Run all available tests
python tests/e2e/run_e2e_tests.py --type all

# Run with verbose output
python tests/e2e/run_e2e_tests.py --type simple --verbose
```

#### Option 2: Run test files directly
```bash
# Run simple E2E tests
python tests/e2e/test_filesystem_simple_e2e.py

# Run comprehensive E2E tests (if agents are implemented)
python tests/e2e/test_filesystem_e2e.py
```

## Test Coverage

### Simple E2E Tests
1. **File Operations Flow**
   - Write file
   - Check existence
   - Read file
   - List directory

2. **Directory Operations Flow**
   - Create nested directories
   - Create files in directories
   - Navigate directory structure

3. **Registry Integration**
   - Tool registration
   - Tool discovery by capability

4. **Error Scenarios**
   - Non-existent file handling
   - Path traversal prevention
   - Invalid operations

5. **Performance Tracking**
   - Batch operations timing
   - Performance assertions

### Comprehensive E2E Tests
1. **Simple File Operations**
   - Natural language query processing
   - File creation and reading

2. **Directory Operations**
   - Complex directory structures
   - Multi-level navigation

3. **Complex Workflows**
   - Multi-step operations
   - Data processing pipelines

4. **Error Handling**
   - Graceful error recovery
   - Security enforcement

5. **Learning Integration**
   - Pattern recognition
   - Performance optimization

6. **Performance Metrics**
   - Operation timing
   - Resource usage tracking

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

## Adding New Tests

To add new E2E tests:

1. Add test methods to existing test classes
2. Follow the naming convention: `test_<feature>_<aspect>`
3. Use descriptive logging for test steps
4. Include assertions for all expected outcomes
5. Clean up any created resources

Example:
```python
async def test_new_feature(self):
    """Test description."""
    logger.info("\n=== Test: New Feature ===")
    
    # Test implementation
    result = await self.filesystem_client.some_operation()
    assert result["success"], "Operation failed"
    
    logger.info("✅ New feature test passed!")
```