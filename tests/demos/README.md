# Demo Scripts

This directory contains demonstration scripts that showcase various features and components of the Auto Tool Discovery system.

## Available Demos

### 1. demo_pipeline_refactor.py
Demonstrates the refactored intent recognition pipeline with modular architecture.

**Key Features:**
- Modular pipeline stages
- Performance monitoring
- Multi-intent handling
- Context enrichment

**Usage:**
```bash
python tests/demos/demo_pipeline_refactor.py
```

### 2. demo_retry_logic.py
Demonstrates the retry and resilience system with exponential backoff and circuit breakers.

**Key Features:**
- Exponential backoff retry policies
- Circuit breaker pattern
- Connection pooling
- Retry metrics collection

**Usage:**
```bash
python tests/demos/demo_retry_logic.py
```

### 3. test_integration_demo.py
Demonstrates the complete integration of all system components.

**Key Features:**
- End-to-end query processing
- Tool discovery and selection
- Parallel tool execution
- Result aggregation

**Usage:**
```bash
python tests/demos/test_integration_demo.py
```

### 4. demo_github_mcp.py
Demonstrates the GitHub MCP integration with mock server.

**Key Features:**
- GitHub repository operations
- Issue and PR management
- Mock MCP server setup

**Usage:**
```bash
python tests/demos/demo_github_mcp.py
```

### 5. demo_github_real.py
Demonstrates real GitHub integration (requires GitHub token).

**Key Features:**
- Real GitHub API interactions
- Repository analysis
- Issue tracking

**Usage:**
```bash
# Set GitHub token first
export GITHUB_TOKEN=your_token_here
python tests/demos/demo_github_real.py
```

## Running All Demos

To run all demos sequentially:

```bash
cd tests/demos
for demo in demo_*.py test_*.py; do
    echo "Running $demo..."
    python "$demo"
    echo "---"
done
```

## Notes

- These demos are for demonstration and testing purposes only
- Some demos require specific environment variables or configurations
- Check individual demo files for specific requirements
- Demos may create temporary files or data - clean up after running