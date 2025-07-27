# Development Commands Reference

This document contains all non-test commands for development, running components, and monitoring the system. For test-specific commands, see `tests/README.md`.

## Setup Commands

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify setup
python verify_setup.py

# Windows-specific setup verification
python tests/utilities/verify_setup_windows.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

## Code Quality Commands

### Code Formatting
```bash
# Format all code with Black
black src/ tests/

# Check formatting without changes
black --check src/ tests/

# Format specific file
black src/main.py
```

### Linting
```bash
# Run flake8 linter
flake8 src/ tests/

# Run with specific config
flake8 --config=.flake8 src/

# Check specific file
flake8 src/agents/intent_recognition_agent.py
```

### Type Checking
```bash
# Run mypy type checker
mypy src/

# Run with strict mode
mypy --strict src/

# Check specific module
mypy src/learning/
```

## Running Components

### Main Application
```bash
# Run the main integrated system
python src/main.py

# Run with custom config
python src/main.py --config config/custom_config.json
```

### Individual Components
```bash
# Test MCP connection
python src/hello_mcp.py

# Run mock MCP servers
python src/tools/mock_mcp_servers.py

# Test logger configuration
cd src && python utils/logger.py
```

### MCP Tool Demos
```bash
# SQLite MCP demo
python src/tools/sqlite_mcp.py

# Filesystem MCP demo
python src/tools/filesystem_mcp.py

# Search MCP demo
python src/tools/search_mcp.py

# Weather MCP demo
python src/tools/custom_wrappers/weather_mcp.py

# GitHub MCP demo
python src/tools/github_mcp.py

# Financial Datasets MCP demo
python src/tools/financial_datasets_mcp.py

# Zerodha MCP demo
python src/tools/zerodha_mcp.py

# Notion MCP demo
python src/tools/notion_mcp.py
```

## Demo Scripts

### Learning System Demos
```bash
# Q-learning demo
python src/learning/test_q_learning.py

# Q-learning with orchestrator
python demos/demo_q_learning_orchestration.py

# Pattern mining demo
python demos/demo_pattern_mining.py

# Deep Q-learning comparison
python demos/demo_dqn_learning.py

# Advanced reward strategies
python demos/demo_advanced_rewards.py
```

### Evaluation Demos
```bash
# A/B testing framework (full demo with 6 scenarios)
python demos/demo_ab_testing_framework.py

# Run only Demo 5 (reward strategy comparison)
python run_demo5_only.py

# Baseline evaluation demo
python demos/demo_baseline_evaluation.py --mode quick

# Real-time monitoring demo
python demos/demo_realtime_monitoring.py
```

### Integration Demos
```bash
# Pipeline refactor demo
python tests/demos/demo_pipeline_refactor.py

# Retry logic demo
python tests/demos/demo_retry_logic.py

# Integration test demo
python tests/demos/test_integration_demo.py

# GitHub MCP demos
python tests/demos/demo_github_mcp.py
python tests/demos/demo_github_real.py

# Notion MCP comprehensive demo
python tests/demos/demo_notion_mcp.py
```

## Monitoring Commands

### Performance Monitoring
```bash
# Monitor Intent Recognition performance
python -c "from src.agents.intent_recognition_agent import IntentRecognitionAgent; agent = IntentRecognitionAgent(); print(agent.get_metrics_summary())"

# Export Intent Recognition metrics
python -c "from src.agents.intent_recognition_agent import IntentRecognitionAgent; agent = IntentRecognitionAgent(); agent.export_metrics('metrics_report.json')"

# Monitor Retry Metrics
python -c "from src.monitoring.retry_metrics import RetryMetricsCollector; from src.core.tool_registry import ToolRegistry; collector = RetryMetricsCollector(ToolRegistry()); print(collector.get_retry_statistics())"
```

### Real-time Monitoring
```bash
# Start real-time monitoring service
python src/evaluation/realtime_monitor.py

# Connect to monitoring WebSocket
# Use WebSocket client to connect to ws://localhost:8000/ws
```

## Database Commands

### SQLite Database Management
```bash
# View tool registry database
sqlite3 data/registry/tools.db ".tables"
sqlite3 data/registry/tools.db "SELECT * FROM tools;"

# Export database schema
sqlite3 data/registry/tools.db ".schema" > schema.sql

# Backup database
sqlite3 data/registry/tools.db ".backup backup.db"
```

### Database Migrations
```bash
# Run database migrations (if implemented)
python scripts/migrate_db.py

# Reset database
rm data/registry/tools.db
python src/database/tool_registry.py  # Recreates database
```

## Environment Variables

### API Keys Setup
```bash
# Brave Search API
export BRAVE_API_KEY='your-api-key'

# GitHub API
export GITHUB_TOKEN='your-github-token'

# OpenWeather API
export OPENWEATHER_API_KEY='your-api-key'

# Notion Integration
export NOTION_INTEGRATION_TOKEN='your-notion-token'

# Financial Datasets API (Note: requires OAuth 2.1 for real server)
export FINANCIAL_DATASETS_API_KEY='your-api-key'

# PostgreSQL connection
export POSTGRES_TEST_URL='postgresql://user:pass@localhost/testdb'
```

### Configuration Override
```bash
# Override config file
export CONFIG_PATH='/path/to/custom/config.json'

# Set logging level
export LOG_LEVEL='DEBUG'

# Enable debug mode
export DEBUG='true'
```

## Development Utilities

### Documentation Generation
```bash
# Generate API documentation (if configured)
python scripts/generate_docs.py

# Serve documentation locally
python -m http.server 8080 --directory docs/
```

### Performance Profiling
```bash
# Profile main application
python -m cProfile -o profile.stats src/main.py

# Analyze profile results
python -m pstats profile.stats

# Memory profiling
python -m memory_profiler src/main.py
```

### Dependency Management
```bash
# Generate requirements file
pip freeze > requirements.txt

# Check for outdated packages
pip list --outdated

# Install specific package version
pip install package==1.2.3

# Uninstall package
pip uninstall package
```

## Jupyter Notebook Commands

```bash
# Start Jupyter server
jupyter notebook

# Start JupyterLab
jupyter lab

# Convert notebook to script
jupyter nbconvert --to script notebook.ipynb
```

## Git Commands (Project Specific)

```bash
# Check project status
git status

# Stage all changes except node_modules
git add . && git reset node_modules/

# Common commit patterns
git commit -m "feat: Add new MCP tool integration"
git commit -m "fix: Resolve connection timeout issue"
git commit -m "docs: Update implementation status"
git commit -m "test: Add unit tests for pattern miner"

# Push to main branch
git push origin main
```

## Troubleshooting Commands

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python src/main.py

# Run with verbose output
python src/main.py -v

# Run with trace logging
python src/main.py --trace
```

### Check System Status
```bash
# Check Python version
python --version

# List installed packages
pip list

# Check available MCP tools
python -c "from src.database.tool_registry import ToolRegistry; registry = ToolRegistry(); print(registry.list_tools())"

# Verify imports
python -c "import src.agents.intent_recognition_agent; print('Import successful')"
```

### Clean Up
```bash
# Remove Python cache files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete

# Clean test artifacts
rm -rf .pytest_cache/
rm -rf htmlcov/
rm -f .coverage

# Clean logs
rm -rf data/logs/*
rm -rf tests/data/logs/*
```

## Docker Commands (If Implemented)

```bash
# Build Docker image
docker build -t auto-tool-disc .

# Run container
docker run -it auto-tool-disc

# Run with volume mount
docker run -v $(pwd)/data:/app/data auto-tool-disc

# Docker Compose
docker-compose up
docker-compose down
```

## Performance Commands

### Load Testing
```bash
# Run load tests (if implemented)
python tests/performance/load_test.py

# Benchmark specific component
python -m timeit -s "from src.agents.intent_recognition_agent import IntentRecognitionAgent; agent = IntentRecognitionAgent()" "agent.recognize_intent('Find Python files')"
```

### Resource Monitoring
```bash
# Monitor system resources during execution
python scripts/monitor_resources.py &
python src/main.py

# Check memory usage
ps aux | grep python
```

## Notes

- Always activate the virtual environment before running commands
- Ensure required environment variables are set for external APIs
- For test-specific commands, refer to `tests/README.md`
- Some commands may require additional setup or configuration
- Mock servers are used by default for testing without external dependencies