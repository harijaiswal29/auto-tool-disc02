# Troubleshooting Guide

This document provides solutions to common issues encountered when working with the Autonomous Tool Discovery and Integration System.

> **Related Documentation**: 
> - [Development Commands Reference](./development/commands-reference.md)
> - [Test Suite Documentation](../tests/README.md)
> - [Configuration Guide](./deployment/configuration.md)
> - [System Architecture](./architecture/system-architecture.md)

## Common Pitfalls & Troubleshooting

### Import Path Issues
**Problem**: `ModuleNotFoundError` when importing project modules
**Solution**: Many scripts modify `sys.path` to add the project root:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### MCP Server Connection Timeouts
**Problem**: MCP servers take too long to start or connection times out
**Solutions**:
- Increase timeout in connection settings
- Use mock servers for development/testing
- Check if required npm packages are installed for real servers
- Verify API keys are correctly set in environment

### Memory Issues with Embeddings
**Problem**: Out of memory errors when processing large batches
**Solutions**:
- Reduce batch size in sentence-transformers
- Use caching to avoid recomputing embeddings
- Consider using a smaller model (e.g., all-MiniLM-L6-v2)

### Async Task Cleanup
**Problem**: "Task was destroyed but it is pending" warnings
**Solution**: Ensure proper cleanup in async functions:
```python
try:
    # async operations
finally:
    await client.disconnect()
    # Cancel any pending tasks
```

### Mock Server Fallback Not Working
**Problem**: System fails when real server unavailable instead of using mock
**Solutions**:
- Check mock server files exist in `src/tools/mock_*.py`
- Ensure fallback logic is implemented in the MCP client
- Verify `use_mock` parameter is properly handled
- Check environment variables are not set (system defaults to mock when API keys missing)
- See [MCP Integration](./architecture/mcp-communication.md) for fallback logic details

### Q-Learning Not Converging
**Problem**: Learning system not improving tool selection
**Solutions**:
- Check reward values are appropriate (not all zero)
- Verify state representation is meaningful (should be 447 dimensions)
- Adjust learning parameters (α, γ, ε) in config.json
- Ensure sufficient exploration (epsilon not too low)
- Review reward calculation logs for anomalies
- See [Learning System](./implementation/learning-system.md) for parameter tuning guidance
- Check [Configuration Guide](./deployment/configuration.md) for optimal settings

### Database Lock Errors
**Problem**: "database is locked" errors with SQLite
**Solutions**:
- Use connection pooling with proper limits
- Implement retry logic for database operations
- Consider using WAL mode for SQLite: `PRAGMA journal_mode=WAL`
- Ensure connections are properly closed
- Check [Database Schema](./architecture/database-schema.md) for connection best practices

## Additional Resources

### Testing Issues
For test-related issues, see:
- [Test Suite Documentation](../tests/README.md)
- [Test Summary](./testing/test-summary.md)

### Performance Issues
For performance optimization:
- [Performance Targets](./evaluation/evaluation-targets.md)
- [Retry Architecture](./architecture/retry-architecture.md) for resilience patterns

### Configuration Issues
For configuration problems:
- [Configuration Guide](./deployment/configuration.md)
- [Environment Setup](./.env.example) for required API keys

### Development Issues
For development workflow issues:
- [Development Commands](./development/commands-reference.md)
- [Project Structure](./project/project-structure.md)