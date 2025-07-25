# GitHub MCP Tool Names Migration Guide

## Overview

The Mock GitHub MCP server has been updated to align tool names with the real GitHub MCP server. This guide helps you migrate existing code to use the new tool names.

## Tool Name Changes

The following tool names have been updated:

| Old Name | New Name | Description |
|----------|----------|-------------|
| `list_repos` | `list_repositories` | List repositories for a user or organization |
| `search_repos` | `search_repositories` | Search for repositories |
| `get_repo` | `get_repository` | Get repository details |
| `create_pull` | `create_pull_request` | Create a pull request |
| `list_pulls` | `list_pull_requests` | List pull requests |

## Backward Compatibility

The mock server maintains backward compatibility by automatically mapping old tool names to new ones. This means:
- Existing code using old names will continue to work
- You can gradually migrate to new names
- No immediate code changes required

## Migration Steps

### 1. Update Tool Calls

Replace old tool names with new ones in your code:

```python
# Old code
result = await client.execute_tool("list_repos", {"username": "test"})

# New code
result = await client.execute_tool("list_repositories", {"username": "test"})
```

### 2. Update Tool Registration

If you're registering tools, update the names:

```python
# Old
tools = ["list_repos", "search_repos", "create_pull"]

# New
tools = ["list_repositories", "search_repositories", "create_pull_request"]
```

### 3. Update Tests

Update your test cases to use new tool names:

```python
# Old test
assert tool["name"] == "list_repos"

# New test
assert tool["name"] == "list_repositories"
```

## New Tools Added

The mock server now includes additional tools to better match the real server:

1. **create_or_update_file** - Create or update a file in a repository
2. **get_file_contents** - Get the contents of a file from a repository
3. **push_files** - Push multiple files to a repository
4. **get_user** - Get user information
5. **get_repository_content** - Get repository content at a specific path

### Example: Using New File Operations

```python
# Create or update a file
result = await client.execute_tool("create_or_update_file", {
    "owner": "username",
    "repo": "repository",
    "path": "README.md",
    "content": base64.b64encode(b"# Hello World").decode(),
    "message": "Update README"
})

# Get file contents
result = await client.execute_tool("get_file_contents", {
    "owner": "username",
    "repo": "repository",
    "path": "README.md"
})
```

## Tool Coverage

### Mock Server (13 tools)
Covers ~80% of common GitHub operations:
- Repository management
- Issue tracking
- Pull requests
- File operations
- Code search
- User information

### Real Server (26 tools)
Full GitHub API coverage including:
- All mock server tools
- Branch management
- Commit operations
- Release management
- Label management
- Comment operations
- And more

## Best Practices

1. **Use Real Tool Names**: When writing new code, use the real server tool names
2. **Test Both Servers**: Ensure your code works with both mock and real servers
3. **Handle Unimplemented Tools**: Check for tools not available in mock server
4. **Document Dependencies**: Note which tools your code requires

## Error Handling

When using tools not implemented in the mock server:

```python
try:
    result = await client.execute_tool("merge_pull_request", {...})
except RuntimeError as e:
    if "not implemented in mock server" in str(e):
        # Handle mock server limitation
        print("This operation requires real GitHub server")
    else:
        raise
```

## Mock-Only Tools

The following tools are available only in the mock server to provide essential GitHub functionality that developers expect:

| Tool | Purpose | Real Server Alternative |
|------|---------|------------------------|
| `list_repositories` | List user/org repositories | Use `search_repositories` with `user:username` query |
| `get_repository` | Get repository details | Use `search_repositories` with `repo:owner/name` query |
| `get_user` | Get user information | Use `search_users` with exact username (if available) |
| `get_repository_content` | Browse repository files/dirs | Use `get_file_contents` for individual files |

### Why These Tools Exist in Mock Only

These tools exist in the mock server because:
1. They represent fundamental GitHub operations that developers naturally expect
2. They're essential for comprehensive testing scenarios
3. They make the mock server more intuitive and useful for development
4. Their absence in the real server appears to be a limitation rather than a design choice

### Using Real Server Alternatives

When working with the real GitHub MCP server, achieve similar functionality using:

```python
# Instead of list_repositories
result = await client.execute_tool("search_repositories", {
    "query": "user:octocat",
    "max_results": 100
})

# Instead of get_repository (specific repo)
result = await client.execute_tool("search_repositories", {
    "query": "repo:octocat/Hello-World"
})

# For user information, you may need to use search or other indirect methods
# as the real server doesn't provide direct user info access
```

## Timeline

- **Phase 1** (Current): Backward compatibility maintained
- **Phase 2** (Future): Deprecation warnings for old names
- **Phase 3** (Future): Old names removed

## Questions?

For questions or issues related to this migration, please refer to:
- Project documentation: `docs/`
- Test examples: `tests/unit/test_github_mcp.py`
- Integration tests: `tests/integration/test_github_mcp.py`