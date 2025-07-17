# Notion MCP Setup Guide

This guide will help you set up and configure the Notion MCP integration for the Auto Tool Discovery system.

## Overview

The Notion MCP (Model Context Protocol) integration enables the Auto Tool Discovery system to interact with Notion workspaces, allowing AI agents to:

- Create, read, update, and delete pages
- Manage databases and records
- Manipulate block-level content
- Search across the workspace
- Export content as Markdown

## Prerequisites

1. **Notion Account**: You need a Notion account with access to a workspace
2. **Python 3.8+**: Required for running the MCP client
3. **Node.js**: Required if you want to use the official Notion MCP server

## Setup Instructions

### Step 1: Create a Notion Integration

1. Go to https://www.notion.so/profile/integrations
2. Click "New integration" or "+ New integration"
3. Fill in the integration details:
   - **Name**: Auto Tool Discovery MCP (or any name you prefer)
   - **Associated workspace**: Select your workspace
   - **Capabilities**: Select all capabilities you need:
     - Read content
     - Update content
     - Insert content
     - Read comments (optional)
   - **User capabilities**: Select if needed
4. Click "Submit"
5. Copy the **Integration Token** (starts with `secret_`)

### Step 2: Share Pages/Databases with Your Integration

**Important**: Your integration can only access pages and databases that have been explicitly shared with it.

1. Open a Notion page or database you want to access
2. Click the "..." menu in the top right
3. Click "Add connections" or "Connections"
4. Search for your integration name
5. Click to add the integration
6. Repeat for all pages/databases you want to access

### Step 3: Configure Environment Variables

Set up the required environment variables:

```bash
# Required: Your Notion integration token
export NOTION_INTEGRATION_TOKEN="secret_YOUR_INTEGRATION_TOKEN_HERE"

# Optional: Custom MCP endpoint (defaults to https://api.notion.com/mcp/v1)
export NOTION_MCP_ENDPOINT="https://api.notion.com/mcp/v1"
```

You can add these to your `.env` file:

```env
NOTION_INTEGRATION_TOKEN=secret_YOUR_INTEGRATION_TOKEN_HERE
NOTION_MCP_ENDPOINT=https://api.notion.com/mcp/v1
```

### Step 4: Install Dependencies

The Notion MCP client is already included in the project. If you want to use the official Notion MCP server:

```bash
# Install the official Notion MCP server (optional)
npm install -g @notionhq/notion-mcp-server

# Or use via npx without installation
npx -y @notionhq/notion-mcp-server
```

### Step 5: Test the Connection

Run the basic test to verify your setup:

```bash
# Test with the built-in client
python src/tools/notion_mcp.py

# Or run the comprehensive demo
python demos/demo_notion_mcp.py
```

If the connection is successful, you should see output indicating that the client connected and discovered available tools.

## Configuration Options

The Notion MCP configuration in `config/config.json`:

```json
{
  "mcp_servers": {
    "notion": {
      "type": "remote",
      "endpoint": "https://api.notion.com/mcp/v1",
      "command": ["npx", "-y", "@notionhq/notion-mcp-server"],
      "auth_type": "integration_token",
      "enabled": true,
      "env": {
        "NOTION_INTEGRATION_TOKEN": "${NOTION_INTEGRATION_TOKEN}",
        "NOTION_MCP_ENDPOINT": "${NOTION_MCP_ENDPOINT}"
      },
      "config": {
        "api_version": "2022-06-28",
        "cache_ttl": 300,
        "markdown_export": true,
        "max_page_size": 100
      }
    }
  }
}
```

### Configuration Parameters

- **type**: Set to "remote" for the hosted Notion MCP server
- **endpoint**: The MCP server endpoint URL
- **auth_type**: Authentication method (integration_token)
- **cache_ttl**: Cache time-to-live in seconds (default: 300)
- **markdown_export**: Enable Notion-flavored Markdown export
- **max_page_size**: Maximum number of results per query

## Usage Examples

### Basic Usage

```python
from src.tools.notion_mcp import NotionMCPClient

# Initialize client
client = NotionMCPClient()

# Connect to Notion
await client.connect()

# Create a page
page = await client.create_page(
    title="My First Page",
    content="# Hello from MCP\n\nThis page was created via the MCP integration."
)

# Search for pages
results = await client.search_pages("MCP")
```

### Using with Tool Registry

```python
from src.core.tool_registry import ToolRegistry

# Register Notion tools
registry = ToolRegistry()
client.register_tools_to_registry(registry)

# Now Notion tools are available for discovery
notion_tools = registry.search_tools_by_domain("documentation")
```

## Available Tools

The Notion MCP integration provides the following tools:

1. **create_page**: Create new pages with content and properties
2. **get_page**: Retrieve page content in Markdown or JSON format
3. **update_page**: Update page title, content, or properties
4. **delete_page**: Archive/delete pages
5. **search_pages**: Search for pages by query
6. **create_database**: Create databases with custom properties
7. **query_database**: Query databases with filters and sorts
8. **create_database_record**: Add records to databases
9. **append_block**: Add blocks (paragraphs, headings, lists) to pages
10. **list_workspace_pages**: List all accessible pages

## Testing

Run the test suite to verify your integration:

```bash
# Unit tests
pytest tests/unit/test_notion_mcp.py -v

# Integration tests
pytest tests/integration/test_notion_mcp.py -v

# Run specific test
pytest tests/integration/test_notion_mcp.py::TestNotionMCPIntegration::test_create_and_get_page -v
```

## Troubleshooting

### Common Issues

1. **"No integration token provided"**
   - Ensure `NOTION_INTEGRATION_TOKEN` is set in your environment
   - Check that the token starts with `secret_`

2. **"Page not found" errors**
   - Ensure the page/database is shared with your integration
   - Check that you're using the correct page/database ID

3. **"Unauthorized" errors**
   - Verify your integration token is correct
   - Check that your integration has the required capabilities

4. **Connection timeouts**
   - Check your internet connection
   - Verify the endpoint URL is correct
   - Try increasing timeout values in configuration

### Using Mock Server

For development and testing without a real Notion workspace:

```python
# Connect to mock server instead
await client.connect(use_mock=True)

# All operations will use the mock server
page = await client.create_page(title="Mock Page")
```

## Security Considerations

1. **Token Security**: Never commit your integration token to version control
2. **Page Permissions**: Only share necessary pages with your integration
3. **Rate Limiting**: The integration includes retry logic and exponential backoff
4. **Data Privacy**: Be mindful of what data you're accessing and storing

## Advanced Features

### Caching

The client includes automatic caching for read operations:

```python
# First call - fetches from server
page1 = await client.get_page("page-id")

# Second call - returns from cache
page2 = await client.get_page("page-id")

# Clear cache when needed
client.clear_cache()
```

### Batch Operations

For better performance with multiple operations:

```python
# Create multiple pages
pages = []
for i in range(10):
    page = await client.create_page(f"Page {i}")
    pages.append(page)
```

### Error Handling

The integration includes comprehensive error handling:

```python
try:
    page = await client.get_page("invalid-id")
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

## Integration with Learning System

The Notion MCP tools are automatically integrated with the Q-learning system:

- Tool usage patterns are learned over time
- Successful tool combinations are rewarded
- The system adapts to prefer efficient Notion operations

## Further Resources

- [Notion API Documentation](https://developers.notion.com/)
- [Model Context Protocol Spec](https://github.com/anthropics/mcp)
- [Notion MCP GitHub Repository](https://github.com/makenotion/notion-mcp-server)
- [Auto Tool Discovery Documentation](../README.md)