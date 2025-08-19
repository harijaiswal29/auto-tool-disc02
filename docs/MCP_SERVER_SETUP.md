# MCP Server Setup Guide

This guide explains how to set up real MCP servers for the demo application.

## Overview

The demo application attempts to connect to real MCP servers first and falls back to mock servers when real servers are unavailable. This provides the best demonstration of the system's capabilities.

## Real MCP Server Setup

### 1. Weather Server (OpenWeather API)
To use the real weather server:
1. Get a free API key from [OpenWeather](https://openweathermap.org/api)
2. Set the environment variable:
   ```bash
   export OPENWEATHER_API_KEY="your_api_key_here"
   ```

### 2. Search Server
The search server currently uses mock implementation. Real search would require:
- Google Custom Search API key
- Or Bing Search API key
- Or other search provider credentials

### 3. Filesystem Server
To use the real filesystem MCP server:
1. Install the official MCP filesystem server:
   ```bash
   npm install -g @modelcontextprotocol/server-filesystem
   ```
2. The demo will automatically:
   - Create a workspace directory at `data/filesystem_workspace/`
   - Use this as the base path for all file operations
   - Connect to the real MCP server if available

### 4. SQLite Server
To use the real SQLite MCP server:
1. Install the official MCP SQLite server:
   ```bash
   npm install -g @modelcontextprotocol/server-sqlite
   ```
2. The demo will create a database at `data/demo.db`

### 5. GitHub Server
To use the real GitHub MCP server:
1. Create a GitHub Personal Access Token:
   - Go to GitHub Settings → Developer Settings → Personal Access Tokens
   - Create a new token with appropriate permissions (repo, read:user)
2. Set the environment variable:
   ```bash
   export GITHUB_TOKEN="your_github_token_here"
   ```
3. Install the official MCP GitHub server (when available):
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

### 6. PostgreSQL Server
To use the real PostgreSQL MCP server:
1. Have a PostgreSQL database running (local or remote)
2. Set the connection string:
   ```bash
   export POSTGRES_CONNECTION_STRING="postgresql://username:password@localhost/dbname"
   ```
3. Install the official MCP PostgreSQL server (when available):
   ```bash
   npm install -g @modelcontextprotocol/server-postgresql
   ```
4. Ensure your database user has appropriate permissions

### 7. Notion Server
To use the real Notion MCP server:
1. Create a Notion Integration:
   - Go to [Notion Integrations](https://www.notion.so/my-integrations)
   - Create a new integration
   - Copy the Integration Token
2. Set the environment variable:
   ```bash
   export NOTION_API_KEY="your_notion_integration_token"
   ```
3. Share pages/databases with your integration:
   - In Notion, open the page/database you want to access
   - Click "Share" → "Invite" → Select your integration
4. The demo will automatically connect to Notion API

### 8. Financial Datasets Server
To use the real Financial Datasets MCP server:
1. Get an API key from [Financial Datasets](https://financialdatasets.ai)
2. Set the environment variable:
   ```bash
   export FINANCIAL_DATASETS_API_KEY="your_financial_api_key"
   ```
3. The demo will automatically connect to the Financial Datasets API
4. Features available:
   - Real-time stock prices
   - Historical market data
   - Financial statements
   - Market news and analysis

### 9. Zerodha Trading Server
To use the real Zerodha MCP server:
1. Get your Kite Connect API credentials:
   - Register at [Kite Trade](https://kite.trade/)
   - Create an app to get API key and secret
   - Generate access token through login flow
2. Set the environment variables:
   ```bash
   export ZERODHA_API_KEY="your_api_key"
   export ZERODHA_API_SECRET="your_api_secret"
   export ZERODHA_ACCESS_TOKEN="your_access_token"
   ```
3. Features available:
   - Portfolio management
   - Live market quotes
   - Order placement and tracking
   - Historical data access
   - P&L reporting

## Running the Demo

1. Set any available API keys as environment variables
2. Run the demo:
   ```bash
   python src/web/demo_app_real.py
   ```
3. Check the console output to see which servers are running in Real vs Mock mode

## Server Status in Demo

The demo will show server status like:
```
MCP Server Status:
  🌐 Search: Real Mode (with Brave API)
  🔧 Weather: Mock Mode (no API key)
  🌐 Filesystem: Real Mode (if npm package installed)
  🔧 SQLite: Mock Mode
  🌐 GitHub: Real Mode (with token)
  🌐 PostgreSQL: Real Mode (if database running)
```

- 🌐 = Real server connected
- 🔧 = Mock server (fallback)

## Benefits of Real Servers

When real servers are connected:
- Actual API calls are made (e.g., real weather data)
- Real filesystem operations are performed
- Database queries run against actual SQLite database
- Q-Learning adapts to real server response times and behaviors

## Mock Server Fallback

Mock servers provide:
- Consistent demo experience without API keys
- Predictable responses for testing
- No external dependencies
- Faster response times for development

The system seamlessly handles both real and mock servers, demonstrating the flexibility of the MCP integration layer.