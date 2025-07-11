# PostgreSQL Setup for Auto Tool Discovery

This directory contains the PostgreSQL database setup scripts for the Auto Tool Discovery project.

## Quick Start

1. **Start PostgreSQL using Docker Compose:**
   ```bash
   docker-compose up -d postgres
   ```

2. **Verify the database is running:**
   ```bash
   docker-compose ps
   docker-compose logs postgres
   ```

3. **Connect to the database:**
   ```bash
   # Using docker exec
   docker exec -it auto_tool_disc_postgres psql -U auto_tool_user -d auto_tool_disc

   # Using psql directly
   psql postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc
   ```

## Database Details

- **Host:** localhost
- **Port:** 5432
- **Database:** auto_tool_disc
- **Username:** auto_tool_user
- **Password:** auto_tool_pass
- **Connection String:** `postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc`

## Schema Overview

The database includes the following tables:

1. **tools** - Registry of available tools
2. **tool_relationships** - Relationships between tools
3. **execution_history** - History of tool executions
4. **q_learning_states** - Q-learning state storage
5. **q_values** - Q-learning value storage
6. **discovered_patterns** - Mined patterns from executions
7. **performance_metrics** - Tool performance metrics

## Testing with PostgreSQL MCP

Once the database is running, you can test the PostgreSQL MCP client:

```bash
# Run the test script
cd src/tools
python postgres_mcp.py

# Or run integration tests
cd src/core
python mcp_integration.py
```

## Troubleshooting

1. **Connection refused error:**
   - Ensure Docker is running
   - Check if port 5432 is available: `lsof -i :5432`
   - Verify container is healthy: `docker-compose ps`

2. **Authentication failed:**
   - Check credentials in docker-compose.yml match your connection string
   - Ensure the database has been created

3. **MCP server fails to start:**
   - Check if the PostgreSQL MCP server binary exists: `ls -la node_modules/.bin/mcp-server-postgres`
   - Verify Node.js dependencies are installed: `npm install`

## Stopping the Database

```bash
# Stop the container
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```