# PostgreSQL Setup Guide for MCP Testing

## Overview

This guide helps you set up PostgreSQL database for testing the PostgreSQL MCP implementation with real server.

## Installation Options

### Option 1: Using Docker (Recommended)

```bash
# Pull PostgreSQL image
docker pull postgres:14

# Run PostgreSQL container
docker run -d \
  --name postgres-mcp-test \
  -e POSTGRES_USER=auto_tool_user \
  -e POSTGRES_PASSWORD=auto_tool_pass \
  -e POSTGRES_DB=auto_tool_disc \
  -p 5432:5432 \
  postgres:14

# Verify it's running
docker ps | grep postgres-mcp-test

# Check logs if needed
docker logs postgres-mcp-test
```

### Option 2: Native Installation (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check status
sudo systemctl status postgresql
```

### Option 3: Native Installation (macOS)

```bash
# Using Homebrew
brew install postgresql@14

# Start PostgreSQL
brew services start postgresql@14

# Or manually
pg_ctl -D /usr/local/var/postgres start
```

### Option 4: WSL2 (Windows)

Since you're on WSL2, here's the specific setup:

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL (WSL2 doesn't use systemctl by default)
sudo service postgresql start

# Check if it's running
sudo service postgresql status

# To make it start automatically
echo "sudo service postgresql start" >> ~/.bashrc
```

## Database Setup

After PostgreSQL is running, set up the test database:

### 1. Connect as postgres user

```bash
# For Docker
docker exec -it postgres-mcp-test psql -U postgres

# For native installation
sudo -u postgres psql
```

### 2. Create user and database

```sql
-- Create user
CREATE USER auto_tool_user WITH PASSWORD 'auto_tool_pass';

-- Create database
CREATE DATABASE auto_tool_disc OWNER auto_tool_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE auto_tool_disc TO auto_tool_user;

-- Exit
\q
```

### 3. Test connection

```bash
# Test with psql
psql -h localhost -U auto_tool_user -d auto_tool_disc

# You should be prompted for password: auto_tool_pass
# Type \q to exit
```

## Configuration

### PostgreSQL Configuration (if needed)

Edit PostgreSQL configuration to allow connections:

```bash
# Find config location
sudo -u postgres psql -c "SHOW config_file;"

# Edit postgresql.conf
sudo nano /etc/postgresql/14/main/postgresql.conf

# Ensure these settings:
listen_addresses = 'localhost'
port = 5432
```

### Authentication Configuration

```bash
# Edit pg_hba.conf
sudo nano /etc/postgresql/14/main/pg_hba.conf

# Add/modify this line for password authentication:
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5

# Restart PostgreSQL after changes
sudo service postgresql restart
```

## Troubleshooting

### Check if PostgreSQL is running

```bash
# Check process
ps aux | grep postgres

# Check port
sudo netstat -tlnp | grep 5432
# or
sudo ss -tlnp | grep 5432

# Check with lsof
sudo lsof -i :5432
```

### Common Issues

1. **Port already in use**
   ```bash
   # Find what's using port 5432
   sudo lsof -i :5432
   
   # Kill the process if needed
   sudo kill -9 <PID>
   ```

2. **Permission denied**
   ```bash
   # Fix PostgreSQL data directory permissions
   sudo chown -R postgres:postgres /var/lib/postgresql/
   sudo chmod 700 /var/lib/postgresql/14/main
   ```

3. **Service won't start**
   ```bash
   # Check logs
   sudo journalctl -xeu postgresql
   # or
   sudo tail -f /var/log/postgresql/postgresql-14-main.log
   ```

## Quick Setup Script

Save this as `setup_postgres.sh`:

```bash
#!/bin/bash

echo "Setting up PostgreSQL for MCP testing..."

# For WSL2/Ubuntu
if command -v apt &> /dev/null; then
    echo "Installing PostgreSQL..."
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
    
    echo "Starting PostgreSQL..."
    sudo service postgresql start
    
    echo "Creating database and user..."
    sudo -u postgres psql <<EOF
CREATE USER auto_tool_user WITH PASSWORD 'auto_tool_pass';
CREATE DATABASE auto_tool_disc OWNER auto_tool_user;
GRANT ALL PRIVILEGES ON DATABASE auto_tool_disc TO auto_tool_user;
EOF
    
    echo "PostgreSQL setup complete!"
    echo "Connection string: postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    
    # Test connection
    PGPASSWORD=auto_tool_pass psql -h localhost -U auto_tool_user -d auto_tool_disc -c "SELECT version();"
fi
```

Make it executable and run:
```bash
chmod +x setup_postgres.sh
./setup_postgres.sh
```

## Verifying Setup for MCP Tests

Once PostgreSQL is running:

```bash
# Set environment variables
export TEST_REAL_POSTGRES=1
export POSTGRES_TEST_CONNECTION="postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"

# Run the test script
python tests/scripts/test_postgres_real_server.py
```

## Docker Compose Alternative

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    container_name: postgres-mcp-test
    environment:
      POSTGRES_USER: auto_tool_user
      POSTGRES_PASSWORD: auto_tool_pass
      POSTGRES_DB: auto_tool_disc
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U auto_tool_user -d auto_tool_disc"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

Run with:
```bash
docker-compose up -d
```