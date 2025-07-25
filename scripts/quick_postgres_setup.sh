#!/bin/bash

# Quick PostgreSQL Setup for MCP Testing
# Automatically chooses the best available method

set -e

echo "======================================"
echo "Quick PostgreSQL Setup for MCP Testing"
echo "======================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
export DB_USER="auto_tool_user"
export DB_PASS="auto_tool_pass"
export DB_NAME="auto_tool_disc"
export DB_PORT="5432"
CONNECTION_STRING="postgresql://$DB_USER:$DB_PASS@localhost:$DB_PORT/$DB_NAME"

# Check if Docker is available
if command -v docker >/dev/null 2>&1; then
    echo -e "${BLUE}Docker is available. Using Docker for PostgreSQL setup.${NC}"
    
    # Check if container already exists
    if docker ps -a | grep -q postgres-mcp-test; then
        echo -e "${YELLOW}Container 'postgres-mcp-test' already exists.${NC}"
        
        # Start if not running
        if ! docker ps | grep -q postgres-mcp-test; then
            echo "Starting existing container..."
            docker start postgres-mcp-test
            sleep 3
        fi
    else
        # Run new container
        echo "Creating new PostgreSQL container..."
        docker run -d \
            --name postgres-mcp-test \
            -e POSTGRES_USER=$DB_USER \
            -e POSTGRES_PASSWORD=$DB_PASS \
            -e POSTGRES_DB=$DB_NAME \
            -p $DB_PORT:5432 \
            postgres:14-alpine
        
        echo "Waiting for PostgreSQL to start..."
        sleep 5
    fi
    
    # Test connection
    echo -e "\n${YELLOW}Testing connection...${NC}"
    if docker exec postgres-mcp-test pg_isready -U $DB_USER >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PostgreSQL is ready!${NC}"
        
        # Create sample tables
        echo "Creating sample tables..."
        docker exec postgres-mcp-test psql -U $DB_USER -d $DB_NAME -c "
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        " >/dev/null 2>&1 || true
        
        METHOD="Docker"
    else
        echo -e "${RED}✗ Failed to connect to PostgreSQL container${NC}"
        exit 1
    fi
    
else
    echo -e "${BLUE}Docker not found. Using native PostgreSQL installation.${NC}"
    
    # Check if PostgreSQL is installed
    if ! command -v psql >/dev/null 2>&1; then
        echo -e "${YELLOW}Installing PostgreSQL...${NC}"
        sudo apt update
        sudo apt install -y postgresql postgresql-contrib
    fi
    
    # Start PostgreSQL
    echo "Starting PostgreSQL service..."
    sudo service postgresql start
    
    # Wait a moment
    sleep 2
    
    # Setup database
    echo "Setting up database..."
    sudo -u postgres psql <<EOF >/dev/null 2>&1 || true
CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';
CREATE DATABASE $DB_NAME OWNER $DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF
    
    METHOD="Native"
fi

# Final connection test
echo -e "\n${YELLOW}Final connection test...${NC}"
export PGPASSWORD=$DB_PASS

if psql -h localhost -U $DB_USER -d $DB_NAME -c "SELECT 'Connection successful!' as status;" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Connection test passed!${NC}"
    
    # Get PostgreSQL version
    VERSION=$(psql -h localhost -U $DB_USER -d $DB_NAME -t -c "SELECT version();" | head -1 | awk '{print $2}')
    
    echo -e "\n${GREEN}======================================"
    echo -e "PostgreSQL Setup Complete!"
    echo -e "======================================${NC}"
    echo -e "Method: ${BLUE}$METHOD${NC}"
    echo -e "Version: ${BLUE}PostgreSQL $VERSION${NC}"
    echo -e "\nConnection String:"
    echo -e "${GREEN}$CONNECTION_STRING${NC}"
    echo -e "\nTo run MCP tests:"
    echo -e "${YELLOW}export TEST_REAL_POSTGRES=1"
    echo -e "export POSTGRES_TEST_CONNECTION=\"$CONNECTION_STRING\""
    echo -e "python tests/scripts/test_postgres_real_server.py${NC}"
    
    # If using Docker, show management commands
    if [ "$METHOD" = "Docker" ]; then
        echo -e "\n${BLUE}Docker Commands:${NC}"
        echo -e "Stop:    docker stop postgres-mcp-test"
        echo -e "Start:   docker start postgres-mcp-test"
        echo -e "Remove:  docker rm -f postgres-mcp-test"
        echo -e "Logs:    docker logs postgres-mcp-test"
    fi
else
    echo -e "${RED}✗ Connection test failed!${NC}"
    echo -e "\nTroubleshooting:"
    
    if [ "$METHOD" = "Docker" ]; then
        echo -e "1. Check Docker logs: docker logs postgres-mcp-test"
        echo -e "2. Check if port 5432 is already in use: sudo lsof -i :5432"
    else
        echo -e "1. Check PostgreSQL status: sudo service postgresql status"
        echo -e "2. Check logs: sudo tail -f /var/log/postgresql/*.log"
    fi
    
    exit 1
fi