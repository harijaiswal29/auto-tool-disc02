#!/bin/bash

# PostgreSQL Setup Script for MCP Testing
# This script sets up PostgreSQL with the required database and user

set -e  # Exit on error

echo "========================================="
echo "PostgreSQL Setup for MCP Testing"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DB_USER="auto_tool_user"
DB_PASS="auto_tool_pass"
DB_NAME="auto_tool_disc"
DB_PORT="5432"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if PostgreSQL is installed
check_postgres_installed() {
    if command_exists psql && command_exists postgres; then
        echo -e "${GREEN}✓ PostgreSQL is installed${NC}"
        return 0
    else
        echo -e "${RED}✗ PostgreSQL is not installed${NC}"
        return 1
    fi
}

# Function to check if PostgreSQL is running
check_postgres_running() {
    if sudo service postgresql status >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PostgreSQL is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ PostgreSQL is not running${NC}"
        return 1
    fi
}

# Function to check if port 5432 is available
check_port() {
    if sudo lsof -i :5432 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Port 5432 is active${NC}"
        return 0
    else
        echo -e "${RED}✗ Port 5432 is not active${NC}"
        return 1
    fi
}

# Function to create database and user
setup_database() {
    echo -e "\n${YELLOW}Setting up database and user...${NC}"
    
    # Check if database already exists
    if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        echo -e "${YELLOW}Database '$DB_NAME' already exists${NC}"
    else
        # Create user and database
        sudo -u postgres psql <<EOF
-- Create user if not exists
DO \$\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_user
      WHERE usename = '$DB_USER'
   ) THEN
      CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';
   END IF;
END
\$\$;

-- Create database
CREATE DATABASE $DB_NAME OWNER $DB_USER;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF
        echo -e "${GREEN}✓ Database and user created successfully${NC}"
    fi
}

# Function to test connection
test_connection() {
    echo -e "\n${YELLOW}Testing connection...${NC}"
    
    export PGPASSWORD=$DB_PASS
    if psql -h localhost -U $DB_USER -d $DB_NAME -c "SELECT version();" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Connection successful!${NC}"
        
        # Show version
        VERSION=$(psql -h localhost -U $DB_USER -d $DB_NAME -t -c "SELECT version();" | head -1)
        echo -e "${GREEN}PostgreSQL version: ${VERSION}${NC}"
        
        return 0
    else
        echo -e "${RED}✗ Connection failed${NC}"
        return 1
    fi
}

# Main script
echo -e "\n1. Checking PostgreSQL installation..."
if ! check_postgres_installed; then
    echo -e "${YELLOW}Installing PostgreSQL...${NC}"
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
fi

echo -e "\n2. Starting PostgreSQL service..."
if ! check_postgres_running; then
    sudo service postgresql start
    sleep 2
fi

echo -e "\n3. Checking port availability..."
check_port

echo -e "\n4. Setting up database..."
setup_database

echo -e "\n5. Testing connection..."
if test_connection; then
    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}PostgreSQL setup complete!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo -e "\nConnection details:"
    echo -e "  Host: localhost"
    echo -e "  Port: $DB_PORT"
    echo -e "  Database: $DB_NAME"
    echo -e "  User: $DB_USER"
    echo -e "  Password: $DB_PASS"
    echo -e "\nConnection string:"
    echo -e "  ${GREEN}postgresql://$DB_USER:$DB_PASS@localhost:$DB_PORT/$DB_NAME${NC}"
    echo -e "\nTo run MCP tests:"
    echo -e "  export TEST_REAL_POSTGRES=1"
    echo -e "  export POSTGRES_TEST_CONNECTION=\"postgresql://$DB_USER:$DB_PASS@localhost:$DB_PORT/$DB_NAME\""
    echo -e "  python tests/scripts/test_postgres_real_server.py"
else
    echo -e "\n${RED}Setup completed but connection test failed.${NC}"
    echo -e "Please check the PostgreSQL logs:"
    echo -e "  sudo tail -f /var/log/postgresql/*.log"
fi