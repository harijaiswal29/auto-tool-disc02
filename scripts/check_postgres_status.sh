#!/bin/bash

# Check PostgreSQL Status and Provide Setup Instructions

echo "========================================"
echo "PostgreSQL Status Check"
echo "========================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DB_USER="auto_tool_user"
DB_PASS="auto_tool_pass"
DB_NAME="auto_tool_disc"
DB_PORT="5432"
CONNECTION_STRING="postgresql://$DB_USER:$DB_PASS@localhost:$DB_PORT/$DB_NAME"

echo -e "\n${BLUE}1. Checking PostgreSQL Installation${NC}"
echo "----------------------------------------"

# Check for PostgreSQL binaries
if command -v psql >/dev/null 2>&1; then
    echo -e "${GREEN}✓ psql command found at: $(which psql)${NC}"
    PSQL_VERSION=$(psql --version 2>/dev/null || echo "unknown")
    echo -e "  Version: $PSQL_VERSION"
else
    echo -e "${RED}✗ psql command not found${NC}"
fi

if command -v postgres >/dev/null 2>&1; then
    echo -e "${GREEN}✓ postgres binary found at: $(which postgres)${NC}"
else
    echo -e "${RED}✗ postgres binary not found${NC}"
fi

echo -e "\n${BLUE}2. Checking PostgreSQL Service${NC}"
echo "----------------------------------------"

# Check service status (try different methods)
if systemctl list-units --type=service 2>/dev/null | grep -q postgres; then
    echo -e "${GREEN}✓ PostgreSQL service found${NC}"
    systemctl status postgresql --no-pager 2>/dev/null || true
elif service --status-all 2>/dev/null | grep -q postgres; then
    echo -e "${GREEN}✓ PostgreSQL service found${NC}"
    service postgresql status 2>/dev/null || true
else
    echo -e "${RED}✗ PostgreSQL service not found${NC}"
fi

echo -e "\n${BLUE}3. Checking Port 5432${NC}"
echo "----------------------------------------"

# Check if port is in use
if ss -tln 2>/dev/null | grep -q :5432; then
    echo -e "${GREEN}✓ Port 5432 is in use${NC}"
    ss -tlnp 2>/dev/null | grep 5432 || ss -tln | grep 5432
elif lsof -i :5432 2>/dev/null; then
    echo -e "${GREEN}✓ Port 5432 is in use${NC}"
else
    echo -e "${RED}✗ Port 5432 is not in use${NC}"
fi

echo -e "\n${BLUE}4. Testing Connection${NC}"
echo "----------------------------------------"

# Try to connect
export PGPASSWORD=$DB_PASS
if psql -h localhost -U $DB_USER -d $DB_NAME -c "SELECT 1;" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Successfully connected to PostgreSQL!${NC}"
    echo -e "  Database: $DB_NAME"
    echo -e "  User: $DB_USER"
else
    echo -e "${RED}✗ Cannot connect to PostgreSQL${NC}"
    
    # Try default postgres user
    if psql -h localhost -U postgres -c "SELECT 1;" >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Can connect as 'postgres' user but not as '$DB_USER'${NC}"
    fi
fi

echo -e "\n${BLUE}5. Installation Instructions${NC}"
echo "========================================"

if ! command -v psql >/dev/null 2>&1; then
    echo -e "${YELLOW}PostgreSQL is not installed. To install:${NC}"
    echo ""
    echo "Option 1: Native Installation (requires sudo)"
    echo "---------------------------------------------"
    echo "sudo apt update"
    echo "sudo apt install postgresql postgresql-contrib"
    echo "sudo service postgresql start"
    echo ""
    echo "Then create the database and user:"
    echo "sudo -u postgres psql"
    echo "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
    echo "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
    echo "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    echo "\q"
    echo ""
    echo "Option 2: Docker (if available)"
    echo "--------------------------------"
    echo "docker run -d \\"
    echo "  --name postgres-mcp-test \\"
    echo "  -e POSTGRES_USER=$DB_USER \\"
    echo "  -e POSTGRES_PASSWORD=$DB_PASS \\"
    echo "  -e POSTGRES_DB=$DB_NAME \\"
    echo "  -p 5432:5432 \\"
    echo "  postgres:14-alpine"
else
    echo -e "${YELLOW}PostgreSQL is installed but may not be running.${NC}"
    echo ""
    echo "To start PostgreSQL:"
    echo "-------------------"
    echo "sudo service postgresql start"
    echo ""
    echo "To create the test database (if needed):"
    echo "---------------------------------------"
    echo "sudo -u postgres psql <<EOF"
    echo "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
    echo "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
    echo "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    echo "EOF"
fi

echo -e "\n${BLUE}6. Environment Variables for Testing${NC}"
echo "========================================"
echo "export TEST_REAL_POSTGRES=1"
echo "export POSTGRES_TEST_CONNECTION=\"$CONNECTION_STRING\""
echo ""
echo "Then run: python tests/scripts/test_postgres_real_server.py"