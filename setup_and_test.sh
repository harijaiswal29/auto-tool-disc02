#!/bin/bash
# Setup and Test Script for Auto Tool Discovery Project

echo "=========================================="
echo "Auto Tool Discovery - Setup and Test"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if activation was successful
if [ "$VIRTUAL_ENV" != "" ]; then
    echo -e "${GREEN}✓ Virtual environment activated: $VIRTUAL_ENV${NC}"
else
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    exit 1
fi

# Install/update requirements
echo -e "\n${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Requirements installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install requirements${NC}"
    exit 1
fi

# Run tests based on argument
if [ "$1" == "test" ]; then
    echo -e "\n${YELLOW}Running Filesystem MCP E2E Tests...${NC}"
    echo "=========================================="
    
    # Run standalone test first (doesn't need all dependencies)
    echo -e "\n${GREEN}1. Running Standalone Tests${NC}"
    python tests/e2e/test_filesystem_standalone.py
    
    # Run simple E2E tests
    echo -e "\n${GREEN}2. Running Simple E2E Tests${NC}"
    python tests/e2e/run_e2e_tests.py --type simple
    
elif [ "$1" == "standalone" ]; then
    echo -e "\n${YELLOW}Running Standalone Tests Only...${NC}"
    python tests/e2e/test_filesystem_standalone.py
    
else
    echo -e "\n${GREEN}Setup complete!${NC}"
    echo ""
    echo "To run tests, use:"
    echo "  ./setup_and_test.sh test       # Run all tests"
    echo "  ./setup_and_test.sh standalone # Run standalone tests only"
    echo ""
    echo "Or manually:"
    echo "  source .venv/bin/activate"
    echo "  python tests/e2e/run_e2e_tests.py --type simple"
fi

echo -e "\n${GREEN}Note: Virtual environment is still active.${NC}"
echo "To deactivate, run: deactivate"