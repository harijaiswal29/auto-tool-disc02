#!/bin/bash
# Script to run different test categories

echo "Auto Tool Discovery - Test Runner"
echo "================================="
echo ""

case "$1" in
    "unit")
        echo "Running unit tests..."
        python -m pytest tests/unit/ -v
        ;;
    "integration")
        echo "Running integration tests..."
        python -m pytest tests/integration/ -v
        ;;
    "quick")
        echo "Running quick unit tests..."
        python -m pytest tests/unit/ -v -x --tb=short
        ;;
    "coverage")
        echo "Running tests with coverage..."
        python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
        ;;
    "search")
        echo "Running search-related tests..."
        python -m pytest tests/ -k "search" -v
        ;;
    "sqlite")
        echo "Running SQLite tests..."
        python -m pytest tests/integration/ -k "sqlite" -v
        ;;
    "all")
        echo "Running all tests..."
        python -m pytest tests/ -v
        ;;
    *)
        echo "Usage: $0 {unit|integration|quick|coverage|search|sqlite|all}"
        echo ""
        echo "Options:"
        echo "  unit         - Run only unit tests"
        echo "  integration  - Run only integration tests"
        echo "  quick        - Run unit tests, stop on first failure"
        echo "  coverage     - Run all tests with coverage report"
        echo "  search       - Run tests related to search functionality"
        echo "  sqlite       - Run SQLite-related tests"
        echo "  all          - Run all tests"
        exit 1
        ;;
esac