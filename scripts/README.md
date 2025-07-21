# Scripts Directory

This directory contains utility and automation scripts for the Auto Tool Discovery project.

## Contents

- `run_tests.sh` - Test runner script with various options (unit, integration, coverage, etc.)
- `setup_and_test.sh` - Setup script that creates virtual environment and runs tests

## Usage

### Run Tests
```bash
./scripts/run_tests.sh unit         # Run unit tests
./scripts/run_tests.sh integration  # Run integration tests
./scripts/run_tests.sh coverage     # Run tests with coverage report
./scripts/run_tests.sh all          # Run all tests
```

### Setup and Test
```bash
./scripts/setup_and_test.sh         # Setup virtual environment
./scripts/setup_and_test.sh test    # Setup and run tests
```

Make sure to give execute permissions if needed:
```bash
chmod +x scripts/*.sh
```