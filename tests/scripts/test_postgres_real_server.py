#!/usr/bin/env python3
"""
PostgreSQL Real Server Test Runner

This script runs PostgreSQL MCP tests using only the real server,
skipping all mock server tests.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_prerequisites():
    """Check if all prerequisites for real server testing are met."""
    print("\n=== Checking Prerequisites for Real PostgreSQL MCP Server Testing ===\n")
    
    issues = []
    
    # Check for PostgreSQL MCP server binary
    mcp_server_path = Path("node_modules/.bin/mcp-server-postgres")
    if mcp_server_path.exists():
        print("✅ PostgreSQL MCP server binary found")
    else:
        issues.append("❌ PostgreSQL MCP server binary not found at node_modules/.bin/mcp-server-postgres")
        print(issues[-1])
    
    # Check for PostgreSQL connection
    connection_string = os.environ.get(
        "POSTGRES_TEST_CONNECTION",
        "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    )
    print(f"\n📋 PostgreSQL connection string: {connection_string}")
    
    # Try to connect to PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(connection_string)
        conn.close()
        print("✅ PostgreSQL database connection successful")
    except ImportError:
        print("⚠️  psycopg2 not installed - cannot verify database connection")
        print("   Install with: pip install psycopg2-binary")
    except Exception as e:
        issues.append(f"❌ PostgreSQL database connection failed: {e}")
        print(issues[-1])
    
    # Check Node.js availability (required for MCP server)
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js available: {result.stdout.strip()}")
        else:
            issues.append("❌ Node.js not available")
            print(issues[-1])
    except FileNotFoundError:
        issues.append("❌ Node.js not found in PATH")
        print(issues[-1])
    
    return len(issues) == 0, issues


def run_real_server_tests():
    """Run PostgreSQL MCP tests with real server only."""
    print("\n=== Running PostgreSQL MCP Real Server Tests ===\n")
    
    # Set environment variables
    os.environ["TEST_REAL_POSTGRES"] = "1"
    if "POSTGRES_TEST_CONNECTION" not in os.environ:
        os.environ["POSTGRES_TEST_CONNECTION"] = "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    
    # Define test commands
    test_commands = [
        # Integration tests - real server tests only
        [
            "pytest", 
            "tests/integration/test_postgres_mcp.py::TestPostgresMCPClient::test_real_server_connection",
            "tests/integration/test_postgres_mcp.py::TestPostgresMCPClient::test_real_execute_query",
            "tests/integration/test_postgres_mcp.py::TestPostgresMCPClient::test_real_list_tables_via_query",
            "-v", "-s"
        ],
        
        # Run demo with real server
        ["python", "tests/demos/demo_postgres_mcp.py"]
    ]
    
    results = []
    
    for i, cmd in enumerate(test_commands):
        print(f"\n📌 Running command {i+1}/{len(test_commands)}: {' '.join(cmd)}")
        print("-" * 80)
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                print(f"\n✅ Command {i+1} passed")
                results.append((cmd, True, None))
            else:
                print(f"\n❌ Command {i+1} failed with return code: {result.returncode}")
                results.append((cmd, False, f"Return code: {result.returncode}"))
        except Exception as e:
            print(f"\n❌ Command {i+1} failed with error: {e}")
            results.append((cmd, False, str(e)))
    
    return results


def print_summary(results):
    """Print test execution summary."""
    print("\n" + "=" * 80)
    print("                        TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"\nTotal test commands: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed commands:")
        for cmd, success, error in results:
            if not success:
                print(f"  - {' '.join(cmd)}")
                print(f"    Error: {error}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    print("🐘 PostgreSQL MCP Real Server Test Runner")
    print("=" * 80)
    
    # Check prerequisites
    ready, issues = check_prerequisites()
    
    if not ready:
        print("\n❌ Prerequisites not met. Please fix the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSetup instructions:")
        print("1. Ensure PostgreSQL is running and accessible")
        print("2. Install PostgreSQL MCP server: npm install @modelcontextprotocol/server-postgres")
        print("3. Create database and user if needed")
        print("4. Set POSTGRES_TEST_CONNECTION environment variable if using different connection")
        return 1
    
    # Run tests
    results = run_real_server_tests()
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on results
    return 0 if all(success for _, success, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())