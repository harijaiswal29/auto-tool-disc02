#!/usr/bin/env python3
"""
Runner script for Filesystem MCP End-to-End Tests

This script provides an easy way to run E2E tests with proper setup and reporting.
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def print_header(message):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {message}")
    print("="*60 + "\n")


async def run_simple_tests():
    """Run simple E2E tests."""
    from test_filesystem_simple_e2e import SimpleFileSystemE2ETests
    
    print_header("Running Simple Filesystem E2E Tests")
    test_suite = SimpleFileSystemE2ETests()
    
    try:
        await test_suite.run_all_tests()
        return True
    except Exception as e:
        print(f"\n❌ Simple E2E tests failed: {str(e)}")
        return False


async def run_comprehensive_tests():
    """Run comprehensive E2E tests (requires full agent implementation)."""
    try:
        from test_filesystem_e2e import FileSystemE2ETests
        
        print_header("Running Comprehensive Filesystem E2E Tests")
        test_suite = FileSystemE2ETests()
        
        await test_suite.run_all_tests()
        return True
    except ImportError as e:
        print(f"\n⚠️  Comprehensive tests require full agent implementation")
        print(f"   Missing components: {str(e)}")
        return None
    except Exception as e:
        print(f"\n❌ Comprehensive E2E tests failed: {str(e)}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Filesystem MCP E2E Tests")
    parser.add_argument(
        "--type",
        choices=["simple", "comprehensive", "all"],
        default="simple",
        help="Type of tests to run (default: simple)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print_header("Filesystem MCP End-to-End Test Runner")
    print(f"Test Type: {args.type}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {"simple": None, "comprehensive": None}
    
    if args.type in ["simple", "all"]:
        results["simple"] = await run_simple_tests()
        
    if args.type in ["comprehensive", "all"]:
        results["comprehensive"] = await run_comprehensive_tests()
    
    # Print summary
    print_header("Test Summary")
    
    for test_type, result in results.items():
        if result is None:
            continue
        elif result:
            print(f"✅ {test_type.capitalize()} Tests: PASSED")
        else:
            print(f"❌ {test_type.capitalize()} Tests: FAILED")
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    if any(r is False for r in results.values()):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())