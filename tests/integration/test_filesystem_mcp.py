#!/usr/bin/env python3
"""
Test script for Filesystem MCP

This script comprehensively tests the filesystem MCP implementation,
including both real and mock servers.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from tests/integration/)
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.tools.filesystem_mcp import FileSystemMCPClient
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry

logger = get_logger(__name__)


async def test_direct_filesystem_client():
    """Test the filesystem MCP client directly."""
    logger.info("=" * 60)
    logger.info("[TEST 1] Direct Filesystem MCP Client Test")
    logger.info("=" * 60)
    
    # Create test directory
    test_dir = Path("data/test_fs_direct")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = FileSystemMCPClient(str(test_dir))
    
    try:
        # Try to connect to real server first
        logger.info("\n[CONNECTING] Attempting to connect to real Filesystem MCP server...")
        connected = await client.connect()
        
        if not connected:
            logger.warning("⚠️  Could not connect to real server, trying mock...")
            connected = await client.connect(use_mock=True)
            
            if not connected:
                logger.error("[ERROR] Could not connect to filesystem server")
                return False
        
        mode = "mock" if client.use_mock else "real"
        logger.info(f"✅ Connected to {mode} filesystem MCP server")
        
        # Test 1: Write a file
        logger.info("\n[TEST] Writing file...")
        write_result = await client.write_file(
            "hello.txt",
            "Hello, World!\nThis is a test file from filesystem MCP."
        )
        logger.info(f"Write result: {write_result}")
        
        # Test 2: Read the file
        logger.info("\n[TEST] Reading file...")
        read_result = await client.read_file("hello.txt")
        logger.info(f"Read result: {read_result}")
        
        # Test 3: Create a directory
        logger.info("\n[TEST] Creating directory...")
        create_result = await client.create_directory("subdir")
        logger.info(f"Create directory result: {create_result}")
        
        # Test 4: Write file in subdirectory
        logger.info("\n[TEST] Writing file in subdirectory...")
        sub_write_result = await client.write_file(
            "subdir/nested.txt",
            "This file is in a subdirectory."
        )
        logger.info(f"Subdirectory write result: {sub_write_result}")
        
        # Test 5: List directory contents
        logger.info("\n[TEST] Listing directory...")
        list_result = await client.list_directory(".")
        logger.info(f"Directory listing: {list_result}")
        
        # Test 6: Check file existence
        logger.info("\n[TEST] Checking file existence...")
        exists_result = await client.file_exists("hello.txt")
        logger.info(f"hello.txt exists: {exists_result}")
        
        not_exists_result = await client.file_exists("nonexistent.txt")
        logger.info(f"nonexistent.txt exists: {not_exists_result}")
        
        # Test 7: Error handling - read non-existent file
        logger.info("\n[TEST] Testing error handling...")
        error_result = await client.read_file("does_not_exist.txt")
        logger.info(f"Error test result: {error_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await client.disconnect()
        logger.info("[CLEANUP] Disconnected from filesystem MCP server")


async def test_mcp_integration():
    """Test filesystem MCP through the integration layer."""
    logger.info("\n" + "=" * 60)
    logger.info("[TEST 2] Filesystem MCP Integration Test")
    logger.info("=" * 60)
    
    # Create registry and integration
    registry = ToolRegistry("data/test_fs_integration_registry.db")
    integration = MCPIntegration(registry)
    
    try:
        # Add filesystem server
        test_dir = "data/test_fs_integration"
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n[SETUP] Adding filesystem server with base path: {test_dir}")
        success = await integration.add_filesystem_server(
            base_path=test_dir,
            server_id="test_fs"
        )
        
        if not success:
            logger.warning("⚠️  Could not add real server, trying mock...")
            success = await integration.add_filesystem_server(
                base_path=test_dir,
                server_id="test_fs",
                use_mock=True
            )
            
            if not success:
                logger.error("[ERROR] Could not add filesystem server")
                return False
        
        # Get server status
        status = integration.get_server_status()
        logger.info(f"\n[STATUS] Server status: {status}")
        
        # Discover tools
        tools = await integration.discover_all_tools()
        fs_tools = [t for t in tools if t['id'].startswith('filesystem.')]
        logger.info(f"\n[TOOLS] Discovered {len(fs_tools)} filesystem tools:")
        for tool in fs_tools:
            logger.info(f"  - {tool['id']}: {tool.get('description', 'No description')}")
        
        # Test tool execution through integration
        logger.info("\n[EXECUTE] Testing tool execution through integration...")
        
        # Write a JSON file
        json_content = '''{"name": "Test Config", "version": "1.0", "settings": {"debug": true}}'''
        write_result = await integration.execute_tool(
            "filesystem.write_file",
            {
                "path": "config.json",
                "content": json_content
            }
        )
        logger.info(f"Write JSON result: {write_result}")
        
        # Read the JSON file
        read_result = await integration.execute_tool(
            "filesystem.read_file",
            {
                "path": "config.json"
            }
        )
        logger.info(f"Read JSON result: {read_result}")
        
        # Create a log directory
        if "filesystem.create_directory" in [t['id'] for t in fs_tools]:
            create_result = await integration.execute_tool(
                "filesystem.create_directory",
                {
                    "path": "logs"
                }
            )
            logger.info(f"Create logs directory result: {create_result}")
        
        # Write a log file
        log_content = "2024-01-15 10:00:00 - Application started\n2024-01-15 10:00:01 - Connected to database"
        log_result = await integration.execute_tool(
            "filesystem.write_file",
            {
                "path": "logs/app.log",
                "content": log_content
            }
        )
        logger.info(f"Write log file result: {log_result}")
        
        # List the root directory
        list_result = await integration.execute_tool(
            "filesystem.list_directory",
            {
                "path": "."
            }
        )
        logger.info(f"Root directory listing: {list_result}")
        
        # Check tool performance
        logger.info("\n[PERFORMANCE] Checking tool performance metrics...")
        for tool in fs_tools[:3]:  # Check first 3 tools
            perf = registry.get_tool_performance(tool['id'])
            logger.info(f"  {tool['id']}: {perf}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await integration.shutdown_all()
        logger.info("[CLEANUP] Shut down all MCP servers")


async def test_combined_operations():
    """Test filesystem MCP working with other tools."""
    logger.info("\n" + "=" * 60)
    logger.info("[TEST 3] Combined Operations Test")
    logger.info("=" * 60)
    
    # Create registry and integration
    registry = ToolRegistry("data/test_combined_registry.db")
    integration = MCPIntegration(registry)
    
    try:
        # Add filesystem server
        fs_dir = "data/test_combined"
        Path(fs_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("\n[SETUP] Adding filesystem server...")
        fs_success = await integration.add_filesystem_server(
            base_path=fs_dir,
            server_id="combined_fs",
            use_mock=True  # Use mock for predictable testing
        )
        
        if not fs_success:
            logger.error("[ERROR] Could not add filesystem server")
            return False
        
        # Add SQLite server
        logger.info("[SETUP] Adding SQLite server...")
        db_path = f"{fs_dir}/test.db"
        sqlite_success = await integration.add_sqlite_server(
            db_path=db_path,
            server_id="combined_sqlite",
            use_mock=True
        )
        
        if not sqlite_success:
            logger.error("[ERROR] Could not add SQLite server")
            return False
        
        # Combined operation: Create database schema and save to file
        logger.info("\n[COMBINED] Creating database schema and saving to file...")
        
        # Create table
        create_table_result = await integration.execute_tool(
            "sqlite.query",
            {
                "sql": """
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    price REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            }
        )
        logger.info(f"Create table result: {create_table_result}")
        
        # Insert sample data
        insert_result = await integration.execute_tool(
            "sqlite.query",
            {
                "sql": "INSERT INTO products (name, price) VALUES (?, ?), (?, ?)",
                "params": ["Widget A", 19.99, "Widget B", 29.99]
            }
        )
        logger.info(f"Insert data result: {insert_result}")
        
        # Query data
        query_result = await integration.execute_tool(
            "sqlite.query",
            {
                "sql": "SELECT * FROM products ORDER BY price"
            }
        )
        logger.info(f"Query result: {query_result}")
        
        # Save query results to file
        if query_result.get("success"):
            # Format results as CSV
            rows = query_result['result'].get('rows', [])
            columns = query_result['result'].get('columns', [])
            
            csv_content = ",".join(columns) + "\n"
            for row in rows:
                csv_content += ",".join(str(val) for val in row) + "\n"
            
            # Write to file
            save_result = await integration.execute_tool(
                "filesystem.write_file",
                {
                    "path": "products.csv",
                    "content": csv_content
                }
            )
            logger.info(f"Save to CSV result: {save_result}")
            
            # Read back the CSV file
            read_csv_result = await integration.execute_tool(
                "filesystem.read_file",
                {
                    "path": "products.csv"
                }
            )
            logger.info(f"Read CSV result: {read_csv_result}")
        
        # Get final status
        final_status = integration.get_server_status()
        logger.info(f"\n[FINAL STATUS] {final_status}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Combined test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await integration.shutdown_all()
        logger.info("[CLEANUP] Shut down all servers")


async def main():
    """Run all filesystem MCP tests."""
    logger.info("🚀 Starting Filesystem MCP Test Suite")
    logger.info("=" * 70)
    
    results = []
    
    # Test 1: Direct client test
    result1 = await test_direct_filesystem_client()
    results.append(("Direct Client Test", result1))
    
    # Test 2: Integration test
    result2 = await test_mcp_integration()
    results.append(("Integration Test", result2))
    
    # Test 3: Combined operations test
    result3 = await test_combined_operations()
    results.append(("Combined Operations Test", result3))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("\n🎉 All filesystem MCP tests passed!")
    else:
        logger.error(f"\n⚠️  {len(results) - passed} tests failed")


if __name__ == "__main__":
    asyncio.run(main())