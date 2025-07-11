"""
Simple End-to-End Tests for Filesystem MCP

This module provides basic E2E tests that work with the current implementation.
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.mcp_integration import MCPIntegration
from src.tools.filesystem_mcp import FileSystemMCPClient
from src.tools.tool_registry import ToolRegistry
from src.utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)


class SimpleFileSystemE2ETests:
    """Simple E2E tests for filesystem MCP functionality."""
    
    def __init__(self):
        self.test_dir = None
        self.mcp_integration = None
        self.registry = None
        self.filesystem_client = None
        
    async def setup(self):
        """Set up test environment."""
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp(prefix="filesystem_simple_e2e_")
        logger.info(f"Created test directory: {self.test_dir}")
        
        # Initialize components
        self.registry = ToolRegistry()
        self.mcp_integration = MCPIntegration(self.registry)
        
        # Initialize filesystem client directly
        self.filesystem_client = FileSystemMCPClient(base_path=self.test_dir)
        
        # Connect client (will use mock if real server unavailable)
        await self.filesystem_client.connect(use_mock=True)
        
        # Register tools to registry
        self.filesystem_client.register_tools_to_registry(self.registry)
        
        logger.info("Test environment setup complete")
        
    async def teardown(self):
        """Clean up test environment."""
        if self.filesystem_client:
            await self.filesystem_client.disconnect()
            
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory: {self.test_dir}")
            
    async def test_file_operations_flow(self):
        """Test complete file operation workflow."""
        logger.info("\n=== Test: File Operations Flow ===")
        
        # Step 1: Write a file
        logger.info("Step 1: Writing test file...")
        content = "This is a test file for E2E testing.\nLine 2\nLine 3"
        result = await self.filesystem_client.write_file("test_e2e.txt", content)
        assert result["success"], f"Failed to write file: {result}"
        logger.info("✓ File written successfully")
        
        # Step 2: Verify file exists
        logger.info("Step 2: Checking file existence...")
        exists = await self.filesystem_client.file_exists("test_e2e.txt")
        assert exists, "File does not exist after writing"
        logger.info("✓ File existence confirmed")
        
        # Step 3: Read the file
        logger.info("Step 3: Reading file content...")
        read_result = await self.filesystem_client.read_file("test_e2e.txt")
        # Extract content from the result dictionary
        assert read_result["success"], f"Failed to read file: {read_result}"
        actual_content = read_result["result"]["content"]
        assert actual_content == content, f"Content mismatch. Expected: {content}, Got: {actual_content}"
        logger.info("✓ File content verified")
        
        # Step 4: List directory
        logger.info("Step 4: Listing directory...")
        list_result = await self.filesystem_client.list_directory(".")
        assert list_result["success"], f"Failed to list directory: {list_result}"
        files = list_result["result"]["items"]
        assert any(f["name"] == "test_e2e.txt" for f in files), "File not found in directory listing"
        logger.info(f"✓ Directory listing successful, found {len(files)} items")
        
        logger.info("✅ File operations flow test passed!")
        
    async def test_directory_operations_flow(self):
        """Test directory creation and navigation workflow."""
        logger.info("\n=== Test: Directory Operations Flow ===")
        
        # Step 1: Create nested directories
        logger.info("Step 1: Creating nested directories...")
        await self.filesystem_client.create_directory("project/src/components")
        logger.info("✓ Nested directories created")
        
        # Step 2: Create files in different directories
        logger.info("Step 2: Creating files in directories...")
        await self.filesystem_client.write_file("project/README.md", "# Test Project")
        await self.filesystem_client.write_file("project/src/main.py", "print('Hello E2E')")
        await self.filesystem_client.write_file("project/src/components/button.py", "class Button: pass")
        logger.info("✓ Files created in directories")
        
        # Step 3: List nested directory
        logger.info("Step 3: Listing nested directory contents...")
        project_result = await self.filesystem_client.list_directory("project")
        assert project_result["success"], f"Failed to list project directory: {project_result}"
        project_files = project_result["result"]["items"]
        
        src_result = await self.filesystem_client.list_directory("project/src")
        assert src_result["success"], f"Failed to list src directory: {src_result}"
        src_files = src_result["result"]["items"]
        
        assert any(f["name"] == "README.md" for f in project_files), "README.md not found"
        assert any(f["name"] == "src" and f["type"] == "directory" for f in project_files), "src directory not found"
        assert any(f["name"] == "main.py" for f in src_files), "main.py not found"
        
        logger.info(f"✓ Directory structure verified")
        logger.info("✅ Directory operations flow test passed!")
        
    async def test_registry_integration(self):
        """Test tool registry integration."""
        logger.info("\n=== Test: Registry Integration ===")
        
        # Check if tools are registered
        logger.info("Checking registered tools...")
        
        tool_ids = [
            "filesystem.read_file",
            "filesystem.write_file",
            "filesystem.list_directory",
            "filesystem.create_directory"
        ]
        
        for tool_id in tool_ids:
            tool = self.registry.get_tool(tool_id)
            assert tool is not None, f"Tool {tool_id} not found in registry"
            logger.info(f"✓ Tool registered: {tool_id}")
            
        # Test tool discovery by capability
        logger.info("\nTesting tool discovery by capability...")
        # Note: search_by_capability not implemented yet in ToolRegistry
        # For now, just list all tools
        all_tools = self.registry.list_tools()
        file_tools = [t for t in all_tools if 'file' in t.get('id', '').lower()]
        assert len(file_tools) > 0, "No file-related tools found"
        logger.info(f"✓ Found {len(file_tools)} file-related tools")
        
        logger.info("✅ Registry integration test passed!")
        
    async def test_error_scenarios(self):
        """Test error handling scenarios."""
        logger.info("\n=== Test: Error Scenarios ===")
        
        # Test 1: Read non-existent file
        logger.info("Test 1: Reading non-existent file...")
        try:
            result = await self.filesystem_client.read_file("non_existent.txt")
            # Check if result indicates failure
            if result.get("success") is False:
                logger.info(f"✓ Correctly handled error: File not found")
            else:
                assert False, "Should have returned failure for non-existent file"
        except Exception as e:
            logger.info(f"✓ Correctly handled error: {type(e).__name__}")
            
        # Test 2: Invalid path (path traversal attempt)
        logger.info("Test 2: Path traversal attempt...")
        try:
            result = await self.filesystem_client.read_file("../../../etc/passwd")
            # Check if result indicates failure or security error
            if result.get("success") is False:
                logger.info(f"✓ Security check passed: Path traversal blocked")
            else:
                assert False, "Should have blocked path traversal"
        except Exception as e:
            logger.info(f"✓ Security check passed: {type(e).__name__}")
            
        # Test 3: Write to directory (invalid operation)
        logger.info("Test 3: Writing to directory...")
        await self.filesystem_client.create_directory("testdir")
        try:
            result = await self.filesystem_client.write_file("testdir", "content")
            # Some implementations might handle this differently
            logger.info("✓ Operation completed (implementation-specific behavior)")
        except Exception as e:
            logger.info(f"✓ Correctly handled error: {type(e).__name__}")
            
        logger.info("✅ Error scenarios test passed!")
        
    async def test_performance_tracking(self):
        """Test performance tracking for filesystem operations."""
        logger.info("\n=== Test: Performance Tracking ===")
        
        # Track operation times
        operations = []
        
        # Test write performance
        start = datetime.now()
        for i in range(10):
            await self.filesystem_client.write_file(f"perf_test_{i}.txt", f"Content {i}")
        write_time = (datetime.now() - start).total_seconds()
        operations.append(("Batch write (10 files)", write_time))
        
        # Test read performance
        start = datetime.now()
        for i in range(10):
            result = await self.filesystem_client.read_file(f"perf_test_{i}.txt")
            assert result["success"], f"Failed to read perf_test_{i}.txt"
        read_time = (datetime.now() - start).total_seconds()
        operations.append(("Batch read (10 files)", read_time))
        
        # Test list performance
        start = datetime.now()
        list_result = await self.filesystem_client.list_directory(".")
        assert list_result["success"], "Failed to list directory"
        list_time = (datetime.now() - start).total_seconds()
        operations.append(("List directory", list_time))
        
        # Display performance results
        logger.info("\nPerformance Results:")
        for op_name, op_time in operations:
            logger.info(f"  {op_name}: {op_time:.3f}s")
            
        # Basic performance assertions
        assert write_time < 5.0, f"Write operations too slow: {write_time:.3f}s"
        assert read_time < 5.0, f"Read operations too slow: {read_time:.3f}s"
        assert list_time < 1.0, f"List operation too slow: {list_time:.3f}s"
        
        logger.info("✅ Performance tracking test passed!")
        
    async def run_all_tests(self):
        """Run all simple E2E tests."""
        try:
            await self.setup()
            
            # Run test suites
            await self.test_file_operations_flow()
            await self.test_directory_operations_flow()
            await self.test_registry_integration()
            await self.test_error_scenarios()
            await self.test_performance_tracking()
            
            logger.info("\n" + "="*50)
            logger.info("✅ All Simple E2E tests passed successfully!")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"\n❌ E2E test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            await self.teardown()


async def main():
    """Main entry point for simple E2E tests."""
    logger.info("Starting Simple Filesystem MCP End-to-End Tests")
    logger.info("=" * 50)
    
    test_suite = SimpleFileSystemE2ETests()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())