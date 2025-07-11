"""
End-to-End Tests for Filesystem MCP Integration

This module tests the complete workflow from user query to filesystem operations
through the autonomous tool discovery system.
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
from src.tools.tool_registry import ToolRegistry
from src.agents.intent_recognition import IntentRecognitionAgent
from src.agents.tool_discovery import ToolDiscoveryAgent
from src.agents.orchestrator import OrchestratorAgent
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


class FileSystemE2ETests:
    """End-to-end tests for filesystem MCP functionality."""
    
    def __init__(self):
        self.test_dir = None
        self.mcp_integration = None
        self.registry = None
        self.intent_agent = None
        self.discovery_agent = None
        self.orchestrator = None
        
    async def setup(self):
        """Set up test environment."""
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp(prefix="filesystem_e2e_test_")
        logger.info(f"Created test directory: {self.test_dir}")
        
        # Initialize components
        self.mcp_integration = MCPIntegration()
        self.registry = ToolRegistry()
        
        # Initialize agents
        self.intent_agent = IntentRecognitionAgent()
        self.discovery_agent = ToolDiscoveryAgent(self.registry)
        self.orchestrator = OrchestratorAgent(
            self.intent_agent,
            self.discovery_agent,
            self.mcp_integration
        )
        
        # Add filesystem server with test directory as base
        await self.mcp_integration.add_filesystem_server(
            server_id="test_filesystem",
            base_path=self.test_dir
        )
        
        logger.info("Test environment setup complete")
        
    async def teardown(self):
        """Clean up test environment."""
        if self.mcp_integration:
            await self.mcp_integration.shutdown()
            
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory: {self.test_dir}")
            
    async def test_simple_file_operations(self):
        """Test basic file operations through full E2E workflow."""
        logger.info("=== Test: Simple File Operations ===")
        
        # Test 1: Create and read a file
        query1 = "Create a file named test.txt with content 'Hello, E2E Testing!'"
        result1 = await self.orchestrator.process_query(query1)
        
        assert result1["success"], f"Failed to process query: {result1.get('error')}"
        assert "test.txt" in result1.get("message", ""), "File creation not confirmed"
        
        # Verify file exists
        test_file = os.path.join(self.test_dir, "test.txt")
        assert os.path.exists(test_file), "Test file was not created"
        
        # Test 2: Read the file
        query2 = "Read the contents of test.txt"
        result2 = await self.orchestrator.process_query(query2)
        
        assert result2["success"], f"Failed to read file: {result2.get('error')}"
        assert "Hello, E2E Testing!" in result2.get("content", ""), "File content mismatch"
        
        logger.info("✓ Simple file operations test passed")
        
    async def test_directory_operations(self):
        """Test directory operations through E2E workflow."""
        logger.info("=== Test: Directory Operations ===")
        
        # Test 1: Create nested directories
        query1 = "Create a directory structure: project/src/components"
        result1 = await self.orchestrator.process_query(query1)
        
        assert result1["success"], f"Failed to create directories: {result1.get('error')}"
        
        # Verify directory structure
        nested_dir = os.path.join(self.test_dir, "project", "src", "components")
        assert os.path.isdir(nested_dir), "Nested directories were not created"
        
        # Test 2: List directory contents
        query2 = "List all files and directories in the project folder"
        result2 = await self.orchestrator.process_query(query2)
        
        assert result2["success"], f"Failed to list directory: {result2.get('error')}"
        assert "src" in str(result2.get("contents", [])), "Directory listing incomplete"
        
        logger.info("✓ Directory operations test passed")
        
    async def test_complex_workflow(self):
        """Test complex multi-step workflow."""
        logger.info("=== Test: Complex Workflow ===")
        
        # Create a project structure with multiple files
        workflows = [
            ("Create a directory named data", "data"),
            ("Create a file data/config.json with content '{\"version\": \"1.0\"}'", "data/config.json"),
            ("Create a file data/users.csv with content 'id,name\\n1,Alice\\n2,Bob'", "data/users.csv"),
            ("Create a directory named output", "output"),
        ]
        
        for query, expected_path in workflows:
            result = await self.orchestrator.process_query(query)
            assert result["success"], f"Failed on query '{query}': {result.get('error')}"
            
            full_path = os.path.join(self.test_dir, expected_path)
            assert os.path.exists(full_path), f"Expected path not created: {expected_path}"
            
        # Test reading and processing
        query = "Read all files in the data directory and list their contents"
        result = await self.orchestrator.process_query(query)
        
        assert result["success"], f"Failed to read data directory: {result.get('error')}"
        assert "config.json" in str(result), "config.json not found in results"
        assert "users.csv" in str(result), "users.csv not found in results"
        
        logger.info("✓ Complex workflow test passed")
        
    async def test_error_handling(self):
        """Test error handling in E2E scenarios."""
        logger.info("=== Test: Error Handling ===")
        
        # Test 1: Try to read non-existent file
        query1 = "Read the contents of non_existent_file.txt"
        result1 = await self.orchestrator.process_query(query1)
        
        # Should handle gracefully
        assert "error" in result1 or "not found" in str(result1).lower(), \
            "Error not properly handled for non-existent file"
        
        # Test 2: Try to create file with invalid path
        query2 = "Create a file at /root/restricted.txt with content 'test'"
        result2 = await self.orchestrator.process_query(query2)
        
        # Should be blocked by security restrictions
        assert not result2.get("success", False) or "error" in result2, \
            "Security restriction not enforced"
        
        logger.info("✓ Error handling test passed")
        
    async def test_learning_integration(self):
        """Test integration with learning system."""
        logger.info("=== Test: Learning Integration ===")
        
        # Perform multiple similar operations to train the system
        training_queries = [
            "Create a log file named app.log",
            "Write 'Application started' to app.log",
            "Read the contents of app.log",
            "Append 'Processing request' to app.log",
            "Read app.log again to see all entries"
        ]
        
        for i, query in enumerate(training_queries):
            logger.info(f"Training query {i+1}: {query}")
            result = await self.orchestrator.process_query(query)
            assert result["success"], f"Training query failed: {query}"
            
            # Check if learning metrics are being updated
            if hasattr(self.orchestrator, "learning_metrics"):
                metrics = self.orchestrator.get_learning_metrics()
                logger.info(f"Learning metrics: {metrics}")
        
        # Test if system learned patterns
        test_query = "Create another log file named system.log and write some initial content"
        result = await self.orchestrator.process_query(test_query)
        
        assert result["success"], "Failed to apply learned patterns"
        
        logger.info("✓ Learning integration test passed")
        
    async def test_performance_metrics(self):
        """Test performance tracking and metrics collection."""
        logger.info("=== Test: Performance Metrics ===")
        
        # Perform timed operations
        start_time = datetime.now()
        
        operations = [
            "Create 10 test files in a batch directory",
            "List all files in the batch directory",
            "Read the first test file",
            "Delete the last test file"
        ]
        
        execution_times = []
        
        for op in operations:
            op_start = datetime.now()
            result = await self.orchestrator.process_query(op)
            op_end = datetime.now()
            
            execution_time = (op_end - op_start).total_seconds()
            execution_times.append(execution_time)
            
            logger.info(f"Operation '{op[:30]}...' took {execution_time:.3f}s")
        
        total_time = (datetime.now() - start_time).total_seconds()
        avg_time = sum(execution_times) / len(execution_times)
        
        logger.info(f"Total execution time: {total_time:.3f}s")
        logger.info(f"Average operation time: {avg_time:.3f}s")
        
        # Performance assertions
        assert avg_time < 2.0, f"Average operation time too high: {avg_time:.3f}s"
        assert max(execution_times) < 5.0, f"Maximum operation time too high: {max(execution_times):.3f}s"
        
        logger.info("✓ Performance metrics test passed")
        
    async def run_all_tests(self):
        """Run all E2E tests."""
        try:
            await self.setup()
            
            # Run test suites
            await self.test_simple_file_operations()
            await self.test_directory_operations()
            await self.test_complex_workflow()
            await self.test_error_handling()
            await self.test_learning_integration()
            await self.test_performance_metrics()
            
            logger.info("\n✅ All E2E tests passed successfully!")
            
        except Exception as e:
            logger.error(f"\n❌ E2E test failed: {str(e)}")
            raise
        finally:
            await self.teardown()


async def main():
    """Main entry point for E2E tests."""
    logger.info("Starting Filesystem MCP End-to-End Tests")
    logger.info("=" * 50)
    
    test_suite = FileSystemE2ETests()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())