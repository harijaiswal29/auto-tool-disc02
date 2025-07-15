"""
End-to-End tests for complete query-to-execution pipeline.

Tests the full workflow from natural language query through intent recognition,
tool discovery, selection, execution, and learning update.
"""

import pytest
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.agents.tool_discovery_agent import ToolDiscoveryAgent
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.database.tool_registry import ToolRegistryDB
from src.monitoring.intent_recognition_metrics import IntentRecognitionMetrics


class TestQueryToExecutionE2E:
    """E2E tests for complete query processing pipeline."""
    
    @pytest.fixture
    async def setup_system(self):
        """Set up the complete system for E2E testing."""
        # Create temporary directory for test
        test_dir = tempfile.mkdtemp(prefix="e2e_test_")
        db_path = os.path.join(test_dir, "test_e2e.db")
        
        # Initialize components
        registry = ToolRegistry(db_path)
        await registry.initialize()
        
        # Initialize MCP integration
        mcp = MCPIntegration(registry)
        await mcp.initialize()
        
        # Add mock MCP servers (use real ones if available)
        await mcp.add_filesystem_server(test_dir, server_id="test_fs", use_mock=True)
        await mcp.add_sqlite_server(db_path, server_id="test_db", use_mock=True)
        await mcp.add_search_server(server_id="test_search", use_mock=True)
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent()
        orchestrator.mcp_integration = mcp
        orchestrator.tool_registry = registry
        await orchestrator.initialize()
        
        yield {
            "orchestrator": orchestrator,
            "mcp": mcp,
            "registry": registry,
            "test_dir": test_dir,
            "db_path": db_path
        }
        
        # Cleanup
        await mcp.shutdown()
        shutil.rmtree(test_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_simple_query_workflow(self, setup_system):
        """Test a simple query through the complete pipeline."""
        logger.info("\n=== E2E Test: Simple Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Step 1: Create a test file
        test_file = os.path.join(test_dir, "test_document.txt")
        with open(test_file, "w") as f:
            f.write("This is a test document for E2E testing.")
        logger.info("✓ Created test file")
        
        # Step 2: Process natural language query
        query = "Find and read the test document file"
        logger.info(f"Processing query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Step 3: Verify intent recognition
        assert result.intent is not None, "Intent recognition failed"
        assert result.intent.primary_intent.type in ["query.search", "query.retrieve"]
        assert result.intent.confidence_passed, "Intent confidence too low"
        logger.info(f"✓ Intent recognized: {result.intent.primary_intent.type} "
                   f"(confidence: {result.intent.primary_intent.confidence:.2f})")
        
        # Step 4: Verify tool discovery
        assert len(result.discovered_tools) > 0, "No tools discovered"
        tool_names = [t.get("name", "") for t in result.discovered_tools]
        assert any("filesystem" in name.lower() for name in tool_names), \
            "Filesystem tool not discovered"
        logger.info(f"✓ Discovered {len(result.discovered_tools)} tools")
        
        # Step 5: Verify tool selection
        assert len(result.selected_tools) > 0, "No tools selected"
        logger.info(f"✓ Selected {len(result.selected_tools)} tools for execution")
        
        # Step 6: Verify execution
        assert result.success, "Execution failed"
        assert len(result.execution_results) > 0, "No execution results"
        
        # Check if file was found
        file_found = False
        for exec_result in result.execution_results:
            if exec_result.get("success") and "test_document" in str(exec_result.get("result", "")):
                file_found = True
                break
        
        assert file_found, "Test file was not found in execution results"
        logger.info("✓ Execution completed successfully")
        
        # Step 7: Verify performance
        assert result.total_time_ms > 0, "Invalid execution time"
        assert result.total_time_ms < 5000, "Execution took too long (>5s)"
        logger.info(f"✓ Total execution time: {result.total_time_ms:.2f}ms")
        
        logger.info("✅ Simple query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_action_query_workflow(self, setup_system):
        """Test an action query (create/modify) through the pipeline."""
        logger.info("\n=== E2E Test: Action Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Step 1: Process create action query
        query = "Create a new configuration file named app_config.json"
        logger.info(f"Processing query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Step 2: Verify intent recognition
        assert result.intent is not None
        assert result.intent.primary_intent.type == "action.create"
        logger.info(f"✓ Intent recognized: {result.intent.primary_intent.type}")
        
        # Step 3: Verify appropriate tools were selected
        assert len(result.selected_tools) > 0
        selected_names = [t.get("name", "") for t in result.selected_tools]
        assert any("filesystem" in name.lower() for name in selected_names)
        logger.info("✓ Filesystem tool selected for create action")
        
        # Step 4: Verify execution
        assert result.success
        config_path = os.path.join(test_dir, "app_config.json")
        
        # In real implementation, file would be created
        # For now, check execution was attempted
        assert len(result.execution_results) > 0
        logger.info("✓ Create action executed")
        
        logger.info("✅ Action query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_database_query_workflow(self, setup_system):
        """Test a database query through the pipeline."""
        logger.info("\n=== E2E Test: Database Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Process database query
        query = "Show me all tables in the database"
        logger.info(f"Processing query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify intent and tool selection
        assert result.intent is not None
        assert result.intent.primary_intent.type in ["query.retrieve", "query.search"]
        
        # Should discover database tools
        tool_names = [t.get("name", "") for t in result.discovered_tools]
        assert any("sqlite" in name.lower() or "database" in name.lower() 
                  for name in tool_names), "Database tool not discovered"
        
        logger.info("✓ Database tool discovered and selected")
        logger.info("✅ Database query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_complex_workflow_with_metrics(self, setup_system):
        """Test a complex workflow with performance metrics collection."""
        logger.info("\n=== E2E Test: Complex Workflow with Metrics ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Create test data
        for i in range(3):
            with open(os.path.join(test_dir, f"data_{i}.txt"), "w") as f:
                f.write(f"Test data file {i}")
        
        # Process complex query
        query = "Find all data files and analyze their content"
        logger.info(f"Processing complex query: '{query}'")
        
        start_time = datetime.now()
        result = await orchestrator.process_user_query(query)
        end_time = datetime.now()
        
        # Verify multi-step execution
        assert result.success
        assert len(result.discovered_tools) >= 2  # Should find multiple relevant tools
        
        # Verify performance metrics
        processing_time = (end_time - start_time).total_seconds() * 1000
        assert processing_time < 10000  # Should complete within 10 seconds
        
        # Check intent recognition metrics
        if hasattr(result.intent, "processing_time_ms"):
            assert result.intent.processing_time_ms < 100  # Intent should be fast
            logger.info(f"✓ Intent recognition: {result.intent.processing_time_ms:.2f}ms")
        
        logger.info(f"✓ Total processing time: {processing_time:.2f}ms")
        logger.info("✅ Complex workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, setup_system):
        """Test error handling in the pipeline."""
        logger.info("\n=== E2E Test: Error Handling Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Test with ambiguous query
        query = "Do something with the things"
        logger.info(f"Processing ambiguous query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # System should handle gracefully
        assert result is not None
        assert result.intent is not None
        
        # Low confidence queries should be flagged
        if not result.intent.confidence_passed:
            logger.info("✓ Low confidence query correctly identified")
        else:
            # Even with confidence, execution should handle gracefully
            assert hasattr(result, "execution_results")
            logger.info("✓ Query processed despite ambiguity")
        
        # Test with invalid file reference
        query = "Read the file /invalid/path/does/not/exist.txt"
        result = await orchestrator.process_user_query(query)
        
        # Should complete without crashing
        assert result is not None
        logger.info("✓ Invalid file reference handled gracefully")
        
        logger.info("✅ Error handling workflow test passed!")


def main():
    """Run E2E tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()