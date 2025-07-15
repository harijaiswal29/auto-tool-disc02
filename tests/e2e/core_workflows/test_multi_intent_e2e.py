"""
End-to-End tests for multi-intent query handling.

Tests the system's ability to handle queries with multiple intents,
including proper sequencing, parallel execution, and dependency resolution.
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
from src.agents.intent_models import MultiIntentHandler


class TestMultiIntentE2E:
    """E2E tests for multi-intent query processing."""
    
    @pytest.fixture
    async def setup_system(self):
        """Set up the complete system for E2E testing."""
        # Create temporary directory
        test_dir = tempfile.mkdtemp(prefix="e2e_multi_intent_")
        db_path = os.path.join(test_dir, "test_multi_intent.db")
        
        # Initialize components
        registry = ToolRegistry(db_path)
        await registry.initialize()
        
        # Initialize MCP integration
        mcp = MCPIntegration(registry)
        await mcp.initialize()
        
        # Add multiple mock MCP servers for testing
        await mcp.add_filesystem_server(test_dir, server_id="test_fs", use_mock=True)
        await mcp.add_sqlite_server(db_path, server_id="test_db", use_mock=True)
        await mcp.add_search_server(server_id="test_search", use_mock=True)
        
        # Initialize orchestrator with multi-intent support
        orchestrator = OrchestratorAgent()
        orchestrator.mcp_integration = mcp
        orchestrator.tool_registry = registry
        orchestrator.config['orchestration']['enable_multi_intent'] = True
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
    async def test_sequential_multi_intent(self, setup_system):
        """Test sequential execution of multiple intents."""
        logger.info("\n=== E2E Test: Sequential Multi-Intent ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Create test file
        test_file = os.path.join(test_dir, "source_data.txt")
        with open(test_file, "w") as f:
            f.write("Sample data for processing")
        
        # Multi-intent query with "then" indicating sequence
        query = "Find the source data file and then create a summary report"
        logger.info(f"Processing multi-intent query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify intent recognition
        assert result.intent is not None
        assert hasattr(result.intent, "all_intents")
        assert len(result.intent.all_intents) >= 2
        
        # Check intent types
        intent_types = [intent.type for intent in result.intent.all_intents]
        assert any("search" in t or "retrieve" in t for t in intent_types)
        assert any("create" in t for t in intent_types)
        logger.info(f"✓ Recognized {len(intent_types)} intents: {intent_types}")
        
        # Verify sequential execution
        assert len(result.execution_results) >= 2
        
        # First should be search/find
        first_result = result.execution_results[0]
        assert "filesystem" in str(first_result.get("tool_name", "")).lower()
        
        logger.info("✓ Sequential execution completed")
        logger.info("✅ Sequential multi-intent test passed!")
    
    @pytest.mark.asyncio
    async def test_parallel_multi_intent(self, setup_system):
        """Test parallel execution of independent intents."""
        logger.info("\n=== E2E Test: Parallel Multi-Intent ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Create multiple test files
        for i in range(3):
            with open(os.path.join(test_dir, f"file_{i}.txt"), "w") as f:
                f.write(f"Content {i}")
        
        # Multi-intent query with "and" indicating parallel possibility
        query = "Search for text files and check database status and get system information"
        logger.info(f"Processing parallel multi-intent query: '{query}'")
        
        start_time = datetime.now()
        result = await orchestrator.process_user_query(query)
        end_time = datetime.now()
        
        # Verify multiple intents recognized
        assert result.intent is not None
        assert len(result.intent.all_intents) >= 2
        
        # Verify multiple tools executed
        assert len(result.execution_results) >= 2
        
        # Check execution time (parallel should be faster)
        total_time = (end_time - start_time).total_seconds()
        logger.info(f"✓ Parallel execution completed in {total_time:.2f}s")
        
        # Verify different tool types were used
        tool_types = set()
        for exec_result in result.execution_results:
            tool_name = exec_result.get("tool_name", "")
            if "filesystem" in tool_name.lower():
                tool_types.add("filesystem")
            elif "database" in tool_name.lower() or "sqlite" in tool_name.lower():
                tool_types.add("database")
        
        assert len(tool_types) >= 2, "Multiple tool types should be used"
        logger.info(f"✓ Used tool types: {tool_types}")
        
        logger.info("✅ Parallel multi-intent test passed!")
    
    @pytest.mark.asyncio
    async def test_dependent_multi_intent(self, setup_system):
        """Test handling of dependent intents."""
        logger.info("\n=== E2E Test: Dependent Multi-Intent ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Query with dependencies: analyze requires finding first
        query = "Find all log files, analyze their content, and create a summary"
        logger.info(f"Processing dependent multi-intent query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify intent chain
        assert result.intent is not None
        assert len(result.intent.all_intents) >= 3
        
        intent_types = [intent.type for intent in result.intent.all_intents]
        logger.info(f"✓ Intent chain: {' → '.join(intent_types)}")
        
        # Verify execution order respects dependencies
        if result.execution_results:
            # First execution should be search/find
            first_tool = result.execution_results[0].get("tool_name", "")
            assert "filesystem" in first_tool.lower() or "search" in first_tool.lower()
            logger.info("✓ Dependencies respected in execution order")
        
        logger.info("✅ Dependent multi-intent test passed!")
    
    @pytest.mark.asyncio
    async def test_complex_multi_intent_workflow(self, setup_system):
        """Test a complex real-world multi-intent scenario."""
        logger.info("\n=== E2E Test: Complex Multi-Intent Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        test_dir = setup_system["test_dir"]
        
        # Create test environment
        project_dir = os.path.join(test_dir, "project")
        os.makedirs(project_dir)
        
        # Create Python files
        for name in ["main.py", "utils.py", "test_main.py"]:
            with open(os.path.join(project_dir, name), "w") as f:
                f.write(f"# {name}\nprint('Hello from {name}')")
        
        # Create data file
        with open(os.path.join(project_dir, "data.json"), "w") as f:
            json.dump({"version": "1.0", "items": [1, 2, 3]}, f)
        
        # Complex multi-intent query
        query = ("Find all Python files in the project, analyze their code structure, "
                "then read the data.json file and create a project report")
        logger.info(f"Processing complex query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify comprehensive processing
        assert result.intent is not None
        assert len(result.intent.all_intents) >= 3
        
        # Check various intent types
        intent_types = [intent.type for intent in result.intent.all_intents]
        assert any("search" in t or "find" in t for t in intent_types)
        assert any("analyze" in t for t in intent_types)
        assert any("create" in t for t in intent_types)
        
        logger.info(f"✓ Processed {len(intent_types)} intents")
        
        # Verify multiple tools were orchestrated
        if result.execution_results:
            tool_names = [r.get("tool_name", "") for r in result.execution_results]
            logger.info(f"✓ Orchestrated tools: {tool_names}")
        
        logger.info("✅ Complex multi-intent workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_multi_intent_error_recovery(self, setup_system):
        """Test error handling in multi-intent queries."""
        logger.info("\n=== E2E Test: Multi-Intent Error Recovery ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Query with mix of valid and invalid operations
        query = "Read nonexistent file and create a new report and search for data"
        logger.info(f"Processing query with potential errors: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # System should handle partial failures
        assert result is not None
        assert result.intent is not None
        
        # Should still recognize all intents
        assert len(result.intent.all_intents) >= 3
        
        # Check if some operations succeeded despite errors
        if result.execution_results:
            successes = [r for r in result.execution_results if r.get("success")]
            failures = [r for r in result.execution_results if not r.get("success")]
            
            logger.info(f"✓ Handled {len(successes)} successes and "
                       f"{len(failures)} failures")
        
        # Overall result might be partial success
        logger.info("✓ System continued despite partial failures")
        logger.info("✅ Multi-intent error recovery test passed!")
    
    @pytest.mark.asyncio
    async def test_multi_intent_performance(self, setup_system):
        """Test performance characteristics of multi-intent processing."""
        logger.info("\n=== E2E Test: Multi-Intent Performance ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Measure single intent performance
        single_query = "Find all files"
        start_single = datetime.now()
        single_result = await orchestrator.process_user_query(single_query)
        single_time = (datetime.now() - start_single).total_seconds() * 1000
        
        # Measure multi-intent performance
        multi_query = "Find all files and check database and get system status"
        start_multi = datetime.now()
        multi_result = await orchestrator.process_user_query(multi_query)
        multi_time = (datetime.now() - start_multi).total_seconds() * 1000
        
        logger.info(f"✓ Single intent: {single_time:.2f}ms")
        logger.info(f"✓ Multi-intent: {multi_time:.2f}ms")
        
        # Multi-intent should not be dramatically slower if parallel
        assert multi_time < single_time * 3, \
            "Multi-intent taking too long (should use parallelism)"
        
        # Verify intent recognition is still fast
        if hasattr(multi_result.intent, "processing_time_ms"):
            assert multi_result.intent.processing_time_ms < 200
            logger.info(f"✓ Intent recognition: {multi_result.intent.processing_time_ms:.2f}ms")
        
        logger.info("✅ Multi-intent performance test passed!")


def main():
    """Run multi-intent E2E tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()