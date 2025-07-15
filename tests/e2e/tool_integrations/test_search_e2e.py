"""
End-to-End tests for Search MCP tool integration.

Tests complete workflows involving search operations through natural
language queries, including web search, result filtering, and analysis.
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
from unittest.mock import AsyncMock, MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.tools.search_mcp import SearchMCP


class TestSearchE2E:
    """E2E tests for Search MCP tool integration."""
    
    @pytest.fixture
    async def setup_system(self):
        """Set up system with Search MCP."""
        # Create temporary directory
        test_dir = tempfile.mkdtemp(prefix="e2e_search_")
        registry_db_path = os.path.join(test_dir, "registry.db")
        
        # Initialize components
        registry = ToolRegistry(registry_db_path)
        await registry.initialize()
        
        # Initialize MCP integration
        mcp = MCPIntegration(registry)
        await mcp.initialize()
        
        # Add Search MCP server (use mock for testing)
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
            "test_dir": test_dir
        }
        
        # Cleanup
        await mcp.shutdown()
        shutil.rmtree(test_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_simple_search_workflow(self, setup_system):
        """Test simple web search query."""
        logger.info("\n=== E2E Test: Simple Search Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Natural language search query
        query = "Search for Python programming tutorials"
        logger.info(f"Processing query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify intent recognition
        assert result.intent is not None
        assert result.intent.primary_intent.type in ["query.search", "query.retrieve"]
        logger.info(f"✓ Intent recognized: {result.intent.primary_intent.type}")
        
        # Verify search tool was selected
        assert len(result.selected_tools) > 0
        tool_names = [t.get("name", "") for t in result.selected_tools]
        assert any("search" in name.lower() for name in tool_names)
        logger.info("✓ Search tool selected")
        
        # Verify execution
        assert result.success
        assert len(result.execution_results) > 0
        logger.info("✓ Search executed successfully")
        
        logger.info("✅ Simple search workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_research_query_workflow(self, setup_system):
        """Test research-oriented search query."""
        logger.info("\n=== E2E Test: Research Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Research query
        query = "Research the latest developments in machine learning for 2024"
        logger.info(f"Processing research query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        assert result.intent.primary_intent.type in ["query.search", "query.analyze"]
        
        # Should use search tool
        assert result.success
        logger.info("✓ Research query processed")
        
        # The system should recognize this as requiring web search
        selected_tools = [t.get("name", "") for t in result.selected_tools]
        assert any("search" in tool.lower() for tool in selected_tools)
        
        logger.info("✅ Research query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_fact_checking_workflow(self, setup_system):
        """Test fact-checking search workflow."""
        logger.info("\n=== E2E Test: Fact Checking Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Fact-checking query
        query = "Verify if Python 3.12 has been released"
        logger.info(f"Processing fact-checking query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        # Could be search or analyze intent
        assert "query" in result.intent.primary_intent.type
        
        assert result.success
        logger.info("✓ Fact-checking query executed")
        
        logger.info("✅ Fact-checking workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_news_search_workflow(self, setup_system):
        """Test news-related search workflow."""
        logger.info("\n=== E2E Test: News Search Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # News query
        query = "Find the latest technology news"
        logger.info(f"Processing news query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        assert result.intent.primary_intent.type == "query.search"
        
        # Should prioritize search tool
        assert result.success
        assert len(result.selected_tools) > 0
        
        logger.info("✓ News search completed")
        logger.info("✅ News search workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_multi_source_search_workflow(self, setup_system):
        """Test searching across multiple sources."""
        logger.info("\n=== E2E Test: Multi-Source Search Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Complex search requiring multiple sources
        query = "Compare different web frameworks and find documentation"
        logger.info(f"Processing multi-source query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        # Might recognize multiple intents (search + compare)
        if hasattr(result.intent, "all_intents"):
            logger.info(f"✓ Recognized {len(result.intent.all_intents)} intents")
        
        assert result.success
        logger.info("✓ Multi-source search completed")
        
        logger.info("✅ Multi-source search workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_search_with_filters_workflow(self, setup_system):
        """Test search with specific filters or constraints."""
        logger.info("\n=== E2E Test: Filtered Search Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Search with time constraint
        query = "Search for AI papers published in the last month"
        logger.info(f"Processing filtered query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        assert result.intent.primary_intent.type == "query.search"
        
        # Should extract time constraint
        if hasattr(result.intent.primary_intent, "parameters"):
            logger.info(f"✓ Extracted parameters: {result.intent.primary_intent.parameters}")
        
        assert result.success
        logger.info("✓ Filtered search executed")
        
        logger.info("✅ Filtered search workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_search_and_summarize_workflow(self, setup_system):
        """Test search followed by summarization."""
        logger.info("\n=== E2E Test: Search and Summarize Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Multi-step query
        query = "Search for quantum computing basics and summarize the key concepts"
        logger.info(f"Processing search-and-summarize query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        
        # Should recognize multiple intents
        if hasattr(result.intent, "all_intents"):
            intent_types = [i.type for i in result.intent.all_intents]
            assert any("search" in t for t in intent_types)
            assert any("analyze" in t or "summary" in t for t in intent_types)
            logger.info(f"✓ Multi-intent recognized: {intent_types}")
        
        assert result.success
        logger.info("✓ Search and summarize completed")
        
        logger.info("✅ Search and summarize workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, setup_system):
        """Test error handling in search operations."""
        logger.info("\n=== E2E Test: Search Error Handling ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Query that might fail
        query = "Search for xyzabc123randomquery that returns no results"
        logger.info(f"Processing query with no results: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Should handle gracefully
        assert result is not None
        assert result.intent is not None
        
        # Even with no/few results, should complete
        logger.info("✓ No results handled gracefully")
        
        # Test with malformed query
        query2 = "Search for [[[invalid<<<query>>>]]]"
        result2 = await orchestrator.process_user_query(query2)
        
        assert result2 is not None
        logger.info("✓ Invalid query handled gracefully")
        
        logger.info("✅ Search error handling test passed!")
    
    @pytest.mark.asyncio
    async def test_search_performance_metrics(self, setup_system):
        """Test performance metrics for search operations."""
        logger.info("\n=== E2E Test: Search Performance Metrics ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Time a search operation
        start_time = datetime.now()
        query = "Quick search for Python documentation"
        result = await orchestrator.process_user_query(query)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds() * 1000
        
        assert result.success
        assert total_time < 10000  # Should complete within 10 seconds
        logger.info(f"✓ Search completed in {total_time:.2f}ms")
        
        # Check if metrics are tracked
        if hasattr(result, "total_time_ms"):
            assert result.total_time_ms > 0
            logger.info(f"✓ Metrics tracked: {result.total_time_ms:.2f}ms")
        
        logger.info("✅ Search performance metrics test passed!")


def main():
    """Run Search E2E tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()