"""
Integration tests for the Autonomous Tool Discovery System.

Tests the complete pipeline from intent recognition to tool execution.
"""

import pytest
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import OrchestratorAgent, IntentRecognitionAgent, ToolDiscoveryAgent
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize an orchestrator for testing."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()
        
        # Add mock servers
        mcp = orchestrator.mcp_integration
        await mcp.add_sqlite_server("data/test_integration.db", "test_sqlite", use_mock=True)
        await mcp.add_search_server(server_id="test_search", use_mock=True)
        await mcp.add_filesystem_server(".", "test_fs", use_mock=True)
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_query(self, orchestrator):
        """Test end-to-end processing of a search query."""
        query = "Find all Python files in the project"
        
        result = await orchestrator.process_user_query(query)
        
        # Check intent recognition
        assert result.intent is not None
        assert result.intent.primary_intent.type == "query.search"
        assert result.intent.confidence_passed
        
        # Check tool discovery
        assert len(result.discovered_tools) > 0
        
        # Check tool selection
        assert len(result.selected_tools) > 0
        
        # Check execution
        assert len(result.execution_results) > 0
        assert result.success  # At least one tool should succeed
        
        # Check timing
        assert result.total_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_create_query(self, orchestrator):
        """Test end-to-end processing of a create action."""
        query = "Create a new configuration file"
        
        result = await orchestrator.process_user_query(query)
        
        # Check intent recognition
        assert result.intent.primary_intent.type == "action.create"
        
        # Check that appropriate tools were discovered
        assert len(result.discovered_tools) > 0
        
        # Verify filesystem tools were found
        tool_types = [t.get('type', '') for t in result.discovered_tools]
        assert any('filesystem' in t for t in tool_types)
    
    @pytest.mark.asyncio
    async def test_intent_to_tool_mapping(self):
        """Test that intents map correctly to tool capabilities."""
        # Initialize components
        intent_agent = IntentRecognitionAgent()
        discovery_agent = ToolDiscoveryAgent()
        await discovery_agent.initialize()
        
        # Test query intent
        intent_result = await intent_agent.process_query("Search for documents")
        tools = await discovery_agent.discover_tools(intent_result)
        
        # Should find search-capable tools
        assert len(tools) > 0
        capabilities = []
        for tool in tools:
            capabilities.extend(tool.capabilities)
        
        assert any('search' in cap.lower() or 'query' in cap.lower() for cap in capabilities)
        
        await discovery_agent.close()
    
    @pytest.mark.asyncio
    async def test_mcp_integration_with_intent(self):
        """Test MCP integration's intent-based methods."""
        registry = ToolRegistry("data/test_integration_registry.db")
        await registry.initialize()
        
        mcp = MCPIntegration(registry=registry)
        await mcp.initialize()
        
        # Add mock servers
        await mcp.add_sqlite_server("data/test.db", use_mock=True)
        await mcp.add_search_server(use_mock=True)
        
        # Test finding tools by intent
        search_tools = await mcp.find_tools_by_intent("query.search")
        assert len(search_tools) > 0
        
        # Test finding tools by capabilities
        query_tools = await mcp.get_tools_by_capabilities(["query"])
        assert len(query_tools) > 0
        
        # Cleanup
        await mcp.shutdown()
    
    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, orchestrator):
        """Test parallel execution of multiple tools."""
        # Ensure parallel execution is enabled
        orchestrator.config['orchestration']['parallel_execution'] = True
        
        query = "Search for files and check the weather"
        result = await orchestrator.process_user_query(query)
        
        # Should execute multiple tools
        assert len(result.execution_results) > 1
        
        # Check that execution was reasonably fast (parallel)
        # If sequential, would take longer
        assert result.total_time_ms < 5000  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test system behavior with errors."""
        # Test with a query that might cause errors
        query = "Execute invalid SQL query: DROP TABLE *"
        
        result = await orchestrator.process_user_query(query)
        
        # System should handle errors gracefully
        assert result is not None
        assert result.query == query
        
        # Should still recognize intent
        assert result.intent is not None
    
    @pytest.mark.asyncio
    async def test_multi_intent_query(self, orchestrator):
        """Test handling of queries with multiple intents."""
        query = "Find log files and then delete old ones"
        
        result = await orchestrator.process_user_query(query)
        
        # Should handle multi-intent
        assert result.intent is not None
        assert len(result.intent.all_intents) >= 1  # At least primary intent
        
        # Should discover tools for multiple intents
        assert len(result.discovered_tools) > 0


class TestComponentIntegration:
    """Test integration between specific components."""
    
    @pytest.mark.asyncio
    async def test_intent_recognition_to_tool_discovery(self):
        """Test flow from intent recognition to tool discovery."""
        intent_agent = IntentRecognitionAgent()
        discovery_agent = ToolDiscoveryAgent()
        
        await discovery_agent.initialize()
        
        # Process a query
        query = "Analyze the code quality"
        intent_result = await intent_agent.process_query(query)
        
        # Discover tools based on intent
        candidates = await discovery_agent.discover_tools(intent_result)
        
        # Verify flow works
        assert intent_result.primary_intent.type == "query.analyze"
        assert len(candidates) > 0
        
        # Check scoring
        for candidate in candidates:
            assert candidate.overall_score > 0
            assert candidate.semantic_score >= 0
            assert candidate.capability_score >= 0
        
        await discovery_agent.close()
    
    @pytest.mark.asyncio
    async def test_tool_discovery_recommendations(self):
        """Test tool recommendation system."""
        discovery_agent = ToolDiscoveryAgent()
        await discovery_agent.initialize()
        
        # Get all tools
        registry = discovery_agent.tool_registry
        all_tools = await registry.get_all_tools()
        
        if all_tools:
            # Get recommendations for first tool
            tool_id = all_tools[0]['id']
            recommendations = await discovery_agent.get_tool_recommendations(tool_id)
            
            # Should return some recommendations
            assert isinstance(recommendations, list)
        
        await discovery_agent.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])