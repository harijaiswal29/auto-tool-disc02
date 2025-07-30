"""
Integration tests for intent-based tool discovery and selection.

This test suite verifies that the system properly:
1. Maps intents to tool capabilities
2. Discovers relevant tools based on intent types
3. Scores and ranks tools by relevance
4. Filters out irrelevant tools
5. Handles complex multi-intent queries
"""

import pytest
import asyncio
import json
import tempfile
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.intent_models import Intent, IntentResult
from src.agents.tool_discovery_agent import ToolDiscoveryAgent
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestIntentBasedDiscovery:
    """Test suite for intent-based tool discovery functionality."""
    
    @pytest.fixture
    async def setup_test_environment(self, tmp_path):
        """Set up test environment with diverse tools."""
        # Create temporary registry
        registry_path = tmp_path / "test_registry.db"
        registry = ToolRegistry(str(registry_path))
        
        # Register diverse tools with different capabilities
        test_tools = [
            # Search tools
            {
                'id': 'search.web',
                'name': 'Web Search',
                'type': 'search',
                'server': 'search_mcp',
                'capabilities': json.dumps({
                    'operations': ['search', 'find', 'query', 'lookup', 'discover']
                }),
                'status': 'active',
                'performance_score': 0.85
            },
            {
                'id': 'search.academic',
                'name': 'Academic Search',
                'type': 'search',
                'server': 'search_mcp',
                'capabilities': json.dumps({
                    'operations': ['search', 'analyze', 'research', 'study']
                }),
                'status': 'active',
                'performance_score': 0.90
            },
            
            # Database tools
            {
                'id': 'database.query',
                'name': 'Database Query',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': json.dumps({
                    'operations': ['query', 'retrieve', 'get', 'fetch', 'analyze']
                }),
                'status': 'active',
                'performance_score': 0.92
            },
            {
                'id': 'database.insert',
                'name': 'Database Insert',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': json.dumps({
                    'operations': ['create', 'insert', 'add', 'write']
                }),
                'status': 'active',
                'performance_score': 0.88
            },
            {
                'id': 'database.update',
                'name': 'Database Update',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': json.dumps({
                    'operations': ['update', 'modify', 'edit', 'change', 'alter']
                }),
                'status': 'active',
                'performance_score': 0.87
            },
            {
                'id': 'database.delete',
                'name': 'Database Delete',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': json.dumps({
                    'operations': ['delete', 'remove', 'drop', 'clear', 'purge']
                }),
                'status': 'active',
                'performance_score': 0.86
            },
            
            # File system tools
            {
                'id': 'filesystem.read',
                'name': 'File Reader',
                'type': 'filesystem',
                'server': 'filesystem_mcp',
                'capabilities': json.dumps({
                    'operations': ['read', 'retrieve', 'get', 'load', 'access']
                }),
                'status': 'active',
                'performance_score': 0.95
            },
            {
                'id': 'filesystem.write',
                'name': 'File Writer',
                'type': 'filesystem',
                'server': 'filesystem_mcp',
                'capabilities': json.dumps({
                    'operations': ['write', 'create', 'save', 'store', 'generate']
                }),
                'status': 'active',
                'performance_score': 0.93
            },
            {
                'id': 'filesystem.analyze',
                'name': 'File Analyzer',
                'type': 'filesystem',
                'server': 'filesystem_mcp',
                'capabilities': json.dumps({
                    'operations': ['analyze', 'inspect', 'examine', 'profile']
                }),
                'status': 'active',
                'performance_score': 0.89
            },
            
            # System tools
            {
                'id': 'system.config',
                'name': 'System Configurator',
                'type': 'system',
                'server': 'system_mcp',
                'capabilities': json.dumps({
                    'operations': ['configure', 'setup', 'initialize', 'customize']
                }),
                'status': 'active',
                'performance_score': 0.91
            },
            {
                'id': 'system.monitor',
                'name': 'System Monitor',
                'type': 'system',
                'server': 'system_mcp',
                'capabilities': json.dumps({
                    'operations': ['monitor', 'track', 'watch', 'observe', 'audit']
                }),
                'status': 'active',
                'performance_score': 0.94
            },
            
            # Weather tool (for testing irrelevant tool filtering)
            {
                'id': 'weather.forecast',
                'name': 'Weather Forecast',
                'type': 'weather',
                'server': 'weather_mcp',
                'capabilities': json.dumps({
                    'operations': ['forecast', 'predict', 'weather']
                }),
                'status': 'active',
                'performance_score': 0.80
            }
        ]
        
        # Register all tools
        for tool in test_tools:
            await registry.register_tool(tool)
        
        # Add some tool relationships
        await registry.add_tool_relationship('database.query', 'database.update', 'complements')
        await registry.add_tool_relationship('filesystem.read', 'filesystem.analyze', 'complements')
        await registry.add_tool_relationship('search.web', 'search.academic', 'alternatives')
        
        # Create test configuration
        config = {
            'orchestration': {
                'max_tools_per_query': 5,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'intent_recognition': {
                'model': 'all-MiniLM-L6-v2',
                'confidence_threshold': 0.7
            },
            'database': {
                'tool_registry': str(registry_path)
            }
        }
        
        return {
            'registry': registry,
            'config': config,
            'test_tools': test_tools
        }
    
    @pytest.mark.asyncio
    async def test_intent_to_capability_mapping(self, setup_test_environment):
        """Test 1: Verify intent-to-capability mapping for all intent types."""
        config = setup_test_environment['config']
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Test intent-capability mappings
        test_cases = [
            ('query.search', ['search', 'find', 'query', 'list']),
            ('query.retrieve', ['read', 'get', 'fetch', 'retrieve']),
            ('query.analyze', ['analyze', 'examine', 'inspect', 'evaluate']),
            ('action.create', ['create', 'write', 'generate', 'make']),
            ('action.modify', ['update', 'edit', 'modify', 'change']),
            ('action.delete', ['delete', 'remove', 'clear', 'drop']),
            ('system.configure', ['configure', 'setup', 'initialize']),
            ('system.monitor', ['monitor', 'track', 'watch', 'observe'])
        ]
        
        for intent_type, expected_capabilities in test_cases:
            capabilities = orchestrator.intent_capability_map.get(intent_type, [])
            logger.info(f"Intent: {intent_type}, Capabilities: {capabilities}")
            
            # Check that expected capabilities are present
            for cap in expected_capabilities:
                assert cap in capabilities, f"Capability '{cap}' not found for intent '{intent_type}'"
        
        logger.info("✅ All intent-to-capability mappings verified")
    
    @pytest.mark.asyncio
    async def test_tool_discovery_by_intent(self, setup_test_environment):
        """Test 2: Test tool discovery based on different intent types."""
        config = setup_test_environment['config']
        registry = setup_test_environment['registry']
        
        # Initialize agents
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Test cases with different intents
        test_cases = [
            {
                'query': 'Search for information about Python',
                'expected_intent': 'query.search',
                'expected_tools': ['search.web', 'search.academic'],
                'unexpected_tools': ['weather.forecast', 'database.delete']
            },
            {
                'query': 'Create a new database record',
                'expected_intent': 'action.create',
                'expected_tools': ['database.insert', 'filesystem.write'],
                'unexpected_tools': ['database.delete', 'search.web']
            },
            {
                'query': 'Analyze the system performance',
                'expected_intent': 'query.analyze',
                'expected_tools': ['database.query', 'filesystem.analyze', 'search.academic'],
                'unexpected_tools': ['weather.forecast', 'database.delete']
            },
            {
                'query': 'Delete old log files',
                'expected_intent': 'action.delete',
                'expected_tools': ['database.delete'],
                'unexpected_tools': ['database.insert', 'filesystem.write']
            },
            {
                'query': 'Monitor system resources',
                'expected_intent': 'system.monitor',
                'expected_tools': ['system.monitor'],
                'unexpected_tools': ['weather.forecast', 'database.insert']
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\nTesting query: {test_case['query']}")
            
            # Process query to get intent
            intent_result = await orchestrator.intent_agent.process_query(test_case['query'])
            assert intent_result.primary_intent.type == test_case['expected_intent'], \
                f"Expected intent '{test_case['expected_intent']}', got '{intent_result.primary_intent.type}'"
            
            # Discover tools based on intent
            discovered_tools = await orchestrator.discover_tools_for_intent(intent_result)
            discovered_tool_ids = [tool['id'] for tool in discovered_tools]
            
            logger.info(f"Discovered tools: {discovered_tool_ids[:5]}")  # Show top 5
            
            # Verify expected tools are discovered
            for expected_tool in test_case['expected_tools']:
                assert expected_tool in discovered_tool_ids, \
                    f"Expected tool '{expected_tool}' not discovered for query '{test_case['query']}'"
            
            # Verify unexpected tools are not in top discoveries
            top_tool_ids = discovered_tool_ids[:5]  # Check top 5 discoveries
            for unexpected_tool in test_case['unexpected_tools']:
                assert unexpected_tool not in top_tool_ids, \
                    f"Unexpected tool '{unexpected_tool}' found in top discoveries for query '{test_case['query']}'"
        
        logger.info("✅ Tool discovery by intent verified for all test cases")
    
    @pytest.mark.asyncio
    async def test_semantic_scoring_and_capability_matching(self, setup_test_environment):
        """Test 3: Verify semantic scoring and capability matching."""
        config = setup_test_environment['config']
        
        # Initialize discovery agent
        discovery_agent = ToolDiscoveryAgent(config)
        
        # Create test intent result
        intent_result = IntentResult(
            primary_intent=Intent(
                type='query.search',
                keywords=['information', 'data', 'find'],
                confidence=0.9,
                entities=[]
            ),
            all_intents=[],
            raw_query='Find information about data processing'
        )
        
        # Discover tools
        candidates = await discovery_agent.discover_tools(intent_result)
        
        # Verify scoring components
        for candidate in candidates[:5]:  # Check top 5
            logger.info(f"\nTool: {candidate.tool_name}")
            logger.info(f"  Semantic score: {candidate.semantic_score:.3f}")
            logger.info(f"  Capability score: {candidate.capability_score:.3f}")
            logger.info(f"  Performance score: {candidate.performance_score:.3f}")
            logger.info(f"  Overall score: {candidate.overall_score:.3f}")
            
            # Verify scores are calculated
            assert candidate.semantic_score >= 0, "Semantic score should be non-negative"
            assert candidate.capability_score >= 0, "Capability score should be non-negative"
            assert candidate.overall_score > 0, "Overall score should be positive for discovered tools"
            
            # Verify search tools have high capability scores
            if 'search' in candidate.tool_id:
                assert candidate.capability_score > 0.5, \
                    f"Search tool '{candidate.tool_name}' should have high capability score for search intent"
        
        logger.info("✅ Semantic scoring and capability matching verified")
    
    @pytest.mark.asyncio
    async def test_tool_ranking_by_relevance(self, setup_test_environment):
        """Test 4: Test tool ranking and selection based on relevance scores."""
        config = setup_test_environment['config']
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Test query with clear intent
        query = "Retrieve user data from the database"
        
        # Process query
        intent_result = await orchestrator.intent_agent.process_query(query)
        discovered_tools = await orchestrator.discover_tools_for_intent(intent_result)
        
        # Verify tools are ranked by relevance score
        previous_score = float('inf')
        for i, tool in enumerate(discovered_tools[:5]):
            current_score = tool.get('relevance_score', 0)
            logger.info(f"Rank {i+1}: {tool['name']} (score: {current_score:.3f})")
            
            # Verify descending order
            assert current_score <= previous_score, \
                "Tools should be ranked in descending order by relevance score"
            previous_score = current_score
        
        # Verify most relevant tools are at the top
        top_tool_ids = [tool['id'] for tool in discovered_tools[:3]]
        assert 'database.query' in top_tool_ids, \
            "Database query tool should be in top 3 for database retrieval query"
        
        logger.info("✅ Tool ranking by relevance verified")
    
    @pytest.mark.asyncio
    async def test_irrelevant_tool_filtering(self, setup_test_environment):
        """Test 5: Verify that irrelevant tools are filtered out."""
        config = setup_test_environment['config']
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Test queries that should NOT match weather tools
        test_queries = [
            "Search for Python tutorials",
            "Create a new file in the system",
            "Monitor database performance",
            "Delete old records from the database"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting filtering for query: {query}")
            
            # Process query
            intent_result = await orchestrator.intent_agent.process_query(query)
            discovered_tools = await orchestrator.discover_tools_for_intent(intent_result)
            
            # Get top discovered tools
            top_tool_ids = [tool['id'] for tool in discovered_tools[:5]]
            logger.info(f"Top tools: {top_tool_ids}")
            
            # Verify weather tool is not in top discoveries
            assert 'weather.forecast' not in top_tool_ids, \
                f"Weather tool should not be in top 5 for query '{query}'"
            
            # If weather tool is discovered at all, it should have low score
            weather_tool = next((tool for tool in discovered_tools if tool['id'] == 'weather.forecast'), None)
            if weather_tool:
                assert weather_tool.get('relevance_score', 0) < 0.3, \
                    f"Weather tool should have low relevance score for query '{query}'"
        
        logger.info("✅ Irrelevant tool filtering verified")
    
    @pytest.mark.asyncio
    async def test_complex_multi_intent_discovery(self, setup_test_environment):
        """Test 6: Test discovery with complex multi-intent queries."""
        config = setup_test_environment['config']
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Complex queries that involve multiple intents/operations
        test_cases = [
            {
                'query': 'Search for user data, analyze it, and save the results',
                'expected_tools': ['search.web', 'database.query', 'filesystem.analyze', 'filesystem.write'],
                'min_tools': 3
            },
            {
                'query': 'Monitor system performance and update configuration if needed',
                'expected_tools': ['system.monitor', 'system.config', 'database.update'],
                'min_tools': 2
            },
            {
                'query': 'Read files, analyze content, and create a summary report',
                'expected_tools': ['filesystem.read', 'filesystem.analyze', 'filesystem.write'],
                'min_tools': 3
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\nTesting complex query: {test_case['query']}")
            
            # Process query
            intent_result = await orchestrator.intent_agent.process_query(test_case['query'])
            discovered_tools = await orchestrator.discover_tools_for_intent(intent_result)
            
            # Get discovered tool IDs
            discovered_tool_ids = [tool['id'] for tool in discovered_tools]
            logger.info(f"Discovered {len(discovered_tools)} tools")
            logger.info(f"Top tools: {discovered_tool_ids[:5]}")
            
            # Verify minimum number of tools discovered
            assert len(discovered_tools) >= test_case['min_tools'], \
                f"Should discover at least {test_case['min_tools']} tools for complex query"
            
            # Verify expected tools are discovered
            found_expected = 0
            for expected_tool in test_case['expected_tools']:
                if expected_tool in discovered_tool_ids[:10]:  # Check top 10
                    found_expected += 1
            
            assert found_expected >= 2, \
                f"Should find at least 2 of the expected tools in top 10 discoveries"
            
            # Verify diversity of tool types
            tool_types = set()
            for tool in discovered_tools[:5]:
                tool_types.add(tool.get('type'))
            
            assert len(tool_types) >= 2, \
                "Complex queries should discover tools from multiple types"
        
        logger.info("✅ Complex multi-intent discovery verified")
    
    @pytest.mark.asyncio
    async def test_intent_confidence_impact(self, setup_test_environment):
        """Test that low confidence intents still discover tools but with adjusted scoring."""
        config = setup_test_environment['config']
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Create two intent results with different confidence levels
        high_confidence_intent = IntentResult(
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'find'],
                confidence=0.95,
                entities=[]
            ),
            all_intents=[],
            raw_query='search for data'
        )
        
        low_confidence_intent = IntentResult(
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'find'],
                confidence=0.55,
                entities=[]
            ),
            all_intents=[],
            raw_query='search for data'
        )
        
        # Discover tools for both
        high_conf_tools = await orchestrator.discover_tools_for_intent(high_confidence_intent)
        low_conf_tools = await orchestrator.discover_tools_for_intent(low_confidence_intent)
        
        # Both should discover tools
        assert len(high_conf_tools) > 0, "High confidence intent should discover tools"
        assert len(low_conf_tools) > 0, "Low confidence intent should still discover tools"
        
        # Compare top tools - should be similar but potentially different ordering
        high_conf_top = [t['id'] for t in high_conf_tools[:3]]
        low_conf_top = [t['id'] for t in low_conf_tools[:3]]
        
        logger.info(f"High confidence top tools: {high_conf_top}")
        logger.info(f"Low confidence top tools: {low_conf_top}")
        
        # At least one tool should be common in top 3
        common_tools = set(high_conf_top) & set(low_conf_top)
        assert len(common_tools) >= 1, \
            "Similar intents should have at least one common tool in top results regardless of confidence"
        
        logger.info("✅ Intent confidence impact on discovery verified")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])