"""
Simplified integration tests for Tool Discovery Agent.

This test file validates the Tool Discovery Agent features without
depending on the full Intent Recognition pipeline.
"""

import pytest
import asyncio
import json
import time
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.tool_discovery_agent import ToolDiscoveryAgent, ToolCandidate
from src.models.intent import Intent, IntentResult
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger


def create_mock_intent_result(intent_type: str, keywords: list, query: str = None) -> IntentResult:
    """Create a mock intent result for testing."""
    if query is None:
        query = " ".join(keywords)
    
    intent = Intent(
        type=intent_type,
        keywords=keywords,
        confidence=0.85,
        entities={}
    )
    
    # Create a simple embedding (normally done by sentence transformer)
    embedding = np.random.rand(384)
    
    return IntentResult(
        raw_query=query,
        primary_intent=intent,
        all_intents=[intent],
        processed_query=query.lower(),
        confidence_threshold_met=True,
        metadata={'features': {'embedding': embedding}}
    )


class TestToolDiscoverySimple:
    """Simplified test suite for Tool Discovery Agent."""
    
    @pytest.fixture
    async def setup_test_environment(self, tmp_path):
        """Set up a complete test environment with tools and relationships."""
        # Create test database
        db_path = tmp_path / "test_discovery_simple.db"
        
        # Initialize registry
        registry = ToolRegistry(str(db_path))
        await registry.initialize()
        
        # Register test tools
        test_tools = [
            {
                'id': 'filesystem.read',
                'name': 'File Reader',
                'server_type': 'filesystem',
                'endpoint': 'mock://filesystem',
                'description': 'Read files from the filesystem',
                'capabilities': {
                    'operations': [
                        {'name': 'read_file', 'category': 'read'},
                        {'name': 'get_content', 'category': 'retrieve'}
                    ]
                }
            },
            {
                'id': 'filesystem.write',
                'name': 'File Writer',
                'server_type': 'filesystem',
                'endpoint': 'mock://filesystem',
                'description': 'Write files to the filesystem',
                'capabilities': {
                    'operations': [
                        {'name': 'write_file', 'category': 'write'},
                        {'name': 'create_file', 'category': 'create'}
                    ]
                }
            },
            {
                'id': 'database.query',
                'name': 'Database Query Tool',
                'server_type': 'database',
                'endpoint': 'mock://database',
                'description': 'Query data from databases',
                'capabilities': {
                    'operations': [
                        {'name': 'execute_query', 'category': 'query'},
                        {'name': 'fetch_data', 'category': 'retrieve'}
                    ]
                }
            },
            {
                'id': 'database.export',
                'name': 'Database Export Tool',
                'server_type': 'database',
                'endpoint': 'mock://database',
                'description': 'Export data from databases',
                'capabilities': {
                    'operations': [
                        {'name': 'export_csv', 'category': 'export'},
                        {'name': 'export_json', 'category': 'export'}
                    ]
                }
            },
            {
                'id': 'search.web',
                'name': 'Web Search Tool',
                'server_type': 'search',
                'endpoint': 'mock://search',
                'description': 'Search the web for information',
                'capabilities': {
                    'operations': [
                        {'name': 'search_web', 'category': 'search'},
                        {'name': 'find_pages', 'category': 'find'}
                    ]
                }
            },
            {
                'id': 'ml.analyze',
                'name': 'ML Analysis Tool',
                'server_type': 'ml',
                'endpoint': 'mock://ml',
                'description': 'Analyze data using machine learning',
                'capabilities': {
                    'operations': [
                        {'name': 'analyze_data', 'category': 'analyze'},
                        {'name': 'predict', 'category': 'analyze'}
                    ]
                }
            }
        ]
        
        # Register all tools
        for tool in test_tools:
            registry.register_tool(tool)
        
        # Add tool relationships for graph testing
        relationships = [
            ('filesystem.read', 'filesystem.write', 'complements', 0.9),
            ('database.query', 'database.export', 'complements', 0.8),
            ('database.export', 'filesystem.write', 'complements', 0.7),
            ('search.web', 'ml.analyze', 'complements', 0.6),
            ('database.query', 'ml.analyze', 'requires', 0.8),
        ]
        
        for tool1, tool2, rel_type, strength in relationships:
            await registry.add_tool_relationship(tool1, tool2, rel_type, strength)
        
        # Initialize discovery agent
        discovery_agent = ToolDiscoveryAgent({
            'database': {'tool_registry': str(db_path)},
            'cache_size': 10,
            'discovery': {
                'semantic_weight': 0.3,
                'capability_weight': 0.3,
                'performance_weight': 0.2,
                'relationship_weight': 0.2
            },
            'min_score_threshold': 0.1  # Lower threshold for testing
        })
        
        await discovery_agent.initialize()
        
        yield {
            'registry': registry,
            'discovery_agent': discovery_agent,
            'db_path': db_path
        }
        
        # Cleanup
        await discovery_agent.close()
        await registry.close()
    
    @pytest.mark.asyncio
    async def test_graph_based_exploration(self, setup_test_environment):
        """Test 1: Check graph-based exploration with NetworkX."""
        discovery_agent = setup_test_environment['discovery_agent']
        
        print("\n=== Testing Graph-Based Exploration ===")
        
        # Test 1: Complementary tool discovery
        print("\n1. Testing complementary tool discovery:")
        complementary_tools = await discovery_agent.find_complementary_tools(['database.query'])
        
        assert len(complementary_tools) > 0, "Should find complementary tools"
        
        # Check that database.export is found as complementary
        tool_ids = [t.tool_id for t in complementary_tools]
        assert 'database.export' in tool_ids, "Should find database.export as complementary"
        
        # Check relationship scores
        for tool in complementary_tools:
            if tool.tool_id == 'database.export':
                assert tool.relationship_score == 0.8, f"Expected 0.8, got {tool.relationship_score}"
                print(f"  ✓ Found complementary tool: {tool.tool_name} (score: {tool.relationship_score})")
        
        # Test 2: Tool recommendations
        print("\n2. Testing tool recommendations:")
        recommendations = await discovery_agent.get_tool_recommendations('database.query', 3)
        
        assert len(recommendations) > 0, "Should get tool recommendations"
        print(f"  ✓ Got {len(recommendations)} recommendations for database.query")
        
        for rec in recommendations:
            print(f"    - {rec.tool_name} ({rec.tool_id})")
        
        # Test 3: Relationship influence on discovery
        print("\n3. Testing relationship influence on discovery:")
        intent_result = create_mock_intent_result(
            'action.create',
            ['export', 'database', 'file'],
            'Export database results to a file'
        )
        
        # Discover tools with recent_tools context
        context = {'recent_tools': ['database.query']}
        candidates = await discovery_agent.discover_tools(intent_result, context)
        
        # Find database.export in results
        export_tool = next((c for c in candidates if c.tool_id == 'database.export'), None)
        assert export_tool is not None, "Should find database.export tool"
        assert export_tool.relationship_score > 0.5, "Should have boosted relationship score"
        
        print(f"  ✓ Relationship scoring working: {export_tool.tool_name} has relationship score {export_tool.relationship_score:.2f}")
        
        # Test 4: Verify graph structure
        print("\n4. Verifying graph structure:")
        graph = discovery_agent.tool_graph
        assert graph.number_of_nodes() == 6, f"Expected 6 nodes, got {graph.number_of_nodes()}"
        assert graph.number_of_edges() == 5, f"Expected 5 edges, got {graph.number_of_edges()}"
        print(f"  ✓ Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        print("\n✅ Graph-based exploration tests passed!")
    
    @pytest.mark.asyncio
    async def test_capability_matching_logic(self, setup_test_environment):
        """Test 2: Confirm capability matching logic."""
        discovery_agent = setup_test_environment['discovery_agent']
        
        print("\n=== Testing Capability Matching Logic ===")
        
        # Test 1: Direct capability matching
        print("\n1. Testing direct capability matching:")
        test_cases = [
            ("I need to read a file", 'query.retrieve', ['read', 'file'], 'filesystem.read'),
            ("Query the database", 'query.search', ['query', 'database'], 'database.query'),
            ("Search for information", 'query.search', ['search', 'information'], 'search.web'),
            ("Analyze this data", 'query.analyze', ['analyze', 'data'], 'ml.analyze')
        ]
        
        for query, intent_type, keywords, expected_tool in test_cases:
            intent_result = create_mock_intent_result(intent_type, keywords, query)
            candidates = await discovery_agent.discover_tools(intent_result)
            
            # Check if expected tool is in top results
            top_tools = [c.tool_id for c in candidates[:3]]
            assert expected_tool in top_tools, f"Expected {expected_tool} in top results for '{query}'"
            
            # Check capability score
            tool_candidate = next(c for c in candidates if c.tool_id == expected_tool)
            assert tool_candidate.capability_score > 0, f"Capability score should be > 0"
            
            print(f"  ✓ '{query}' matched to {expected_tool} (score: {tool_candidate.capability_score:.2f})")
        
        # Test 2: Synonym handling
        print("\n2. Testing capability synonym handling:")
        
        # Test with 'lookup' keyword (synonym for 'search')
        intent_result = create_mock_intent_result('query.search', ['lookup', 'information'])
        
        # Calculate capability score for search tool
        score = discovery_agent._calculate_capability_score(['search'], intent_result)
        assert score > 0, "Should recognize 'lookup' as synonym for 'search'"
        print(f"  ✓ Synonym handling working: 'lookup' recognized as 'search' synonym")
        
        print("\n✅ Capability matching logic tests passed!")
    
    @pytest.mark.asyncio
    async def test_pattern_based_discovery(self, setup_test_environment):
        """Test 3: Validate pattern-based discovery."""
        discovery_agent = setup_test_environment['discovery_agent']
        
        print("\n=== Testing Pattern-Based Discovery ===")
        
        # Test 1: Basic pattern search
        print("\n1. Testing basic pattern-based discovery:")
        patterns = [
            "database query tools",
            "file handling utilities",
            "web search engines",
            "data analysis tools"
        ]
        
        for pattern in patterns:
            candidates = await discovery_agent.discover_tools_by_pattern(pattern)
            
            assert len(candidates) > 0, f"Should find tools for pattern '{pattern}'"
            print(f"  ✓ Pattern '{pattern}' found {len(candidates)} tools:")
            
            for candidate in candidates[:2]:
                print(f"    - {candidate.tool_name} (score: {candidate.semantic_score:.2f})")
        
        # Test 2: Pattern specificity
        print("\n2. Testing pattern specificity:")
        
        specific_patterns = [
            ("filesystem read operations", 'filesystem.read'),
            ("database export functionality", 'database.export'),
            ("machine learning analysis", 'ml.analyze')
        ]
        
        for pattern, expected_tool in specific_patterns:
            candidates = await discovery_agent.discover_tools_by_pattern(pattern)
            
            # Expected tool should be in top results
            top_tools = [c.tool_id for c in candidates[:2]]
            assert expected_tool in top_tools, f"Pattern '{pattern}' should find {expected_tool}"
            
            expected_candidate = next(c for c in candidates if c.tool_id == expected_tool)
            print(f"  ✓ Pattern '{pattern}' found {expected_tool} (score: {expected_candidate.semantic_score:.2f})")
        
        print("\n✅ Pattern-based discovery tests passed!")
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, setup_test_environment):
        """Test 4: Test caching mechanism."""
        discovery_agent = setup_test_environment['discovery_agent']
        
        print("\n=== Testing Caching Mechanism ===")
        
        # Test 1: Tool embedding cache
        print("\n1. Testing tool embedding cache:")
        
        # Clear cache
        discovery_agent.tool_embeddings.clear()
        assert len(discovery_agent.tool_embeddings) == 0, "Cache should be empty"
        
        # Generate embeddings for tools
        tool_ids = ['filesystem.read', 'database.query', 'search.web']
        
        for tool_id in tool_ids:
            embedding = await discovery_agent._get_tool_embedding(tool_id, f"Test tool {tool_id}")
            assert tool_id in discovery_agent.tool_embeddings, f"Tool {tool_id} should be cached"
        
        assert len(discovery_agent.tool_embeddings) == 3, "Should have 3 cached embeddings"
        print(f"  ✓ Successfully cached {len(discovery_agent.tool_embeddings)} tool embeddings")
        
        # Test 2: Cache hit performance
        print("\n2. Testing cache hit performance:")
        
        # Time embedding generation without cache
        start_time = time.perf_counter()
        _ = await discovery_agent._get_tool_embedding('new_tool', "New tool description")
        uncached_time = time.perf_counter() - start_time
        
        # Time embedding retrieval with cache
        start_time = time.perf_counter()
        _ = await discovery_agent._get_tool_embedding('filesystem.read', "Test tool filesystem.read")
        cached_time = time.perf_counter() - start_time
        
        # Cache should be significantly faster
        assert cached_time < uncached_time, "Cached retrieval should be faster"
        speedup = uncached_time / cached_time if cached_time > 0 else float('inf')
        print(f"  ✓ Cache speedup: {speedup:.1f}x faster")
        
        # Test 3: Cache size limit
        print("\n3. Testing cache size limit:")
        
        # Set small cache size
        discovery_agent.embedding_cache_size = 5
        discovery_agent.tool_embeddings.clear()
        
        # Add more embeddings than cache size
        for i in range(7):
            tool_id = f"test_tool_{i}"
            await discovery_agent._get_tool_embedding(tool_id, f"Description {i}")
        
        # Cache should not exceed limit
        assert len(discovery_agent.tool_embeddings) <= 5, "Cache should not exceed size limit"
        
        # First tools should be evicted
        assert 'test_tool_0' not in discovery_agent.tool_embeddings, "Oldest entry should be evicted"
        assert 'test_tool_6' in discovery_agent.tool_embeddings, "Newest entry should be in cache"
        
        print(f"  ✓ Cache size limit working: {len(discovery_agent.tool_embeddings)} entries (limit: 5)")
        
        print("\n✅ Caching mechanism tests passed!")
    
    @pytest.mark.asyncio
    async def test_all_features_integration(self, setup_test_environment):
        """Integration test combining all features."""
        discovery_agent = setup_test_environment['discovery_agent']
        
        print("\n=== Integration Test: All Features Combined ===")
        
        # Complex query that should use all features
        intent_result = create_mock_intent_result(
            'action.create',
            ['query', 'database', 'export', 'file', 'analyze'],
            "Query the database and export results to a file for analysis"
        )
        
        print(f"\n1. Intent: {intent_result.primary_intent.type}")
        print(f"   Keywords: {intent_result.primary_intent.keywords}")
        
        # Discover tools with context
        context = {'recent_tools': ['database.query']}
        candidates = await discovery_agent.discover_tools(intent_result, context)
        
        print(f"\n2. Discovered {len(candidates)} tools:")
        for i, candidate in enumerate(candidates[:5]):
            print(f"   {i+1}. {candidate.tool_name} ({candidate.tool_id})")
            print(f"      - Semantic: {candidate.semantic_score:.2f}")
            print(f"      - Capability: {candidate.capability_score:.2f}")
            print(f"      - Relationship: {candidate.relationship_score:.2f}")
            print(f"      - Overall: {candidate.overall_score:.2f}")
        
        # Verify all features were used
        assert len(candidates) >= 3, "Should find multiple relevant tools"
        
        # Check specific tools are found
        tool_ids = [c.tool_id for c in candidates]
        assert 'database.export' in tool_ids, "Should find database export tool"
        assert any('filesystem' in tid for tid in tool_ids), "Should find filesystem tools"
        
        # Check scores reflect all factors
        export_tool = next(c for c in candidates if c.tool_id == 'database.export')
        assert export_tool.semantic_score > 0, "Should have semantic score"
        assert export_tool.capability_score > 0, "Should have capability score"
        assert export_tool.relationship_score > 0, "Should have relationship score (due to context)"
        
        print("\n✅ All features working together successfully!")


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])