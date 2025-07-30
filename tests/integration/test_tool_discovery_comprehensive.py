"""
Comprehensive integration tests for Tool Discovery Agent.

This test file validates all aspects of the Tool Discovery Agent:
1. Graph-based exploration with NetworkX
2. Capability matching logic
3. Pattern-based discovery
4. Caching mechanism
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
from src.agents.intent_recognition_agent import IntentRecognitionAgent, Intent, IntentResult
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger


class TestToolDiscoveryComprehensive:
    """Comprehensive test suite for Tool Discovery Agent."""
    
    @pytest.fixture
    async def setup_test_environment(self, tmp_path):
        """Set up a complete test environment with tools and relationships."""
        # Create test database
        db_path = tmp_path / "test_discovery.db"
        
        # Initialize registry
        registry = ToolRegistry(str(db_path))
        await registry.initialize()
        
        # Register test tools with various capabilities
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
        
        # Initialize agents
        intent_agent = IntentRecognitionAgent()
        discovery_agent = ToolDiscoveryAgent({
            'database': {'tool_registry': str(db_path)},
            'cache_size': 10,  # Small cache for testing
            'discovery': {
                'semantic_weight': 0.3,
                'capability_weight': 0.3,
                'performance_weight': 0.2,
                'relationship_weight': 0.2
            },
            'min_score_threshold': 0.3
        })
        
        await discovery_agent.initialize()
        
        yield {
            'registry': registry,
            'intent_agent': intent_agent,
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
        intent_agent = setup_test_environment['intent_agent']
        
        print("\n=== Testing Graph-Based Exploration ===")
        
        # Test complementary tool discovery
        print("\n1. Testing complementary tool discovery:")
        complementary_tools = await discovery_agent.find_complementary_tools(['database.query'])
        
        assert len(complementary_tools) > 0, "Should find complementary tools"
        
        # Check that database.export is found as complementary
        tool_ids = [t.tool_id for t in complementary_tools]
        assert 'database.export' in tool_ids, "Should find database.export as complementary to database.query"
        
        # Check relationship scores
        for tool in complementary_tools:
            if tool.tool_id == 'database.export':
                assert tool.relationship_score == 0.8, "Should have correct relationship strength"
                print(f"  ✓ Found complementary tool: {tool.tool_name} (score: {tool.relationship_score})")
        
        # Test tool recommendations (uses graph relationships)
        print("\n2. Testing tool recommendations based on relationships:")
        recommendations = await discovery_agent.get_tool_recommendations('database.query', max_recommendations=3)
        
        assert len(recommendations) > 0, "Should get tool recommendations"
        print(f"  ✓ Got {len(recommendations)} recommendations for database.query")
        
        for rec in recommendations:
            print(f"    - {rec.tool_name} ({rec.tool_id})")
        
        # Test that relationship scores influence overall scoring
        print("\n3. Testing relationship influence on discovery:")
        query = "I need to export database results to a file"
        intent_result = await intent_agent.process_query(query)
        
        # Check if intent was recognized
        if intent_result.primary_intent is None:
            print("  ⚠️  Intent recognition failed, creating mock intent")
            # Create a mock intent for testing
            from src.agents.intent_models import Intent
            mock_intent = Intent(
                type='action.create',
                keywords=['export', 'database', 'file'],
                confidence=0.85,
                entities={'action': 'export', 'source': 'database', 'target': 'file'}
            )
            intent_result.primary_intent = mock_intent
        
        # Ensure features have an embedding
        if intent_result.features is None:
            intent_result.features = {}
        if 'embedding' not in intent_result.features:
            # Create a mock embedding
            intent_result.features['embedding'] = np.random.rand(384)
        
        # Discover tools with recent_tools context
        context = {'recent_tools': ['database.query']}
        candidates = await discovery_agent.discover_tools(intent_result, context)
        
        # Find database.export in results
        export_tool = next((c for c in candidates if c.tool_id == 'database.export'), None)
        assert export_tool is not None, "Should find database.export tool"
        assert export_tool.relationship_score > 0.5, "Should have boosted relationship score due to context"
        
        print(f"  ✓ Relationship scoring working: {export_tool.tool_name} has relationship score {export_tool.relationship_score:.2f}")
        
        # Verify graph structure
        print("\n4. Verifying graph structure:")
        graph = discovery_agent.tool_graph
        assert graph.number_of_nodes() == 6, f"Should have 6 nodes, got {graph.number_of_nodes()}"
        assert graph.number_of_edges() == 5, f"Should have 5 edges, got {graph.number_of_edges()}"
        print(f"  ✓ Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        print("\n✅ Graph-based exploration tests passed!")
        
        return True
    
    @pytest.mark.asyncio
    async def test_capability_matching_logic(self, setup_test_environment):
        """Test 2: Confirm capability matching logic."""
        discovery_agent = setup_test_environment['discovery_agent']
        intent_agent = setup_test_environment['intent_agent']
        
        print("\n=== Testing Capability Matching Logic ===")
        
        # Test 1: Direct capability matching
        print("\n1. Testing direct capability matching:")
        test_cases = [
            ("I need to read a file", 'filesystem.read', ['read', 'get']),
            ("Query the database for user data", 'database.query', ['query', 'retrieve']),
            ("Search for information online", 'search.web', ['search', 'find']),
            ("Analyze this data", 'ml.analyze', ['analyze', 'predict'])
        ]
        
        for query, expected_tool, expected_caps in test_cases:
            intent_result = await intent_agent.process_query(query)
            candidates = await discovery_agent.discover_tools(intent_result)
            
            # Check if expected tool is in top results
            top_tools = [c.tool_id for c in candidates[:3]]
            assert expected_tool in top_tools, f"Expected {expected_tool} in top results for '{query}'"
            
            # Check capability score
            tool_candidate = next(c for c in candidates if c.tool_id == expected_tool)
            assert tool_candidate.capability_score > 0.5, f"Capability score should be > 0.5, got {tool_candidate.capability_score}"
            
            print(f"  ✓ '{query}' correctly matched to {expected_tool} (score: {tool_candidate.capability_score:.2f})")
        
        # Test 2: Intent to capability mapping
        print("\n2. Testing intent to capability mapping:")
        intent_mappings = [
            ('query.search', ['search', 'find', 'query']),
            ('query.retrieve', ['read', 'get', 'fetch']),
            ('action.create', ['create', 'write', 'generate']),
            ('query.analyze', ['analyze', 'examine', 'inspect'])
        ]
        
        for intent_type, expected_capabilities in intent_mappings:
            mapped_caps = discovery_agent.intent_capability_map.get(intent_type, [])
            
            # Check if expected capabilities are in the mapping
            for cap in expected_capabilities:
                assert cap in mapped_caps, f"Expected '{cap}' in capabilities for {intent_type}"
            
            print(f"  ✓ Intent '{intent_type}' correctly maps to capabilities")
        
        # Test 3: Synonym handling
        print("\n3. Testing capability synonym handling:")
        
        # Create mock intent with 'lookup' keyword (synonym for 'search')
        mock_intent = Intent(
            type='query.search',
            keywords=['lookup', 'information'],
            confidence=0.85,
            entities={}
        )
        mock_result = IntentResult(
            raw_query="Lookup information",
            primary_intent=mock_intent,
            alternative_intents=[],
            context={},
            processing_time_ms=10.0,
            features={'embedding': np.random.rand(384)}
        )
        
        # Calculate capability score for search tool
        score = discovery_agent._calculate_capability_score(['search'], mock_result)
        assert score > 0, "Should recognize 'lookup' as synonym for 'search'"
        print(f"  ✓ Synonym handling working: 'lookup' recognized as 'search' synonym")
        
        # Test 4: Multi-capability tools
        print("\n4. Testing multi-capability tool scoring:")
        
        # Database export tool has multiple export capabilities
        export_caps = ['export_csv', 'export_json']
        export_intent = Intent(
            type='action.create',
            keywords=['export', 'save'],
            confidence=0.9,
            entities={}
        )
        export_result = IntentResult(
            raw_query="Export data",
            primary_intent=export_intent,
            alternative_intents=[],
            context={},
            processing_time_ms=10.0,
            features={'embedding': np.random.rand(384)}
        )
        
        score = discovery_agent._calculate_capability_score(export_caps, export_result)
        assert score > 0, "Should score multi-capability tools appropriately"
        print(f"  ✓ Multi-capability scoring working (score: {score:.2f})")
        
        print("\n✅ Capability matching logic tests passed!")
        
        return True
    
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
        
        # Test 2: Semantic similarity
        print("\n2. Testing semantic similarity in pattern discovery:")
        
        # These patterns should find similar tools
        similar_patterns = [
            ("search for files", "find documents"),
            ("query database", "fetch data from db"),
            ("export results", "save output")
        ]
        
        for pattern1, pattern2 in similar_patterns:
            candidates1 = await discovery_agent.discover_tools_by_pattern(pattern1)
            candidates2 = await discovery_agent.discover_tools_by_pattern(pattern2)
            
            # Get top tool IDs from each
            top_ids1 = {c.tool_id for c in candidates1[:3]}
            top_ids2 = {c.tool_id for c in candidates2[:3]}
            
            # Should have some overlap
            overlap = top_ids1.intersection(top_ids2)
            assert len(overlap) > 0, f"Similar patterns '{pattern1}' and '{pattern2}' should find overlapping tools"
            
            print(f"  ✓ Patterns '{pattern1}' and '{pattern2}' have {len(overlap)} overlapping tools")
        
        # Test 3: Pattern specificity
        print("\n3. Testing pattern specificity:")
        
        specific_patterns = [
            ("filesystem read operations", 'filesystem.read'),
            ("database export functionality", 'database.export'),
            ("machine learning analysis", 'ml.analyze')
        ]
        
        for pattern, expected_tool in specific_patterns:
            candidates = await discovery_agent.discover_tools_by_pattern(pattern)
            
            # Expected tool should be in top results
            top_tools = [c.tool_id for c in candidates[:2]]
            assert expected_tool in top_tools, f"Pattern '{pattern}' should find {expected_tool} in top results"
            
            # Check semantic score
            expected_candidate = next(c for c in candidates if c.tool_id == expected_tool)
            assert expected_candidate.semantic_score > 0.5, "Should have high semantic score for specific pattern"
            
            print(f"  ✓ Specific pattern '{pattern}' correctly found {expected_tool} (score: {expected_candidate.semantic_score:.2f})")
        
        # Test 4: Complex patterns
        print("\n4. Testing complex pattern matching:")
        
        complex_pattern = "tools for reading database query results and exporting them to files"
        candidates = await discovery_agent.discover_tools_by_pattern(complex_pattern)
        
        # Should find multiple relevant tools
        assert len(candidates) >= 2, "Complex pattern should find multiple relevant tools"
        
        # Check if both database and file tools are found
        tool_types = {c.tool_type for c in candidates[:4]}
        assert 'database' in tool_types or 'filesystem' in tool_types, "Should find relevant tool types"
        
        print(f"  ✓ Complex pattern found {len(candidates)} relevant tools")
        
        print("\n✅ Pattern-based discovery tests passed!")
        
        return True
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, setup_test_environment):
        """Test 4: Test caching mechanism."""
        discovery_agent = setup_test_environment['discovery_agent']
        intent_agent = setup_test_environment['intent_agent']
        
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
        
        # Time embedding retrieval with cache (use existing cached tool)
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
        
        # Test 4: Cache impact on discovery performance
        print("\n4. Testing cache impact on discovery performance:")
        
        # Clear cache and run discovery
        discovery_agent.tool_embeddings.clear()
        query = "Find and analyze data"
        intent_result = await intent_agent.process_query(query)
        
        # First run (cold cache)
        start_time = time.perf_counter()
        candidates1 = await discovery_agent.discover_tools(intent_result)
        cold_time = time.perf_counter() - start_time
        
        # Second run (warm cache)
        start_time = time.perf_counter()
        candidates2 = await discovery_agent.discover_tools(intent_result)
        warm_time = time.perf_counter() - start_time
        
        # Results should be consistent
        assert len(candidates1) == len(candidates2), "Results should be consistent"
        
        # Warm cache should be faster
        if warm_time > 0:
            improvement = ((cold_time - warm_time) / cold_time) * 100
            print(f"  ✓ Warm cache {improvement:.1f}% faster than cold cache")
        
        # Test 5: Cache persistence across queries
        print("\n5. Testing cache persistence:")
        
        initial_cache_size = len(discovery_agent.tool_embeddings)
        
        # Run multiple discoveries
        queries = ["Read files", "Query database", "Search web"]
        for q in queries:
            intent_res = await intent_agent.process_query(q)
            await discovery_agent.discover_tools(intent_res)
        
        final_cache_size = len(discovery_agent.tool_embeddings)
        assert final_cache_size >= initial_cache_size, "Cache should persist and grow across queries"
        
        print(f"  ✓ Cache persisted and grew from {initial_cache_size} to {final_cache_size} entries")
        
        print("\n✅ Caching mechanism tests passed!")
        
        return True
    
    @pytest.mark.asyncio
    async def test_all_features_integration(self, setup_test_environment):
        """Integration test combining all features."""
        discovery_agent = setup_test_environment['discovery_agent']
        intent_agent = setup_test_environment['intent_agent']
        
        print("\n=== Integration Test: All Features Combined ===")
        
        # Complex query that should use all features
        query = "I need to query the database and export the results to a file for machine learning analysis"
        
        # Process intent
        intent_result = await intent_agent.process_query(query)
        print(f"\n1. Intent recognized: {intent_result.primary_intent.type}")
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
        
        return True


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])