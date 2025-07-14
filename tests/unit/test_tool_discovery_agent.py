"""
Unit tests for Tool Discovery Agent.

Tests the tool discovery functionality including:
- Semantic search for tools
- Capability-based matching
- Tool relationship graph exploration
- Scoring algorithms
- Complementary tool discovery
- Pattern-based discovery
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import networkx as nx

from src.agents.tool_discovery_agent import ToolDiscoveryAgent, ToolCandidate
from src.agents.intent_recognition_agent import IntentResult, Intent


class TestToolDiscoveryAgent:
    """Test cases for ToolDiscoveryAgent class."""
    
    @pytest.fixture
    def default_config(self):
        """Default configuration for testing."""
        return {
            'model': 'all-MiniLM-L6-v2',
            'cache_size': 100,
            'discovery': {
                'semantic_weight': 0.3,
                'capability_weight': 0.3,
                'performance_weight': 0.2,
                'relationship_weight': 0.2
            },
            'min_score_threshold': 0.3,
            'database': {
                'tool_registry': 'test_registry.db'
            }
        }
    
    @pytest.fixture
    def mock_intent_result(self):
        """Create a mock intent result."""
        primary_intent = Intent(
            type='query.search',
            keywords=['find', 'files', 'project'],
            confidence=0.85,
            entities={'target': 'files'}
        )
        return IntentResult(
            raw_query="Find files in the project",
            primary_intent=primary_intent,
            alternative_intents=[],
            context={'domain': 'development'},
            processing_time_ms=50.0,
            features={'embedding': np.random.rand(384)}  # Mock embedding
        )
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tool data."""
        return [
            {
                'id': 'filesystem.search',
                'name': 'File Search',
                'type': 'filesystem',
                'performance_score': 0.9,
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'search_files', 'category': 'search'},
                        {'name': 'find_files', 'category': 'find'}
                    ]
                })
            },
            {
                'id': 'grep.search',
                'name': 'Grep Search',
                'type': 'search',
                'performance_score': 0.8,
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'search_content', 'category': 'search'},
                        {'name': 'find_pattern', 'category': 'find'}
                    ]
                })
            },
            {
                'id': 'database.query',
                'name': 'Database Query',
                'type': 'database',
                'performance_score': 0.7,
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'query', 'category': 'query'},
                        {'name': 'retrieve', 'category': 'retrieve'}
                    ]
                })
            }
        ]
    
    @pytest.fixture
    async def discovery_agent(self, default_config):
        """Create ToolDiscoveryAgent instance with mocks."""
        with patch('src.agents.tool_discovery_agent.SentenceTransformer') as mock_transformer:
            with patch('src.agents.tool_discovery_agent.ToolRegistry') as mock_registry:
                # Mock sentence transformer
                mock_model = Mock()
                mock_model.encode = Mock(return_value=np.random.rand(384))
                mock_transformer.return_value = mock_model
                
                # Create agent
                agent = ToolDiscoveryAgent(config=default_config)
                
                # Setup registry mock
                agent.tool_registry = mock_registry.return_value
                agent.tool_registry.get_all_tools = AsyncMock()
                agent.tool_registry.get_tool_relationships = AsyncMock(return_value=[])
                agent.tool_registry.initialize = AsyncMock()
                agent.tool_registry.close = AsyncMock()
                
                yield agent
    
    def test_initialization(self, default_config):
        """Test ToolDiscoveryAgent initialization."""
        with patch('src.agents.tool_discovery_agent.SentenceTransformer'):
            with patch('src.agents.tool_discovery_agent.ToolRegistry'):
                agent = ToolDiscoveryAgent(config=default_config)
                
                assert agent.config == default_config
                assert agent.intent_capability_map is not None
                assert 'query.search' in agent.intent_capability_map
                assert agent.tool_embeddings == {}
                assert isinstance(agent.tool_graph, nx.DiGraph)
    
    def test_load_default_config(self, discovery_agent):
        """Test loading default configuration."""
        config = discovery_agent._load_default_config()
        
        assert 'model' in config
        assert 'cache_size' in config
        assert 'discovery' in config
        assert config['discovery']['semantic_weight'] == 0.3
    
    @pytest.mark.asyncio
    async def test_discover_tools_success(self, discovery_agent, mock_intent_result, mock_tools):
        """Test successful tool discovery."""
        discovery_agent.tool_registry.get_all_tools.return_value = mock_tools
        
        candidates = await discovery_agent.discover_tools(mock_intent_result)
        
        assert len(candidates) > 0
        assert all(isinstance(c, ToolCandidate) for c in candidates)
        # Should be sorted by overall score
        scores = [c.overall_score for c in candidates]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_discover_tools_no_tools(self, discovery_agent, mock_intent_result):
        """Test tool discovery when no tools are available."""
        discovery_agent.tool_registry.get_all_tools.return_value = []
        
        candidates = await discovery_agent.discover_tools(mock_intent_result)
        
        assert len(candidates) == 0
    
    @pytest.mark.asyncio
    async def test_score_tool(self, discovery_agent, mock_intent_result, mock_tools):
        """Test individual tool scoring."""
        tool = mock_tools[0]
        
        candidate = await discovery_agent._score_tool(tool, mock_intent_result, {})
        
        assert isinstance(candidate, ToolCandidate)
        assert candidate.tool_id == tool['id']
        assert candidate.tool_name == tool['name']
        assert 0 <= candidate.semantic_score <= 1.0
        assert 0 <= candidate.capability_score <= 1.0
        assert 0 <= candidate.performance_score <= 1.0
        assert 0 <= candidate.relationship_score <= 1.0
        assert 0 <= candidate.overall_score <= 1.0
    
    def test_parse_capabilities_json(self, discovery_agent):
        """Test parsing capabilities from JSON format."""
        capabilities = json.dumps({
            'operations': [
                {'name': 'search'},
                {'name': 'find'}
            ]
        })
        
        parsed = discovery_agent._parse_capabilities(capabilities)
        
        assert parsed == ['search', 'find']
    
    def test_parse_capabilities_dict(self, discovery_agent):
        """Test parsing capabilities from dict format."""
        capabilities = {
            'operations': [
                {'name': 'query'},
                {'name': 'retrieve'}
            ]
        }
        
        parsed = discovery_agent._parse_capabilities(capabilities)
        
        assert parsed == ['query', 'retrieve']
    
    def test_parse_capabilities_list(self, discovery_agent):
        """Test parsing capabilities from list format."""
        capabilities = ['read', 'write', 'delete']
        
        parsed = discovery_agent._parse_capabilities(capabilities)
        
        assert parsed == ['read', 'write', 'delete']
    
    def test_parse_capabilities_string(self, discovery_agent):
        """Test parsing capabilities from string format."""
        capabilities = "search"
        
        parsed = discovery_agent._parse_capabilities(capabilities)
        
        assert parsed == ['search']
    
    def test_parse_capabilities_invalid(self, discovery_agent):
        """Test parsing invalid capabilities format."""
        capabilities = "invalid json {"
        
        parsed = discovery_agent._parse_capabilities(capabilities)
        
        assert parsed == ['invalid json {']
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_score(self, discovery_agent, mock_intent_result, mock_tools):
        """Test semantic score calculation."""
        tool = mock_tools[0]
        
        # Mock embeddings
        discovery_agent.model.encode.return_value = np.random.rand(384)
        
        score = await discovery_agent._calculate_semantic_score(tool, mock_intent_result)
        
        assert 0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_score_no_embedding(self, discovery_agent, mock_tools):
        """Test semantic score with no query embedding."""
        tool = mock_tools[0]
        intent_result = Mock()
        intent_result.features = {'embedding': None}
        
        score = await discovery_agent._calculate_semantic_score(tool, intent_result)
        
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_get_tool_embedding_caching(self, discovery_agent):
        """Test tool embedding caching."""
        tool_id = "test_tool"
        tool_desc = "Test tool description"
        
        # First call - compute embedding
        embedding1 = await discovery_agent._get_tool_embedding(tool_id, tool_desc)
        assert discovery_agent.model.encode.call_count == 1
        
        # Second call - use cache
        embedding2 = await discovery_agent._get_tool_embedding(tool_id, tool_desc)
        assert discovery_agent.model.encode.call_count == 1  # Not called again
        assert np.array_equal(embedding1, embedding2)
    
    @pytest.mark.asyncio
    async def test_get_tool_embedding_cache_limit(self, discovery_agent):
        """Test tool embedding cache size limit."""
        discovery_agent.embedding_cache_size = 2
        
        # Add embeddings up to limit
        await discovery_agent._get_tool_embedding("tool1", "desc1")
        await discovery_agent._get_tool_embedding("tool2", "desc2")
        assert len(discovery_agent.tool_embeddings) == 2
        
        # Add one more - should evict oldest
        await discovery_agent._get_tool_embedding("tool3", "desc3")
        assert len(discovery_agent.tool_embeddings) == 2
        assert "tool1" not in discovery_agent.tool_embeddings
        assert "tool2" in discovery_agent.tool_embeddings
        assert "tool3" in discovery_agent.tool_embeddings
    
    def test_calculate_capability_score(self, discovery_agent, mock_intent_result):
        """Test capability score calculation."""
        tool_capabilities = ['search', 'find', 'query']
        
        score = discovery_agent._calculate_capability_score(
            tool_capabilities, mock_intent_result
        )
        
        assert 0 <= score <= 1.0
        # Should have good score since capabilities match intent
        assert score > 0.5
    
    def test_calculate_capability_score_with_synonyms(self, discovery_agent, mock_intent_result):
        """Test capability score with synonyms."""
        tool_capabilities = ['lookup']  # Synonym for search
        
        score = discovery_agent._calculate_capability_score(
            tool_capabilities, mock_intent_result
        )
        
        assert score > 0  # Should recognize synonym
    
    def test_calculate_capability_score_no_capabilities(self, discovery_agent, mock_intent_result):
        """Test capability score with no capabilities."""
        score = discovery_agent._calculate_capability_score([], mock_intent_result)
        
        assert score == 0.0
    
    def test_calculate_capability_score_no_requirements(self, discovery_agent):
        """Test capability score with no requirements."""
        intent_result = Mock()
        intent_result.primary_intent = Mock(type='unknown.intent', keywords=[])
        
        score = discovery_agent._calculate_capability_score(['search'], intent_result)
        
        assert score == 0.5  # Neutral score
    
    @pytest.mark.asyncio
    async def test_calculate_relationship_score(self, discovery_agent):
        """Test relationship score calculation."""
        # Add nodes to graph
        discovery_agent.tool_graph.add_node('tool1')
        discovery_agent.tool_graph.add_node('tool2')
        discovery_agent.tool_graph.add_edge('tool1', 'tool2', type='complements')
        
        context = {'recent_tools': ['tool1']}
        
        score = await discovery_agent._calculate_relationship_score('tool2', context)
        
        assert score > 0.5  # Should be boosted by complement relationship
    
    @pytest.mark.asyncio
    async def test_calculate_relationship_score_no_node(self, discovery_agent):
        """Test relationship score for non-existent node."""
        score = await discovery_agent._calculate_relationship_score('unknown_tool', {})
        
        assert score == 0.5  # Neutral score
    
    @pytest.mark.asyncio
    async def test_build_tool_graph(self, discovery_agent, mock_tools):
        """Test building tool relationship graph."""
        discovery_agent.tool_registry.get_all_tools.return_value = mock_tools
        discovery_agent.tool_registry.get_tool_relationships.return_value = [
            {
                'tool1_id': 'filesystem.search',
                'tool2_id': 'grep.search',
                'relationship_type': 'complements',
                'strength': 0.8
            }
        ]
        
        await discovery_agent._build_tool_graph()
        
        assert discovery_agent.tool_graph.number_of_nodes() == len(mock_tools)
        assert discovery_agent.tool_graph.number_of_edges() == 1
        assert discovery_agent.tool_graph.has_edge('filesystem.search', 'grep.search')
    
    @pytest.mark.asyncio
    async def test_find_complementary_tools(self, discovery_agent, mock_tools):
        """Test finding complementary tools."""
        # Setup graph
        for tool in mock_tools:
            discovery_agent.tool_graph.add_node(tool['id'], **tool)
        discovery_agent.tool_graph.add_edge(
            'filesystem.search', 'grep.search',
            type='complements', strength=0.9
        )
        
        complements = await discovery_agent.find_complementary_tools(['filesystem.search'])
        
        assert len(complements) > 0
        assert complements[0].tool_id == 'grep.search'
        assert complements[0].relationship_score == 0.9
    
    @pytest.mark.asyncio
    async def test_find_complementary_tools_no_complements(self, discovery_agent):
        """Test finding complementary tools when none exist."""
        discovery_agent.tool_graph.add_node('lonely_tool')
        
        complements = await discovery_agent.find_complementary_tools(['lonely_tool'])
        
        assert len(complements) == 0
    
    @pytest.mark.asyncio
    async def test_discover_tools_by_pattern(self, discovery_agent, mock_tools):
        """Test pattern-based tool discovery."""
        discovery_agent.tool_registry.get_all_tools.return_value = mock_tools
        
        # Mock high similarity for first tool
        discovery_agent.model.encode.side_effect = [
            np.array([1.0] * 384),  # Pattern embedding
            np.array([0.9] * 384),  # First tool (high similarity)
            np.array([0.3] * 384),  # Second tool (low similarity)
            np.array([0.2] * 384),  # Third tool (low similarity)
        ]
        
        candidates = await discovery_agent.discover_tools_by_pattern("file search tools")
        
        assert len(candidates) >= 1
        assert candidates[0].semantic_score > 0.5
    
    @pytest.mark.asyncio
    async def test_get_tool_recommendations(self, discovery_agent, mock_tools):
        """Test getting tool recommendations."""
        # Setup graph
        for tool in mock_tools:
            discovery_agent.tool_graph.add_node(tool['id'], **tool)
        discovery_agent.tool_graph.add_edge(
            'filesystem.search', 'grep.search',
            type='complements', strength=0.8
        )
        
        discovery_agent.tool_registry.get_all_tools.return_value = mock_tools
        
        recommendations = await discovery_agent.get_tool_recommendations(
            'filesystem.search', max_recommendations=3
        )
        
        assert len(recommendations) <= 3
        # Should not include the original tool
        assert all(r.tool_id != 'filesystem.search' for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_initialize_and_close(self, discovery_agent):
        """Test agent initialization and cleanup."""
        await discovery_agent.initialize()
        
        discovery_agent.tool_registry.initialize.assert_called_once()
        
        await discovery_agent.close()
        
        discovery_agent.tool_registry.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_tools_with_context(self, discovery_agent, mock_intent_result, mock_tools):
        """Test tool discovery with context information."""
        discovery_agent.tool_registry.get_all_tools.return_value = mock_tools
        
        # Add graph relationships
        for tool in mock_tools:
            discovery_agent.tool_graph.add_node(tool['id'], **tool)
        
        context = {
            'recent_tools': ['filesystem.search'],
            'domain': 'development'
        }
        
        candidates = await discovery_agent.discover_tools(mock_intent_result, context)
        
        assert len(candidates) > 0
        # Context should influence scoring
        
    @pytest.mark.asyncio
    async def test_score_filtering(self, discovery_agent, mock_intent_result, mock_tools):
        """Test that low-scoring tools are filtered out."""
        discovery_agent.tool_registry.get_all_tools.return_value = mock_tools
        discovery_agent.config['min_score_threshold'] = 0.8  # High threshold
        
        # Mock low scores
        original_score_tool = discovery_agent._score_tool
        async def mock_score_tool(tool, intent, context):
            candidate = await original_score_tool(tool, intent, context)
            candidate.overall_score = 0.2  # Low score
            return candidate
        
        discovery_agent._score_tool = mock_score_tool
        
        candidates = await discovery_agent.discover_tools(mock_intent_result)
        
        assert len(candidates) == 0  # All filtered out due to low scores


if __name__ == '__main__':
    pytest.main([__file__, '-v'])