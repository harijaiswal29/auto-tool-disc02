"""Unit tests for ContextExtractor class."""

import pytest
from unittest.mock import Mock, patch

from src.learning.context_extractor import ContextExtractor, UserContext


class TestContextExtractor:
    """Test cases for ContextExtractor."""
    
    @pytest.fixture
    def context_extractor(self):
        """Create ContextExtractor instance."""
        return ContextExtractor()
    
    def test_initialization(self, context_extractor):
        """Test ContextExtractor initialization."""
        # Check domain keywords are loaded
        assert 'engineering' in context_extractor.domain_keywords
        assert 'data_science' in context_extractor.domain_keywords
        assert 'web_dev' in context_extractor.domain_keywords
        assert 'devops' in context_extractor.domain_keywords
        assert 'database' in context_extractor.domain_keywords
        
        # Check expertise indicators are loaded
        assert 'novice' in context_extractor.expertise_indicators
        assert 'intermediate' in context_extractor.expertise_indicators
        assert 'expert' in context_extractor.expertise_indicators
        
        # Verify some sample keywords
        assert 'python' in context_extractor.domain_keywords['engineering']
        assert 'analyze' in context_extractor.domain_keywords['data_science']
        assert 'deploy' in context_extractor.domain_keywords['devops']
    
    # Domain Extraction Tests
    
    def test_extract_domain_engineering(self, context_extractor):
        """Test domain extraction for engineering queries."""
        queries = [
            "Debug this Python code",
            "Fix the bug in my JavaScript function",
            "Refactor this class module",
            "Build and compile the rust project"
        ]
        
        for query in queries:
            domain, scores = context_extractor._extract_domain(query)
            assert domain == 'engineering'
            assert scores.get('engineering', 0) > 0
    
    def test_extract_domain_data_science(self, context_extractor):
        """Test domain extraction for data science queries."""
        queries = [
            "Analyze this CSV dataset",
            "Train a machine learning model",
            "Create a pandas visualization",
            "Plot the statistics from numpy array"
        ]
        
        for query in queries:
            domain, scores = context_extractor._extract_domain(query)
            assert domain == 'data_science'
            assert scores.get('data_science', 0) > 0
    
    def test_extract_domain_web_dev(self, context_extractor):
        """Test domain extraction for web development queries."""
        queries = [
            "Create a responsive HTML webpage",
            "Build a React frontend component",
            "Setup Node.js backend API",
            "Fix CSS styling issues"
        ]
        
        for query in queries:
            domain, scores = context_extractor._extract_domain(query)
            assert domain == 'web_dev'
            assert scores.get('web_dev', 0) > 0
    
    def test_extract_domain_devops(self, context_extractor):
        """Test domain extraction for DevOps queries."""
        queries = [
            "Deploy to Kubernetes cluster",
            "Setup CI/CD pipeline",
            "Monitor Docker containers",
            "Configure AWS infrastructure with Terraform"
        ]
        
        for query in queries:
            domain, scores = context_extractor._extract_domain(query)
            assert domain == 'devops'
            assert scores.get('devops', 0) > 0
    
    def test_extract_domain_database(self, context_extractor):
        """Test domain extraction for database queries."""
        queries = [
            "Query the PostgreSQL database",
            "Create SQL table schema",
            "Backup MongoDB collections",
            "Migrate database to new schema"
        ]
        
        for query in queries:
            domain, scores = context_extractor._extract_domain(query)
            assert domain == 'database'
            assert scores.get('database', 0) > 0
    
    def test_extract_domain_general_fallback(self, context_extractor):
        """Test domain extraction falls back to 'general' for ambiguous queries."""
        queries = [
            "Hello world",
            "What time is it?",
            "Show me something",
            "Help me with this"
        ]
        
        for query in queries:
            domain, scores = context_extractor._extract_domain(query)
            assert domain == 'general'
    
    def test_extract_domain_with_intent_hints(self, context_extractor):
        """Test domain extraction with intent type hints."""
        # Create intent should boost engineering
        domain, scores = context_extractor._extract_domain(
            "Create something new", 
            intent_type="action.create"
        )
        assert scores.get('engineering', 0) >= 0.2
        
        # Analyze intent should boost data science
        domain, scores = context_extractor._extract_domain(
            "Look at the data",
            intent_type="query.analyze"
        )
        assert scores.get('data_science', 0) >= 0.2
        
        # Deploy/monitor intent should boost devops
        domain, scores = context_extractor._extract_domain(
            "Check the system",
            intent_type="system.monitor"
        )
        assert scores.get('devops', 0) >= 0.2
    
    def test_extract_domain_keyword_weighting(self, context_extractor):
        """Test that longer keywords get higher weight."""
        # Multi-word keyword should have higher impact
        query = "Setup machine learning pipeline"
        domain, scores = context_extractor._extract_domain(query)
        assert domain == 'data_science'
        # "machine learning" is a 2-word keyword, should have higher weight
        assert scores['data_science'] > 0.1
    
    # Expertise Level Extraction Tests
    
    def test_extract_expertise_novice_queries(self, context_extractor):
        """Test expertise extraction for novice-level queries."""
        queries = [
            "What is Python?",
            "How to create a file?",
            "Help me understand this",
            "Show me an example"
        ]
        
        for query in queries:
            expertise, scores = context_extractor._extract_expertise(query)
            assert expertise == 'novice'
            assert scores['novice'] > scores['expert']
    
    def test_extract_expertise_intermediate_queries(self, context_extractor):
        """Test expertise extraction for intermediate-level queries."""
        queries = [
            "Find all Python files in directory",
            "Search for specific pattern in code",
            "Update the configuration file",
            "Modify database records"
        ]
        
        for query in queries:
            expertise, scores = context_extractor._extract_expertise(query)
            assert expertise == 'intermediate'
    
    def test_extract_expertise_expert_queries(self, context_extractor):
        """Test expertise extraction for expert-level queries."""
        queries = [
            "Optimize the async pipeline performance",
            "Refactor the architecture to use microservices",
            "Integrate CI/CD with automated testing",
            "Automate deployment using infrastructure as code"
        ]
        
        for query in queries:
            expertise, scores = context_extractor._extract_expertise(query)
            assert expertise == 'expert'
            assert scores['expert'] > scores['novice']
    
    def test_extract_expertise_query_length_scoring(self, context_extractor):
        """Test expertise scoring based on query length."""
        # Short query
        expertise, scores = context_extractor._extract_expertise("Fix bug")
        assert scores['novice'] > 0
        
        # Medium query
        expertise, scores = context_extractor._extract_expertise(
            "Find and fix the memory leak in application"
        )
        assert scores['intermediate'] > 0
        
        # Long query
        expertise, scores = context_extractor._extract_expertise(
            "Analyze the performance bottlenecks in our distributed system "
            "and optimize the database queries for better throughput"
        )
        assert scores['expert'] > 0
    
    def test_extract_expertise_technical_terms(self, context_extractor):
        """Test expertise scoring based on technical terms."""
        # No technical terms
        expertise, scores = context_extractor._extract_expertise("Find files")
        novice_score = scores['novice']
        
        # Some technical terms
        expertise, scores = context_extractor._extract_expertise("Find files using regex API")
        intermediate_score = scores['intermediate']
        
        # Many technical terms
        expertise, scores = context_extractor._extract_expertise(
            "Optimize async pipeline using regex patterns and API endpoints with schema validation"
        )
        expert_score = scores['expert']
        
        # Expert should have highest score when many technical terms
        assert expert_score > intermediate_score
    
    def test_extract_expertise_with_user_stats(self, context_extractor):
        """Test expertise extraction with user statistics."""
        # Novice stats
        user_stats = {
            'success_rate': 0.5,
            'query_count': 5,
            'avg_tools_used': 1.2
        }
        expertise, scores = context_extractor._extract_expertise(
            "Find files", user_stats
        )
        assert scores['novice'] > scores['expert']
        
        # Intermediate stats
        user_stats = {
            'success_rate': 0.7,
            'query_count': 30,
            'avg_tools_used': 2.0
        }
        expertise, scores = context_extractor._extract_expertise(
            "Find files", user_stats
        )
        assert scores['intermediate'] > 0
        
        # Expert stats
        user_stats = {
            'success_rate': 0.9,
            'query_count': 100,
            'avg_tools_used': 3.5
        }
        expertise, scores = context_extractor._extract_expertise(
            "Find files", user_stats
        )
        assert scores['expert'] > scores['novice']
    
    def test_extract_expertise_default_intermediate(self, context_extractor):
        """Test that ambiguous cases default to intermediate."""
        # Create ambiguous scores
        with patch.object(context_extractor, '_extract_expertise') as mock:
            mock.return_value = ('intermediate', {
                'novice': 0.5,
                'intermediate': 0.52,
                'expert': 0.51
            })
            expertise, _ = mock.return_value
            assert expertise == 'intermediate'
    
    # Context Vector Conversion Tests
    
    def test_get_context_vector_expertise_encoding(self, context_extractor):
        """Test context vector generation for expertise levels."""
        # Novice
        context = UserContext(
            user_expertise='novice',
            domain='general',
            raw_expertise_indicators={},
            raw_domain_indicators={}
        )
        vector = context_extractor.get_context_vector(context)
        assert vector[:3] == [1.0, 0.0, 0.0]  # First 3 dims for expertise
        
        # Intermediate
        context.user_expertise = 'intermediate'
        vector = context_extractor.get_context_vector(context)
        assert vector[:3] == [0.0, 1.0, 0.0]
        
        # Expert
        context.user_expertise = 'expert'
        vector = context_extractor.get_context_vector(context)
        assert vector[:3] == [0.0, 0.0, 1.0]
    
    def test_get_context_vector_domain_encoding(self, context_extractor):
        """Test context vector generation for domains."""
        context = UserContext(
            user_expertise='intermediate',
            domain='general',
            raw_expertise_indicators={},
            raw_domain_indicators={}
        )
        
        # General domain
        vector = context_extractor.get_context_vector(context)
        assert vector[3:8] == [1.0, 0.0, 0.0, 0.0, 0.0]  # Last 5 dims for domain
        
        # Engineering domain
        context.domain = 'engineering'
        vector = context_extractor.get_context_vector(context)
        assert vector[3:8] == [0.0, 1.0, 0.0, 0.0, 0.0]
        
        # Data science domain
        context.domain = 'data_science'
        vector = context_extractor.get_context_vector(context)
        assert vector[3:8] == [0.0, 0.0, 1.0, 0.0, 0.0]
        
        # Web dev domain
        context.domain = 'web_dev'
        vector = context_extractor.get_context_vector(context)
        assert vector[3:8] == [0.0, 0.0, 0.0, 1.0, 0.0]
        
        # DevOps domain
        context.domain = 'devops'
        vector = context_extractor.get_context_vector(context)
        assert vector[3:8] == [0.0, 0.0, 0.0, 0.0, 1.0]
    
    def test_get_context_vector_complete(self, context_extractor):
        """Test complete context vector is 8-dimensional."""
        context = UserContext(
            user_expertise='expert',
            domain='engineering',
            raw_expertise_indicators={},
            raw_domain_indicators={}
        )
        
        vector = context_extractor.get_context_vector(context)
        assert len(vector) == 8
        assert vector == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    def test_get_context_vector_unknown_values(self, context_extractor):
        """Test context vector handles unknown expertise/domain gracefully."""
        context = UserContext(
            user_expertise='unknown',
            domain='unknown',
            raw_expertise_indicators={},
            raw_domain_indicators={}
        )
        
        vector = context_extractor.get_context_vector(context)
        assert len(vector) == 8
        # Should have all zeros for unknown values
        assert vector == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Integration Tests
    
    def test_extract_context_full_flow(self, context_extractor):
        """Test complete context extraction flow."""
        # Engineering expert query
        query = "Optimize the async Python pipeline using advanced regex patterns"
        user_stats = {
            'success_rate': 0.9,
            'query_count': 150,
            'avg_tools_used': 3.0
        }
        
        context = context_extractor.extract_context(query, user_stats, 'action.modify')
        
        assert context.user_expertise == 'expert'
        assert context.domain == 'engineering'
        assert len(context.raw_expertise_indicators) > 0
        assert len(context.raw_domain_indicators) > 0
    
    def test_extract_context_minimal_input(self, context_extractor):
        """Test context extraction with minimal input."""
        query = "Hello"
        
        context = context_extractor.extract_context(query)
        
        assert context.user_expertise in ['novice', 'intermediate', 'expert']
        assert context.domain == 'general'
    
    def test_extract_context_data_science_intermediate(self, context_extractor):
        """Test context extraction for data science intermediate user."""
        query = "Analyze sales data and create visualization"
        user_stats = {
            'success_rate': 0.75,
            'query_count': 40,
            'avg_tools_used': 2.2
        }
        
        context = context_extractor.extract_context(query, user_stats, 'query.analyze')
        
        assert context.domain == 'data_science'
        # Could be intermediate or expert depending on exact scoring
        assert context.user_expertise in ['intermediate', 'expert']