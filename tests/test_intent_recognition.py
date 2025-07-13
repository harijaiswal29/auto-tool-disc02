"""
Unit tests for the Intent Recognition Agent.
"""

import pytest
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.agents.intent_models import (
    Intent,
    IntentResult,
    TextPreprocessor,
    MultiIntentHandler
)
from src.agents.intent_recognition_agent_legacy import (
    IntentClassifier,
    IntentConfidenceScorer
)


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    @pytest.mark.asyncio
    async def test_lowercase(self, preprocessor):
        """Test lowercase conversion."""
        result = preprocessor.lowercase("Hello WORLD")
        assert result == "hello world"
    
    @pytest.mark.asyncio
    async def test_expand_contractions(self, preprocessor):
        """Test contraction expansion."""
        result = preprocessor.expand_contractions("don't can't won't")
        assert result == "do not cannot will not"
    
    @pytest.mark.asyncio
    async def test_remove_special_chars(self, preprocessor):
        """Test special character removal."""
        result = preprocessor.remove_special_chars("Hello@World#123!")
        assert result == "Hello World 123!"
    
    @pytest.mark.asyncio
    async def test_normalize_whitespace(self, preprocessor):
        """Test whitespace normalization."""
        result = preprocessor.normalize_whitespace("Hello    World  ")
        assert result == "Hello World"
    
    @pytest.mark.asyncio
    async def test_full_preprocessing(self, preprocessor):
        """Test complete preprocessing pipeline."""
        result = await preprocessor.process("Don't   forget@to   NORMALIZE!")
        assert result == "do not forget to normalize!"


class TestIntentClassifier:
    """Test intent classification functionality."""
    
    @pytest.fixture
    def classifier(self):
        return IntentClassifier()
    
    def test_extract_keywords(self, classifier):
        """Test keyword extraction."""
        keywords = classifier.extract_keywords("find and search for files")
        assert "find" in keywords
        assert "search" in keywords
    
    def test_classify_search_intent(self, classifier):
        """Test classification of search intent."""
        text = "find all python files"
        classifications = classifier.classify(text, {})
        
        # Should classify as query.search
        assert len(classifications) > 0
        assert classifications[0][0] == "query.search"
        assert classifications[0][1] > 0
    
    def test_classify_create_intent(self, classifier):
        """Test classification of create intent."""
        text = "create a new directory"
        classifications = classifier.classify(text, {})
        
        # Should classify as action.create
        assert len(classifications) > 0
        assert classifications[0][0] == "action.create"
    
    def test_classify_mixed_keywords(self, classifier):
        """Test classification with mixed keywords."""
        text = "find files and create report"
        classifications = classifier.classify(text, {})
        
        # Should have multiple classifications
        intent_types = [c[0] for c in classifications]
        assert "query.search" in intent_types
        assert "action.create" in intent_types


class TestIntentConfidenceScorer:
    """Test confidence scoring functionality."""
    
    @pytest.fixture
    def scorer(self):
        return IntentConfidenceScorer()
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, scorer):
        """Test confidence calculation."""
        features = {
            'semantic_scores': {'query.search': 0.8},
            'keyword_scores': {'query.search': 0.9},
            'context_score': 0.5,
            'pattern_scores': {'query.search': 0.7}
        }
        
        confidence = await scorer.calculate_confidence('query.search', features)
        
        # Should be weighted average
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be relatively high


class TestMultiIntentHandler:
    """Test multi-intent handling functionality."""
    
    @pytest.fixture
    def handler(self):
        return MultiIntentHandler()
    
    @pytest.mark.asyncio
    async def test_detect_multi_intent(self, handler):
        """Test multi-intent detection."""
        assert await handler.detect_multi_intent("find files and create report") == True
        assert await handler.detect_multi_intent("search for python files") == False
    
    @pytest.mark.asyncio
    async def test_split_intents(self, handler):
        """Test intent splitting."""
        segments = await handler.split_intents("find files and create report then analyze results")
        assert len(segments) == 3
        assert "find files" in segments
        assert "create report" in segments
        assert "analyze results" in segments
    
    def test_determine_execution_order(self, handler):
        """Test execution order determination."""
        segments = ["find files", "then create report", "and analyze"]
        execution_plan = handler.determine_execution_order(segments)
        
        assert len(execution_plan) == 3
        assert execution_plan[1][1] == "sequential"  # "then" indicates sequential


class TestIntentRecognitionAgent:
    """Test the main Intent Recognition Agent."""
    
    @pytest.fixture
    def agent(self):
        return IntentRecognitionAgent()
    
    @pytest.mark.asyncio
    async def test_process_simple_query(self, agent):
        """Test processing a simple query."""
        result = await agent.process_query("find all python files")
        
        assert isinstance(result, IntentResult)
        assert result.primary_intent.type == "query.search"
        assert result.confidence_passed  # Should have high confidence
        assert "find" in result.features['keywords']
    
    @pytest.mark.asyncio
    async def test_process_create_query(self, agent):
        """Test processing a create action query."""
        result = await agent.process_query("create a new configuration file")
        
        assert result.primary_intent.type == "action.create"
        assert "create" in result.features['keywords']
    
    @pytest.mark.asyncio
    async def test_process_multi_intent_query(self, agent):
        """Test processing a multi-intent query."""
        result = await agent.process_query("find log files and delete old ones")
        
        # Should have multiple intents
        assert len(result.all_intents) >= 2
        intent_types = [i.type for i in result.all_intents]
        assert any("search" in t for t in intent_types)
        assert any("delete" in t for t in intent_types)
    
    @pytest.mark.asyncio
    async def test_process_with_context(self, agent):
        """Test processing with context."""
        context = {
            'history': ['previous query'],
            'domain': 'engineering',
            'user_profile': {'role': 'developer'}
        }
        
        result = await agent.process_query("analyze the code", context)
        
        assert result.primary_intent.type == "query.analyze"
        assert result.features['context_score'] > 0.5  # Should have higher context score
    
    @pytest.mark.asyncio
    async def test_get_intent_details(self, agent):
        """Test getting intent details."""
        details = await agent.get_intent_details("query.search")
        
        assert details['type'] == "query.search"
        assert details['category'] == "query"
        assert details['subcategory'] == "search"
        assert len(details['keywords']) > 0
        assert len(details['example_patterns']) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, agent):
        """Test that embeddings are cached."""
        # Process same query twice
        query = "find python files"
        
        result1 = await agent.process_query(query)
        time1 = result1.processing_time_ms
        
        result2 = await agent.process_query(query)
        time2 = result2.processing_time_ms
        
        # Second query should be faster due to caching
        assert time2 < time1
        assert query in agent.embedding_cache
    
    @pytest.mark.asyncio
    async def test_low_confidence_fallback(self, agent):
        """Test fallback behavior for low confidence queries."""
        # Use a gibberish query
        result = await agent.process_query("xyzabc qwerty")
        
        assert result.primary_intent is not None
        # Should have low confidence (not meeting threshold)
        assert result.primary_intent.confidence < 0.7
        assert not result.confidence_passed  # Should not pass threshold


@pytest.mark.asyncio
async def test_end_to_end_scenarios():
    """Test end-to-end scenarios."""
    agent = IntentRecognitionAgent()
    
    # Test various real-world queries
    test_cases = [
        ("Show me all test files in the project", "query.retrieve"),
        ("Delete temporary files older than 7 days", "action.delete"),
        ("Monitor system performance and alert on high CPU", "system.monitor"),
        ("Setup development environment", "system.configure"),
        ("Analyze code quality and generate report", "query.analyze")
    ]
    
    for query, expected_category in test_cases:
        result = await agent.process_query(query)
        # Check if the expected category is part of the intent type
        assert expected_category in result.primary_intent.type or result.primary_intent.type.startswith(expected_category.split('.')[0])
        assert result.processing_time_ms < 1000  # Should be fast


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])