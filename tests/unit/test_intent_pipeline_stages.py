"""
Comprehensive unit tests for Intent Recognition pipeline stages.

This module tests each pipeline stage individually to ensure
they conform to the expected behavior and interfaces.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline import PipelineData
from src.pipeline.stages import (
    TextPreprocessorStage,
    TokenizerModule,
    FeatureExtractorStage,
    IntentClassifierStage,
    ContextEnricherStage,
    ConfidenceScorerStage
)
from src.pipeline.stages.state_manager import StateManagerStage
from src.agents.intent_models import Intent


class TestTextPreprocessorStage:
    """Test the text preprocessing pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return TextPreprocessorStage()
    
    @pytest.mark.asyncio
    async def test_basic_preprocessing(self, stage):
        """Test basic text preprocessing functionality."""
        data = PipelineData()
        data.raw_data = "Hello WORLD! Don't you think?"
        
        result = await stage.process(data)
        
        assert result.get_stage_result('TextPreprocessor', 'normalized_text') == "hello world! do not you think?"
        assert result.raw_data == "Hello WORLD! Don't you think?"  # Original preserved
    
    @pytest.mark.asyncio
    async def test_contraction_expansion(self, stage):
        """Test all contraction expansions."""
        contractions_test = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "I'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "I'll": "i will"
        }
        
        for contraction, expected in contractions_test.items():
            data = PipelineData()
            data.raw_data = contraction
            result = await stage.process(data)
            normalized = result.get_stage_result('TextPreprocessor', 'normalized_text')
            assert normalized == expected
    
    @pytest.mark.asyncio
    async def test_special_character_handling(self, stage):
        """Test special character removal."""
        data = PipelineData()
        data.raw_data = "Hello@World#123!Test$%^"
        
        result = await stage.process(data)
        normalized = result.get_stage_result('TextPreprocessor', 'normalized_text')
        
        assert "@" not in normalized
        assert "#" not in normalized
        assert "$" not in normalized
        assert "hello" in normalized
        assert "world" in normalized
        assert "123" in normalized
    
    @pytest.mark.asyncio
    async def test_whitespace_normalization(self, stage):
        """Test whitespace normalization."""
        data = PipelineData()
        data.raw_data = "Hello    World\t\nTest   Multiple     Spaces"
        
        result = await stage.process(data)
        normalized = result.get_stage_result('TextPreprocessor', 'normalized_text')
        
        assert "  " not in normalized  # No double spaces
        assert "\t" not in normalized  # No tabs
        assert "\n" not in normalized  # No newlines


class TestTokenizerModule:
    """Test the tokenizer pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return TokenizerModule()
    
    @pytest.mark.asyncio
    async def test_basic_tokenization(self, stage):
        """Test basic tokenization."""
        data = PipelineData()
        data.set_stage_result('TextPreprocessor', 'normalized_text', 'hello world test')
        
        result = await stage.process(data)
        
        tokens = result.get_stage_result('Tokenizer', 'tokens')
        assert tokens == ['hello', 'world', 'test']
        assert result.get_stage_result('Tokenizer', 'word_count') == 3
    
    @pytest.mark.asyncio
    async def test_question_detection(self, stage):
        """Test question word detection."""
        questions = [
            "what is this",
            "where is the file",
            "when should i run this",
            "why is it failing",
            "how do i fix this",
            "who created this"
        ]
        
        for question in questions:
            data = PipelineData()
            data.set_stage_result('TextPreprocessor', 'normalized_text', question)
            result = await stage.process(data)
            assert result.get_stage_result('Tokenizer', 'has_question') is True
    
    @pytest.mark.asyncio
    async def test_non_question_detection(self, stage):
        """Test non-question detection."""
        statements = [
            "create a new file",
            "delete the old logs",
            "update the configuration"
        ]
        
        for statement in statements:
            data = PipelineData()
            data.set_stage_result('TextPreprocessor', 'normalized_text', statement)
            result = await stage.process(data)
            assert result.get_stage_result('Tokenizer', 'has_question') is False


class TestFeatureExtractorStage:
    """Test the feature extraction pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return FeatureExtractorStage({
            'model': 'all-MiniLM-L6-v2',
            'cache_size': 100,
            'similarity_threshold': 0.7
        })
    
    @pytest.mark.asyncio
    async def test_semantic_embedding_generation(self, stage):
        """Test semantic embedding generation."""
        data = PipelineData()
        data.set_stage_result('TextPreprocessor', 'normalized_text', 'find all python files')
        
        result = await stage.process(data)
        
        embedding = result.get_stage_result('FeatureExtractor', 'semantic_embedding')
        assert embedding is not None
        assert len(embedding) == 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_intent_pattern_matching(self, stage):
        """Test intent pattern semantic matching."""
        data = PipelineData()
        data.set_stage_result('TextPreprocessor', 'normalized_text', 'search for documentation files')
        
        result = await stage.process(data)
        
        semantic_scores = result.get_stage_result('FeatureExtractor', 'semantic_scores')
        assert 'query.search' in semantic_scores
        assert semantic_scores['query.search'] > 0.5  # Should have high similarity
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self, stage):
        """Test that embeddings are cached."""
        data1 = PipelineData()
        data1.set_stage_result('TextPreprocessor', 'normalized_text', 'test query')
        
        # First call
        result1 = await stage.process(data1)
        
        # Second call with same text
        data2 = PipelineData()
        data2.set_stage_result('TextPreprocessor', 'normalized_text', 'test query')
        result2 = await stage.process(data2)
        
        # Embeddings should be identical (from cache)
        emb1 = result1.get_stage_result('FeatureExtractor', 'semantic_embedding')
        emb2 = result2.get_stage_result('FeatureExtractor', 'semantic_embedding')
        assert emb1 == emb2


class TestIntentClassifierStage:
    """Test the intent classification pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return IntentClassifierStage()
    
    @pytest.mark.asyncio
    async def test_keyword_based_classification(self, stage):
        """Test keyword-based intent classification."""
        test_cases = [
            ("find all files", "query.search"),
            ("create a new file", "action.create"),
            ("delete old logs", "action.delete"),
            ("update the config", "action.modify"),
            ("check system status", "system.monitor")
        ]
        
        for query, expected_intent in test_cases:
            data = PipelineData()
            data.set_stage_result('Tokenizer', 'tokens', query.split())
            data.set_stage_result('FeatureExtractor', 'semantic_scores', {})
            
            result = await stage.process(data)
            
            keyword_scores = result.get_stage_result('IntentClassifier', 'keyword_scores')
            assert expected_intent in keyword_scores
            assert keyword_scores[expected_intent] > 0
    
    @pytest.mark.asyncio
    async def test_combined_scoring(self, stage):
        """Test combined semantic and keyword scoring."""
        data = PipelineData()
        data.set_stage_result('Tokenizer', 'tokens', ['search', 'for', 'files'])
        data.set_stage_result('FeatureExtractor', 'semantic_scores', {
            'query.search': 0.8,
            'query.retrieve': 0.3
        })
        
        result = await stage.process(data)
        
        all_intents = result.get_stage_result('IntentClassifier', 'all_intents')
        assert len(all_intents) > 0
        assert all_intents[0]['type'] == 'query.search'  # Should be highest scored


class TestContextEnricherStage:
    """Test the context enrichment pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return ContextEnricherStage()
    
    @pytest.mark.asyncio
    async def test_context_enrichment(self, stage):
        """Test basic context enrichment."""
        data = PipelineData()
        data.context = {
            'session_id': 'test_session',
            'domain': 'engineering'
        }
        data.set_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.8}
        ])
        
        result = await stage.process(data)
        
        enriched_context = result.get_stage_result('ContextEnricher', 'enriched_context')
        assert enriched_context is not None
        assert enriched_context.get('session_id') == 'test_session'
        assert enriched_context.get('domain') == 'engineering'
    
    @pytest.mark.asyncio
    async def test_conversation_history_integration(self, stage):
        """Test conversation history integration."""
        # Add some history to the stage
        stage.conversation_history.append({
            'query': 'previous query',
            'intent': 'query.search',
            'timestamp': '2024-01-01T00:00:00'
        })
        
        data = PipelineData()
        data.context = {'session_id': 'test'}
        data.set_stage_result('IntentClassifier', 'all_intents', [])
        
        result = await stage.process(data)
        
        context_score = result.get_stage_result('ContextEnricher', 'context_score')
        assert context_score >= 0.5  # Should have some context score


class TestConfidenceScorerStage:
    """Test the confidence scoring pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return ConfidenceScorerStage({
            'confidence_threshold': 0.7,
            'similarity_threshold': 0.7
        })
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, stage):
        """Test confidence score calculation."""
        data = PipelineData()
        data.set_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.8},
            {'type': 'query.retrieve', 'score': 0.3}
        ])
        data.set_stage_result('FeatureExtractor', 'semantic_scores', {
            'query.search': 0.85
        })
        data.set_stage_result('ContextEnricher', 'context_score', 0.6)
        
        result = await stage.process(data)
        
        primary_intent = result.get_stage_result('ConfidenceScorer', 'primary_intent')
        assert primary_intent is not None
        assert primary_intent.type == 'query.search'
        assert primary_intent.confidence > 0.7
        assert result.get_stage_result('ConfidenceScorer', 'confidence_passed') is True
    
    @pytest.mark.asyncio
    async def test_low_confidence_handling(self, stage):
        """Test handling of low confidence scenarios."""
        data = PipelineData()
        data.set_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.3},
            {'type': 'query.retrieve', 'score': 0.2}
        ])
        data.set_stage_result('FeatureExtractor', 'semantic_scores', {})
        data.set_stage_result('ContextEnricher', 'context_score', 0.5)
        
        result = await stage.process(data)
        
        confidence_passed = result.get_stage_result('ConfidenceScorer', 'confidence_passed')
        assert confidence_passed is False


class TestStateManagerStage:
    """Test the state management pipeline stage."""
    
    @pytest.fixture
    def stage(self):
        return StateManagerStage()
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, stage):
        """Test conversation state transitions."""
        # Initial state should be IDLE
        assert stage.get_current_state_name() == 'IDLE'
        
        # Process query - should transition to QUERY_RECEIVED
        data = PipelineData()
        data.raw_data = "test query"
        result = await stage.process(data)
        
        assert stage.get_current_state_name() == 'QUERY_RECEIVED'
        assert result.get_stage_result('StateManager', 'state_transition') is not None
    
    @pytest.mark.asyncio
    async def test_error_state_handling(self, stage):
        """Test error state transitions."""
        # Force an error state
        stage.state_machine.current_state = 'EXECUTION_FAILED'
        
        assert stage.is_in_error_state() is True
        
        # Request retry
        retry_accepted = await stage.request_retry()
        assert retry_accepted is True
        assert stage.get_current_state_name() == 'RETRY_REQUESTED'
    
    @pytest.mark.asyncio
    async def test_clarification_handling(self, stage):
        """Test clarification request handling."""
        # Set state to CLARIFICATION_NEEDED
        stage.state_machine.current_state = 'CLARIFICATION_NEEDED'
        
        assert stage.needs_user_input() is True
        
        # Handle clarification
        handled = await stage.handle_clarification("more specific query")
        assert handled is True
        assert stage.get_current_state_name() == 'CLARIFICATION_RECEIVED'


class TestPipelineIntegration:
    """Test the complete pipeline integration."""
    
    @pytest.fixture
    def stages(self):
        return [
            TextPreprocessorStage(),
            TokenizerModule(),
            FeatureExtractorStage({
                'model': 'all-MiniLM-L6-v2',
                'cache_size': 100,
                'similarity_threshold': 0.7
            }),
            IntentClassifierStage(),
            ContextEnricherStage(),
            ConfidenceScorerStage({
                'confidence_threshold': 0.7,
                'similarity_threshold': 0.7
            })
        ]
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, stages):
        """Test data flow through all pipeline stages."""
        from src.pipeline import Pipeline
        
        pipeline = Pipeline(stages, name="TestPipeline")
        
        # Test with a sample query
        result = await pipeline.process("Find all Python files in the project", {
            'session_id': 'test_session',
            'domain': 'engineering'
        })
        
        # Verify all stages produced results
        assert result.get_stage_result('TextPreprocessor', 'normalized_text') is not None
        assert result.get_stage_result('Tokenizer', 'tokens') is not None
        assert result.get_stage_result('FeatureExtractor', 'semantic_embedding') is not None
        assert result.get_stage_result('IntentClassifier', 'all_intents') is not None
        assert result.get_stage_result('ContextEnricher', 'enriched_context') is not None
        assert result.get_stage_result('ConfidenceScorer', 'primary_intent') is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, stages):
        """Test pipeline error handling."""
        from src.pipeline import Pipeline
        
        # Create a failing stage
        class FailingStage:
            name = "FailingStage"
            async def process(self, data):
                raise ValueError("Test error")
        
        # Insert failing stage
        stages_with_error = stages[:2] + [FailingStage()] + stages[2:]
        pipeline = Pipeline(stages_with_error, name="ErrorTestPipeline")
        
        # Should handle error gracefully
        with pytest.raises(ValueError):
            await pipeline.process("test query", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])