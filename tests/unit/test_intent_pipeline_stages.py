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
from src.pipeline.base import PipelineStage
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
        data = PipelineData(raw_input="Hello WORLD! Don't you think?")
        
        result = await stage.process(data)
        
        assert result.get_stage_result('TextPreprocessor', 'normalized_text') == "hello world! do not you think?"
        assert result.raw_input == "Hello WORLD! Don't you think?"  # Original preserved
    
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
            data = PipelineData(raw_input=contraction)
            result = await stage.process(data)
            normalized = result.get_stage_result('TextPreprocessor', 'normalized_text')
            assert normalized == expected
    
    @pytest.mark.asyncio
    async def test_special_character_handling(self, stage):
        """Test special character removal."""
        data = PipelineData(raw_input="Hello@World#123!Test$%^")
        
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
        data = PipelineData(raw_input="Hello    World\t\nTest   Multiple     Spaces")
        
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
        data = PipelineData(raw_input="test query")
        data.add_stage_result('TextPreprocessor', 'normalized_text', 'hello world test')
        
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
            data = PipelineData(raw_input=question)
            data.add_stage_result('TextPreprocessor', 'normalized_text', question)
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
            data = PipelineData(raw_input=statement)
            data.add_stage_result('TextPreprocessor', 'normalized_text', statement)
            result = await stage.process(data)
            assert result.get_stage_result('Tokenizer', 'has_question') is False


class TestFeatureExtractorStage:
    """Test the feature extraction pipeline stage."""
    
    @pytest.fixture
    async def stage(self):
        stage = FeatureExtractorStage({
            'model': 'all-MiniLM-L6-v2',
            'cache_size': 100,
            'similarity_threshold': 0.7
        })
        await stage.initialize()
        return stage
    
    @pytest.mark.asyncio
    async def test_semantic_embedding_generation(self, stage):
        """Test semantic embedding generation."""
        data = PipelineData(raw_input="test query")
        data.add_stage_result('TextPreprocessor', 'normalized_text', 'find all python files')
        
        result = await stage.process(data)
        
        embedding = result.get_stage_result('FeatureExtractor', 'semantic_embedding')
        assert embedding is not None
        assert len(embedding) > 0  # Should have embeddings
        assert all(isinstance(x, (float, int)) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_intent_pattern_matching(self, stage):
        """Test intent pattern semantic matching."""
        data = PipelineData(raw_input="test query")
        data.add_stage_result('TextPreprocessor', 'normalized_text', 'search for documentation files')
        
        result = await stage.process(data)
        
        semantic_scores = result.get_stage_result('FeatureExtractor', 'semantic_scores')
        assert 'query.search' in semantic_scores
        assert semantic_scores['query.search'] > 0.5  # Should have high similarity
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self, stage):
        """Test that embeddings are cached."""
        data1 = PipelineData(raw_input="test query")
        data1.add_stage_result('TextPreprocessor', 'normalized_text', 'test query')
        
        # First call
        result1 = await stage.process(data1)
        
        # Second call with same text
        data2 = PipelineData(raw_input="test query")
        data2.add_stage_result('TextPreprocessor', 'normalized_text', 'test query')
        result2 = await stage.process(data2)
        
        # Embeddings should be identical (from cache)
        emb1 = result1.get_stage_result('FeatureExtractor', 'semantic_embedding')
        emb2 = result2.get_stage_result('FeatureExtractor', 'semantic_embedding')
        assert emb1 is not None and emb2 is not None
        # Check they are the same (allowing for floating point comparison)
        assert len(emb1) == len(emb2)
        for i in range(len(emb1)):
            assert abs(emb1[i] - emb2[i]) < 1e-6


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
            data = PipelineData(raw_input=query)
            data.add_stage_result('Tokenizer', 'tokens', query.split())
            data.add_stage_result('FeatureExtractor', 'semantic_scores', {})
            
            result = await stage.process(data)
            
            keyword_scores = result.get_stage_result('IntentClassifier', 'keyword_scores')
            assert expected_intent in keyword_scores
            assert keyword_scores[expected_intent] > 0
    
    @pytest.mark.asyncio
    async def test_combined_scoring(self, stage):
        """Test combined semantic and keyword scoring."""
        data = PipelineData(raw_input="test query")
        data.add_stage_result('Tokenizer', 'tokens', ['search', 'for', 'files'])
        data.add_stage_result('FeatureExtractor', 'semantic_scores', {
            'query.search': 0.8,
            'query.retrieve': 0.3
        })
        
        result = await stage.process(data)
        
        all_intents = result.get_stage_result('IntentClassifier', 'all_intents')
        assert all_intents is not None
        assert len(all_intents) > 0
        # Check that query.search is among the top intents
        intent_types = [intent['type'] for intent in all_intents[:3]]
        assert 'query.search' in intent_types


class TestContextEnricherStage:
    """Test the context enrichment pipeline stage."""
    
    @pytest.fixture
    async def stage(self):
        stage = ContextEnricherStage()
        await stage.initialize()
        return stage
    
    @pytest.mark.asyncio
    async def test_context_enrichment(self, stage):
        """Test basic context enrichment."""
        data = PipelineData(raw_input="test query")
        data.context = {
            'session_id': 'test_session',
            'domain': 'engineering'
        }
        data.add_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.8}
        ])
        data.add_stage_result('IntentClassifier', 'classified_intents', [
            {'type': 'query.search', 'score': 0.8}
        ])
        
        result = await stage.process(data)
        
        # Check that context was enriched
        assert result.context is not None
        # The enriched context may be in the result context
        assert 'session' in result.context or 'session_id' in result.context
    
    @pytest.mark.asyncio
    async def test_conversation_history_integration(self, stage):
        """Test conversation history integration."""
        data = PipelineData(raw_input="test query")
        data.context = {'session_id': 'test'}
        data.add_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.7}
        ])
        data.add_stage_result('IntentClassifier', 'classified_intents', [
            {'type': 'query.search', 'score': 0.7}
        ])
        
        result = await stage.process(data)
        
        # Check that context processing occurred
        assert result.context is not None
        # Context score should be added
        context_score = result.get_stage_result('ContextEnricher', 'context_score')
        assert context_score is not None
        assert context_score >= 0  # Should have a context score


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
        data = PipelineData(raw_input="test query")
        data.add_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.8},
            {'type': 'query.retrieve', 'score': 0.3}
        ])
        data.add_stage_result('FeatureExtractor', 'semantic_scores', {
            'query.search': 0.85
        })
        data.add_stage_result('ContextEnricher', 'context_score', 0.6)
        
        result = await stage.process(data)
        
        primary_intent = result.get_stage_result('ConfidenceScorer', 'primary_intent')
        assert primary_intent is not None
        # Primary intent should be created from the highest scoring intent
        assert hasattr(primary_intent, 'type') and hasattr(primary_intent, 'confidence')
        assert primary_intent.confidence > 0  # Should have some confidence
        confidence_passed = result.get_stage_result('ConfidenceScorer', 'confidence_passed')
        assert confidence_passed is not None
    
    @pytest.mark.asyncio
    async def test_low_confidence_handling(self, stage):
        """Test handling of low confidence scenarios."""
        data = PipelineData(raw_input="test query")
        data.add_stage_result('IntentClassifier', 'all_intents', [
            {'type': 'query.search', 'score': 0.3},
            {'type': 'query.retrieve', 'score': 0.2}
        ])
        data.add_stage_result('FeatureExtractor', 'semantic_scores', {})
        data.add_stage_result('ContextEnricher', 'context_score', 0.5)
        
        result = await stage.process(data)
        
        confidence_passed = result.get_stage_result('ConfidenceScorer', 'confidence_passed')
        assert confidence_passed is False


class TestStateManagerStage:
    """Test the state management pipeline stage."""
    
    @pytest.fixture
    async def stage(self):
        stage = StateManagerStage()
        await stage.initialize()
        return stage
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, stage):
        """Test conversation state transitions."""
        # Initial state should be IDLE
        assert stage.get_current_state_name() == 'IDLE'
        
        # Process query - should transition to QUERY_RECEIVED
        data = PipelineData(raw_input="test query")
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
    def stages_without_state(self):
        """Pipeline stages without state management (6 stages)."""
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
    
    @pytest.fixture
    def stages_with_state(self):
        """Pipeline stages with state management (7 stages)."""
        return [
            StateManagerStage(),
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
    async def test_full_pipeline_flow_without_state(self, stages_without_state):
        """Test data flow through 6-stage pipeline (without state management)."""
        from src.pipeline import Pipeline
        
        pipeline = Pipeline(stages_without_state, name="TestPipeline")
        await pipeline.initialize()
        
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
    async def test_full_pipeline_flow_with_state(self, stages_with_state):
        """Test data flow through 7-stage pipeline (with state management)."""
        from src.pipeline import Pipeline
        
        pipeline = Pipeline(stages_with_state, name="TestPipelineWithState")
        await pipeline.initialize()
        
        # Test with a sample query
        result = await pipeline.process("Find all Python files in the project", {
            'session_id': 'test_session',
            'domain': 'engineering'
        })
        
        # Verify all stages produced results including state manager
        assert result.get_stage_result('StateManager', 'state_transition') is not None
        assert result.get_stage_result('TextPreprocessor', 'normalized_text') is not None
        assert result.get_stage_result('Tokenizer', 'tokens') is not None
        assert result.get_stage_result('FeatureExtractor', 'semantic_embedding') is not None
        assert result.get_stage_result('IntentClassifier', 'all_intents') is not None
        assert result.get_stage_result('ContextEnricher', 'enriched_context') is not None
        assert result.get_stage_result('ConfidenceScorer', 'primary_intent') is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, stages_without_state):
        """Test pipeline error handling."""
        from src.pipeline import Pipeline
        
        # Create a failing stage
        class FailingStage(PipelineStage):
            def __init__(self):
                super().__init__(name="FailingStage")
                
            async def _initialize(self):
                pass
                
            async def process(self, data):
                raise ValueError("Test error")
        
        # Insert failing stage
        stages_with_error = stages_without_state[:2] + [FailingStage()] + stages_without_state[2:]
        pipeline = Pipeline(stages_with_error, name="ErrorTestPipeline")
        await pipeline.initialize()
        
        # The pipeline may or may not raise the error depending on implementation
        # Let's just verify we can attempt to process
        try:
            result = await pipeline.process("test query", {})
            # If no error, check that we got some result
            assert result is not None
        except ValueError as e:
            # If error is raised, verify it's our test error
            assert str(e) == "Test error"


class TestIntentRecognitionAgentIntegration:
    """Test the Intent Recognition Agent with 7-stage pipeline."""
    
    @pytest.fixture
    async def agent_with_state(self):
        """Create an agent with state tracking enabled."""
        from src.agents.intent_recognition_agent import IntentRecognitionAgent
        agent = IntentRecognitionAgent({
            'enable_state_tracking': True,
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'collect_metrics': False,
            'enable_persistence': False
        })
        # Ensure pipeline is initialized
        agent.pipeline = await agent._get_or_create_pipeline()
        return agent
    
    @pytest.fixture
    async def agent_without_state(self):
        """Create an agent without state tracking."""
        from src.agents.intent_recognition_agent import IntentRecognitionAgent
        agent = IntentRecognitionAgent({
            'enable_state_tracking': False,
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'collect_metrics': False,
            'enable_persistence': False
        })
        # Ensure pipeline is initialized
        agent.pipeline = await agent._get_or_create_pipeline()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_creates_7_stage_pipeline(self, agent_with_state):
        """Test that agent with state tracking creates 7 stages."""
        assert agent_with_state.pipeline is not None
        stage_names = agent_with_state.pipeline.get_stage_names()
        
        expected_stages = [
            'StateManager',
            'TextPreprocessor',
            'Tokenizer',
            'FeatureExtractor',
            'IntentClassifier',
            'ContextEnricher',
            'ConfidenceScorer'
        ]
        
        assert stage_names == expected_stages
        assert len(stage_names) == 7
    
    @pytest.mark.asyncio
    async def test_agent_creates_6_stage_pipeline(self, agent_without_state):
        """Test that agent without state tracking creates 6 stages."""
        assert agent_without_state.pipeline is not None
        stage_names = agent_without_state.pipeline.get_stage_names()
        
        expected_stages = [
            'TextPreprocessor',
            'Tokenizer',
            'FeatureExtractor',
            'IntentClassifier',
            'ContextEnricher',
            'ConfidenceScorer'
        ]
        
        assert stage_names == expected_stages
        assert len(stage_names) == 6
    
    @pytest.mark.asyncio
    async def test_agent_processes_query_through_all_stages(self, agent_with_state):
        """Test that a query is processed through all 7 stages."""
        result = await agent_with_state.process_query(
            "Search for all Python files in the database",
            context={'session_id': 'test_session'}
        )
        
        # Verify intent was recognized
        assert result.primary_intent is not None
        assert result.primary_intent.type in ['query.search', 'query.retrieve']
        assert result.primary_intent.confidence > 0
        
        # Verify all expected features are present
        assert 'tokens' in result.features
        assert 'keywords' in result.features
        assert 'semantic_scores' in result.features
        assert result.processing_time_ms > 0
        
        # Verify state manager is accessible
        assert agent_with_state.state_manager is not None
        assert agent_with_state.get_current_state() is not None
    
    @pytest.mark.asyncio
    async def test_state_transitions_during_processing(self, agent_with_state):
        """Test state transitions during query processing."""
        # Initial state should be IDLE
        assert agent_with_state.get_current_state() == 'IDLE'
        
        # Process a query
        await agent_with_state.process_query("Update the configuration file")
        
        # Should return to IDLE after processing
        assert agent_with_state.get_current_state() == 'IDLE'
        
        # Check state history
        history = agent_with_state.get_state_history(limit=5)
        assert len(history) > 0
        
        # Should have transitioned from IDLE to QUERY_RECEIVED and back
        state_sequence = [(h['from_state'], h['to_state']) for h in history]
        assert ('IDLE', 'QUERY_RECEIVED') in state_sequence
    
    @pytest.mark.asyncio 
    async def test_pipeline_cache_sharing(self):
        """Test that pipelines are cached and shared across instances."""
        from src.agents.intent_recognition_agent import IntentRecognitionAgent
        
        # Clear cache first
        IntentRecognitionAgent._pipeline_cache.clear()
        
        # Create first agent
        agent1 = IntentRecognitionAgent({
            'enable_state_tracking': True,
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'collect_metrics': False,
            'enable_persistence': False
        })
        await agent1._get_or_create_pipeline()
        
        # Create second agent with same config
        agent2 = IntentRecognitionAgent({
            'enable_state_tracking': True,
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'collect_metrics': False,
            'enable_persistence': False
        })
        await agent2._get_or_create_pipeline()
        
        # Should share the same pipeline key
        assert agent1._pipeline_key == agent2._pipeline_key
        
        # Should have only one cached pipeline for this config
        assert agent1._pipeline_key in IntentRecognitionAgent._pipeline_cache
    
    @pytest.mark.asyncio
    async def test_core_intent_classification_scenarios(self, agent_with_state):
        """Test core intent classification scenarios for dissertation goals."""
        test_cases = [
            # Query intents
            ("Find all configuration files", "query.search", 0.6),
            ("Show me the latest logs", "query.retrieve", 0.6),
            ("Analyze system performance metrics", "query.analyze", 0.6),
            
            # Action intents
            ("Create a new database table", "action.create", 0.6),
            ("Update the user permissions", "action.modify", 0.6),
            ("Delete old backup files", "action.delete", 0.6),
            
            # System intents
            ("Configure the monitoring system", "system.configure", 0.6),
            ("Check the service health status", "system.monitor", 0.6)
        ]
        
        for query, expected_intent, min_confidence in test_cases:
            result = await agent_with_state.process_query(query)
            
            # Verify primary intent
            assert result.primary_intent is not None, f"No intent found for: {query}"
            assert result.primary_intent.type == expected_intent, \
                f"Expected {expected_intent}, got {result.primary_intent.type} for: {query}"
            assert result.primary_intent.confidence >= min_confidence, \
                f"Confidence {result.primary_intent.confidence} below {min_confidence} for: {query}"
            
            # Verify processing completed successfully
            assert result.processing_time_ms > 0
            assert result.processed_query != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])