"""
Tests for the refactored pipeline architecture.

This module tests the pipeline infrastructure and individual stages.
"""

import asyncio
import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import Pipeline, PipelineData, PipelineStage
from src.pipeline.stages import (
    TextPreprocessorStage,
    TokenizerModule,
    FeatureExtractorStage,
    IntentClassifierStage,
    ContextEnricherStage,
    ConfidenceScorerStage
)
from src.agents.intent_recognition_agent_v2 import IntentRecognitionAgentV2


class TestPipelineInfrastructure:
    """Test the base pipeline infrastructure."""
    
    @pytest.mark.asyncio
    async def test_pipeline_data(self):
        """Test PipelineData container functionality."""
        data = PipelineData(raw_input="test query")
        
        # Test adding stage results
        data.add_stage_result("Stage1", "key1", "value1")
        assert data.get_stage_result("Stage1", "key1") == "value1"
        
        # Test metadata
        data.add_metadata("meta_key", "meta_value")
        assert data.metadata["meta_key"] == "meta_value"
        
        # Test timestamps
        data.record_timestamp("Stage1", 100.5)
        assert data.timestamps["Stage1"] == 100.5
    
    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        """Test a simple pipeline with mock stages."""
        
        class MockStage(PipelineStage):
            async def _initialize(self):
                pass
            
            async def process(self, data: PipelineData) -> PipelineData:
                data.add_stage_result(self.name, "processed", True)
                return data
        
        # Create pipeline with mock stages
        stages = [MockStage("Stage1"), MockStage("Stage2")]
        pipeline = Pipeline(stages, "TestPipeline")
        
        # Process data
        result = await pipeline.process("test input")
        
        assert result.raw_input == "test input"
        assert result.get_stage_result("Stage1", "processed") is True
        assert result.get_stage_result("Stage2", "processed") is True
        assert "pipeline_total_time_ms" in result.metadata


class TestPipelineStages:
    """Test individual pipeline stages."""
    
    @pytest.mark.asyncio
    async def test_text_preprocessor(self):
        """Test TextPreprocessorStage."""
        stage = TextPreprocessorStage()
        await stage.initialize()
        
        data = PipelineData(raw_input="Don't you think it's GREAT?")
        result = await stage.process(data)
        
        normalized = result.get_stage_result("TextPreprocessor", "normalized_text")
        assert normalized == "do not you think it is great ?"
        assert result.metadata["normalized_query"] == normalized
    
    @pytest.mark.asyncio
    async def test_tokenizer(self):
        """Test TokenizerModule."""
        stage = TokenizerModule()
        await stage.initialize()
        
        # Prepare input with preprocessed text
        data = PipelineData(raw_input="find python files")
        data.add_stage_result("TextPreprocessor", "normalized_text", "find python files")
        
        result = await stage.process(data)
        
        tokens = result.get_stage_result("Tokenizer", "tokens")
        assert tokens == ["find", "python", "files"]
        assert result.get_stage_result("Tokenizer", "word_count") == 3
        assert result.get_stage_result("Tokenizer", "has_question") is False
    
    @pytest.mark.asyncio
    async def test_feature_extractor(self):
        """Test FeatureExtractorStage."""
        stage = FeatureExtractorStage({'model': 'all-MiniLM-L6-v2'})
        await stage.initialize()
        
        # Prepare input
        data = PipelineData(raw_input="analyze the system performance")
        data.add_stage_result("TextPreprocessor", "normalized_text", "analyze the system performance")
        data.add_stage_result("Tokenizer", "tokens", ["analyze", "the", "system", "performance"])
        data.add_stage_result("Tokenizer", "has_question", False)
        data.add_stage_result("Tokenizer", "word_count", 4)
        
        result = await stage.process(data)
        
        features = result.get_stage_result("FeatureExtractor", "features")
        assert "embedding" in features
        assert "semantic_scores" in features
        assert len(features["semantic_scores"]) > 0
    
    @pytest.mark.asyncio
    async def test_intent_classifier(self):
        """Test IntentClassifierStage."""
        stage = IntentClassifierStage()
        await stage.initialize()
        
        # Prepare input
        data = PipelineData(raw_input="create a new file")
        data.add_stage_result("TextPreprocessor", "normalized_text", "create a new file")
        data.add_stage_result("Tokenizer", "tokens", ["create", "a", "new", "file"])
        data.add_stage_result("Tokenizer", "content_tokens", ["create", "new", "file"])
        data.add_stage_result("FeatureExtractor", "semantic_scores", {
            "action.create": 0.8,
            "query.search": 0.3
        })
        
        result = await stage.process(data)
        
        keywords = result.get_stage_result("IntentClassifier", "keywords")
        assert "create" in keywords
        assert "new" in keywords
        
        intents = result.get_stage_result("IntentClassifier", "classified_intents")
        assert len(intents) > 0
        assert intents[0].type == "action.create"


class TestIntentRecognitionAgentV2:
    """Test the refactored Intent Recognition Agent."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization with pipeline."""
        agent = IntentRecognitionAgentV2()
        
        # Check pipeline is created
        assert agent.pipeline is not None
        assert len(agent.stages) == 6  # All stages present
        
        # Check pipeline info
        info = agent.get_pipeline_info()
        assert info["pipeline_name"] == "IntentRecognitionPipeline"
        assert len(info["stages"]) == 6
    
    @pytest.mark.asyncio
    async def test_single_intent_processing(self):
        """Test processing a single intent query."""
        agent = IntentRecognitionAgentV2()
        
        result = await agent.process_query("Find all Python files")
        
        assert result.query == "Find all Python files"
        assert result.primary_intent is not None
        assert result.primary_intent.type in ["query.search", "query.retrieve"]
        assert result.primary_intent.confidence > 0
        assert "find" in result.features.get("keywords", [])
    
    @pytest.mark.asyncio
    async def test_multi_intent_processing(self):
        """Test processing a multi-intent query."""
        agent = IntentRecognitionAgentV2()
        
        result = await agent.process_query("Create a file and then analyze it")
        
        assert result.primary_intent is not None
        assert len(result.all_intents) >= 1
        # Should detect either create or analyze as primary
        assert result.primary_intent.type in ["action.create", "query.analyze"]
    
    @pytest.mark.asyncio
    async def test_context_integration(self):
        """Test context integration in pipeline."""
        agent = IntentRecognitionAgentV2()
        
        context = {
            "domain": "software_development",
            "history": [{"query": "search for bugs", "intent_type": "query.search"}]
        }
        
        result = await agent.process_query("Find more issues", context=context)
        
        assert result.primary_intent is not None
        assert result.features.get("context_score", 0) > 0.5


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_pipeline_timing(self):
        """Test that pipeline tracks timing correctly."""
        agent = IntentRecognitionAgentV2()
        
        result = await agent.process_query("Analyze system performance metrics")
        
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 5000  # Should be reasonably fast
    
    @pytest.mark.asyncio
    async def test_caching_performance(self):
        """Test that caching improves performance."""
        agent = IntentRecognitionAgentV2()
        
        # First query
        result1 = await agent.process_query("Find Python files")
        time1 = result1.processing_time_ms
        
        # Same query again (should use cache)
        result2 = await agent.process_query("Find Python files")
        time2 = result2.processing_time_ms
        
        # Second query should be faster due to caching
        assert time2 <= time1


if __name__ == "__main__":
    # Run specific test
    asyncio.run(TestIntentRecognitionAgentV2().test_single_intent_processing())
    
    # Run all tests with pytest
    # pytest tests/test_pipeline_architecture.py -v