"""
Demo script showcasing the refactored Intent Recognition Agent with pipeline architecture.

This script demonstrates:
1. The modular pipeline architecture
2. Individual stage processing
3. Performance comparison between v1 and v2
4. Extensibility of the pipeline
"""

import asyncio
import time
import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents import IntentRecognitionAgent, IntentRecognitionAgentV2
from src.pipeline import Pipeline, PipelineStage, PipelineData


class CustomLoggingStage(PipelineStage):
    """Example custom stage that can be added to the pipeline."""
    
    async def _initialize(self):
        self.logger.info("Custom logging stage initialized")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """Log information about the pipeline processing."""
        self.logger.info(f"Processing query: {data.raw_input}")
        self.logger.info(f"Stages completed: {list(data.processed_data.keys())}")
        
        # Add custom metadata
        data.add_metadata('custom_processed', True)
        data.add_metadata('stage_count', len(data.processed_data))
        
        return data


async def demo_basic_pipeline():
    """Demonstrate basic pipeline functionality."""
    print("=" * 60)
    print("DEMO 1: Basic Pipeline Functionality")
    print("=" * 60)
    
    # Create agent with pipeline architecture
    agent = IntentRecognitionAgentV2()
    
    # Show pipeline configuration
    print("\nPipeline Configuration:")
    print(json.dumps(agent.get_pipeline_info(), indent=2))
    
    # Process some queries
    queries = [
        "Find all Python files in the src directory",
        "Create a new database table for users",
        "Monitor system performance metrics"
    ]
    
    print("\nProcessing Queries:")
    for query in queries:
        result = await agent.process_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {result.primary_intent.type} (confidence: {result.primary_intent.confidence:.2f})")
        print(f"Keywords: {result.features.get('keywords', [])}")
        print(f"Processing time: {result.processing_time_ms:.2f}ms")


async def demo_performance_comparison():
    """Compare performance between v1 and v2."""
    print("\n" + "=" * 60)
    print("DEMO 2: Performance Comparison (V1 vs V2)")
    print("=" * 60)
    
    # Create both versions
    agent_v1 = IntentRecognitionAgent()
    agent_v2 = IntentRecognitionAgentV2()
    
    # Test queries
    test_queries = [
        "Search for configuration files",
        "Update the user profile settings",
        "Delete temporary cache files",
        "Analyze code quality metrics",
        "Create backup and compress files"
    ]
    
    # Warm up (for fair comparison)
    await agent_v1.process_query("warm up query")
    await agent_v2.process_query("warm up query")
    
    # Process with v1
    print("\nProcessing with V1 (Monolithic):")
    v1_times = []
    for query in test_queries:
        start = time.time()
        result = await agent_v1.process_query(query)
        elapsed = (time.time() - start) * 1000
        v1_times.append(elapsed)
        print(f"  {query[:30]:<30} -> {result.primary_intent.type:<20} ({elapsed:.2f}ms)")
    
    # Process with v2
    print("\nProcessing with V2 (Pipeline):")
    v2_times = []
    for query in test_queries:
        start = time.time()
        result = await agent_v2.process_query(query)
        elapsed = (time.time() - start) * 1000
        v2_times.append(elapsed)
        print(f"  {query[:30]:<30} -> {result.primary_intent.type:<20} ({elapsed:.2f}ms)")
    
    # Summary
    print(f"\nPerformance Summary:")
    print(f"  V1 Average: {sum(v1_times)/len(v1_times):.2f}ms")
    print(f"  V2 Average: {sum(v2_times)/len(v2_times):.2f}ms")


async def demo_pipeline_extensibility():
    """Demonstrate how to extend the pipeline with custom stages."""
    print("\n" + "=" * 60)
    print("DEMO 3: Pipeline Extensibility")
    print("=" * 60)
    
    # Create agent and add custom stage
    agent = IntentRecognitionAgentV2()
    
    # Add custom logging stage at the end
    custom_stage = CustomLoggingStage("CustomLogger")
    agent.pipeline.add_stage(custom_stage)
    
    print("\nExtended Pipeline Stages:")
    print(agent.pipeline.get_stage_names())
    
    # Process query with extended pipeline
    print("\nProcessing with Extended Pipeline:")
    result = await agent.process_query("Analyze system logs for errors")
    
    print(f"\nIntent: {result.primary_intent.type}")
    print(f"Custom metadata added: {result.features.get('custom_processed', 'Not found')}")


async def demo_stage_inspection():
    """Demonstrate inspecting individual stage results."""
    print("\n" + "=" * 60)
    print("DEMO 4: Stage-by-Stage Processing Inspection")
    print("=" * 60)
    
    # Create agent
    agent = IntentRecognitionAgentV2()
    
    # Process a query and inspect stage results
    query = "Can you help me find and analyze Python code quality issues?"
    
    # Process through pipeline directly to get full pipeline data
    pipeline_data = await agent.pipeline.process(query, {})
    
    print(f"\nQuery: '{query}'")
    print("\nStage-by-Stage Results:")
    
    # Inspect each stage's output
    stages_to_inspect = [
        ("TextPreprocessor", ["normalized_text"]),
        ("Tokenizer", ["tokens", "word_count", "has_question"]),
        ("FeatureExtractor", ["semantic_scores"]),
        ("IntentClassifier", ["keywords", "classified_intents"]),
        ("ContextEnricher", ["context_score"]),
        ("ConfidenceScorer", ["primary_intent", "confidence_passed"])
    ]
    
    for stage_name, keys in stages_to_inspect:
        print(f"\n{stage_name}:")
        for key in keys:
            value = pipeline_data.get_stage_result(stage_name, key)
            if isinstance(value, dict) and "semantic_scores" in key:
                # Pretty print semantic scores
                print(f"  {key}:")
                for intent, score in sorted(value.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    {intent}: {score:.3f}")
            elif isinstance(value, list) and "intents" in key:
                # Show intent objects
                print(f"  {key}: {len(value)} intents found")
                for intent in value[:3]:
                    print(f"    - {intent.type}: {intent.confidence:.3f}")
            else:
                print(f"  {key}: {value}")
    
    # Show timing information
    print("\nStage Timing:")
    for stage_name, time_ms in pipeline_data.timestamps.items():
        print(f"  {stage_name}: {time_ms:.2f}ms")
    print(f"  Total: {pipeline_data.metadata.get('pipeline_total_time_ms', 0):.2f}ms")


async def main():
    """Run all demos."""
    print("INTENT RECOGNITION PIPELINE REFACTORING DEMO")
    print("=" * 60)
    
    # Run demos
    await demo_basic_pipeline()
    await demo_performance_comparison()
    await demo_pipeline_extensibility()
    await demo_stage_inspection()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())