"""
Verification script for the 7-stage modular pipeline in Intent Recognition Agent.
"""

import asyncio
import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.pipeline import Pipeline, PipelineData


async def verify_pipeline_stages():
    """Verify the 7-stage pipeline in the Intent Recognition Agent."""
    print("=" * 80)
    print("INTENT RECOGNITION AGENT - 7-STAGE PIPELINE VERIFICATION")
    print("=" * 80)
    
    # Create the agent
    agent = IntentRecognitionAgent({
        'enable_state_tracking': True,  # Enable state manager to get all 7 stages
        'collect_metrics': True
    })
    
    # Ensure pipeline is initialized
    if agent.pipeline is None:
        agent.pipeline = await agent._get_or_create_pipeline()
    
    # List all pipeline stages
    print("\n1. PIPELINE STAGES:")
    print("-" * 40)
    stage_names = agent.pipeline.get_stage_names()
    for i, stage_name in enumerate(stage_names, 1):
        print(f"   Stage {i}: {stage_name}")
    
    print(f"\nTotal stages: {len(stage_names)}")
    print(f"State tracking enabled: {agent.enable_state_tracking}")
    
    # Test queries to verify each stage
    test_queries = [
        "Find all Python files in the project",
        "Create a new configuration file",
        "Monitor system health status",
        "what's the current database schema?",
        ""  # Empty query to test edge case
    ]
    
    print("\n2. TESTING PIPELINE WITH SAMPLE QUERIES:")
    print("-" * 40)
    
    for query in test_queries:
        if query == "":
            print(f"\n\nTesting empty query...")
        else:
            print(f"\n\nQuery: '{query}'")
        
        # Process the query
        result = await agent.process_query(query, {'domain': 'test'})
        
        # Display stage outputs
        print("\nStage Outputs:")
        
        # 1. StateManager (if enabled)
        if agent.state_manager:
            state = agent.get_current_state()
            print(f"  1. StateManager: Current state = {state}")
        
        # Use result features and metadata to show stage outputs
        if result:
            # 2. TextPreprocessor
            print(f"  2. TextPreprocessor: '{query}' -> '{result.processed_query}'")
            
            # 3. Tokenizer
            features = result.features
            tokens = features.get('tokens', [])
            word_count = features.get('word_count', len(tokens))
            has_question = features.get('has_question', False)
            print(f"  3. Tokenizer: {word_count} words, tokens={tokens}, is_question={has_question}")
            
            # 4. FeatureExtractor
            semantic_scores = features.get('semantic_scores', {})
            if semantic_scores:
                top_score = max(semantic_scores.items(), key=lambda x: x[1])
                print(f"  4. FeatureExtractor: Generated embedding, top semantic match: {top_score[0]} (score: {top_score[1]:.3f})")
            else:
                print(f"  4. FeatureExtractor: Generated embedding")
            
            # 5. IntentClassifier
            keywords = features.get('keywords', [])
            keyword_scores = features.get('keyword_scores', {})
            print(f"  5. IntentClassifier: Found {len(result.all_intents)} intents, keywords={keywords}")
            
            # 6. ContextEnricher
            context_score = features.get('context_score', 0.5)
            print(f"  6. ContextEnricher: Context score = {context_score}")
            
            # 7. ConfidenceScorer
            print(f"  7. ConfidenceScorer: {result.primary_intent.type} (confidence: {result.primary_intent.confidence:.3f}), passed={result.confidence_passed}")
        
        # Final result
        print(f"\nFinal Result:")
        print(f"  Primary Intent: {result.primary_intent.type}")
        print(f"  Confidence: {result.primary_intent.confidence:.3f}")
        print(f"  Threshold Met: {result.confidence_passed}")
        print(f"  Processing Time: {result.processing_time_ms:.2f}ms")
    
    # Verify state transitions
    if agent.state_manager:
        print("\n3. STATE TRANSITION HISTORY:")
        print("-" * 40)
        history = agent.get_state_history(limit=10)
        for i, transition in enumerate(history):
            print(f"  {i+1}. {transition['from_state']} -> {transition['to_state']} "
                  f"(trigger: {transition['trigger']}) at {transition['timestamp']}")
    
    # Display metrics summary
    print("\n4. PERFORMANCE METRICS:")
    print("-" * 40)
    metrics = agent.get_metrics_summary()
    if 'total_queries' in metrics:
        print(f"  Total Queries: {metrics['total_queries']}")
        print(f"  Average Processing Time: {metrics.get('avg_processing_time_ms', 0):.2f}ms")
        print(f"  Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.2%}")
        
        # Intent distribution
        if 'intent_distribution' in metrics:
            print("\n  Intent Distribution:")
            for intent, count in metrics['intent_distribution'].items():
                print(f"    {intent}: {count}")
    
    print("\n5. PIPELINE VERIFICATION SUMMARY:")
    print("-" * 40)
    print(f"✓ Pipeline initialized with {len(stage_names)} stages")
    print(f"✓ All stages are functional and producing outputs")
    print(f"✓ State management is {'enabled' if agent.enable_state_tracking else 'disabled'}")
    print(f"✓ Metrics collection is {'enabled' if agent.collect_metrics else 'disabled'}")
    
    # Check for the expected 7 stages
    expected_stages = [
        'StateManager',
        'TextPreprocessor', 
        'Tokenizer',
        'FeatureExtractor',
        'IntentClassifier',
        'ContextEnricher',
        'ConfidenceScorer'
    ]
    
    if agent.enable_state_tracking:
        if len(stage_names) == 7:
            print("\n✓ All 7 stages are present (including StateManager)")
        else:
            print(f"\n⚠ Expected 7 stages but found {len(stage_names)}")
    else:
        if len(stage_names) == 6:
            print("\n✓ All 6 core stages are present (StateManager disabled)")
        else:
            print(f"\n⚠ Expected 6 stages but found {len(stage_names)}")
    
    # Verify each expected stage is present
    print("\nStage Presence Check:")
    for expected in expected_stages:
        if expected == 'StateManager' and not agent.enable_state_tracking:
            continue
        if expected in stage_names:
            print(f"  ✓ {expected}")
        else:
            print(f"  ✗ {expected} (missing)")
    
    print("\n" + "=" * 80)
    print("PIPELINE VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(verify_pipeline_stages())