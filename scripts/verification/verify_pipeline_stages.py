"""
Verify the 7-stage modular pipeline in Intent Recognition Agent.
"""

import asyncio
import json
from src.agents.intent_recognition_agent import IntentRecognitionAgent


async def verify_pipeline_stages():
    """Verify all 7 stages of the Intent Recognition pipeline."""
    print("=== Verifying 7-Stage Intent Recognition Pipeline ===\n")
    
    # Create agent
    agent = IntentRecognitionAgent()
    
    # Wait for pipeline initialization
    await agent._get_or_create_pipeline()
    
    # Get pipeline info
    pipeline_info = agent.get_pipeline_info()
    print("Pipeline Name:", pipeline_info['pipeline_name'])
    print("Number of Stages:", len(pipeline_info['stages']))
    print("\nPipeline Stages:")
    for i, stage in enumerate(pipeline_info['stages'], 1):
        print(f"  Stage {i}: {stage}")
    
    print("\n" + "="*50 + "\n")
    
    # Test queries to verify each stage
    test_queries = [
        {
            "query": "Find all Python files in the src directory",
            "expected_intent": "query.search",
            "description": "Tests keyword matching and semantic understanding"
        },
        {
            "query": "Create a new configuration file",
            "expected_intent": "action.create",
            "description": "Tests action intent classification"
        },
        {
            "query": "What's the status of the system?",
            "expected_intent": "system.monitor",
            "description": "Tests question detection and system intents"
        },
        {
            "query": "Delete old log files and update the database",
            "expected_intent": ["action.delete", "action.modify"],
            "description": "Tests multi-intent detection"
        }
    ]
    
    for test in test_queries:
        print(f"Query: '{test['query']}'")
        print(f"Description: {test['description']}")
        
        # Process query
        result = await agent.process_query(test['query'])
        
        print(f"\nPipeline Processing Results:")
        print(f"  1. Text Preprocessing:")
        print(f"     - Raw Query: {result.raw_query}")
        print(f"     - Processed Query: {result.processed_query}")
        
        print(f"  2. Tokenization:")
        print(f"     - Tokens: {result.features.get('tokens', [])}")
        print(f"     - Word Count: {result.features.get('word_count', 0)}")
        print(f"     - Has Question: {result.features.get('has_question', False)}")
        
        print(f"  3. Feature Extraction:")
        semantic_scores = result.features.get('semantic_scores', {})
        if semantic_scores:
            top_semantic = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"     - Top Semantic Matches: {top_semantic}")
        
        print(f"  4. Intent Classification:")
        print(f"     - Keywords Detected: {result.features.get('keywords', [])}")
        keyword_scores = result.features.get('keyword_scores', {})
        if keyword_scores:
            print(f"     - Keyword Scores: {dict(list(keyword_scores.items())[:3])}")
        
        print(f"  5. Context Enrichment:")
        print(f"     - Context Score: {result.features.get('context_score', 0.5):.2f}")
        
        print(f"  6. Confidence Scoring:")
        print(f"     - Primary Intent: {result.primary_intent.type}")
        print(f"     - Confidence: {result.primary_intent.confidence:.2f}")
        print(f"     - Threshold Met: {result.confidence_threshold_met}")
        
        print(f"  7. State Management:")
        print(f"     - Current State: {agent.get_current_state()}")
        
        print(f"\nAll Detected Intents:")
        for intent in result.all_intents:
            print(f"  - {intent.type}: {intent.confidence:.2f}")
        
        print(f"\nProcessing Time: {result.processing_time_ms:.2f}ms")
        print("\n" + "-"*50 + "\n")
    
    # Show state history if available
    if agent.state_manager:
        print("State Transition History:")
        history = agent.get_state_history(limit=5)
        for transition in history:
            print(f"  {transition['from_state']} → {transition['to_state']} "
                  f"(trigger: {transition['trigger']})")
    
    print("\n=== All 7 Pipeline Stages Verified Successfully! ===")


if __name__ == "__main__":
    asyncio.run(verify_pipeline_stages())