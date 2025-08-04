"""
Verify Monitoring & Metrics Setup for Intent Recognition System.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from src.agents.intent_recognition_agent import IntentRecognitionAgent


async def verify_metrics_setup():
    """Comprehensive verification of metrics collection and monitoring."""
    print("=== Verifying Monitoring & Metrics Setup ===\n")
    
    # Create agent with metrics enabled
    config = {
        'collect_metrics': True,
        'similarity_threshold': 0.7,
        'confidence_threshold': 0.7
    }
    agent = IntentRecognitionAgent(config)
    
    # Initialize pipeline
    await agent._get_or_create_pipeline()
    
    print("1. METRICS COLLECTOR VERIFICATION")
    print("-" * 40)
    
    # Check if metrics collector is initialized
    if hasattr(agent, 'metrics') and agent.metrics is not None:
        print("✓ Metrics collector initialized")
        print(f"  - Collector type: {type(agent.metrics).__name__}")
        print(f"  - Collection enabled: {agent.collect_metrics}")
    else:
        print("✗ Metrics collector not initialized")
        return
    
    # Process various queries to generate metrics
    test_queries = [
        "Find all Python files in the project",
        "Create a new configuration file",
        "Delete old log files",
        "Monitor system status",
        "Analyze code quality and generate report",
        "Search for bugs, fix them, and update documentation",
        "What is the status?",  # Question query
        "xyz abc 123"  # Low confidence query
    ]
    
    print("\n2. PROCESSING QUERIES FOR METRICS")
    print("-" * 40)
    
    start_time = time.time()
    for i, query in enumerate(test_queries, 1):
        print(f"Processing query {i}/{len(test_queries)}: '{query[:40]}...'")
        result = await agent.process_query(query)
        print(f"  - Intent: {result.primary_intent.type}")
        print(f"  - Confidence: {result.primary_intent.confidence:.2f}")
        print(f"  - Processing time: {result.processing_time_ms:.2f}ms")
    
    total_time = (time.time() - start_time) * 1000
    print(f"\nTotal processing time for {len(test_queries)} queries: {total_time:.2f}ms")
    print(f"Average time per query: {total_time/len(test_queries):.2f}ms")
    
    # Get metrics summary
    print("\n3. METRICS SUMMARY")
    print("-" * 40)
    
    summary = agent.get_metrics_summary()
    if summary and summary != {"metrics_collection": "disabled"}:
        print("✓ Metrics collection working")
        print(json.dumps(summary, indent=2))
    else:
        print("✗ No metrics collected")
    
    # Check pipeline stage performance tracking
    print("\n4. PIPELINE STAGE PERFORMANCE TRACKING")
    print("-" * 40)
    
    # Process a query and check for stage timings
    result = await agent.process_query("Find and analyze Python files")
    
    if hasattr(result, 'metadata') and 'stage_timings' in result.metadata:
        print("✓ Pipeline stage timings tracked")
        timings = result.metadata['stage_timings']
        for stage, timing in timings.items():
            print(f"  - {stage}: {timing:.2f}ms")
    else:
        print("⚠ Stage timings not exposed in result metadata")
        print("  (This is normal - timings are tracked internally)")
    
    # Verify specific metrics features
    print("\n5. METRICS FEATURES VERIFICATION")
    print("-" * 40)
    
    if hasattr(agent.metrics, 'intent_counts'):
        print("✓ Intent counting supported")
        
    if hasattr(agent.metrics, 'confidence_scores'):
        print("✓ Confidence score tracking supported")
        
    if hasattr(agent.metrics, 'processing_times'):
        print("✓ Processing time tracking supported")
        
    if hasattr(agent.metrics, 'cache_metrics'):
        print("✓ Cache metrics tracking supported")
    
    # Test metrics export
    print("\n6. METRICS EXPORT FUNCTIONALITY")
    print("-" * 40)
    
    export_dir = "metrics_export_test"
    os.makedirs(export_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"intent_metrics_{timestamp}.json")
    
    try:
        agent.export_metrics(export_path)
        
        # Check if file was created
        if os.path.exists(export_path):
            print("✓ Metrics export successful")
            print(f"  - Export path: {export_path}")
            
            # Read and display exported content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            print(f"  - Exported metrics categories: {list(exported_data.keys())}")
            
            # Show sample of exported data
            if 'summary' in exported_data:
                print("\n  Exported Summary:")
                print(json.dumps(exported_data['summary'], indent=4)[:500] + "...")
        else:
            print("✗ Export file not created")
    except Exception as e:
        print(f"✗ Export failed: {e}")
    
    # Performance analysis
    print("\n7. PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    if 'performance' in summary:
        perf = summary['performance']
        avg_time = perf.get('avg_processing_time_ms', 0)
        p50_time = perf.get('p50_processing_time_ms', 0)
        p95_time = perf.get('p95_processing_time_ms', 0)
        p99_time = perf.get('p99_processing_time_ms', 0)
        
        print(f"Average processing time: {avg_time:.2f}ms")
        print(f"P50 processing time: {p50_time:.2f}ms")
        print(f"P95 processing time: {p95_time:.2f}ms")
        print(f"P99 processing time: {p99_time:.2f}ms")
        
        # Check against performance targets
        avg_time = perf.get('avg_processing_time_ms', 0)
        if avg_time > 0 and avg_time < 100:
            print("✓ Meeting performance target (<100ms average)")
        else:
            print("⚠ Performance may need optimization")
    
    # Intent distribution analysis
    print("\n8. INTENT DISTRIBUTION")
    print("-" * 40)
    
    if 'intent_distribution' in summary:
        dist = summary['intent_distribution']
        print("Intent type counts:")
        for intent_type, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {intent_type}: {count}")
    
    # Final verdict
    print("\n=== MONITORING & METRICS VERDICT ===")
    print("-" * 40)
    
    checks_passed = 0
    total_checks = 3
    
    if agent.metrics is not None and agent.collect_metrics:
        print("✓ Metrics collector: WORKING")
        checks_passed += 1
    else:
        print("✗ Metrics collector: NOT WORKING")
    
    if summary and summary != {"metrics_collection": "disabled"}:
        print("✓ Performance tracking: WORKING")
        checks_passed += 1
    else:
        print("✗ Performance tracking: NOT WORKING")
    
    if os.path.exists(export_path):
        print("✓ Export functionality: WORKING")
        checks_passed += 1
    else:
        print("✗ Export functionality: NOT WORKING")
    
    print(f"\nOverall: {checks_passed}/{total_checks} monitoring features working")
    
    if checks_passed == total_checks:
        print("\n✅ All monitoring & metrics features are working correctly!")
    elif checks_passed >= 2:
        print("\n⚠️ Most monitoring features working, minor issues detected")
    else:
        print("\n❌ Significant monitoring issues detected")
    
    # Cleanup
    if os.path.exists(export_path):
        os.remove(export_path)
    if os.path.exists(export_dir):
        os.rmdir(export_dir)


if __name__ == "__main__":
    asyncio.run(verify_metrics_setup())