#!/usr/bin/env python3
"""
Test script to debug reward recording in A/B testing framework
"""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.evaluation.ab_testing_framework import ABTestingFramework, ExperimentConfig, AssignmentStrategy
from src.evaluation.ab_test_manager import ABTestManager

async def test_reward_recording():
    """Test that rewards are properly recorded."""
    print("Testing reward recording in A/B testing framework...\n")
    
    # Create framework and manager
    framework = ABTestingFramework()
    manager = ABTestManager({'database': {'ab_test_db': 'data/test_reward_debug.db'}})
    await manager.initialize()
    
    # Create simple experiment
    config = ExperimentConfig(
        name="test_reward_recording",
        description="Test reward recording",
        variants=["control", "treatment"],
        primary_metric="reward",
        min_sample_size=10
    )
    
    # Create and start experiment
    experiment = await framework.create_experiment(config)
    await experiment.start()
    
    print(f"Created experiment: {config.name}")
    print(f"Variants: {config.variants}")
    print(f"Primary metric: {config.primary_metric}")
    
    # Record some events
    print("\nRecording events...")
    for i in range(20):
        user_id = f"user_{i}"
        
        # Assign user
        variant = await framework.assign_user(config.name, user_id)
        print(f"User {user_id} assigned to {variant}")
        
        # Record reward
        reward_value = 0.5 if variant == "control" else 0.7
        await framework.record_event(config.name, user_id, "reward", reward_value, True)
        print(f"  Recorded reward: {reward_value}")
    
    # Check variant metrics directly
    print("\nDirect metric inspection:")
    for variant, metrics_dict in experiment.variant_metrics.items():
        print(f"\n{variant}:")
        for metric_name, metrics in metrics_dict.items():
            print(f"  {metric_name}:")
            print(f"    sample_size: {metrics.sample_size}")
            print(f"    total_value: {metrics.total_value}")
            print(f"    mean: {metrics.mean}")
            print(f"    values array length: {len(metrics.values)}")
            if metrics.values:
                print(f"    first few values: {metrics.values[:5]}")
    
    # Analyze results
    print("\nAnalyzing experiment...")
    results = await experiment.analyze()
    
    print("\nAnalysis results:")
    for variant, data in results['variant_results'].items():
        print(f"\n{variant}:")
        print(f"  Users: {data['users']}")
        for metric_name, metric_data in data['metrics'].items():
            print(f"  {metric_name}:")
            for key, value in metric_data.items():
                print(f"    {key}: {value}")
    
    # Clean up
    await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_reward_recording())