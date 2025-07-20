#!/usr/bin/env python3
"""
Demo: A/B Testing with Reward Strategy Integration

This demo shows how to use A/B testing to compare different reward strategies
in the autonomous tool discovery system.
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from src.evaluation.ab_testing_framework import (
    ABTestingFramework, ExperimentConfig, ExperimentStatus,
    AssignmentStrategy, StatisticalMethod
)
from src.evaluation.ab_test_manager import ABTestManager
from src.learning.advanced_rewards.strategy_manager import RewardStrategyManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RewardStrategyExperiment:
    """Simulates comparing different reward strategies using A/B testing."""
    
    def __init__(self):
        self.ab_manager = None
        self.strategy_manager = None
        self.experiment_name = "reward_strategy_comparison"
        
    async def setup(self):
        """Initialize the experiment components."""
        # Initialize A/B test manager
        config = {
            'database': {
                'ab_test_db': 'data/ab_test_rewards_demo.db'
            },
            'ab_testing': {
                'enable_monitoring': False
            }
        }
        
        self.ab_manager = ABTestManager(config)
        await self.ab_manager.initialize()
        
        # Initialize reward strategy manager
        strategy_config = {
            "advanced_reward_strategies": {
                "enabled": True,
                "combination_method": "weighted_average",
                "strategy_weights": {
                    "temporal": 0.25,
                    "hierarchical": 0.25,
                    "adaptive": 0.25,
                    "information_theoretic": 0.25
                }
            }
        }
        self.strategy_manager = RewardStrategyManager(strategy_config)
        await self.strategy_manager.initialize()
        
    async def create_experiment(self):
        """Create the A/B test experiment."""
        config = ExperimentConfig(
            name=self.experiment_name,
            description="Compare base reward vs advanced reward strategies",
            variants=["base_reward", "advanced_reward"],
            primary_metric="reward",
            secondary_metrics=["task_completion_rate", "exploration_rate"],
            assignment_strategy=AssignmentStrategy.RANDOM,
            assignment_weights={"base_reward": 0.5, "advanced_reward": 0.5},
            statistical_method=StatisticalMethod.FREQUENTIST,
            min_sample_size=50,
            target_sample_size=200,
            confidence_level=0.95,
            minimum_detectable_effect=0.1,
            enable_early_stopping=True,
            early_stopping_threshold=0.001
        )
        
        await self.ab_manager.create_experiment(config)
        await self.ab_manager.start_experiment(self.experiment_name)
        logger.info(f"Started experiment: {self.experiment_name}")
        
    async def simulate_tool_execution(self, user_id: str) -> Dict[str, Any]:
        """Simulate a tool execution scenario."""
        # Get variant assignment for user
        variant = await self.ab_manager.get_variant_for_user(
            self.experiment_name, user_id
        )
        
        if not variant:
            return {}
        
        # Simulate different execution scenarios
        import random
        
        scenarios = [
            # Successful execution
            {
                "success": True,
                "execution_time": random.uniform(100, 500),
                "tools_used": ["tool1", "tool2"],
                "partial_completion": 1.0,
                "error_type": None,
                "resource_usage": {
                    "cpu_percent": random.uniform(10, 50),
                    "memory_mb": random.uniform(100, 500)
                }
            },
            # Partial success
            {
                "success": False,
                "execution_time": random.uniform(200, 800),
                "tools_used": ["tool1"],
                "partial_completion": random.uniform(0.3, 0.8),
                "error_type": "timeout",
                "resource_usage": {
                    "cpu_percent": random.uniform(30, 70),
                    "memory_mb": random.uniform(200, 600)
                }
            },
            # Failure
            {
                "success": False,
                "execution_time": random.uniform(50, 200),
                "tools_used": ["tool3"],
                "partial_completion": 0.0,
                "error_type": random.choice(["permission_error", "rate_limit", "network_timeout"]),
                "resource_usage": {
                    "cpu_percent": random.uniform(5, 20),
                    "memory_mb": random.uniform(50, 200)
                }
            }
        ]
        
        # Select scenario based on variant (advanced strategy should perform better)
        if variant == "advanced_reward":
            # 70% success, 20% partial, 10% failure
            weights = [0.7, 0.2, 0.1]
        else:
            # 50% success, 30% partial, 20% failure  
            weights = [0.5, 0.3, 0.2]
            
        scenario = random.choices(scenarios, weights=weights)[0]
        
        # Calculate reward based on variant
        if variant == "base_reward":
            # Simple reward calculation
            if scenario["success"]:
                reward = 1.0
            elif scenario["partial_completion"] > 0:
                reward = 0.3 * scenario["partial_completion"]
            else:
                reward = -0.5
        else:
            # Advanced reward calculation (simulated)
            base_reward = 1.0 if scenario["success"] else -0.5
            
            # Partial success bonus
            if scenario["partial_completion"] > 0:
                base_reward += 0.5 * scenario["partial_completion"]
            
            # Error type adjustment
            if scenario["error_type"] == "network_timeout":
                base_reward += 0.3  # Less penalty for transient errors
            elif scenario["error_type"] == "permission_error":
                base_reward -= 0.3  # More penalty for permanent errors
                
            # Resource efficiency
            resource_penalty = (
                scenario["resource_usage"]["cpu_percent"] / 1000 +
                scenario["resource_usage"]["memory_mb"] / 10000
            )
            
            reward = base_reward - resource_penalty
            
        # Exploration bonus for advanced strategy
        if variant == "advanced_reward" and random.random() < 0.3:
            reward += 0.2  # Exploration bonus
            
        return {
            "variant": variant,
            "reward": reward,
            "success": scenario["success"],
            "completion_rate": scenario["partial_completion"],
            "exploration": random.random() < 0.3
        }
        
    async def run_experiment(self, num_users: int = 100):
        """Run the experiment with simulated users."""
        logger.info(f"Running experiment with {num_users} users...")
        
        for i in range(num_users):
            user_id = f"user_{i:04d}"
            
            # Simulate tool execution
            result = await self.simulate_tool_execution(user_id)
            
            if result:
                # Record metrics
                await self.ab_manager.record_metric(
                    self.experiment_name, user_id,
                    "reward", result["reward"], result["success"]
                )
                
                await self.ab_manager.record_metric(
                    self.experiment_name, user_id,
                    "task_completion_rate", 
                    1.0 if result["success"] else result["completion_rate"],
                    result["success"]
                )
                
                await self.ab_manager.record_metric(
                    self.experiment_name, user_id,
                    "exploration_rate",
                    1.0 if result["exploration"] else 0.0,
                    True
                )
                
            # Small delay to simulate real usage
            await asyncio.sleep(0.01)
            
            # Show progress
            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1} users...")
                
    async def analyze_results(self):
        """Analyze and display experiment results."""
        # Get current status
        status = await self.ab_manager.get_experiment_status(self.experiment_name)
        
        print("\n" + "="*80)
        print("EXPERIMENT STATUS")
        print("="*80)
        print(f"Name: {status['name']}")
        print(f"Status: {status['status']}")
        print(f"Total Users: {status['total_users']}")
        
        print("\nVariant Metrics:")
        for metric in status.get('variant_metrics', []):
            print(f"\n  {metric['variant']}:")
            print(f"    Users: {metric['users']}")
            print(f"    Events: {metric['events']}")
            print(f"    Avg Value: {metric['avg_value']:.3f}")
        
        # Stop experiment and get final results
        results = await self.ab_manager.stop_experiment(self.experiment_name)
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        # Display variant results
        for variant, variant_data in results['variant_results'].items():
            print(f"\n{variant.upper()}:")
            print(f"  Users: {variant_data['users']}")
            
            for metric_name, metric_data in variant_data['metrics'].items():
                print(f"\n  {metric_name}:")
                print(f"    Sample Size: {metric_data['sample_size']}")
                print(f"    Mean: {metric_data['mean']:.4f}")
                print(f"    Std Dev: {metric_data['std_dev']:.4f}")
                
                if 'value' in metric_data:
                    print(f"    Value: {metric_data['value']:.4f}")
                if 'conversion_rate' in metric_data:
                    print(f"    Conversion Rate: {metric_data['conversion_rate']:.4f}")
                if 'success_rate' in metric_data:
                    print(f"    Success Rate: {metric_data['success_rate']:.4f}")
        
        # Statistical significance
        print("\n" + "-"*40)
        print("STATISTICAL ANALYSIS")
        print("-"*40)
        sig = results.get('statistical_significance', {})
        print(f"Method: {sig.get('method', 'N/A')}")
        print(f"P-value: {sig.get('p_value', 0):.6f}")
        print(f"Statistically Significant: {sig.get('significant', False)}")
        
        # Winner determination
        if results.get('winner'):
            print(f"\nWINNER: {results['winner']}")
            print(f"Confidence: {results.get('confidence', 0):.4f}")
        else:
            print("\nNo statistically significant winner yet.")
            
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if sig.get('significant') and results.get('winner') == 'advanced_reward':
            print("✓ The advanced reward strategy shows statistically significant improvement")
            print("✓ Consider adopting the advanced reward strategy for better performance")
            
            # Calculate improvement
            base_reward = results['variant_results']['base_reward']['metrics']['reward']['mean']
            adv_reward = results['variant_results']['advanced_reward']['metrics']['reward']['mean']
            improvement = ((adv_reward - base_reward) / base_reward) * 100
            
            print(f"✓ Average reward improvement: {improvement:.1f}%")
        elif not sig.get('significant'):
            print("→ Results are not yet statistically significant")
            print("→ Continue running the experiment to gather more data")
        else:
            print("→ The base reward strategy is performing adequately")
            print("→ Advanced strategies may not be necessary for your use case")


async def main():
    """Run the complete demo."""
    print("\n" + "="*80)
    print("A/B TESTING DEMO: REWARD STRATEGY COMPARISON")
    print("="*80)
    print("\nThis demo compares base reward calculation vs advanced reward strategies")
    print("using A/B testing to determine which performs better.\n")
    
    experiment = RewardStrategyExperiment()
    
    try:
        # Setup
        print("Setting up experiment...")
        await experiment.setup()
        
        # Create experiment
        print("Creating A/B test experiment...")
        await experiment.create_experiment()
        
        # Run experiment
        print("\nRunning experiment simulation...")
        await experiment.run_experiment(num_users=150)
        
        # Analyze results
        print("\nAnalyzing results...")
        await experiment.analyze_results()
        
    finally:
        # Cleanup
        if experiment.ab_manager:
            await experiment.ab_manager.cleanup()
            
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())