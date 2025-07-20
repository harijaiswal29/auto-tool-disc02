#!/usr/bin/env python3
"""
Comprehensive Demo of A/B Testing Framework

This demo showcases:
1. Creating and managing A/B test experiments
2. Testing different reward strategies
3. Statistical analysis of results
4. Multi-armed bandit functionality
5. Integration with the evaluation system
"""

import asyncio
import numpy as np
import random
from datetime import datetime
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.ab_testing_framework import (
    ABTestingFramework, ExperimentConfig, AssignmentStrategy, StatisticalMethod
)
from src.evaluation.ab_test_manager import ABTestManager
from src.learning.advanced_rewards.strategy_manager import StrategyManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ABTestingDemo:
    """Demo class for A/B testing framework."""
    
    def __init__(self):
        self.framework = ABTestingFramework()
        self.config = self._load_config()
        self.ab_test_manager = ABTestManager(self.config)
        # Pass the same ab_test_manager to strategy_manager
        self.config['_shared_ab_test_manager'] = self.ab_test_manager
        self.strategy_manager = StrategyManager(self.config)
        self.timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_config(self):
        """Load configuration."""
        return {
            'database': {'ab_test_db': 'data/demo_ab_tests.db'},
            'ab_testing': {
                'enable_monitoring': True,
                'monitoring_interval': 5  # 5 seconds for demo
            },
            'advanced_reward_strategies': {
                'enabled': True,
                'use_advanced_ab_testing': True,
                'strategies': {
                    'temporal_difference': {'enabled': True},
                    'hierarchical': {'enabled': True},
                    'adaptive_shaping': {'enabled': True},
                    'information_theoretic': {'enabled': True}
                }
            }
        }
    
    async def demo_basic_experiment(self):
        """Demo 1: Basic A/B test experiment."""
        print("\n" + "="*60)
        print("Demo 1: Basic A/B Test Experiment")
        print("="*60)
        
        # Create experiment config with unique name
        config = ExperimentConfig(
            name=f"demo_basic_conversion_{self.timestamp_suffix}",
            description="Testing button color impact on conversion",
            variants=["control_blue", "treatment_green"],
            primary_metric="conversion_rate",
            assignment_strategy=AssignmentStrategy.RANDOM,
            min_sample_size=50,
            confidence_level=0.95,
            enable_early_stopping=False
        )
        
        # Create and start experiment
        experiment = await self.framework.create_experiment(config)
        await experiment.start()
        
        print(f"Created experiment: {config.name}")
        print(f"Variants: {config.variants}")
        
        # Simulate users
        print("\nSimulating user interactions...")
        for i in range(100):
            user_id = f"user_{i}"
            variant = await self.framework.assign_user(config.name, user_id)
            
            # Simulate conversion based on variant
            if variant == "control_blue":
                # 20% conversion rate for blue button
                converted = random.random() < 0.20
            else:
                # 25% conversion rate for green button
                converted = random.random() < 0.25
            
            # Record event
            await self.framework.record_event(
                config.name, user_id, "conversion_rate", 
                1.0 if converted else 0.0, converted
            )
        
        # Analyze results
        results = await experiment.analyze()
        self._print_results(results)
    
    async def demo_multi_variant_experiment(self):
        """Demo 2: Multi-variant experiment with weighted assignment."""
        print("\n" + "="*60)
        print("Demo 2: Multi-Variant Experiment")
        print("="*60)
        
        config = ExperimentConfig(
            name=f"demo_multi_variant_{self.timestamp_suffix}",
            description="Testing multiple page layouts",
            variants=["control", "layout_a", "layout_b", "layout_c"],
            primary_metric="engagement_score",
            secondary_metrics=["time_on_page", "bounce_rate"],
            assignment_strategy=AssignmentStrategy.RANDOM,
            assignment_weights={
                "control": 0.4,
                "layout_a": 0.2,
                "layout_b": 0.2,
                "layout_c": 0.2
            },
            min_sample_size=100
        )
        
        experiment = await self.framework.create_experiment(config)
        await experiment.start()
        
        print(f"Created experiment: {config.name}")
        print(f"Variants with weights: {config.assignment_weights}")
        
        # Simulate users
        print("\nSimulating user interactions...")
        engagement_means = {
            "control": 5.0,
            "layout_a": 5.5,
            "layout_b": 6.0,
            "layout_c": 5.2
        }
        
        for i in range(200):
            user_id = f"user_{i}"
            variant = await self.framework.assign_user(config.name, user_id)
            
            # Simulate engagement score
            mean = engagement_means[variant]
            engagement = max(0, np.random.normal(mean, 1.5))
            
            # Simulate secondary metrics
            time_on_page = max(0, np.random.normal(mean * 60, 20))
            bounce_rate = max(0, min(1, np.random.normal(0.3 - mean/20, 0.1)))
            
            # Record metrics
            await self.framework.record_event(
                config.name, user_id, "engagement_score", engagement
            )
            await self.framework.record_event(
                config.name, user_id, "time_on_page", time_on_page
            )
            await self.framework.record_event(
                config.name, user_id, "bounce_rate", bounce_rate
            )
        
        # Analyze results
        results = await experiment.analyze()
        self._print_results(results)
    
    async def demo_bayesian_ab_test(self):
        """Demo 3: Bayesian A/B testing."""
        print("\n" + "="*60)
        print("Demo 3: Bayesian A/B Testing")
        print("="*60)
        
        config = ExperimentConfig(
            name=f"demo_bayesian_{self.timestamp_suffix}",
            description="Bayesian analysis of feature adoption",
            variants=["control", "new_feature"],
            primary_metric="adoption_rate",
            statistical_method=StatisticalMethod.BAYESIAN,
            min_sample_size=80
        )
        
        experiment = await self.framework.create_experiment(config)
        await experiment.start()
        
        print(f"Created Bayesian experiment: {config.name}")
        
        # Simulate users with slight improvement in new feature
        adoption_rates = {
            "control": 0.15,
            "new_feature": 0.18
        }
        
        print("\nSimulating user interactions...")
        for i in range(150):
            user_id = f"user_{i}"
            variant = await self.framework.assign_user(config.name, user_id)
            
            adopted = random.random() < adoption_rates[variant]
            
            await self.framework.record_event(
                config.name, user_id, "adoption_rate",
                1.0 if adopted else 0.0, adopted
            )
        
        # Analyze with Bayesian method
        results = await experiment.analyze()
        self._print_bayesian_results(results)
    
    async def demo_multi_armed_bandit(self):
        """Demo 4: Multi-armed bandit experiment."""
        print("\n" + "="*60)
        print("Demo 4: Multi-Armed Bandit Experiment")
        print("="*60)
        
        config = ExperimentConfig(
            name=f"demo_mab_{self.timestamp_suffix}",
            description="MAB for optimizing recommendation algorithms",
            variants=["algo_baseline", "algo_collaborative", "algo_content", "algo_hybrid"],
            primary_metric="click_through_rate",
            assignment_strategy=AssignmentStrategy.MULTI_ARMED_BANDIT,
            enable_multi_armed_bandit=True,
            min_sample_size=50
        )
        
        experiment = await self.framework.create_experiment(config)
        await experiment.start()
        
        print(f"Created MAB experiment: {config.name}")
        print("Thompson sampling will adapt based on performance")
        
        # True click-through rates (unknown to the algorithm)
        true_ctrs = {
            "algo_baseline": 0.05,
            "algo_collaborative": 0.08,
            "algo_content": 0.06,
            "algo_hybrid": 0.09  # Best performer
        }
        
        print("\nSimulating adaptive user assignment...")
        assignment_counts = {v: 0 for v in config.variants}
        
        for i in range(300):
            user_id = f"user_{i}"
            variant = await self.framework.assign_user(config.name, user_id)
            assignment_counts[variant] += 1
            
            # Simulate click based on true CTR
            clicked = random.random() < true_ctrs[variant]
            
            await self.framework.record_event(
                config.name, user_id, "click_through_rate",
                1.0 if clicked else 0.0, clicked
            )
            
            # Show assignment distribution every 50 users
            if (i + 1) % 50 == 0:
                print(f"\nAfter {i+1} users - Assignment distribution:")
                for v, count in sorted(assignment_counts.items()):
                    print(f"  {v}: {count} ({count/(i+1)*100:.1f}%)")
        
        # Final analysis
        results = await experiment.analyze()
        self._print_results(results)
    
    async def demo_strategy_comparison(self):
        """Demo 5: A/B test for reward strategy comparison."""
        print("\n" + "="*60)
        print("Demo 5: Reward Strategy A/B Test")
        print("="*60)
        
        # Import necessary classes for creating execution results
        from dataclasses import dataclass
        from typing import Optional, Any
        
        @dataclass
        class MockToolExecutionResult:
            """Mock tool execution result for testing."""
            tool_id: str
            tool_name: str
            success: bool
            result: Any
            error: Optional[str] = None
            execution_time_ms: float = 0.0
            partial_success: bool = False
            completion_percentage: float = 0.0
            error_type: Optional[str] = None
            retry_count: int = 0
            resource_usage: Optional[dict] = None
            result_quality: float = 1.0
        
        # Initialize both A/B test manager and strategy manager
        await self.ab_test_manager.initialize()
        await self.strategy_manager.initialize()
        
        # Create strategy experiment with unique name
        experiment_id = await self.strategy_manager.create_strategy_experiment(
            name=f"demo_strategy_comparison_{self.timestamp_suffix}",
            description="Comparing reward calculation strategies",
            strategies_to_test=["temporal", "hierarchical", "adaptive"],
            min_sample_size=50,
            max_duration_days=1
        )
        
        if not experiment_id:
            print("Failed to create strategy experiment")
            return
        
        print(f"Created strategy experiment: {experiment_id}")
        print("Testing strategies: temporal, hierarchical, adaptive (vs control ensemble)")
        
        # Simulate tool executions with different contexts
        print("\nSimulating tool executions...")
        
        for i in range(200):
            user_id = f"execution_{i}"
            
            # Get variant assignment
            variant = await self.ab_test_manager.get_variant_for_user(
                experiment_id, user_id
            )
            
            # Simulate execution results
            success = random.random() < 0.7
            execution_time_ms = np.random.exponential(2000.0)  # in milliseconds
            
            # Create mock execution results
            execution_results = [
                MockToolExecutionResult(
                    tool_id="tool1",
                    tool_name="mock_tool_1",
                    success=success,
                    result={"data": "mock_result"} if success else None,
                    error="Execution failed" if not success else None,
                    execution_time_ms=execution_time_ms * 0.6,
                    partial_success=not success and random.random() < 0.3,
                    completion_percentage=1.0 if success else random.uniform(0.0, 0.8),
                    error_type="timeout" if not success and random.random() < 0.3 else None,
                    resource_usage={
                        "memory_mb": random.uniform(50, 200),
                        "cpu_percent": random.uniform(10, 80),
                        "api_calls": random.randint(1, 5)
                    }
                ),
                MockToolExecutionResult(
                    tool_id="tool2",
                    tool_name="mock_tool_2",
                    success=success,
                    result={"data": "mock_result_2"} if success else None,
                    error="Execution failed" if not success else None,
                    execution_time_ms=execution_time_ms * 0.4,
                    partial_success=not success and random.random() < 0.2,
                    completion_percentage=1.0 if success else random.uniform(0.0, 0.7),
                    resource_usage={
                        "memory_mb": random.uniform(30, 150),
                        "cpu_percent": random.uniform(5, 60),
                        "api_calls": random.randint(0, 3)
                    }
                )
            ]
            
            # Create context for reward calculation
            context = {
                "intent": {"type": "query.search", "confidence": 0.85},
                "domain": "test",
                "exploration_mode": random.random() < 0.3,
                "user_initiated": True,
                "execution_id": user_id,
                "confidence_score": random.uniform(0.6, 0.95),
                "ab_test_variant": variant,  # Important: specify which variant to use
                "experiment_name": experiment_id  # Required for A/B test integration
            }
            
            # Create mock state vectors for advanced strategies
            state = np.random.rand(439)  # 439-dimensional state vector
            action = ["tool1", "tool2"]
            next_state = np.random.rand(439)
            
            # Use strategy manager to calculate reward with the appropriate strategy
            # The strategy manager will use the variant specified in context
            reward, breakdown = self.strategy_manager.calculate_reward(
                state, action, next_state, execution_results, context
            )
            
            # Debug: Print reward calculation details
            if i < 5:  # Only print first few for debugging
                print(f"  User {user_id}: variant={variant}, reward={reward:.3f}, success={success}")
            
            # Record results
            await self.strategy_manager.record_experiment_result(
                experiment_id, user_id, reward,
                1.0 if success else 0.0, execution_time_ms / 1000.0  # convert to seconds
            )
        
        # Analyze results
        results = await self.strategy_manager.analyze_strategy_experiment(experiment_id)
        self._print_results(results)
    
    async def demo_with_manager(self):
        """Demo 6: Using A/B Test Manager for full lifecycle."""
        print("\n" + "="*60)
        print("Demo 6: A/B Test Manager Full Lifecycle")
        print("="*60)
        
        # Initialize manager
        await self.ab_test_manager.initialize()
        
        # Create experiment through manager
        config = ExperimentConfig(
            name=f"demo_manager_experiment_{self.timestamp_suffix}",
            description="Full lifecycle demo with persistence",
            variants=["version_a", "version_b"],
            primary_metric="task_completion_rate",
            secondary_metrics=["user_satisfaction"],
            min_sample_size=75,
            max_duration_days=7,
            enable_early_stopping=True,
            early_stopping_threshold=0.001
        )
        
        experiment_name = await self.ab_test_manager.create_experiment(config)
        await self.ab_test_manager.start_experiment(experiment_name)
        
        print(f"Created and started experiment: {experiment_name}")
        
        # Simulate users over time
        print("\nSimulating users over time...")
        
        # Version B is slightly better
        completion_rates = {"version_a": 0.75, "version_b": 0.80}
        satisfaction_means = {"version_a": 3.5, "version_b": 3.8}
        
        for i in range(150):
            user_id = f"user_{i}"
            
            # Get variant assignment
            variant = await self.ab_test_manager.get_variant_for_user(
                experiment_name, user_id
            )
            
            # Simulate metrics
            completed = random.random() < completion_rates[variant]
            satisfaction = min(5, max(1, np.random.normal(
                satisfaction_means[variant], 0.8
            )))
            
            # Record metrics
            await self.ab_test_manager.record_metric(
                experiment_name, user_id, "task_completion_rate",
                1.0 if completed else 0.0, completed
            )
            await self.ab_test_manager.record_metric(
                experiment_name, user_id, "user_satisfaction",
                satisfaction
            )
            
            # Check status periodically
            if (i + 1) % 50 == 0:
                status = await self.ab_test_manager.get_experiment_status(experiment_name)
                print(f"\nStatus after {i+1} users:")
                print(f"  Total users: {status['total_users']}")
                print(f"  Status: {status['status']}")
        
        # Get final results
        results = await self.ab_test_manager.get_experiment_results(experiment_name)
        if results:
            self._print_results(results)
        
        # List all experiments
        print("\nAll experiments:")
        experiments = await self.ab_test_manager.list_experiments()
        for exp in experiments:
            print(f"  - {exp['name']}: {exp['status']}")
    
    def _print_results(self, results):
        """Pretty print experiment results."""
        print("\n" + "-"*40)
        print("EXPERIMENT RESULTS")
        print("-"*40)
        
        print(f"Experiment: {results['experiment_name']}")
        print(f"Status: {results['status']}")
        print(f"Total users: {results['total_users']}")
        
        if results.get('duration_days') is not None:
            print(f"Duration: {results['duration_days']} days")
        
        print("\nVariant Results:")
        for variant, data in results['variant_results'].items():
            print(f"\n  {variant}:")
            print(f"    Users: {data['users']}")
            for metric, values in data['metrics'].items():
                print(f"    {metric}:")
                print(f"      Sample size: {values['sample_size']}")
                if values.get('conversion_rate') is not None:
                    print(f"      Conversion rate: {values['conversion_rate']:.3f}")
                if values.get('value') is not None:
                    print(f"      Value: {values['value']:.3f}")
                if values.get('mean') is not None and values['mean'] != 0:
                    print(f"      Mean: {values['mean']:.3f}")
                if values.get('std_dev') is not None and values['std_dev'] != 0:
                    print(f"      Std dev: {values['std_dev']:.3f}")
                if values.get('success_rate') is not None:
                    print(f"      Success rate: {values['success_rate']:.3f}")
        
        if 'statistical_significance' in results:
            sig = results['statistical_significance']
            print(f"\nStatistical Analysis ({sig['method']}):")
            if sig['method'] == 'frequentist':
                print(f"  P-value: {sig.get('p_value', 'N/A'):.4f}")
                print(f"  Significant: {sig.get('significant', False)}")
            
            if 'winner' in results and results['winner']:
                print(f"\nWinner: {results['winner']} 🎉")
    
    def _print_bayesian_results(self, results):
        """Pretty print Bayesian experiment results."""
        self._print_results(results)
        
        if 'statistical_significance' in results:
            sig = results['statistical_significance']
            if sig['method'] == 'bayesian':
                print("\nBayesian Analysis:")
                print(f"  P(treatment > control): {sig.get('probability_treatment_better', 0):.3f}")
                print(f"  Expected lift: {sig.get('expected_lift', 0):.3%}")
                if 'credible_interval_95' in sig:
                    ci = sig['credible_interval_95']
                    print(f"  95% Credible interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    async def run_all_demos(self):
        """Run all demos in sequence."""
        demos = [
            ("Basic A/B Test", self.demo_basic_experiment),
            ("Multi-Variant Test", self.demo_multi_variant_experiment),
            ("Bayesian A/B Test", self.demo_bayesian_ab_test),
            ("Multi-Armed Bandit", self.demo_multi_armed_bandit),
            ("Strategy Comparison", self.demo_strategy_comparison),
            ("Full Lifecycle with Manager", self.demo_with_manager)
        ]
        
        print("\n" + "="*60)
        print("A/B TESTING FRAMEWORK DEMO")
        print("="*60)
        print(f"Running {len(demos)} demonstrations...")
        
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n[{i}/{len(demos)}] {name}")
            try:
                await demo_func()
            except Exception as e:
                logger.error(f"Demo failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Cleanup
        if self.ab_test_manager:
            await self.ab_test_manager.cleanup()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED!")
        print("="*60)


async def main():
    """Main entry point."""
    # setup_logging()  # Not available in current logger module
    
    demo = ABTestingDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())