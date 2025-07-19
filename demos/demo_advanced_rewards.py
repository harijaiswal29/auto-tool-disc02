"""Demonstration of advanced reward calculation strategies."""

import sys
import os
import numpy as np
import json
from typing import List, Dict, Any
import asyncio
import time

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics
from src.learning.advanced_rewards.strategy_manager import StrategyManager
from src.utils.logger import get_logger

# Set up logging
logger = get_logger(__name__)


def create_sample_execution_results(scenario: str) -> List[ExecutionMetrics]:
    """Create sample execution results for different scenarios."""
    
    if scenario == "all_success":
        return [
            ExecutionMetrics(
                tool_id="filesystem_mcp",
                success=True,
                execution_time_ms=150,
                resource_usage={"cpu_percent": 20, "memory_mb": 100}
            ),
            ExecutionMetrics(
                tool_id="search_mcp",
                success=True,
                execution_time_ms=300,
                resource_usage={"cpu_percent": 30, "memory_mb": 150}
            )
        ]
    
    elif scenario == "partial_success":
        return [
            ExecutionMetrics(
                tool_id="filesystem_mcp",
                success=False,
                partial_success=True,
                completion_percentage=0.7,
                execution_time_ms=200,
                error_type="timeout",
                resource_usage={"cpu_percent": 40, "memory_mb": 200}
            ),
            ExecutionMetrics(
                tool_id="search_mcp",
                success=True,
                execution_time_ms=250,
                resource_usage={"cpu_percent": 25, "memory_mb": 120}
            )
        ]
    
    elif scenario == "exploration":
        return [
            ExecutionMetrics(
                tool_id="new_tool_mcp",
                success=True,
                execution_time_ms=500,
                resource_usage={"cpu_percent": 50, "memory_mb": 300}
            )
        ]
    
    elif scenario == "failure":
        return [
            ExecutionMetrics(
                tool_id="database_mcp",
                success=False,
                execution_time_ms=1000,
                error_type="permission_error",
                retry_count=3,
                resource_usage={"cpu_percent": 60, "memory_mb": 400}
            )
        ]
    
    else:
        return []


def create_sample_states() -> tuple:
    """Create sample state vectors for demonstration."""
    # 439-dimensional state vectors
    state_dim = 439
    
    # Current state (random for demo)
    current_state = np.random.randn(state_dim) * 0.1
    
    # Next state (slightly perturbed)
    next_state = current_state + np.random.randn(state_dim) * 0.05
    
    return current_state, next_state


def demo_basic_vs_advanced():
    """Compare basic reward calculation with advanced strategies."""
    logger.info("=== Comparing Basic vs Advanced Reward Calculation ===")
    
    # Load configuration
    config = {
        "reward_calculation": {
            "base_weights": {"success": 1.0, "failure": -0.5, "partial_success": 0.3}
        },
        "advanced_reward_strategies": {
            "enabled": True,
            "combination_method": "weighted_average",
            "strategies": {
                "temporal_difference": {"enabled": True, "lambda": 0.9, "n_steps": 5},
                "hierarchical": {"enabled": True},
                "adaptive_shaping": {"enabled": True},
                "information_theoretic": {"enabled": True}
            }
        }
    }
    
    # Create calculators
    basic_calc = RewardCalculator(config, use_advanced_strategies=False)
    advanced_calc = RewardCalculator(config, use_advanced_strategies=True)
    
    # Test scenarios
    scenarios = ["all_success", "partial_success", "exploration", "failure"]
    
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario} ---")
        
        # Create execution results
        results = create_sample_execution_results(scenario)
        
        # Create states
        state, next_state = create_sample_states()
        action = [r.tool_id for r in results]
        
        # Context
        context = {
            "mode": "exploration" if scenario == "exploration" else "production",
            "intent_confidence": 0.8,
            "user_initiated": True
        }
        
        # Calculate basic reward
        basic_reward, basic_breakdown = basic_calc.calculate_reward(
            results, context, user_feedback=None
        )
        
        # Calculate advanced reward
        advanced_reward, advanced_breakdown = advanced_calc.calculate_reward(
            results, context, user_feedback=None,
            state=state, action=action, next_state=next_state
        )
        
        logger.info(f"Basic Reward: {basic_reward:.3f}")
        logger.info(f"Advanced Reward: {advanced_reward:.3f}")
        
        if isinstance(advanced_breakdown, dict) and 'advanced_components' in advanced_breakdown:
            adv_comp = advanced_breakdown['advanced_components']
            if 'strategy_rewards' in adv_comp:
                logger.info("Strategy Contributions:")
                for strategy, reward in adv_comp['strategy_rewards'].items():
                    logger.info(f"  {strategy}: {reward:.3f}")


def demo_strategy_specific_features():
    """Demonstrate specific features of each advanced strategy."""
    logger.info("\n=== Demonstrating Strategy-Specific Features ===")
    
    config = {
        "advanced_reward_strategies": {
            "enabled": True,
            "strategies": {
                "temporal_difference": {"enabled": True},
                "hierarchical": {"enabled": True},
                "adaptive_shaping": {"enabled": True},
                "information_theoretic": {"enabled": True}
            }
        }
    }
    
    manager = StrategyManager(config)
    
    # Create test data
    state, next_state = create_sample_states()
    action = ["tool1", "tool2"]
    results = create_sample_execution_results("all_success")
    context = {"mode": "exploration"}
    
    logger.info("\n--- Temporal Difference Features ---")
    # The temporal calculator tracks n-step returns and eligibility traces
    logger.info("- Uses TD(λ) with eligibility traces for credit assignment")
    logger.info("- Calculates n-step returns for better long-term rewards")
    logger.info("- Maintains experience buffer for temporal dependencies")
    
    logger.info("\n--- Hierarchical Goal Features ---")
    # The hierarchical calculator tracks goal achievement
    logger.info("- Tracks multi-level goal hierarchy (primary, secondary, tertiary)")
    logger.info("- Provides milestone bonuses for significant achievements")
    logger.info("- Supports partial progress rewards")
    
    logger.info("\n--- Adaptive Shaping Features ---")
    # The adaptive shaper adjusts weights dynamically
    logger.info("- Dynamically adjusts component weights based on performance")
    logger.info("- Implements curriculum learning with multiple stages")
    logger.info("- Uses meta-learning for parameter adaptation")
    
    logger.info("\n--- Information-Theoretic Features ---")
    # The information calculator encourages exploration
    logger.info("- Provides curiosity bonuses for novel state-action pairs")
    logger.info("- Calculates entropy rewards for diverse exploration")
    logger.info("- Tracks information gain and mutual information")


def demo_ab_testing():
    """Demonstrate A/B testing capabilities."""
    logger.info("\n=== Demonstrating A/B Testing ===")
    
    config = {
        "advanced_reward_strategies": {
            "enabled": True,
            "ab_testing_enabled": True,
            "strategies": {
                "temporal_difference": {"enabled": True},
                "hierarchical": {"enabled": True},
                "adaptive_shaping": {"enabled": True},
                "information_theoretic": {"enabled": True}
            }
        }
    }
    
    manager = StrategyManager(config)
    manager.enable_ab_testing(['control', 'temporal', 'hierarchical'])
    
    # Simulate multiple executions with different groups
    for i in range(30):
        group = ['control', 'temporal', 'hierarchical'][i % 3]
        
        state, next_state = create_sample_states()
        action = ["tool1", "tool2"]
        results = create_sample_execution_results(
            ["all_success", "partial_success", "failure"][i % 3]
        )
        
        context = {
            "ab_test_group": group,
            "execution_id": f"exec_{i}"
        }
        
        reward, _ = manager.calculate_reward(state, action, next_state, results, context)
    
    # Get A/B test results
    ab_results = manager.get_ab_test_results()
    logger.info("\nA/B Test Results:")
    for group, stats in ab_results.items():
        logger.info(f"\n{group}:")
        logger.info(f"  Average Reward: {stats['avg_reward']:.3f}")
        logger.info(f"  Std Dev: {stats['std_reward']:.3f}")
        logger.info(f"  Sample Size: {stats['sample_size']}")
        if 'significant' in stats:
            logger.info(f"  Statistically Significant: {stats['significant']}")


def demo_performance_analysis():
    """Analyze performance of different strategies."""
    logger.info("\n=== Strategy Performance Analysis ===")
    
    config = {
        "advanced_reward_strategies": {
            "enabled": True,
            "strategies": {
                "temporal_difference": {"enabled": True},
                "hierarchical": {"enabled": True},
                "adaptive_shaping": {"enabled": True},
                "information_theoretic": {"enabled": True}
            }
        }
    }
    
    manager = StrategyManager(config)
    
    # Run multiple scenarios
    scenarios_tested = 0
    for _ in range(20):
        state, next_state = create_sample_states()
        scenario = np.random.choice(["all_success", "partial_success", "exploration", "failure"])
        results = create_sample_execution_results(scenario)
        action = [r.tool_id for r in results]
        context = {"scenario": scenario}
        
        reward, breakdown = manager.calculate_reward(state, action, next_state, results, context)
        scenarios_tested += 1
    
    # Get performance report
    performance = manager.get_strategy_performance_report()
    
    logger.info(f"\nTested {scenarios_tested} scenarios")
    logger.info("\nStrategy Performance Report:")
    for strategy, metrics in performance.items():
        logger.info(f"\n{strategy}:")
        logger.info(f"  Average Reward: {metrics['average_reward']:.3f}")
        logger.info(f"  Executions: {metrics['execution_count']}")
        logger.info(f"  Avg Computation Time: {metrics['avg_computation_time_ms']:.2f}ms")
        logger.info(f"  Current Weight: {metrics['current_weight']:.3f}")


def main():
    """Run all demonstrations."""
    logger.info("Advanced Reward Calculation Strategies Demo")
    logger.info("=" * 50)
    
    # Run demonstrations
    demo_basic_vs_advanced()
    demo_strategy_specific_features()
    demo_ab_testing()
    demo_performance_analysis()
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()