#!/usr/bin/env python3
"""Test H1: Q-learning achieves >30% improvement over baseline strategies.

This test validates the primary research hypothesis by comparing Q-learning
performance against multiple baseline strategies with statistical validation.
"""

import pytest
import numpy as np
from scipy import stats
import asyncio
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.baseline_strategies import (
    RandomStrategy, PopularToolsStrategy, FixedPolicyStrategy,
    GreedyStrategy, ContextAgnosticQLearning
)
from src.orchestration.orchestrator_agent import OrchestratorAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.hypothesis
@pytest.mark.asyncio
class TestLearningImprovement:
    """Test suite for validating H1: Q-learning improvement hypothesis."""
    
    @pytest.fixture
    async def evaluation_engine(self):
        """Create evaluation engine for testing."""
        config = {
            'episodes': 100,  # Reduced for testing, use 1000 for dissertation
            'random_seed': 42,
            'confidence_level': 0.95,
            'min_improvement': 0.30  # 30% improvement threshold
        }
        engine = EvaluationEngine(config)
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def test_queries(self):
        """Standard test queries for evaluation."""
        return [
            "Find all Python files in the project",
            "Search for TODO comments in code",
            "Get current weather in London",
            "Query database for user information",
            "Analyze code complexity metrics",
            "Find and fix security vulnerabilities",
            "Generate API documentation",
            "Run performance benchmarks"
        ]
    
    async def test_qlearning_vs_random_baseline(self, evaluation_engine, test_queries):
        """Test that Q-learning significantly outperforms random selection."""
        logger.info("Testing H1: Q-learning vs Random Baseline")
        
        # Run experiments
        qlearning_results = []
        random_results = []
        
        # Multiple runs for statistical validity
        num_runs = 5  # Use 30 for dissertation
        
        for run in range(num_runs):
            # Q-learning performance
            ql_metrics = await evaluation_engine.evaluate_strategy(
                'qlearning',
                test_queries,
                episodes=100
            )
            qlearning_results.append(ql_metrics['task_completion_rate'])
            
            # Random baseline
            random_metrics = await evaluation_engine.evaluate_strategy(
                'random',
                test_queries,
                episodes=100
            )
            random_results.append(random_metrics['task_completion_rate'])
        
        # Statistical analysis
        ql_mean = np.mean(qlearning_results)
        random_mean = np.mean(random_results)
        improvement = (ql_mean - random_mean) / random_mean
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(qlearning_results, random_results)
        
        # Cohen's d for effect size
        pooled_std = np.sqrt((np.std(qlearning_results)**2 + np.std(random_results)**2) / 2)
        cohens_d = (ql_mean - random_mean) / pooled_std
        
        # Log results
        logger.info(f"Q-learning mean: {ql_mean:.3f}")
        logger.info(f"Random mean: {random_mean:.3f}")
        logger.info(f"Improvement: {improvement:.1%}")
        logger.info(f"p-value: {p_value:.4f}")
        logger.info(f"Cohen's d: {cohens_d:.3f}")
        
        # Assertions
        assert improvement > 0.30, f"Improvement {improvement:.1%} < 30%"
        assert p_value < 0.05, f"Not statistically significant (p={p_value:.4f})"
        assert cohens_d > 0.8, f"Effect size too small (d={cohens_d:.3f})"
        
        # Save results for dissertation
        results = {
            'hypothesis': 'H1',
            'comparison': 'qlearning_vs_random',
            'qlearning_mean': ql_mean,
            'baseline_mean': random_mean,
            'improvement': improvement,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'num_runs': num_runs
        }
        self._save_results(results)
    
    async def test_qlearning_vs_all_baselines(self, evaluation_engine, test_queries):
        """Test Q-learning against all baseline strategies."""
        logger.info("Testing H1: Q-learning vs All Baselines")
        
        strategies = {
            'qlearning': None,
            'random': RandomStrategy(),
            'popular': PopularToolsStrategy(),
            'fixed_policy': FixedPolicyStrategy(),
            'greedy': GreedyStrategy(),
            'context_agnostic': ContextAgnosticQLearning()
        }
        
        results = {}
        num_runs = 5  # Use 30 for dissertation
        
        for strategy_name, strategy in strategies.items():
            run_results = []
            
            for run in range(num_runs):
                metrics = await evaluation_engine.evaluate_strategy(
                    strategy_name,
                    test_queries,
                    episodes=100,
                    strategy_instance=strategy
                )
                run_results.append(metrics['task_completion_rate'])
            
            results[strategy_name] = {
                'mean': np.mean(run_results),
                'std': np.std(run_results),
                'values': run_results
            }
        
        # Compare Q-learning to each baseline
        ql_mean = results['qlearning']['mean']
        comparisons = []
        
        for baseline_name, baseline_data in results.items():
            if baseline_name == 'qlearning':
                continue
                
            baseline_mean = baseline_data['mean']
            improvement = (ql_mean - baseline_mean) / baseline_mean
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(
                results['qlearning']['values'],
                baseline_data['values']
            )
            
            # Effect size
            pooled_std = np.sqrt(
                (results['qlearning']['std']**2 + baseline_data['std']**2) / 2
            )
            cohens_d = (ql_mean - baseline_mean) / pooled_std
            
            comparisons.append({
                'baseline': baseline_name,
                'improvement': improvement,
                'p_value': p_value,
                'cohens_d': cohens_d
            })
            
            logger.info(f"Q-learning vs {baseline_name}: "
                       f"improvement={improvement:.1%}, "
                       f"p={p_value:.4f}, d={cohens_d:.3f}")
        
        # Assert all comparisons show significant improvement
        for comp in comparisons:
            if comp['baseline'] in ['random', 'greedy']:  # Expected large improvements
                assert comp['improvement'] > 0.30, \
                    f"Insufficient improvement vs {comp['baseline']}"
            assert comp['p_value'] < 0.05, \
                f"Not significant vs {comp['baseline']}"
        
        # Save comprehensive results
        self._save_results({
            'hypothesis': 'H1',
            'comparison': 'all_baselines',
            'results': results,
            'comparisons': comparisons
        })
    
    async def test_learning_curve_superiority(self, evaluation_engine, test_queries):
        """Test that Q-learning shows superior learning curves."""
        logger.info("Testing H1: Learning Curve Analysis")
        
        episodes = [10, 50, 100, 200, 500]
        
        # Track performance over episodes
        qlearning_curve = []
        random_curve = []
        
        for ep in episodes:
            # Q-learning
            ql_metrics = await evaluation_engine.evaluate_strategy(
                'qlearning',
                test_queries,
                episodes=ep
            )
            qlearning_curve.append(ql_metrics['task_completion_rate'])
            
            # Random baseline (should be flat)
            random_metrics = await evaluation_engine.evaluate_strategy(
                'random',
                test_queries,
                episodes=ep
            )
            random_curve.append(random_metrics['task_completion_rate'])
        
        # Calculate improvement rate
        ql_improvement_rate = (qlearning_curve[-1] - qlearning_curve[0]) / qlearning_curve[0]
        random_improvement_rate = (random_curve[-1] - random_curve[0]) / random_curve[0]
        
        logger.info(f"Q-learning improvement rate: {ql_improvement_rate:.1%}")
        logger.info(f"Random improvement rate: {random_improvement_rate:.1%}")
        
        # Q-learning should show significant improvement
        assert ql_improvement_rate > 0.20, "Q-learning not improving sufficiently"
        assert ql_improvement_rate > random_improvement_rate + 0.15, \
            "Q-learning not outpacing random baseline"
        
        # Save learning curves
        self._save_results({
            'hypothesis': 'H1',
            'test': 'learning_curves',
            'episodes': episodes,
            'qlearning_curve': qlearning_curve,
            'random_curve': random_curve,
            'ql_improvement_rate': ql_improvement_rate
        })
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "hypothesis_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"h1_results_{results.get('comparison', 'test')}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / filename}")


if __name__ == "__main__":
    # Run tests directly for dissertation data collection
    pytest.main([__file__, "-v", "-s", "--tb=short"])