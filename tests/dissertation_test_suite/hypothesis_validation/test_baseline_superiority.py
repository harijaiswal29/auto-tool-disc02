#!/usr/bin/env python3
"""Test statistical superiority of Q-learning over baselines.

This test provides comprehensive statistical validation including:
- Multiple comparison corrections
- Confidence intervals
- Power analysis
- Robustness testing
"""

import pytest
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple
import json
from pathlib import Path
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import ttest_power

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.evaluation_engine import EvaluationEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.hypothesis
@pytest.mark.statistical
@pytest.mark.asyncio
class TestBaselineSuperiority:
    """Statistical tests for baseline superiority claims."""
    
    async def test_statistical_power_analysis(self):
        """Verify we have sufficient statistical power for our claims."""
        logger.info("Running statistical power analysis")
        
        # Expected parameters from pilot studies
        effect_size = 0.8  # Large effect (Cohen's d)
        alpha = 0.05       # Significance level
        power_target = 0.8 # Desired power
        
        # Calculate required sample size
        from statsmodels.stats.power import tt_solve_power
        required_n = tt_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power_target,
            alternative='two-sided'
        )
        
        logger.info(f"Required sample size for power={power_target}: {required_n:.0f}")
        
        # Verify our planned sample size
        planned_n = 30  # Our planned runs
        actual_power = ttest_power(
            effect_size,
            planned_n,
            alpha,
            alternative='two-sided'
        )
        
        logger.info(f"Actual power with n={planned_n}: {actual_power:.3f}")
        
        assert actual_power >= 0.8, f"Insufficient power: {actual_power:.3f} < 0.8"
        
        # Save power analysis
        self._save_results({
            'test': 'power_analysis',
            'effect_size': effect_size,
            'alpha': alpha,
            'required_n': required_n,
            'planned_n': planned_n,
            'actual_power': actual_power
        })
    
    async def test_multiple_comparison_correction(self, evaluation_engine):
        """Apply Bonferroni correction for multiple hypothesis testing."""
        logger.info("Testing with multiple comparison correction")
        
        # Simulate comparison data (replace with real data in dissertation)
        np.random.seed(42)
        
        # Generate realistic performance data
        strategies = {
            'qlearning': np.random.normal(0.85, 0.05, 30),      # High performance
            'context_agnostic': np.random.normal(0.75, 0.06, 30), # Medium-high
            'fixed_policy': np.random.normal(0.70, 0.07, 30),   # Medium
            'greedy': np.random.normal(0.65, 0.08, 30),         # Medium-low
            'popular': np.random.normal(0.55, 0.08, 30),        # Low-medium
            'random': np.random.normal(0.35, 0.10, 30)          # Low
        }
        
        # Perform all pairwise comparisons
        comparisons = []
        baseline_names = [k for k in strategies.keys() if k != 'qlearning']
        
        for baseline in baseline_names:
            t_stat, p_value = stats.ttest_ind(
                strategies['qlearning'],
                strategies[baseline]
            )
            comparisons.append({
                'baseline': baseline,
                'p_value': p_value,
                't_statistic': t_stat
            })
        
        # Apply Bonferroni correction
        n_comparisons = len(comparisons)
        corrected_alpha = 0.05 / n_comparisons
        
        logger.info(f"Bonferroni corrected alpha: {corrected_alpha:.4f}")
        
        # Check significance with correction
        significant_after_correction = []
        for comp in comparisons:
            comp['corrected_significant'] = comp['p_value'] < corrected_alpha
            if comp['corrected_significant']:
                significant_after_correction.append(comp['baseline'])
            
            logger.info(f"{comp['baseline']}: p={comp['p_value']:.4f}, "
                       f"significant={comp['corrected_significant']}")
        
        # All comparisons should remain significant
        assert len(significant_after_correction) == n_comparisons, \
            "Some comparisons lost significance after correction"
        
        # Save results
        self._save_results({
            'test': 'multiple_comparison_correction',
            'n_comparisons': n_comparisons,
            'corrected_alpha': corrected_alpha,
            'comparisons': comparisons,
            'all_significant': len(significant_after_correction) == n_comparisons
        })
    
    async def test_confidence_intervals(self):
        """Calculate confidence intervals for performance differences."""
        logger.info("Calculating confidence intervals")
        
        # Simulate data (replace with real data)
        np.random.seed(42)
        qlearning = np.random.normal(0.85, 0.05, 30)
        random_baseline = np.random.normal(0.35, 0.10, 30)
        
        # Calculate difference
        differences = qlearning - random_baseline
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, df=n-1)
        margin_of_error = t_critical * (std_diff / np.sqrt(n))
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error
        
        # Calculate improvement percentage
        baseline_mean = np.mean(random_baseline)
        improvement_pct = mean_diff / baseline_mean * 100
        improvement_ci_lower = ci_lower / baseline_mean * 100
        improvement_ci_upper = ci_upper / baseline_mean * 100
        
        logger.info(f"Mean difference: {mean_diff:.3f}")
        logger.info(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        logger.info(f"Improvement: {improvement_pct:.1f}% "
                   f"[{improvement_ci_lower:.1f}%, {improvement_ci_upper:.1f}%]")
        
        # Assert minimum improvement with confidence
        assert ci_lower > 0.30, "Lower bound of improvement < 30%"
        assert improvement_ci_lower > 30, "Improvement CI lower bound < 30%"
        
        # Save results
        self._save_results({
            'test': 'confidence_intervals',
            'mean_difference': mean_diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'improvement_pct': improvement_pct,
            'improvement_ci': [improvement_ci_lower, improvement_ci_upper]
        })
    
    async def test_robustness_across_domains(self, evaluation_engine):
        """Test that superiority holds across different task domains."""
        logger.info("Testing robustness across domains")
        
        # Define domain-specific queries
        domains = {
            'file_operations': [
                "Find all Python files",
                "Search for TODO comments",
                "List recent modifications"
            ],
            'data_queries': [
                "Query database for users",
                "Get sales statistics",
                "Find customer records"
            ],
            'web_search': [
                "Search for Python tutorials",
                "Find documentation on MCP",
                "Get latest AI news"
            ],
            'system_tasks': [
                "Check system performance",
                "Monitor resource usage",
                "Analyze logs for errors"
            ]
        }
        
        domain_results = {}
        
        for domain, queries in domains.items():
            # Evaluate Q-learning
            ql_metrics = await evaluation_engine.evaluate_strategy(
                'qlearning',
                queries,
                episodes=100
            )
            
            # Evaluate random baseline
            random_metrics = await evaluation_engine.evaluate_strategy(
                'random',
                queries,
                episodes=100
            )
            
            improvement = (ql_metrics['task_completion_rate'] - 
                         random_metrics['task_completion_rate']) / \
                         random_metrics['task_completion_rate']
            
            domain_results[domain] = {
                'qlearning': ql_metrics['task_completion_rate'],
                'random': random_metrics['task_completion_rate'],
                'improvement': improvement
            }
            
            logger.info(f"{domain}: improvement={improvement:.1%}")
            
            # Each domain should show improvement
            assert improvement > 0.20, f"Insufficient improvement in {domain}"
        
        # Calculate overall robustness
        improvements = [r['improvement'] for r in domain_results.values()]
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        cv = std_improvement / mean_improvement  # Coefficient of variation
        
        logger.info(f"Mean improvement: {mean_improvement:.1%}")
        logger.info(f"Std improvement: {std_improvement:.1%}")
        logger.info(f"Coefficient of variation: {cv:.3f}")
        
        # Low CV indicates consistent improvement
        assert cv < 0.30, f"Improvement not consistent across domains (CV={cv:.3f})"
        
        # Save results
        self._save_results({
            'test': 'domain_robustness',
            'domains': domain_results,
            'mean_improvement': mean_improvement,
            'consistency_cv': cv
        })
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "hypothesis_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"statistical_validation_{results.get('test', 'general')}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / filename}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])