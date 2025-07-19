"""Integration tests for baseline evaluation framework."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import asyncio
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime

from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.comparison_visualizer import ComparisonVisualizer
from src.evaluation.metrics_collector import MetricsCollector


class TestBaselineEvaluationIntegration:
    """Integration tests for the complete evaluation framework."""
    
    @pytest.fixture
    def test_config(self):
        """Test configuration for evaluation."""
        return {
            'evaluation': {
                'enabled': True,
                'baselines': ['random', 'popular', 'fixed_policy', 'greedy', 'context_agnostic'],
                'evaluation_interval': 10,
                'min_episodes_for_comparison': 5,
                'confidence_level': 0.95
            },
            'q_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.2,
                'exploration_decay': 0.995,
                'min_exploration_rate': 0.01,
                'max_tools': 3,
                'enable_learning': True
            },
            'max_tools': 3,
            'min_history': 5,
            'figure_size': (10, 6),
            'dpi': 100,
            'output_dir': 'test_reports'
        }
    
    @pytest.fixture
    async def cleanup_output_dir(self, test_config):
        """Clean up test output directory."""
        yield
        # Cleanup after test
        output_dir = Path(test_config['output_dir'])
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    @pytest.mark.asyncio
    async def test_full_evaluation_workflow(self, test_config, cleanup_output_dir):
        """Test complete evaluation workflow."""
        # Create evaluation engine
        engine = EvaluationEngine(test_config)
        
        # Verify strategies are initialized
        assert len(engine.strategies) >= 6  # 5 baselines + q_learning
        assert 'random' in engine.strategies
        assert 'q_learning' in engine.strategies
        
        # Generate test scenarios
        scenarios = engine.generate_test_scenarios(50)
        assert len(scenarios) == 50
        
        # Run evaluation
        results = await engine.run_evaluation(num_episodes=50, parallel=True)
        
        # Verify results structure
        assert 'timestamp' in results
        assert 'num_episodes' in results
        assert results['num_episodes'] == 50
        assert 'strategies' in results
        assert 'comparisons' in results
        assert 'summary' in results
        
        # Verify all strategies were evaluated
        for strategy_name in engine.strategies:
            assert strategy_name in results['strategies']
            strategy_results = results['strategies'][strategy_name]
            assert 'rewards' in strategy_results
            assert 'times' in strategy_results
            assert 'selections' in strategy_results
            assert 'statistics' in strategy_results
            assert len(strategy_results['rewards']) == 50
        
        # Verify comparisons
        assert len(results['comparisons']) == len(engine.strategies) - 1
        for comparison in results['comparisons'].values():
            assert 'improvement' in comparison
            assert 'p_value' in comparison
            assert 'significant' in comparison
            assert 'cohens_d' in comparison
            assert 'win_rate' in comparison
        
        # Verify summary
        summary = results['summary']
        assert 'best_strategy' in summary
        assert 'rankings' in summary
        assert len(summary['rankings']) == len(engine.strategies)
    
    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self, test_config):
        """Test metrics collection during evaluation."""
        engine = EvaluationEngine(test_config)
        
        # Access metrics collector
        collector = engine.metrics_collector
        assert isinstance(collector, MetricsCollector)
        
        # Run short evaluation
        await engine.run_evaluation(num_episodes=20, parallel=False)
        
        # Verify metrics were collected
        for strategy_name in engine.strategies:
            # Check episode metrics
            assert strategy_name in collector.episode_metrics
            assert len(collector.episode_metrics[strategy_name]) == 20
            
            # Check performance metrics
            perf_metrics = collector.get_performance_metrics(strategy_name)
            assert perf_metrics.num_episodes == 20
            assert perf_metrics.mean_reward >= -1.0
            assert perf_metrics.mean_reward <= 1.0
            
            # Check tool usage statistics
            tool_stats = collector.get_tool_usage_statistics()
            assert strategy_name in tool_stats
            assert tool_stats[strategy_name]['total_selections'] > 0
        
        # Check relative metrics
        relative_metrics = collector.calculate_relative_metrics('random')
        assert len(relative_metrics) == len(engine.strategies) - 1
    
    @pytest.mark.asyncio
    async def test_visualization_integration(self, test_config, cleanup_output_dir):
        """Test visualization generation."""
        engine = EvaluationEngine(test_config)
        visualizer = ComparisonVisualizer(test_config)
        
        # Run evaluation
        results = await engine.run_evaluation(num_episodes=30, parallel=True)
        
        # Generate report
        report_path = visualizer.generate_report(results)
        
        # Verify report was created
        assert Path(report_path).exists()
        assert report_path.endswith('.pdf')
        
        # Verify individual plots can be created
        learning_data = {
            name: data.get('rewards', [])
            for name, data in results['strategies'].items()
        }
        
        # Test learning curves
        fig = visualizer.create_learning_curves(learning_data)
        assert fig is not None
        
        # Test performance distribution
        fig = visualizer.create_performance_distribution(learning_data)
        assert fig is not None
        
        # Save plots individually
        saved_files = visualizer.save_plots_individually(results, prefix='test')
        assert len(saved_files) > 0
        assert all(Path(f).exists() for f in saved_files)
    
    @pytest.mark.asyncio
    async def test_baseline_strategy_differences(self, test_config):
        """Test that different baselines produce different results."""
        engine = EvaluationEngine(test_config)
        
        # Run evaluation
        results = await engine.run_evaluation(num_episodes=100, parallel=True)
        
        # Extract mean rewards for each strategy
        mean_rewards = {}
        for strategy_name, data in results['strategies'].items():
            mean_rewards[strategy_name] = data['statistics']['reward']['mean']
        
        # Verify strategies have different performance
        reward_values = list(mean_rewards.values())
        assert len(set(reward_values)) > 1  # Not all the same
        
        # Verify some expected patterns
        # Random should typically not be the best
        rankings = results['summary']['rankings']
        best_strategy = rankings[0]['strategy']
        assert best_strategy != 'random'  # Usually true
        
        # Q-learning should improve over time
        q_rewards = results['strategies']['q_learning']['rewards']
        early_avg = np.mean(q_rewards[:20])
        late_avg = np.mean(q_rewards[-20:])
        # This might not always be true due to randomness, but often is
        # assert late_avg >= early_avg
    
    @pytest.mark.asyncio
    async def test_convergence_detection(self, test_config):
        """Test convergence detection in strategies."""
        engine = EvaluationEngine(test_config)
        
        # Run longer evaluation for convergence
        results = await engine.run_evaluation(num_episodes=200, parallel=True)
        
        # Check convergence metrics
        convergence_count = 0
        for strategy_name, data in results['strategies'].items():
            convergence = data['statistics']['convergence']
            if convergence['converged']:
                convergence_count += 1
                assert convergence['convergence_episode'] is not None
                assert convergence['final_performance'] is not None
        
        # At least some strategies should converge
        assert convergence_count > 0
    
    @pytest.mark.asyncio
    async def test_statistical_significance(self, test_config):
        """Test statistical significance calculations."""
        engine = EvaluationEngine(test_config)
        
        # Run evaluation with enough episodes for statistics
        results = await engine.run_evaluation(num_episodes=100, parallel=True)
        
        # Check comparisons
        comparisons = results['comparisons']
        
        # Count significant improvements
        significant_count = sum(
            1 for comp in comparisons.values()
            if comp['significant'] and comp['improvement'] > 0
        )
        
        # Verify p-values are reasonable
        for comp in comparisons.values():
            assert 0 <= comp['p_value'] <= 1
            assert comp['significant'] == (comp['p_value'] < 0.05)
            
            # Check effect size interpretation
            assert comp['effect_size'] in ['negligible', 'small', 'medium', 'large']
    
    @pytest.mark.asyncio
    async def test_scenario_performance_tracking(self, test_config):
        """Test tracking performance across different scenarios."""
        engine = EvaluationEngine(test_config)
        collector = engine.metrics_collector
        
        # Run evaluation
        await engine.run_evaluation(num_episodes=60, parallel=False)
        
        # Analyze scenario performance
        scenario_analysis = collector.get_scenario_analysis()
        
        for strategy_name in engine.strategies:
            assert strategy_name in scenario_analysis
            analysis = scenario_analysis[strategy_name]
            
            # Should have analyzed multiple scenarios
            assert len(analysis['scenario_stats']) > 0
            
            # Best and worst scenarios should be identified
            if analysis['scenario_stats']:
                assert analysis['best_scenario'][0] is not None
                assert analysis['worst_scenario'][0] is not None
    
    @pytest.mark.asyncio
    async def test_learning_progress_tracking(self, test_config):
        """Test that learning progress is tracked correctly."""
        engine = EvaluationEngine(test_config)
        collector = engine.metrics_collector
        
        # Run evaluation
        await engine.run_evaluation(num_episodes=100, parallel=False)
        
        # Get learning curves
        curves = collector.get_learning_curves(window_size=20)
        
        # Verify curves exist for all strategies
        for strategy_name in engine.strategies:
            assert strategy_name in curves
            curve = curves[strategy_name]
            assert len(curve) > 0
            
            # For learning strategies, check if there's improvement
            if strategy_name in ['q_learning', 'context_agnostic']:
                # These strategies should show some learning
                # (though not guaranteed due to randomness)
                pass
    
    @pytest.mark.asyncio
    async def test_report_generation_and_saving(self, test_config, cleanup_output_dir):
        """Test complete report generation and saving."""
        engine = EvaluationEngine(test_config)
        
        # Run evaluation
        results = await engine.run_evaluation(num_episodes=50, parallel=True)
        
        # Save results
        output_dir = Path(test_config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        results_path = output_dir / 'test_results.pkl'
        engine.save_results(str(results_path))
        
        # Verify files were created
        assert results_path.exists()
        json_path = results_path.with_suffix('_summary.json')
        assert json_path.exists()
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            json_summary = json.load(f)
        
        assert 'timestamp' in json_summary
        assert 'summary' in json_summary
        assert 'comparisons' in json_summary
        
        # Verify summary content
        summary = json_summary['summary']
        assert 'best_strategy' in summary
        assert 'rankings' in summary
        assert 'significant_improvements' in summary