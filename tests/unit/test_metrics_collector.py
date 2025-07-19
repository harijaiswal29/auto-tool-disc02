"""Unit tests for metrics collector."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

from src.evaluation.metrics_collector import (
    MetricsCollector, EpisodeMetrics, AggregatedMetrics
)


class TestEpisodeMetrics:
    """Test EpisodeMetrics dataclass."""
    
    def test_episode_metrics_creation(self):
        """Test creating episode metrics."""
        timestamp = datetime.now()
        metrics = EpisodeMetrics(
            strategy_name='test_strategy',
            episode_id=1,
            reward=0.8,
            tools_selected=['tool1', 'tool2'],
            selection_time=0.15,
            timestamp=timestamp,
            scenario_id='scenario_001'
        )
        
        assert metrics.strategy_name == 'test_strategy'
        assert metrics.episode_id == 1
        assert metrics.reward == 0.8
        assert metrics.tools_selected == ['tool1', 'tool2']
        assert metrics.selection_time == 0.15
        assert metrics.timestamp == timestamp
        assert metrics.scenario_id == 'scenario_001'
    
    def test_episode_metrics_to_dict(self):
        """Test converting episode metrics to dictionary."""
        timestamp = datetime.now()
        metrics = EpisodeMetrics(
            strategy_name='test',
            episode_id=1,
            reward=0.5,
            tools_selected=['tool1'],
            selection_time=0.1,
            timestamp=timestamp
        )
        
        data = metrics.to_dict()
        
        assert data['strategy_name'] == 'test'
        assert data['episode_id'] == 1
        assert data['reward'] == 0.5
        assert data['tools_selected'] == ['tool1']
        assert data['selection_time'] == 0.1
        assert data['timestamp'] == timestamp.isoformat()


class TestAggregatedMetrics:
    """Test AggregatedMetrics dataclass."""
    
    def test_aggregated_metrics_creation(self):
        """Test creating aggregated metrics."""
        metrics = AggregatedMetrics(
            strategy_name='test',
            num_episodes=100,
            mean_reward=0.75,
            std_reward=0.15,
            mean_selection_time=0.2,
            convergence_rate=0.8,
            sample_efficiency=0.9,
            exploration_efficiency=0.7
        )
        
        assert metrics.strategy_name == 'test'
        assert metrics.num_episodes == 100
        assert metrics.mean_reward == 0.75
        assert metrics.convergence_rate == 0.8
    
    def test_aggregated_metrics_to_dict(self):
        """Test converting to dictionary."""
        metrics = AggregatedMetrics(
            strategy_name='test',
            num_episodes=50,
            mean_reward=0.6,
            std_reward=0.1,
            mean_selection_time=0.15
        )
        
        data = metrics.to_dict()
        
        assert data['strategy_name'] == 'test'
        assert data['num_episodes'] == 50
        assert data['mean_reward'] == 0.6
        assert data['convergence_rate'] is None


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector."""
        config = {}
        return MetricsCollector(config)
    
    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.config == {}
        assert len(collector.episode_metrics) == 0
        assert len(collector.window_metrics) == 0
        assert len(collector.cumulative_metrics) == 0
        assert isinstance(collector.start_time, datetime)
    
    def test_record_episode(self, collector):
        """Test recording episode metrics."""
        collector.record_episode(
            strategy_name='test',
            episode_id=1,
            reward=0.8,
            tools_selected=['tool1', 'tool2'],
            selection_time=0.15,
            scenario_id='s1'
        )
        
        assert len(collector.episode_metrics['test']) == 1
        assert len(collector.window_metrics['test']) == 1
        assert collector.performance_history['test'] == [0.8]
        assert collector.tool_usage_counts['test']['tool1'] == 1
        assert collector.tool_usage_counts['test']['tool2'] == 1
        assert collector.scenario_performance['test']['s1'] == [0.8]
    
    def test_window_metrics_limit(self, collector):
        """Test that window metrics respect maxlen."""
        # Record 150 episodes (window size is 100)
        for i in range(150):
            collector.record_episode('test', i, 0.5, ['tool1'], 0.1)
        
        assert len(collector.window_metrics['test']) == 100
        assert len(collector.episode_metrics['test']) == 150
    
    def test_cumulative_metrics_update(self, collector):
        """Test cumulative metrics update."""
        # Record episodes to trigger update
        for i in range(11):  # Update happens every 10 episodes
            collector.record_episode('test', i, 0.5 + i * 0.05, ['tool1'], 0.1)
        
        assert 'test' in collector.cumulative_metrics
        metrics = collector.cumulative_metrics['test']
        assert metrics['total_episodes'] == 11
        assert metrics['mean_reward'] == pytest.approx(0.75, 0.01)
        assert 'convergence_metrics' in metrics
        assert 'efficiency_metrics' in metrics
    
    def test_calculate_convergence_metrics(self, collector):
        """Test convergence metrics calculation."""
        # Non-converged case
        rewards1 = [0.5 + i * 0.01 for i in range(100)]
        conv1 = collector._calculate_convergence_metrics(rewards1)
        assert not conv1['converged']
        
        # Converged case (constant rewards)
        rewards2 = [0.8] * 100
        conv2 = collector._calculate_convergence_metrics(rewards2)
        assert conv2['converged']
        assert conv2['final_performance'] == 0.8
        assert conv2['performance_variance'] < 0.01
    
    def test_calculate_efficiency_metrics(self, collector):
        """Test efficiency metrics calculation."""
        # Create test episodes
        episodes = []
        for i in range(10):
            episodes.append(EpisodeMetrics(
                strategy_name='test',
                episode_id=i,
                reward=0.8,
                tools_selected=['tool1', 'tool2'] if i % 2 == 0 else ['tool3'],
                selection_time=0.1,
                timestamp=datetime.now()
            ))
        
        metrics = collector._calculate_efficiency_metrics(episodes)
        
        assert metrics['sample_efficiency'] == 0.8
        assert metrics['unique_combinations'] == 2
        assert metrics['exploration_efficiency'] == 0.2
        assert 'cumulative_regret' in metrics
        assert 'milestones' in metrics
    
    def test_get_performance_metrics(self, collector):
        """Test getting aggregated performance metrics."""
        # No data case
        metrics1 = collector.get_performance_metrics('nonexistent')
        assert metrics1.num_episodes == 0
        assert metrics1.mean_reward == 0
        
        # With data
        for i in range(50):
            collector.record_episode('test', i, 0.7, ['tool1'], 0.15)
        
        metrics2 = collector.get_performance_metrics('test')
        assert metrics2.strategy_name == 'test'
        assert metrics2.num_episodes == 50
        assert metrics2.mean_reward == 0.7
        assert metrics2.mean_selection_time == 0.15
    
    def test_get_learning_curves(self, collector):
        """Test learning curves generation."""
        # Add performance history
        for i in range(200):
            reward = 0.5 + 0.3 * (1 - np.exp(-i/50))  # Learning curve
            collector.performance_history['test'].append(reward)
        
        curves = collector.get_learning_curves(window_size=50)
        
        assert 'test' in curves
        assert len(curves['test']) == 151  # 200 - 50 + 1
        # Should be smoothed (increasing trend)
        assert curves['test'][-1] > curves['test'][0]
    
    def test_get_tool_usage_statistics(self, collector):
        """Test tool usage statistics."""
        # Record tool usage
        collector.tool_usage_counts['test'] = {
            'tool1': 50,
            'tool2': 30,
            'tool3': 20
        }
        
        stats = collector.get_tool_usage_statistics()
        
        assert 'test' in stats
        test_stats = stats['test']
        assert test_stats['total_selections'] == 100
        assert test_stats['unique_tools'] == 3
        assert test_stats['usage_percentages']['tool1'] == 50.0
        assert test_stats['most_used_tool'] == ('tool1', 50)
        assert test_stats['least_used_tool'] == ('tool3', 20)
        assert test_stats['usage_entropy'] > 0
    
    def test_calculate_usage_entropy(self, collector):
        """Test entropy calculation."""
        # Uniform distribution (high entropy)
        counts1 = {'tool1': 33, 'tool2': 33, 'tool3': 34}
        entropy1 = collector._calculate_usage_entropy(counts1)
        
        # Skewed distribution (low entropy)
        counts2 = {'tool1': 90, 'tool2': 5, 'tool3': 5}
        entropy2 = collector._calculate_usage_entropy(counts2)
        
        assert entropy1 > entropy2
        assert entropy1 > 1.5  # Close to log2(3) ≈ 1.58
        assert entropy2 < 1.0
    
    def test_get_scenario_analysis(self, collector):
        """Test scenario performance analysis."""
        # Add scenario data
        collector.scenario_performance['test'] = {
            'scenario1': [0.8, 0.9, 0.85],
            'scenario2': [0.3, 0.4, 0.35],
            'scenario3': [0.6, 0.7]
        }
        
        analysis = collector.get_scenario_analysis()
        
        assert 'test' in analysis
        test_analysis = analysis['test']
        
        assert test_analysis['best_scenario'][0] == 'scenario1'
        assert test_analysis['worst_scenario'][0] == 'scenario2'
        assert test_analysis['performance_variance'] > 0
        
        # Check scenario stats
        assert test_analysis['scenario_stats']['scenario1']['mean_reward'] == pytest.approx(0.85, 0.01)
        assert test_analysis['scenario_stats']['scenario1']['success_rate'] == 1.0
    
    def test_calculate_relative_metrics(self, collector):
        """Test relative metrics calculation."""
        # Add baseline data
        for i in range(50):
            collector.record_episode('random', i, 0.5, ['tool1'], 0.2)
            collector.record_episode('strategy1', i, 0.7, ['tool1'], 0.15)
        
        relative = collector.calculate_relative_metrics('random')
        
        assert 'strategy1' in relative
        metrics = relative['strategy1']
        
        assert metrics['reward_improvement'] == pytest.approx(0.2, 0.01)
        assert metrics['reward_improvement_pct'] == pytest.approx(40.0, 0.1)
        assert metrics['time_difference'] == pytest.approx(-0.05, 0.01)
        assert metrics['time_ratio'] == pytest.approx(0.75, 0.01)
    
    def test_export_to_dataframe(self, collector):
        """Test DataFrame export."""
        # Add some episodes
        for i in range(5):
            collector.record_episode(
                strategy_name='test',
                episode_id=i,
                reward=0.5 + i * 0.1,
                tools_selected=['tool1'],
                selection_time=0.1,
                scenario_id=f's{i}'
            )
        
        df = collector.export_to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'strategy_name' in df.columns
        assert 'reward' in df.columns
        assert 'tools_selected' in df.columns
        assert df['reward'].mean() == 0.7
    
    def test_get_summary_statistics(self, collector):
        """Test comprehensive summary statistics."""
        # Add data for multiple strategies
        for strategy in ['random', 'test']:
            for i in range(20):
                reward = 0.5 if strategy == 'random' else 0.7
                collector.record_episode(strategy, i, reward, ['tool1'], 0.1)
        
        summary = collector.get_summary_statistics()
        
        assert summary['total_episodes'] == 40
        assert summary['strategies_evaluated'] == 2
        assert 'performance_summary' in summary
        assert 'learning_summary' in summary
        assert 'efficiency_summary' in summary
        
        # Check learning summary
        for strategy in ['random', 'test']:
            assert strategy in summary['learning_summary']
            learning = summary['learning_summary'][strategy]
            assert 'early_performance' in learning
            assert 'late_performance' in learning
            assert 'improvement' in learning