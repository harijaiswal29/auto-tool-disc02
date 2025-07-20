"""Metrics collector for evaluation framework.

This module collects and aggregates performance metrics during evaluation,
tracking both individual strategy performance and comparative metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import asyncio

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    strategy_name: str
    episode_id: int
    reward: float
    tools_selected: List[str]
    selection_time: float
    timestamp: datetime
    scenario_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over multiple episodes."""
    strategy_name: str
    num_episodes: int
    mean_reward: float
    std_reward: float
    mean_selection_time: float
    convergence_rate: Optional[float] = None
    sample_efficiency: Optional[float] = None
    exploration_efficiency: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_metrics = defaultdict(list)
        self.window_metrics = defaultdict(lambda: deque(maxlen=100))
        self.cumulative_metrics = defaultdict(dict)
        self.comparison_metrics = {}
        self.start_time = datetime.now()
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.tool_usage_counts = defaultdict(lambda: defaultdict(int))
        self.scenario_performance = defaultdict(lambda: defaultdict(list))
        
        # Real-time tracking
        self.real_time_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.metric_observers: List[Callable] = []
        self.baseline_tracker = defaultdict(lambda: {'values': deque(maxlen=100), 'baseline': None})
        self.streaming_enabled = config.get('streaming_enabled', False)
        self.update_interval = config.get('update_interval', 1.0)  # seconds
        
    def record_episode(self, strategy_name: str, episode_id: int, reward: float,
                      tools_selected: List[str], selection_time: float,
                      scenario_id: Optional[str] = None):
        """Record metrics for a single episode."""
        metrics = EpisodeMetrics(
            strategy_name=strategy_name,
            episode_id=episode_id,
            reward=reward,
            tools_selected=tools_selected,
            selection_time=selection_time,
            timestamp=datetime.now(),
            scenario_id=scenario_id
        )
        
        # Store episode metrics
        self.episode_metrics[strategy_name].append(metrics)
        
        # Update window metrics
        self.window_metrics[strategy_name].append(metrics)
        
        # Update performance history
        self.performance_history[strategy_name].append(reward)
        
        # Update tool usage
        for tool in tools_selected:
            self.tool_usage_counts[strategy_name][tool] += 1
            
        # Update scenario performance
        if scenario_id:
            self.scenario_performance[strategy_name][scenario_id].append(reward)
            
        # Update cumulative metrics periodically
        if episode_id % 10 == 0:
            self._update_cumulative_metrics(strategy_name)
            
    def _update_cumulative_metrics(self, strategy_name: str):
        """Update cumulative metrics for a strategy."""
        episodes = self.episode_metrics[strategy_name]
        if not episodes:
            return
            
        rewards = [e.reward for e in episodes]
        times = [e.selection_time for e in episodes]
        
        self.cumulative_metrics[strategy_name] = {
            'total_episodes': len(episodes),
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_time': np.mean(times),
            'convergence_metrics': self._calculate_convergence_metrics(rewards),
            'efficiency_metrics': self._calculate_efficiency_metrics(episodes)
        }
        
    def _calculate_convergence_metrics(self, rewards: List[float]) -> Dict[str, Any]:
        """Calculate convergence-related metrics."""
        if len(rewards) < 10:
            return {'converged': False, 'convergence_rate': None}
            
        # Calculate moving averages
        window_size = min(50, len(rewards) // 4)
        if window_size < 10:
            return {'converged': False, 'convergence_rate': None}
            
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Check convergence (variance in last window < threshold)
        last_window_var = np.var(rewards[-window_size:])
        converged = last_window_var < 0.01
        
        # Calculate convergence rate (episodes to reach 90% of final performance)
        final_performance = np.mean(rewards[-window_size:])
        target_performance = 0.9 * final_performance
        
        convergence_episode = None
        for i in range(window_size, len(moving_avg)):
            if moving_avg[i] >= target_performance:
                convergence_episode = i
                break
                
        convergence_rate = convergence_episode / len(rewards) if convergence_episode else None
        
        return {
            'converged': converged,
            'convergence_rate': convergence_rate,
            'convergence_episode': convergence_episode,
            'final_performance': final_performance,
            'performance_variance': last_window_var
        }
        
    def _calculate_efficiency_metrics(self, episodes: List[EpisodeMetrics]) -> Dict[str, Any]:
        """Calculate efficiency-related metrics."""
        # Sample efficiency: reward per episode
        rewards = [e.reward for e in episodes]
        cumulative_rewards = np.cumsum(rewards)
        
        # Find episodes to reach performance milestones
        milestones = {}
        total_possible_reward = len(episodes)  # Assuming max reward of 1 per episode
        for threshold in [0.5, 0.7, 0.9]:
            target = threshold * total_possible_reward
            milestone_episode = None
            for i, cum_reward in enumerate(cumulative_rewards):
                if cum_reward >= target:
                    milestone_episode = i
                    break
            milestones[f'episodes_to_{int(threshold*100)}pct'] = milestone_episode
            
        # Exploration efficiency: unique tool combinations tried
        tool_combinations = set()
        for episode in episodes:
            combination = tuple(sorted(episode.tools_selected))
            tool_combinations.add(combination)
            
        exploration_efficiency = len(tool_combinations) / len(episodes) if episodes else 0
        
        # Regret calculation (compared to best possible performance)
        best_possible_reward = len(episodes) * 1.0  # Assuming max reward of 1
        actual_total_reward = sum(rewards)
        cumulative_regret = best_possible_reward - actual_total_reward
        
        return {
            'sample_efficiency': np.mean(rewards) if rewards else 0,
            'milestones': milestones,
            'exploration_efficiency': exploration_efficiency,
            'unique_combinations': len(tool_combinations),
            'cumulative_regret': cumulative_regret,
            'regret_per_episode': cumulative_regret / len(episodes) if episodes else 0
        }
        
    def get_performance_metrics(self, strategy_name: str) -> AggregatedMetrics:
        """Get aggregated performance metrics for a strategy."""
        episodes = self.episode_metrics[strategy_name]
        if not episodes:
            return AggregatedMetrics(
                strategy_name=strategy_name,
                num_episodes=0,
                mean_reward=0,
                std_reward=0,
                mean_selection_time=0
            )
            
        rewards = [e.reward for e in episodes]
        times = [e.selection_time for e in episodes]
        
        convergence = self._calculate_convergence_metrics(rewards)
        efficiency = self._calculate_efficiency_metrics(episodes)
        
        return AggregatedMetrics(
            strategy_name=strategy_name,
            num_episodes=len(episodes),
            mean_reward=np.mean(rewards),
            std_reward=np.std(rewards),
            mean_selection_time=np.mean(times),
            convergence_rate=convergence.get('convergence_rate'),
            sample_efficiency=efficiency.get('sample_efficiency'),
            exploration_efficiency=efficiency.get('exploration_efficiency')
        )
        
    def get_learning_curves(self, window_size: int = 100) -> Dict[str, List[float]]:
        """Get smoothed learning curves for all strategies."""
        curves = {}
        
        for strategy_name, rewards in self.performance_history.items():
            if len(rewards) < window_size:
                curves[strategy_name] = rewards
            else:
                # Calculate moving average
                smoothed = np.convolve(
                    rewards, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                curves[strategy_name] = smoothed.tolist()
                
        return curves
        
    def get_tool_usage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get tool usage statistics for each strategy."""
        stats = {}
        
        for strategy_name, tool_counts in self.tool_usage_counts.items():
            total_selections = sum(tool_counts.values())
            
            # Calculate usage percentages
            usage_percentages = {
                tool: count / total_selections * 100
                for tool, count in tool_counts.items()
            } if total_selections > 0 else {}
            
            # Find most/least used tools
            if tool_counts:
                most_used = max(tool_counts.items(), key=lambda x: x[1])
                least_used = min(tool_counts.items(), key=lambda x: x[1])
            else:
                most_used = least_used = (None, 0)
                
            stats[strategy_name] = {
                'total_selections': total_selections,
                'unique_tools': len(tool_counts),
                'usage_counts': dict(tool_counts),
                'usage_percentages': usage_percentages,
                'most_used_tool': most_used,
                'least_used_tool': least_used,
                'usage_entropy': self._calculate_usage_entropy(tool_counts)
            }
            
        return stats
        
    def _calculate_usage_entropy(self, tool_counts: Dict[str, int]) -> float:
        """Calculate entropy of tool usage distribution."""
        if not tool_counts:
            return 0
            
        total = sum(tool_counts.values())
        if total == 0:
            return 0
            
        # Calculate probabilities
        probs = [count / total for count in tool_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        
        return entropy
        
    def get_scenario_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance across different scenarios."""
        analysis = {}
        
        for strategy_name, scenario_data in self.scenario_performance.items():
            scenario_stats = {}
            
            for scenario_id, rewards in scenario_data.items():
                scenario_stats[scenario_id] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'num_episodes': len(rewards),
                    'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
                }
                
            # Find best/worst scenarios
            if scenario_stats:
                best_scenario = max(scenario_stats.items(), 
                                  key=lambda x: x[1]['mean_reward'])
                worst_scenario = min(scenario_stats.items(), 
                                   key=lambda x: x[1]['mean_reward'])
            else:
                best_scenario = worst_scenario = (None, {})
                
            analysis[strategy_name] = {
                'scenario_stats': scenario_stats,
                'best_scenario': best_scenario,
                'worst_scenario': worst_scenario,
                'performance_variance': np.var([
                    s['mean_reward'] for s in scenario_stats.values()
                ]) if scenario_stats else 0
            }
            
        return analysis
        
    def calculate_relative_metrics(self, baseline_strategy: str = 'random') -> Dict[str, Dict[str, float]]:
        """Calculate metrics relative to a baseline strategy."""
        relative_metrics = {}
        
        baseline_metrics = self.get_performance_metrics(baseline_strategy)
        if baseline_metrics.num_episodes == 0:
            logger.warning(f"Baseline strategy '{baseline_strategy}' has no episodes")
            return relative_metrics
            
        for strategy_name in self.episode_metrics:
            if strategy_name == baseline_strategy:
                continue
                
            metrics = self.get_performance_metrics(strategy_name)
            
            relative_metrics[strategy_name] = {
                'reward_improvement': metrics.mean_reward - baseline_metrics.mean_reward,
                'reward_improvement_pct': (
                    (metrics.mean_reward - baseline_metrics.mean_reward) / 
                    baseline_metrics.mean_reward * 100
                ) if baseline_metrics.mean_reward != 0 else 0,
                'time_difference': metrics.mean_selection_time - baseline_metrics.mean_selection_time,
                'time_ratio': (
                    metrics.mean_selection_time / baseline_metrics.mean_selection_time
                ) if baseline_metrics.mean_selection_time > 0 else 1,
                'efficiency_ratio': (
                    metrics.sample_efficiency / baseline_metrics.sample_efficiency
                ) if baseline_metrics.sample_efficiency and baseline_metrics.sample_efficiency > 0 else 1
            }
            
        return relative_metrics
        
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all episode metrics to a pandas DataFrame."""
        all_episodes = []
        
        for strategy_name, episodes in self.episode_metrics.items():
            for episode in episodes:
                data = episode.to_dict()
                all_episodes.append(data)
                
        return pd.DataFrame(all_episodes)
        
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        summary = {
            'evaluation_duration': (datetime.now() - self.start_time).total_seconds(),
            'total_episodes': sum(len(episodes) for episodes in self.episode_metrics.values()),
            'strategies_evaluated': len(self.episode_metrics),
            'performance_summary': {},
            'learning_summary': {},
            'efficiency_summary': {}
        }
        
        # Performance summary
        for strategy_name in self.episode_metrics:
            metrics = self.get_performance_metrics(strategy_name)
            summary['performance_summary'][strategy_name] = metrics.to_dict()
            
        # Learning summary
        for strategy_name, rewards in self.performance_history.items():
            if len(rewards) > 10:
                early_performance = np.mean(rewards[:10])
                late_performance = np.mean(rewards[-10:])
                improvement = late_performance - early_performance
                
                summary['learning_summary'][strategy_name] = {
                    'early_performance': early_performance,
                    'late_performance': late_performance,
                    'improvement': improvement,
                    'improvement_pct': (improvement / early_performance * 100) if early_performance != 0 else 0
                }
                
        # Efficiency summary
        summary['efficiency_summary'] = self.calculate_relative_metrics()
        
        return summary
    
    # Real-time tracking methods
    def record_real_time_metric(self, metric_name: str, value: float, 
                               timestamp: Optional[datetime] = None):
        """Record a real-time metric value."""
        timestamp = timestamp or datetime.now()
        
        # Store in real-time metrics
        self.real_time_metrics[metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
        
        # Update baseline tracker
        baseline_data = self.baseline_tracker[metric_name]
        baseline_data['values'].append(value)
        
        # Update baseline if enough samples
        if len(baseline_data['values']) >= 20 and baseline_data['baseline'] is None:
            baseline_data['baseline'] = {
                'mean': np.mean(list(baseline_data['values'])),
                'std': np.std(list(baseline_data['values']))
            }
        elif baseline_data['baseline'] and len(baseline_data['values']) % 10 == 0:
            # Update baseline periodically using exponential moving average
            alpha = 0.1
            # Convert deque to list and get last 10 values
            values_list = list(baseline_data['values'])
            last_10_values = values_list[-10:]
            baseline_data['baseline']['mean'] = (
                alpha * np.mean(last_10_values) + 
                (1 - alpha) * baseline_data['baseline']['mean']
            )
            baseline_data['baseline']['std'] = (
                alpha * np.std(last_10_values) + 
                (1 - alpha) * baseline_data['baseline']['std']
            )
        
        # Notify observers
        for observer in self.metric_observers:
            try:
                observer(metric_name, value, timestamp)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")
    
    def add_metric_observer(self, observer: Callable):
        """Add an observer for real-time metric updates."""
        self.metric_observers.append(observer)
    
    def remove_metric_observer(self, observer: Callable):
        """Remove a metric observer."""
        if observer in self.metric_observers:
            self.metric_observers.remove(observer)
    
    def get_real_time_metrics(self, metric_name: str, 
                             time_window: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Get real-time metrics within a time window."""
        metrics = self.real_time_metrics.get(metric_name, [])
        
        if not time_window:
            return list(metrics)
        
        cutoff = datetime.now() - time_window
        return [m for m in metrics if m['timestamp'] >= cutoff]
    
    def get_metric_baseline(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for a metric."""
        baseline_data = self.baseline_tracker.get(metric_name)
        return baseline_data.get('baseline') if baseline_data else None
    
    def calculate_metric_deviation(self, metric_name: str, value: float) -> Optional[float]:
        """Calculate deviation from baseline in standard deviations."""
        baseline = self.get_metric_baseline(metric_name)
        if not baseline or baseline['std'] == 0:
            return None
        
        return (value - baseline['mean']) / baseline['std']
    
    async def stream_metrics(self, callback: Callable, metrics: List[str], 
                           interval: Optional[float] = None):
        """Stream real-time metrics to a callback."""
        interval = interval or self.update_interval
        
        while self.streaming_enabled:
            try:
                # Collect latest metrics
                updates = {}
                for metric_name in metrics:
                    latest = self.real_time_metrics.get(metric_name)
                    if latest:
                        updates[metric_name] = latest[-1]
                
                # Send update
                if updates:
                    await callback(updates)
                
                # Wait for next update
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error streaming metrics: {e}")
                break
    
    def enable_streaming(self):
        """Enable metric streaming."""
        self.streaming_enabled = True
    
    def disable_streaming(self):
        """Disable metric streaming."""
        self.streaming_enabled = False
    
    def get_performance_trends(self, strategy_name: str, 
                              window_size: int = 50) -> Dict[str, Any]:
        """Analyze performance trends for a strategy."""
        rewards = self.performance_history.get(strategy_name, [])
        
        if len(rewards) < window_size:
            return {'trend': 'insufficient_data', 'confidence': 0}
        
        # Calculate moving averages
        recent_avg = np.mean(rewards[-window_size:])
        previous_avg = np.mean(rewards[-2*window_size:-window_size])
        
        # Calculate trend
        if recent_avg > previous_avg * 1.05:
            trend = 'improving'
        elif recent_avg < previous_avg * 0.95:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        # Calculate confidence based on variance
        recent_std = np.std(rewards[-window_size:])
        confidence = 1 - min(recent_std / abs(recent_avg) if recent_avg != 0 else 1, 1)
        
        return {
            'trend': trend,
            'confidence': confidence,
            'recent_performance': recent_avg,
            'previous_performance': previous_avg,
            'change_percentage': ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg != 0 else 0
        }
    
    def detect_anomalies(self, metric_name: str, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalies in real-time metrics using z-score."""
        metrics = list(self.real_time_metrics.get(metric_name, []))
        baseline = self.get_metric_baseline(metric_name)
        
        if not baseline or len(metrics) < 10:
            return []
        
        anomalies = []
        for metric in metrics:
            z_score = abs((metric['value'] - baseline['mean']) / baseline['std']) if baseline['std'] > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'timestamp': metric['timestamp'],
                    'value': metric['value'],
                    'z_score': z_score,
                    'baseline_mean': baseline['mean'],
                    'baseline_std': baseline['std']
                })
        
        return anomalies