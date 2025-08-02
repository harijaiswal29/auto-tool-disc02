"""
Cache Metrics Dashboard for Visualization.

This module provides a real-time dashboard for visualizing cache performance metrics,
trends, and patterns. Essential for demonstrating learning effectiveness in the dissertation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pathlib import Path

from src.monitoring.cache_metrics_monitor import CacheMetricsMonitor
from src.utils.logger import get_logger


class CacheDashboard:
    """
    Real-time dashboard for cache metrics visualization.
    
    Features:
    - Live performance graphs
    - Hit rate trends over time
    - Query pattern analysis
    - Learning effectiveness visualization
    - Export capabilities for dissertation
    """
    
    def __init__(self, monitor: CacheMetricsMonitor, config: Optional[Dict[str, Any]] = None):
        """Initialize the cache dashboard."""
        self.logger = get_logger(__name__)
        self.monitor = monitor
        self.config = config or {}
        
        # Dashboard configuration
        self.update_interval = self.config.get('update_interval', 5000)  # ms
        self.figure_size = self.config.get('figure_size', (15, 10))
        self.export_path = Path(self.config.get('export_path', 'data/visualizations/cache'))
        self.export_path.mkdir(parents=True, exist_ok=True)
        
        # Setup style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Initialize figure and subplots
        self.fig = None
        self.axes = None
        self.animation = None
        
        self.logger.info("Cache dashboard initialized")
    
    def create_dashboard(self):
        """Create the dashboard layout."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=self.figure_size)
        self.fig.suptitle('Cache Performance Dashboard', fontsize=16)
        
        # Configure subplots
        self.axes[0, 0].set_title('Hit Rate Over Time')
        self.axes[0, 1].set_title('Response Time Trends')
        self.axes[0, 2].set_title('Cache Efficiency')
        self.axes[1, 0].set_title('Query Pattern Distribution')
        self.axes[1, 1].set_title('Learning Effectiveness')
        self.axes[1, 2].set_title('System Metrics')
        
        plt.tight_layout()
    
    def update_dashboard(self, frame):
        """Update dashboard with latest metrics."""
        try:
            # Get current metrics
            current_metrics = self.monitor.get_current_metrics()
            historical_metrics = self.monitor.get_historical_metrics(hours=1)
            
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Update each subplot
            self._plot_hit_rate(self.axes[0, 0], historical_metrics)
            self._plot_response_times(self.axes[0, 1], historical_metrics)
            self._plot_cache_efficiency(self.axes[0, 2], current_metrics)
            self._plot_query_patterns(self.axes[1, 0], current_metrics)
            self._plot_learning_effectiveness(self.axes[1, 1])
            self._plot_system_metrics(self.axes[1, 2], current_metrics)
            
            # Refresh layout
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"Dashboard update error: {e}")
    
    def _plot_hit_rate(self, ax, historical_metrics: List[Dict[str, Any]]):
        """Plot hit rate over time."""
        if not historical_metrics:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('Hit Rate Over Time')
            return
        
        # Extract data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in historical_metrics]
        hit_rates = [m['hit_rate'] for m in historical_metrics]
        
        # Plot
        ax.plot(timestamps, hit_rates, 'b-', linewidth=2, label='Hit Rate')
        
        # Add moving average
        if len(hit_rates) > 10:
            window = min(10, len(hit_rates) // 3)
            ma = np.convolve(hit_rates, np.ones(window)/window, mode='valid')
            ma_times = timestamps[window-1:]
            ax.plot(ma_times, ma, 'r--', linewidth=2, label=f'{window}-point MA')
        
        # Format
        ax.set_xlabel('Time')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend()
        ax.set_title('Hit Rate Over Time')
        
        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_response_times(self, ax, historical_metrics: List[Dict[str, Any]]):
        """Plot response time trends."""
        if not historical_metrics:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('Response Time Trends')
            return
        
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in historical_metrics]
        response_times = [m['avg_retrieval_time_ms'] for m in historical_metrics]
        
        # Plot with fill
        ax.fill_between(timestamps, response_times, alpha=0.3)
        ax.plot(timestamps, response_times, 'g-', linewidth=2)
        
        # Add threshold line
        if response_times:
            avg_response = np.mean(response_times)
            ax.axhline(y=avg_response, color='r', linestyle='--', 
                      label=f'Avg: {avg_response:.2f}ms')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Response Time (ms)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend()
        ax.set_title('Response Time Trends')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_cache_efficiency(self, ax, current_metrics: Dict[str, Any]):
        """Plot cache efficiency metrics."""
        indicators = current_metrics.get('performance_indicators', {})
        
        metrics = {
            'Hit Rate\nTrend': indicators.get('hit_rate_trend', 0),
            'Response\nImprovement': indicators.get('avg_response_improvement', 0),
            'Cache\nEfficiency': indicators.get('cache_efficiency', 0) / 100,  # Scale down
            'Warming\nEffectiveness': indicators.get('warming_effectiveness', 0)
        }
        
        # Create bar chart
        x = list(metrics.keys())
        y = list(metrics.values())
        colors = ['green' if val > 0 else 'red' for val in y]
        
        bars = ax.bar(x, y, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Performance Indicator')
        ax.set_title('Cache Efficiency Metrics')
    
    def _plot_query_patterns(self, ax, current_metrics: Dict[str, Any]):
        """Plot query pattern distribution."""
        patterns = current_metrics.get('top_patterns', [])
        
        if not patterns:
            ax.text(0.5, 0.5, 'No pattern data available', ha='center', va='center')
            ax.set_title('Query Pattern Distribution')
            return
        
        # Extract top patterns
        pattern_names = [p['pattern'][:20] + '...' if len(p['pattern']) > 20 
                        else p['pattern'] for p in patterns[:5]]
        hit_rates = [p['hit_rate'] for p in patterns[:5]]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(pattern_names))
        ax.barh(y_pos, hit_rates, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pattern_names)
        ax.set_xlabel('Hit Rate')
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add value labels
        for i, v in enumerate(hit_rates):
            ax.text(v + 0.01, i, f'{v:.1%}', va='center')
        
        ax.set_title('Top Query Patterns by Hit Rate')
    
    def _plot_learning_effectiveness(self, ax):
        """Plot learning effectiveness metrics."""
        effectiveness = self.monitor.calculate_learning_effectiveness()
        
        # Create radar chart
        categories = list(effectiveness.keys())
        values = list(effectiveness.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot
        ax = plt.subplot(2, 3, 5, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Learning Effectiveness', y=1.08)
        
        # Add grid
        ax.grid(True)
    
    def _plot_system_metrics(self, ax, current_metrics: Dict[str, Any]):
        """Plot system-level cache metrics."""
        current = current_metrics.get('current', {})
        
        if not current:
            ax.text(0.5, 0.5, 'No system metrics available', ha='center', va='center')
            ax.set_title('System Metrics')
            return
        
        # Prepare metrics
        metrics_text = f"""
        Total Queries: {current.get('total_queries', 0):,}
        Cache Size: {current.get('current_size', 0)} / {current.get('max_size', 0)}
        Memory Usage: {current.get('cache_size_bytes', 0) / 1024:.1f} KB
        Evictions: {current.get('evictions', 0):,}
        Expirations: {current.get('expirations', 0):,}
        
        Overall Hit Rate: {current.get('hit_rate', 0):.1%}
        Avg Response: {current.get('avg_retrieval_time_ms', 0):.2f} ms
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        ax.set_title('System Metrics')
    
    async def start_live_dashboard(self):
        """Start the live dashboard with real-time updates."""
        self.create_dashboard()
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig, 
            self.update_dashboard,
            interval=self.update_interval,
            cache_frame_data=False
        )
        
        plt.show()
    
    def export_current_view(self, filename: Optional[str] = None):
        """Export current dashboard view."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'cache_dashboard_{timestamp}.png'
        
        filepath = self.export_path / filename
        
        if self.fig:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard exported to {filepath}")
            return str(filepath)
        
        return None
    
    def generate_dissertation_figures(self):
        """Generate specific figures for dissertation."""
        self.logger.info("Generating dissertation figures...")
        
        # Get metrics
        historical = self.monitor.get_historical_metrics(hours=24)
        effectiveness = self.monitor.calculate_learning_effectiveness()
        
        # Figure 1: Hit Rate Improvement Over Time
        plt.figure(figsize=(10, 6))
        if historical:
            timestamps = [datetime.fromisoformat(m['timestamp']) for m in historical]
            hit_rates = [m['hit_rate'] for m in historical]
            plt.plot(timestamps, hit_rates, 'b-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Cache Hit Rate')
            plt.title('Cache Hit Rate Improvement Over Time')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.export_path / 'hit_rate_improvement.png', dpi=300)
            plt.close()
        
        # Figure 2: Learning Effectiveness Summary
        plt.figure(figsize=(8, 8))
        categories = list(effectiveness.keys())
        values = list(effectiveness.values())
        
        # Create bar chart
        plt.bar(categories, values, color=['green' if v > 0 else 'red' for v in values])
        plt.ylabel('Score')
        plt.title('Learning Effectiveness Metrics')
        plt.ylim(-1, 1)
        
        # Add value labels
        for i, (cat, val) in enumerate(zip(categories, values)):
            plt.text(i, val + 0.02 if val > 0 else val - 0.02, 
                    f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top')
        
        plt.axhline(y=0, color='black', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.export_path / 'learning_effectiveness.png', dpi=300)
        plt.close()
        
        # Figure 3: Performance Comparison
        if len(historical) > 20:
            plt.figure(figsize=(10, 6))
            
            # Compare early vs recent performance
            early = historical[:10]
            recent = historical[-10:]
            
            metrics_comparison = {
                'Early Period': {
                    'Hit Rate': np.mean([m['hit_rate'] for m in early]),
                    'Avg Response (ms)': np.mean([m['avg_retrieval_time_ms'] for m in early])
                },
                'Recent Period': {
                    'Hit Rate': np.mean([m['hit_rate'] for m in recent]),
                    'Avg Response (ms)': np.mean([m['avg_retrieval_time_ms'] for m in recent])
                }
            }
            
            # Create grouped bar chart
            x = np.arange(2)
            width = 0.35
            
            early_vals = list(metrics_comparison['Early Period'].values())
            recent_vals = list(metrics_comparison['Recent Period'].values())
            
            # Normalize response time for visualization
            early_vals[1] = early_vals[1] / 100
            recent_vals[1] = recent_vals[1] / 100
            
            plt.bar(x - width/2, early_vals, width, label='Early Period', alpha=0.8)
            plt.bar(x + width/2, recent_vals, width, label='Recent Period', alpha=0.8)
            
            plt.xlabel('Metric')
            plt.ylabel('Value (normalized)')
            plt.title('Performance Comparison: Early vs Recent')
            plt.xticks(x, ['Hit Rate', 'Response Time\n(normalized)'])
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.export_path / 'performance_comparison.png', dpi=300)
            plt.close()
        
        self.logger.info(f"Dissertation figures saved to {self.export_path}")
        
        return {
            'hit_rate_improvement': str(self.export_path / 'hit_rate_improvement.png'),
            'learning_effectiveness': str(self.export_path / 'learning_effectiveness.png'),
            'performance_comparison': str(self.export_path / 'performance_comparison.png')
        }