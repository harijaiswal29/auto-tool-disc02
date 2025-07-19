"""Comparison visualizer for evaluation results.

This module creates visualizations and reports for comparing strategy performance
across multiple dimensions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import json
import logging
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

logger = get_logger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ComparisonVisualizer:
    """Creates visualizations for strategy comparisons."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.figure_size = config.get('figure_size', (10, 6))
        self.dpi = config.get('dpi', 100)
        self.output_dir = Path(config.get('output_dir', 'src/evaluation/reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_learning_curves(self, learning_curves: Dict[str, List[float]], 
                             title: str = "Learning Curves Comparison") -> plt.Figure:
        """Create learning curves comparison plot."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        for strategy_name, curve in learning_curves.items():
            episodes = range(len(curve))
            ax.plot(episodes, curve, label=strategy_name, linewidth=2)
            
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_distribution(self, performance_data: Dict[str, List[float]],
                                      title: str = "Performance Distribution") -> plt.Figure:
        """Create violin plots showing performance distribution."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Prepare data for violin plot
        data = []
        labels = []
        for strategy_name, rewards in performance_data.items():
            data.extend(rewards)
            labels.extend([strategy_name] * len(rewards))
            
        df = pd.DataFrame({'Strategy': labels, 'Reward': data})
        
        # Create violin plot
        sns.violinplot(data=df, x='Strategy', y='Reward', ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        
        # Rotate x-axis labels if needed
        if len(performance_data) > 5:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        return fig
    
    def create_comparison_heatmap(self, comparison_matrix: Dict[str, Dict[str, float]],
                                 metric: str = "improvement_percent",
                                 title: str = "Strategy Comparison Heatmap") -> plt.Figure:
        """Create heatmap showing pairwise comparisons."""
        strategies = list(comparison_matrix.keys())
        n = len(strategies) + 1  # +1 for baseline
        
        # Create matrix
        matrix = np.zeros((n, n))
        labels = ['random'] + strategies  # Assuming random is baseline
        
        # Fill matrix
        for i, strat1 in enumerate(labels):
            for j, strat2 in enumerate(labels):
                if i == j:
                    matrix[i, j] = 0
                elif strat1 == 'random' and strat2 in comparison_matrix:
                    matrix[i, j] = comparison_matrix[strat2].get(metric, 0)
                elif strat2 == 'random' and strat1 in comparison_matrix:
                    matrix[i, j] = -comparison_matrix[strat1].get(metric, 0)
                    
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': f'{metric} (%)'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_radar_chart(self, metrics_data: Dict[str, Dict[str, float]],
                          title: str = "Multi-Metric Comparison") -> plt.Figure:
        """Create radar chart for multi-metric comparison."""
        # Select metrics to display
        metric_names = ['mean_reward', 'convergence_rate', 'sample_efficiency', 
                       'exploration_efficiency', 'time_efficiency']
        
        # Prepare data
        strategies = list(metrics_data.keys())
        n_metrics = len(metric_names)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='polar')
        
        # Set up angles
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each strategy
        for strategy in strategies:
            values = []
            for metric in metric_names:
                if metric == 'mean_reward':
                    val = metrics_data[strategy].get('mean_reward', 0)
                elif metric == 'time_efficiency':
                    # Inverse of mean time (lower is better)
                    mean_time = metrics_data[strategy].get('mean_selection_time', 1)
                    val = 1 / mean_time if mean_time > 0 else 0
                else:
                    val = metrics_data[strategy].get(metric, 0)
                    
                # Normalize to 0-1 range
                values.append(min(max(val, 0), 1))
                
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy)
            ax.fill(angles, values, alpha=0.25)
        
        # Customize plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_convergence_plot(self, convergence_data: Dict[str, Dict[str, Any]],
                               title: str = "Convergence Analysis") -> plt.Figure:
        """Create convergence analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Convergence episodes bar chart
        strategies = []
        convergence_episodes = []
        colors = []
        
        for strategy, data in convergence_data.items():
            strategies.append(strategy)
            conv_ep = data.get('convergence_episode', float('inf'))
            if conv_ep is None:
                conv_ep = float('inf')
            convergence_episodes.append(conv_ep if conv_ep != float('inf') else 0)
            colors.append('green' if data.get('converged', False) else 'red')
            
        bars = ax1.bar(strategies, convergence_episodes, color=colors, alpha=0.7)
        ax1.set_xlabel('Strategy', fontsize=12)
        ax1.set_ylabel('Episodes to Convergence', fontsize=12)
        ax1.set_title('Convergence Speed', fontsize=13)
        
        # Add convergence status labels
        for bar, converged in zip(bars, [d.get('converged', False) 
                                        for d in convergence_data.values()]):
            if not converged:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        'NC', ha='center', va='bottom', fontsize=10)
                
        # Final performance comparison
        final_performances = [data.get('final_performance', 0) 
                            for data in convergence_data.values()]
        
        ax2.bar(strategies, final_performances, alpha=0.7)
        ax2.set_xlabel('Strategy', fontsize=12)
        ax2.set_ylabel('Final Performance', fontsize=12)
        ax2.set_title('Final Performance Comparison', fontsize=13)
        
        # Rotate labels if needed
        if len(strategies) > 5:
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
                
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_tool_usage_heatmap(self, tool_usage_data: Dict[str, Dict[str, int]],
                                 title: str = "Tool Usage Patterns") -> plt.Figure:
        """Create heatmap showing tool usage patterns."""
        # Get all unique tools
        all_tools = set()
        for usage in tool_usage_data.values():
            all_tools.update(usage.keys())
        all_tools = sorted(list(all_tools))
        
        # Create usage matrix
        strategies = list(tool_usage_data.keys())
        matrix = np.zeros((len(strategies), len(all_tools)))
        
        for i, strategy in enumerate(strategies):
            for j, tool in enumerate(all_tools):
                matrix[i, j] = tool_usage_data[strategy].get(tool, 0)
                
        # Normalize by row (percentage of total usage)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_normalized = np.divide(matrix, row_sums, 
                                    where=row_sums != 0, out=np.zeros_like(matrix))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        sns.heatmap(matrix_normalized * 100, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=all_tools, yticklabels=strategies, ax=ax,
                   cbar_kws={'label': 'Usage Percentage (%)'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Tools', fontsize=12)
        ax.set_ylabel('Strategies', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_statistical_summary_plot(self, comparison_results: Dict[str, Dict[str, Any]],
                                       title: str = "Statistical Significance") -> plt.Figure:
        """Create plot showing statistical significance of comparisons."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        strategies = list(comparison_results.keys())
        p_values = [comp.get('p_value', 1.0) for comp in comparison_results.values()]
        effect_sizes = [comp.get('cohens_d', 0) for comp in comparison_results.values()]
        
        # P-value plot
        bars1 = ax1.bar(strategies, p_values, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        ax1.set_xlabel('Strategy', fontsize=12)
        ax1.set_ylabel('P-value', fontsize=12)
        ax1.set_title('Statistical Significance (vs baseline)', fontsize=13)
        ax1.legend()
        ax1.set_ylim(0, max(p_values) * 1.1)
        
        # Color bars based on significance
        for bar, p_val in zip(bars1, p_values):
            bar.set_color('green' if p_val < 0.05 else 'gray')
            
        # Effect size plot
        bars2 = ax2.bar(strategies, effect_sizes, alpha=0.7)
        ax2.set_xlabel('Strategy', fontsize=12)
        ax2.set_ylabel("Cohen's d", fontsize=12)
        ax2.set_title('Effect Size', fontsize=13)
        
        # Add effect size interpretation lines
        ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax2.axhline(y=0.8, color='gray', linestyle='-', alpha=0.5, label='Large')
        ax2.legend()
        
        # Color bars based on effect size
        for bar, d in zip(bars2, effect_sizes):
            if abs(d) < 0.2:
                bar.set_color('gray')
            elif abs(d) < 0.5:
                bar.set_color('yellow')
            elif abs(d) < 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
                
        # Rotate labels if needed
        if len(strategies) > 5:
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
                
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       output_filename: str = None) -> str:
        """Generate comprehensive PDF report with all visualizations."""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluation_report_{timestamp}.pdf"
            
        output_path = self.output_dir / output_filename
        
        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'Evaluation Report', size=24, ha='center', weight='bold')
            fig.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    size=14, ha='center')
            fig.text(0.5, 0.5, f"Episodes: {evaluation_results.get('num_episodes', 'N/A')}", 
                    size=14, ha='center')
            fig.text(0.5, 0.4, f"Strategies: {len(evaluation_results.get('strategies', {}))}", 
                    size=14, ha='center')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Summary statistics table
            summary = evaluation_results.get('summary', {})
            if summary:
                fig = self._create_summary_table(summary)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
            # Learning curves
            if 'strategies' in evaluation_results:
                learning_data = {name: data.get('rewards', []) 
                               for name, data in evaluation_results['strategies'].items()}
                fig = self.create_learning_curves(learning_data)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
            # Performance distribution
            fig = self.create_performance_distribution(learning_data)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Comparison heatmap
            if 'comparisons' in evaluation_results:
                fig = self.create_comparison_heatmap(evaluation_results['comparisons'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
            # Convergence analysis
            convergence_data = {
                name: data.get('statistics', {}).get('convergence', {})
                for name, data in evaluation_results.get('strategies', {}).items()
            }
            fig = self.create_convergence_plot(convergence_data)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Statistical significance
            if 'comparisons' in evaluation_results:
                fig = self.create_statistical_summary_plot(evaluation_results['comparisons'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
        logger.info(f"Report saved to: {output_path}")
        return str(output_path)
    
    def _create_summary_table(self, summary: Dict[str, Any]) -> plt.Figure:
        """Create summary statistics table."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        rankings = summary.get('rankings', [])
        if rankings:
            headers = ['Strategy', 'Mean Reward', 'Converged']
            rows = [[r['strategy'], f"{r['mean_reward']:.3f}", 
                    '✓' if r['convergence'] else '✗'] for r in rankings]
            
            table = ax.table(cellText=rows, colLabels=headers, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            # Color best strategy
            if rankings:
                table[(1, 0)].set_facecolor('#90EE90')
                table[(1, 1)].set_facecolor('#90EE90')
                table[(1, 2)].set_facecolor('#90EE90')
                
        ax.set_title('Strategy Rankings', fontsize=14, fontweight='bold', pad=20)
        return fig
    
    def save_plots_individually(self, evaluation_results: Dict[str, Any], 
                               prefix: str = "eval") -> List[str]:
        """Save all plots as individual files."""
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectory
        plot_dir = self.output_dir / f"{prefix}_{timestamp}"
        plot_dir.mkdir(exist_ok=True)
        
        # Learning curves
        if 'strategies' in evaluation_results:
            learning_data = {name: data.get('rewards', []) 
                           for name, data in evaluation_results['strategies'].items()}
            fig = self.create_learning_curves(learning_data)
            path = plot_dir / "learning_curves.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(str(path))
            plt.close()
            
        # Add more individual plot saves as needed...
        
        return saved_files