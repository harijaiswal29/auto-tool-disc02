#!/usr/bin/env python3
"""
Chart Generation Script for Dissertation

This script generates publication-quality visualizations from experiment results
for inclusion in the dissertation document.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Color palette for consistency
COLORS = {
    'q_learning': '#2E86AB',           # Blue (backward compatibility)
    'q_learning_tabular': '#2E86AB',   # Blue
    'q_learning_dqn': '#1B4F72',       # Dark Blue (for DQN)
    'random': '#E63946',               # Red
    'popular': '#F77F00',              # Orange
    'fixed': '#06D6A0',                # Green
    'fixed_policy': '#06D6A0',         # Green (alternative name)
    'greedy': '#7209B7',               # Purple
    'context_agnostic': '#F72585'      # Pink
}

# Strategy display names
STRATEGY_NAMES = {
    'q_learning': 'Q-Learning',
    'q_learning_tabular': 'Q-Learning (Tabular)',
    'q_learning_dqn': 'Q-Learning (DQN)',
    'random': 'Random Selection',
    'popular': 'Most Popular Tools',
    'fixed': 'Fixed Policy',
    'fixed_policy': 'Fixed Policy',
    'greedy': 'Greedy Selection',
    'context_agnostic': 'Q-Learning (No Context)'
}


class ChartGenerator:
    """Generates dissertation charts from experiment results."""
    
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.charts_dir = self.output_dir / "comparison_charts"
        self.curves_dir = self.output_dir / "learning_curves"
        self.stats_dir = self.output_dir / "statistical_reports"
        self.dissertation_dir = self.output_dir / "dissertation_figures"
        
        for dir in [self.charts_dir, self.curves_dir, self.stats_dir, self.dissertation_dir]:
            dir.mkdir(parents=True, exist_ok=True)
    
    def load_experiment_results(self, filename: str) -> Dict[str, Any]:
        """Load experiment results from JSON file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            # Try parent directory
            filepath = self.results_dir.parent / filename
        
        with open(filepath) as f:
            return json.load(f)
    
    def generate_baseline_comparison_chart(self, results: Dict[str, Any]):
        """Generate bar chart comparing baseline strategies."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        strategies = []
        completion_rates = []
        errors = []
        
        for strategy, summary in results['strategy_summaries'].items():
            strategies.append(STRATEGY_NAMES.get(strategy, strategy))
            tcr = summary['task_completion_rate']
            completion_rates.append(tcr['mean'])
            # Use confidence interval for error bars
            errors.append(tcr['mean'] - tcr['ci_lower'])
        
        # Sort by performance
        sorted_indices = np.argsort(completion_rates)
        strategies = [strategies[i] for i in sorted_indices]
        completion_rates = [completion_rates[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
        colors = [COLORS.get(k, '#666666') for k in results['strategy_summaries'].keys()]
        colors = [colors[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax.barh(strategies, completion_rates, xerr=errors, 
                       color=colors, alpha=0.8, capsize=5)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, completion_rates)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{rate:.1%}', ha='left', va='center')
        
        # Customization
        ax.set_xlabel('Task Completion Rate')
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.grid(axis='x', alpha=0.3)
        ax.set_title('Task Completion Rate by Strategy\n(with 95% Confidence Intervals)')
        
        # Add significance markers
        if 'statistical_comparison' in results:
            # Mark significant improvements
            y_pos = len(strategies) - 1  # Q-learning position
            for i, (strategy, comp) in enumerate(results['statistical_comparison'].items()):
                if comp.get('significant_corrected', False):
                    ax.text(0.02, i, '***', fontsize=16, color='green', 
                           weight='bold', va='center')
                elif comp.get('significant', False):
                    ax.text(0.02, i, '*', fontsize=16, color='green', 
                           weight='bold', va='center')
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in ['png', 'pdf']:
            filepath = self.dissertation_dir / f"baseline_comparison.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=300)
        
        plt.close()
    
    def generate_learning_curves(self, results: Dict[str, Any]):
        """Generate learning curves showing performance over episodes."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot completion rates over episodes
        for strategy_name in ['random', 'greedy', 'q_learning_tabular', 'q_learning_dqn']:
            if strategy_name not in results['strategy_summaries']:
                continue
            
            # Get time series data from raw results
            strategy_data = [r for r in results.get('raw_results', []) 
                           if r['strategy'] == strategy_name]
            
            if strategy_data:
                # Average across runs
                all_curves = [r['time_series']['completion_rates'] 
                             for r in strategy_data]
                
                # Pad to same length
                max_len = max(len(curve) for curve in all_curves)
                padded_curves = []
                for curve in all_curves:
                    padded = curve + [curve[-1]] * (max_len - len(curve))
                    padded_curves.append(padded)
                
                mean_curve = np.mean(padded_curves, axis=0)
                std_curve = np.std(padded_curves, axis=0)
                episodes = range(len(mean_curve))
                
                # Plot with confidence band
                color = COLORS.get(strategy_name, '#666666')
                ax1.plot(episodes, mean_curve, color=color, linewidth=2,
                        label=STRATEGY_NAMES.get(strategy_name, strategy_name))
                ax1.fill_between(episodes, 
                               mean_curve - std_curve, 
                               mean_curve + std_curve,
                               color=color, alpha=0.2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Task Completion Rate')
        ax1.set_title('Learning Curves: Task Completion Rate')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot cumulative rewards
        for strategy_name in ['random', 'greedy', 'qlearning']:
            if strategy_name not in results['strategy_summaries']:
                continue
            
            strategy_data = [r for r in results.get('raw_results', []) 
                           if r['strategy'] == strategy_name]
            
            if strategy_data:
                # Get episode rewards
                all_rewards = [r['time_series']['episode_rewards'] 
                             for r in strategy_data]
                
                # Calculate cumulative rewards
                cumulative_rewards = []
                for rewards in all_rewards:
                    cumsum = np.cumsum(rewards)
                    cumulative_rewards.append(cumsum)
                
                # Pad and average
                max_len = max(len(curve) for curve in cumulative_rewards)
                padded_rewards = []
                for curve in cumulative_rewards:
                    padded = list(curve) + [curve[-1]] * (max_len - len(curve))
                    padded_rewards.append(padded)
                
                mean_rewards = np.mean(padded_rewards, axis=0)
                episodes = range(len(mean_rewards))
                
                color = COLORS.get(strategy_name, '#666666')
                ax2.plot(episodes, mean_rewards, color=color, linewidth=2,
                        label=STRATEGY_NAMES.get(strategy_name, strategy_name))
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Learning Curves: Cumulative Reward')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.dissertation_dir / f"learning_curves.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=300)
        
        plt.close()
    
    def generate_convergence_analysis(self, results: Dict[str, Any]):
        """Generate convergence analysis visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract convergence data
        strategies = []
        convergence_episodes = []
        
        for strategy, summary in results['strategy_summaries'].items():
            if 'convergence' in summary.get('metrics', {}):
                strategies.append(STRATEGY_NAMES.get(strategy, strategy))
                conv_data = summary['metrics']['convergence']
                convergence_episodes.append(conv_data.get('episodes_to_stable', 1000))
        
        # Create horizontal bar chart
        colors = [COLORS.get(k, '#666666') for k in results['strategy_summaries'].keys()
                 if 'convergence' in results['strategy_summaries'][k].get('metrics', {})]
        
        bars = ax.barh(strategies, convergence_episodes, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, episodes in zip(bars, convergence_episodes):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                   f'{int(episodes)}', ha='left', va='center')
        
        # Add target line
        ax.axvline(x=1000, color='red', linestyle='--', alpha=0.7, 
                  label='Target (1000 episodes)')
        
        ax.set_xlabel('Episodes to Convergence')
        ax.set_title('Convergence Analysis by Strategy')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        # Handle empty convergence_episodes list
        if convergence_episodes:
            ax.set_xlim(0, max(convergence_episodes) * 1.1)
        else:
            ax.set_xlim(0, 1000)  # Default to max episodes if no convergence data
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.dissertation_dir / f"convergence_analysis.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=300)
        
        plt.close()
    
    def generate_statistical_significance_plot(self, results: Dict[str, Any]):
        """Generate visualization of statistical significance."""
        if 'statistical_comparison' not in results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Extract comparison data
        baselines = []
        p_values = []
        effect_sizes = []
        improvements = []
        
        for baseline, comp in results['statistical_comparison'].items():
            baselines.append(STRATEGY_NAMES.get(baseline, baseline))
            p_values.append(comp['p_value'])
            effect_sizes.append(abs(comp['cohens_d']))
            improvements.append(comp['improvement_percent'])
        
        # P-value plot
        bars1 = ax1.barh(baselines, p_values, color='lightcoral', alpha=0.8)
        ax1.axvline(x=0.05, color='green', linestyle='--', label='p = 0.05')
        ax1.axvline(x=0.05/len(baselines), color='darkgreen', linestyle='--', 
                   label=f'Bonferroni corrected\n(p = {0.05/len(baselines):.4f})')
        
        # Add value labels
        for bar, p_val in zip(bars1, p_values):
            ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{p_val:.4f}', ha='left', va='center', fontsize=10)
        
        ax1.set_xlabel('p-value')
        ax1.set_title('Statistical Significance')
        ax1.legend()
        ax1.set_xlim(0, max(p_values) * 1.2)
        ax1.grid(axis='x', alpha=0.3)
        
        # Effect size plot
        bars2 = ax2.barh(baselines, effect_sizes, color='lightblue', alpha=0.8)
        ax2.axvline(x=0.8, color='green', linestyle='--', label="Cohen's d = 0.8\n(large effect)")
        
        # Add value labels
        for bar, d in zip(bars2, effect_sizes):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{d:.2f}', ha='left', va='center', fontsize=10)
        
        ax2.set_xlabel("Cohen's d (Effect Size)")
        ax2.set_title('Effect Size Analysis')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Statistical Analysis: Q-Learning vs Baselines')
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.dissertation_dir / f"statistical_significance.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=300)
        
        plt.close()
    
    def generate_dqn_comparison(self, results: Dict[str, Any]):
        """Generate DQN vs Q-Learning comparison chart."""
        # Check if both strategies are present
        if 'q_learning_tabular' not in results['strategy_summaries'] or 'q_learning_dqn' not in results['strategy_summaries']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Comparison metrics
        metrics = ['task_completion_rate', 'average_reward']
        metric_labels = ['Task Completion Rate', 'Average Reward']
        
        for ax, metric, label in zip(axes, metrics, metric_labels):
            ql_data = results['strategy_summaries']['q_learning_tabular'].get(metric, {})
            dqn_data = results['strategy_summaries']['q_learning_dqn'].get(metric, {})
            
            if not ql_data or not dqn_data:
                continue
            
            strategies = ['Q-Learning', 'DQN']
            means = [ql_data.get('mean', 0), dqn_data.get('mean', 0)]
            errors = [
                ql_data.get('mean', 0) - ql_data.get('ci_lower', 0),
                dqn_data.get('mean', 0) - dqn_data.get('ci_lower', 0)
            ]
            
            bars = ax.bar(strategies, means, yerr=errors, 
                          color=[COLORS['q_learning_tabular'], COLORS['q_learning_dqn']], 
                          alpha=0.8, capsize=10)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{mean:.3f}' if metric == 'average_reward' else f'{mean:.1%}',
                       ha='center', va='bottom')
            
            # Calculate improvement
            if ql_data.get('mean', 0) > 0:
                improvement = ((dqn_data.get('mean', 0) - ql_data.get('mean', 0)) / 
                              ql_data.get('mean', 0)) * 100
                ax.text(0.5, 0.95, f'DQN Improvement: {improvement:+.1f}%',
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Deep Q-Network vs Standard Q-Learning', fontsize=16)
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.dissertation_dir / f"dqn_comparison.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=300)
        
        plt.close()
    
    def generate_improvement_summary(self, results: Dict[str, Any]):
        """Generate improvement percentage visualization."""
        if 'statistical_comparison' not in results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract improvement data
        baselines = []
        improvements = []
        significant = []
        
        for baseline, comp in results['statistical_comparison'].items():
            baselines.append(STRATEGY_NAMES.get(baseline, baseline))
            improvements.append(comp['improvement_percent'])
            significant.append(comp.get('significant_corrected', False))
        
        # Sort by improvement
        sorted_indices = np.argsort(improvements)[::-1]
        baselines = [baselines[i] for i in sorted_indices]
        improvements = [improvements[i] for i in sorted_indices]
        significant = [significant[i] for i in sorted_indices]
        
        # Color based on significance
        colors = ['darkgreen' if sig else 'gray' for sig in significant]
        
        bars = ax.bar(baselines, improvements, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, imp, sig in zip(bars, improvements, significant):
            label = f'{imp:.0f}%'
            if sig:
                label += ' ***'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   label, ha='center', va='bottom', fontweight='bold' if sig else 'normal')
        
        # Add 30% target line
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.7,
                  label='Target (30% improvement)')
        
        ax.set_ylabel('Improvement over Baseline (%)')
        ax.set_title('Q-Learning Performance Improvement')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(improvements) * 1.2)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.dissertation_dir / f"improvement_summary.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=300)
        
        plt.close()
    
    def generate_all_charts(self, results_filename: str = "baseline_comparison_final_*.json"):
        """Generate all charts from results file."""
        # Find most recent results file
        import glob
        pattern = str(self.results_dir.parent / results_filename)
        files = glob.glob(pattern)
        
        if not files:
            print(f"No results files found matching {pattern}")
            return
        
        # Use most recent
        results_file = max(files, key=lambda f: Path(f).stat().st_mtime)
        print(f"Loading results from {results_file}")
        
        results = self.load_experiment_results(Path(results_file).name)
        
        # Generate all charts
        print("Generating baseline comparison chart...")
        self.generate_baseline_comparison_chart(results)
        
        print("Generating learning curves...")
        self.generate_learning_curves(results)
        
        print("Generating convergence analysis...")
        self.generate_convergence_analysis(results)
        
        print("Generating statistical significance plot...")
        self.generate_statistical_significance_plot(results)
        
        print("Generating improvement summary...")
        self.generate_improvement_summary(results)
        
        print("Generating DQN comparison...")
        self.generate_dqn_comparison(results)
        
        print(f"\nAll charts saved to {self.dissertation_dir}")
        
        # Create LaTeX include file
        self._create_latex_includes()
    
    def _create_latex_includes(self):
        """Create LaTeX file with figure includes."""
        latex_content = """% Dissertation Figures - Auto-generated
% Include in your LaTeX document

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{figures/baseline_comparison.pdf}
\\caption{Comparison of task completion rates across different strategies. Error bars represent 95\\% confidence intervals. *** indicates statistical significance after Bonferroni correction.}
\\label{fig:baseline-comparison}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{figures/learning_curves.pdf}
\\caption{Learning curves showing (a) task completion rate and (b) cumulative reward over episodes for different strategies. Shaded regions represent standard deviation across runs.}
\\label{fig:learning-curves}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{figures/convergence_analysis.pdf}
\\caption{Episodes required for convergence by strategy. The red dashed line indicates the target of 1000 episodes.}
\\label{fig:convergence-analysis}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{figures/statistical_significance.pdf}
\\caption{Statistical analysis of Q-learning performance compared to baselines: (a) p-values with significance thresholds, (b) Cohen's d effect sizes.}
\\label{fig:statistical-significance}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{figures/improvement_summary.pdf}
\\caption{Percentage improvement of Q-learning over baseline strategies. The red dashed line indicates the 30\\% improvement target. *** indicates statistical significance.}
\\label{fig:improvement-summary}
\\end{figure}
"""
        
        latex_file = self.dissertation_dir / "figure_includes.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX includes saved to {latex_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate dissertation charts")
    parser.add_argument("--results-dir", type=Path, 
                       default=Path(__file__).parent.parent / "results",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent.parent / "results",
                       help="Output directory for charts")
    parser.add_argument("--results-file", type=str,
                       default="baseline_comparison_final_*.json",
                       help="Results filename pattern")
    
    args = parser.parse_args()
    
    generator = ChartGenerator(args.results_dir, args.output_dir)
    generator.generate_all_charts(args.results_file)


if __name__ == "__main__":
    main()