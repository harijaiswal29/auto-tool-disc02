#!/usr/bin/env python3
"""Evaluate the impact of the supervised encoder on Q-learning performance.

This script compares performance with and without the encoder to measure
its effectiveness in improving learning speed and final performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import subprocess
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation(config_path: str, episodes: int, use_encoder: bool,
                  encoder_path: str = None, output_dir: str = None) -> Dict:
    """Run evaluation with or without encoder.
    
    Args:
        config_path: Path to configuration file
        episodes: Number of episodes to run
        use_encoder: Whether to use encoder
        encoder_path: Path to encoder model
        output_dir: Directory for results
        
    Returns:
        Dictionary of evaluation results
    """
    # Load and modify configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update encoder configuration
    config['state_encoder']['enabled'] = use_encoder
    if encoder_path:
        config['state_encoder']['model_path'] = encoder_path
    
    # Create temporary config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_config = f"/tmp/config_encoder_{timestamp}.json"
    
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare command
    cmd = [
        'python', 'tests/dissertation_test_suite/scripts/run_baseline_comparison.py',
        '--config', temp_config,
        '--episodes', str(episodes),
        '--strategies', 'q_learning_tabular', 'q_learning_dqn', 'random'
    ]
    
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    
    logger.info(f"Running evaluation {'with' if use_encoder else 'without'} encoder")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run evaluation
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed_time = time.time() - start_time
        
        # Parse output to find results file
        output_lines = result.stdout.split('\n')
        results_file = None
        for line in output_lines:
            if 'baseline_comparison_final' in line and '.json' in line:
                # Extract file path from output
                parts = line.split()
                for part in parts:
                    if 'baseline_comparison_final' in part and '.json' in part:
                        results_file = part
                        break
        
        if not results_file:
            # Try to find it in the output directory
            if output_dir:
                for file in Path(output_dir).glob('**/baseline_comparison_final*.json'):
                    results_file = str(file)
                    break
        
        if results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                results['elapsed_time'] = elapsed_time
                results['encoder_used'] = use_encoder
                return results
        else:
            logger.error("Could not find results file")
            return {
                'error': 'Results file not found',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed_time': elapsed_time,
                'encoder_used': use_encoder
            }
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr,
            'elapsed_time': time.time() - start_time,
            'encoder_used': use_encoder
        }
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config):
            os.remove(temp_config)


def plot_comparison(results_without: Dict, results_with: Dict, save_path: str = None):
    """Plot comparison of results with and without encoder.
    
    Args:
        results_without: Results without encoder
        results_with: Results with encoder
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    strategies = ['q_learning_tabular', 'q_learning_dqn', 'random']
    
    for idx, strategy in enumerate(strategies):
        # Extract data
        without_data = results_without.get('strategies', {}).get(strategy, {})
        with_data = results_with.get('strategies', {}).get(strategy, {})
        
        # Completion rates
        without_rates = without_data.get('completion_rates', [])
        with_rates = with_data.get('completion_rates', [])
        
        if without_rates and with_rates:
            episodes = range(1, min(len(without_rates), len(with_rates)) + 1)
            
            # Plot completion rates
            axes[0, idx].plot(episodes, without_rates[:len(episodes)], 
                            'r-', label='Without Encoder', alpha=0.7)
            axes[0, idx].plot(episodes, with_rates[:len(episodes)], 
                            'b-', label='With Encoder', alpha=0.7)
            axes[0, idx].set_title(f'{strategy.replace("_", " ").title()}')
            axes[0, idx].set_xlabel('Episode')
            axes[0, idx].set_ylabel('Completion Rate')
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)
            
            # Calculate moving average for smoother comparison
            window = min(10, len(episodes) // 10)
            if window > 1:
                without_smooth = np.convolve(without_rates[:len(episodes)], 
                                            np.ones(window)/window, mode='valid')
                with_smooth = np.convolve(with_rates[:len(episodes)], 
                                         np.ones(window)/window, mode='valid')
                
                smooth_episodes = range(window//2, len(episodes) - window//2 + 1)
                axes[0, idx].plot(smooth_episodes, without_smooth, 
                                'r--', linewidth=2, alpha=0.8)
                axes[0, idx].plot(smooth_episodes, with_smooth, 
                                'b--', linewidth=2, alpha=0.8)
    
    # Performance metrics comparison
    metrics_to_compare = ['mean_completion_rate', 'final_completion_rate', 'convergence_speed']
    
    for idx, metric in enumerate(metrics_to_compare):
        values_without = []
        values_with = []
        labels = []
        
        for strategy in strategies:
            without_val = results_without.get('strategies', {}).get(strategy, {}).get(metric, 0)
            with_val = results_with.get('strategies', {}).get(strategy, {}).get(metric, 0)
            
            values_without.append(without_val)
            values_with.append(with_val)
            labels.append(strategy.replace('_', ' ').title())
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1, idx].bar(x - width/2, values_without, width, 
                        label='Without Encoder', color='red', alpha=0.7)
        axes[1, idx].bar(x + width/2, values_with, width, 
                        label='With Encoder', color='blue', alpha=0.7)
        
        axes[1, idx].set_xlabel('Strategy')
        axes[1, idx].set_ylabel(metric.replace('_', ' ').title())
        axes[1, idx].set_title(metric.replace('_', ' ').title())
        axes[1, idx].set_xticks(x)
        axes[1, idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Supervised Encoder on Learning Performance', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()


def calculate_improvement_metrics(results_without: Dict, results_with: Dict) -> Dict:
    """Calculate improvement metrics from using encoder.
    
    Args:
        results_without: Results without encoder
        results_with: Results with encoder
        
    Returns:
        Dictionary of improvement metrics
    """
    improvements = {}
    
    for strategy in ['q_learning_tabular', 'q_learning_dqn']:
        without = results_without.get('strategies', {}).get(strategy, {})
        with_encoder = results_with.get('strategies', {}).get(strategy, {})
        
        if without and with_encoder:
            strategy_improvements = {}
            
            # Completion rate improvement
            without_mean = without.get('mean_completion_rate', 0)
            with_mean = with_encoder.get('mean_completion_rate', 0)
            if without_mean > 0:
                strategy_improvements['mean_improvement'] = (with_mean - without_mean) / without_mean * 100
            else:
                strategy_improvements['mean_improvement'] = 0
            
            # Final performance improvement
            without_final = without.get('final_completion_rate', 0)
            with_final = with_encoder.get('final_completion_rate', 0)
            if without_final > 0:
                strategy_improvements['final_improvement'] = (with_final - without_final) / without_final * 100
            else:
                strategy_improvements['final_improvement'] = 0
            
            # Convergence speed improvement
            without_conv = without.get('convergence_episode', float('inf'))
            with_conv = with_encoder.get('convergence_episode', float('inf'))
            if without_conv < float('inf') and with_conv < float('inf'):
                strategy_improvements['convergence_speedup'] = (without_conv - with_conv) / without_conv * 100
            else:
                strategy_improvements['convergence_speedup'] = 0
            
            # Training time comparison
            strategy_improvements['time_ratio'] = (
                results_with.get('elapsed_time', 0) / 
                results_without.get('elapsed_time', 1)
            )
            
            improvements[strategy] = strategy_improvements
    
    return improvements


def main():
    parser = argparse.ArgumentParser(description="Evaluate encoder impact on learning")
    
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Configuration file path')
    parser.add_argument('--encoder-path', type=str, 
                       default='models/supervised_encoder/best_encoder.pth',
                       help='Path to trained encoder model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to run')
    parser.add_argument('--output-dir', type=str, 
                       default='tests/dissertation_test_suite/results/encoder_evaluation',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip encoder training (use existing model)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train encoder if not skipping
    if not args.skip_training and not os.path.exists(args.encoder_path):
        logger.info("Training encoder first...")
        train_cmd = [
            'python', 'scripts/train_supervised_encoder.py',
            '--epochs', '50',
            '--batch-size', '256',
            '--save-dir', os.path.dirname(args.encoder_path)
        ]
        
        try:
            subprocess.run(train_cmd, check=True)
            logger.info("Encoder training completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Encoder training failed: {e}")
            return
    
    # Run evaluation without encoder
    logger.info("\n" + "="*50)
    logger.info("Running baseline evaluation WITHOUT encoder")
    logger.info("="*50)
    
    results_without = run_evaluation(
        args.config,
        args.episodes,
        use_encoder=False,
        output_dir=os.path.join(args.output_dir, f"without_encoder_{timestamp}")
    )
    
    # Run evaluation with encoder
    logger.info("\n" + "="*50)
    logger.info("Running evaluation WITH encoder")
    logger.info("="*50)
    
    results_with = run_evaluation(
        args.config,
        args.episodes,
        use_encoder=True,
        encoder_path=args.encoder_path,
        output_dir=os.path.join(args.output_dir, f"with_encoder_{timestamp}")
    )
    
    # Calculate improvements
    improvements = calculate_improvement_metrics(results_without, results_with)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    for strategy, metrics in improvements.items():
        logger.info(f"\n{strategy.upper()}:")
        logger.info(f"  Mean Completion Rate Improvement: {metrics.get('mean_improvement', 0):.1f}%")
        logger.info(f"  Final Performance Improvement: {metrics.get('final_improvement', 0):.1f}%")
        logger.info(f"  Convergence Speedup: {metrics.get('convergence_speedup', 0):.1f}%")
        logger.info(f"  Time Ratio (with/without): {metrics.get('time_ratio', 1):.2f}x")
    
    # Save comparison results
    comparison_results = {
        'timestamp': timestamp,
        'episodes': args.episodes,
        'encoder_path': args.encoder_path,
        'results_without_encoder': results_without,
        'results_with_encoder': results_with,
        'improvements': improvements
    }
    
    results_path = os.path.join(args.output_dir, f'encoder_comparison_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\nSaved comparison results to {results_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = os.path.join(args.output_dir, f'encoder_comparison_{timestamp}.png')
        plot_comparison(results_without, results_with, plot_path)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    
    avg_improvement = np.mean([
        m.get('mean_improvement', 0) 
        for m in improvements.values()
    ])
    
    if avg_improvement > 0:
        logger.info(f"✓ Encoder provides average {avg_improvement:.1f}% improvement")
        logger.info("✓ Recommendation: Enable encoder for better performance")
    else:
        logger.info("✗ Encoder does not provide significant improvement")
        logger.info("✗ Recommendation: Continue with raw state representation")


if __name__ == "__main__":
    main()