#!/usr/bin/env python3
"""Demo script for automated baseline evaluation.

This script demonstrates the evaluation framework by comparing Q-learning
and DQN approaches against various baseline strategies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.comparison_visualizer import ComparisonVisualizer
from src.utils.logger import get_logger

# Setup logging
logger = get_logger("BaselineEvaluationDemo", log_dir="data/logs")


def load_config() -> dict:
    """Load configuration with evaluation settings."""
    config_path = Path("config/config.json")
    
    # Load base config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Add evaluation-specific settings
    config['evaluation'] = {
        'enabled': True,
        'baselines': ['random', 'popular', 'fixed_policy', 'greedy', 'context_agnostic'],
        'evaluation_interval': 100,
        'min_episodes_for_comparison': 50,
        'confidence_level': 0.95,
        'report_generation': {
            'format': ['html', 'pdf'],
            'frequency': 'daily',
            'include_visualizations': True
        }
    }
    
    # Q-learning settings
    config['q_learning'] = config.get('q_learning', {})
    config['q_learning'].update({
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'exploration_rate': 0.2,
        'exploration_decay': 0.995,
        'min_exploration_rate': 0.01,
        'max_tools': 3,
        'enable_learning': True
    })
    
    # DQN settings (optional)
    config['dqn'] = config.get('dqn', {})
    config['dqn'].update({
        'enabled': False,  # Set to True to include DQN in evaluation
        'network_type': 'standard',
        'learning_rate': 0.0001,
        'batch_size': 64,
        'memory_size': 10000
    })
    
    # General settings
    config['max_tools'] = 3
    config['figure_size'] = (12, 8)
    config['dpi'] = 100
    config['output_dir'] = 'src/evaluation/reports'
    
    return config


async def run_quick_evaluation(num_episodes: int = 500):
    """Run a quick evaluation with fewer episodes."""
    logger.info("=" * 60)
    logger.info("Starting Quick Baseline Evaluation Demo")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Create evaluation engine
    logger.info("Initializing evaluation engine...")
    engine = EvaluationEngine(config)
    
    # Generate test scenarios
    logger.info(f"Generating {num_episodes} test scenarios...")
    scenarios = engine.generate_test_scenarios(num_episodes)
    logger.info(f"Created {len(scenarios)} unique test scenarios")
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    logger.info("Strategies being evaluated:")
    for strategy_name in engine.strategies:
        logger.info(f"  - {strategy_name}")
    
    start_time = asyncio.get_event_loop().time()
    results = await engine.run_evaluation(num_episodes=num_episodes, parallel=True)
    elapsed_time = asyncio.get_event_loop().time() - start_time
    
    logger.info(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    
    # Display results summary
    display_results_summary(results)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    visualizer = ComparisonVisualizer(config)
    report_path = visualizer.generate_report(results)
    logger.info(f"Report saved to: {report_path}")
    
    # Save raw results
    results_path = Path(config['output_dir']) / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results_json(results, results_path)
    
    return results


async def run_full_evaluation(num_episodes: int = 2000):
    """Run a comprehensive evaluation."""
    logger.info("=" * 60)
    logger.info("Starting Comprehensive Baseline Evaluation")
    logger.info("=" * 60)
    
    # Load configuration with DQN enabled
    config = load_config()
    config['dqn']['enabled'] = True
    
    # Create evaluation engine
    engine = EvaluationEngine(config)
    
    # Generate diverse test scenarios
    logger.info(f"Generating {num_episodes} diverse test scenarios...")
    scenarios = engine.generate_test_scenarios(num_episodes)
    
    # Run evaluation
    logger.info("\nRunning comprehensive evaluation...")
    results = await engine.run_evaluation(num_episodes=num_episodes, parallel=True)
    
    # Detailed analysis
    logger.info("\nPerforming detailed analysis...")
    analyze_convergence(results)
    analyze_statistical_significance(results)
    analyze_tool_usage_patterns(results)
    
    # Generate comprehensive report
    visualizer = ComparisonVisualizer(config)
    report_path = visualizer.generate_report(results)
    
    # Save individual plots
    plot_paths = visualizer.save_plots_individually(results, prefix="full_eval")
    logger.info(f"Saved {len(plot_paths)} individual plots")
    
    return results


def display_results_summary(results: dict):
    """Display a summary of evaluation results."""
    summary = results.get('summary', {})
    comparisons = results.get('comparisons', {})
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Best strategy
    logger.info(f"\nBest Strategy: {summary.get('best_strategy', 'N/A')}")
    
    # Rankings
    logger.info("\nStrategy Rankings:")
    rankings = summary.get('rankings', [])
    for i, rank in enumerate(rankings, 1):
        logger.info(f"  {i}. {rank['strategy']:<20} "
                   f"Mean Reward: {rank['mean_reward']:.3f} "
                   f"Converged: {'Yes' if rank['convergence'] else 'No'}")
    
    # Statistical comparisons
    logger.info("\nStatistical Comparisons (vs random baseline):")
    for strategy, comp in comparisons.items():
        logger.info(f"\n  {strategy}:")
        logger.info(f"    Improvement: {comp['improvement']:.3f} ({comp['improvement_percent']:.1f}%)")
        logger.info(f"    P-value: {comp['p_value']:.4f} {'(Significant)' if comp['significant'] else '(Not significant)'}")
        logger.info(f"    Effect size: {comp['cohens_d']:.3f} ({comp['effect_size']})")
        logger.info(f"    Win rate: {comp['win_rate']:.2%}")


def analyze_convergence(results: dict):
    """Analyze convergence properties of strategies."""
    logger.info("\n" + "=" * 40)
    logger.info("CONVERGENCE ANALYSIS")
    logger.info("=" * 40)
    
    for strategy_name, data in results.get('strategies', {}).items():
        convergence = data.get('statistics', {}).get('convergence', {})
        
        logger.info(f"\n{strategy_name}:")
        if convergence.get('converged'):
            logger.info(f"  Converged at episode: {convergence.get('convergence_episode', 'N/A')}")
            logger.info(f"  Final performance: {convergence.get('final_performance', 0):.3f}")
        else:
            logger.info("  Did not converge")


def analyze_statistical_significance(results: dict):
    """Analyze statistical significance of improvements."""
    comparisons = results.get('comparisons', {})
    
    significant_improvements = [
        (name, comp) for name, comp in comparisons.items()
        if comp.get('significant', False) and comp.get('improvement', 0) > 0
    ]
    
    logger.info("\n" + "=" * 40)
    logger.info("SIGNIFICANT IMPROVEMENTS")
    logger.info("=" * 40)
    
    if significant_improvements:
        for name, comp in significant_improvements:
            logger.info(f"\n{name}:")
            logger.info(f"  Improvement: {comp['improvement']:.3f}")
            logger.info(f"  P-value: {comp['p_value']:.4f}")
            logger.info(f"  Cohen's d: {comp['cohens_d']:.3f}")
    else:
        logger.info("\nNo statistically significant improvements found.")


def analyze_tool_usage_patterns(results: dict):
    """Analyze how different strategies use tools."""
    logger.info("\n" + "=" * 40)
    logger.info("TOOL USAGE PATTERNS")
    logger.info("=" * 40)
    
    # This would require additional data from metrics collector
    # For now, just log a summary
    for strategy_name in results.get('strategies', {}):
        logger.info(f"\n{strategy_name}: Analysis would show tool preference patterns")


def save_results_json(results: dict, filepath: Path):
    """Save results in JSON format."""
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # Prepare JSON-serializable version
    json_results = {
        'timestamp': results.get('timestamp', datetime.now().isoformat()),
        'num_episodes': results.get('num_episodes'),
        'summary': results.get('summary', {}),
        'comparisons': results.get('comparisons', {}),
        'strategy_statistics': {}
    }
    
    # Add strategy statistics
    for name, data in results.get('strategies', {}).items():
        stats = data.get('statistics', {})
        json_results['strategy_statistics'][name] = {
            'mean_reward': convert_numpy(stats.get('reward', {}).get('mean', 0)),
            'std_reward': convert_numpy(stats.get('reward', {}).get('std', 0)),
            'converged': stats.get('convergence', {}).get('converged', False)
        }
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2, default=convert_numpy)
    
    logger.info(f"Results saved to: {filepath}")


async def main():
    """Main entry point for the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Evaluation Demo")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help="Evaluation mode: 'quick' (500 episodes) or 'full' (2000 episodes)")
    parser.add_argument('--episodes', type=int, default=None,
                       help="Override number of episodes")
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'quick':
            num_episodes = args.episodes or 500
            results = await run_quick_evaluation(num_episodes)
        else:
            num_episodes = args.episodes or 2000
            results = await run_full_evaluation(num_episodes)
            
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())