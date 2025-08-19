#!/usr/bin/env python3
"""
Baseline comparison test for tool accuracy experiments.
This module provides functions to run evaluations with different configurations.
"""

import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.evaluation_engine import EvaluationEngine, TestScenario
from src.models.intent import Intent
from tests.dissertation_test_suite.data.test_queries import get_evaluation_sets, TestQuery

logger = logging.getLogger(__name__)

def create_test_scenarios_from_queries(queries: List[TestQuery]) -> List[TestScenario]:
    """Convert TestQuery objects to TestScenario objects for evaluation."""
    scenarios = []
    
    for i, query in enumerate(queries):
        # Create a mock intent with the query details
        class MockIntent:
            def __init__(self, q):
                self.type = q.intents[0] if q.intents else 'unknown'
                self.embedding = np.random.randn(384)  # Mock embedding
                self.confidence = 0.8
                self.query = q.query
        
        scenario = TestScenario(
            scenario_id=f"query_{i}",
            intent=MockIntent(query),
            available_tools=query.optimal_tools + ['search_mcp', 'filesystem_mcp'],  # Add some extras
            constraints={},
            expected_reward_range=(0.3, 1.0),
            description=query.query
        )
        # Add optimal tools as an attribute for tool-optimized calculator
        scenario.optimal_tools = query.optimal_tools
        scenarios.append(scenario)
    
    return scenarios

def run_evaluation_with_config(
    config: Dict[str, Any],
    query_set: str = 'dissertation_core',
    episodes: int = 100,
    output_dir: Optional[str] = None,
    track_tool_accuracy: bool = True
) -> Dict[str, Any]:
    """
    Run evaluation with specified configuration.
    
    Args:
        config: Configuration dictionary
        query_set: Name of query set to use
        episodes: Number of episodes to run
        output_dir: Directory to save results
        track_tool_accuracy: Whether to track tool selection accuracy
    
    Returns:
        Evaluation results dictionary
    """
    
    # Get queries for the specified set
    query_sets = get_evaluation_sets()
    queries = query_sets.get(query_set, query_sets['dissertation_core'])
    
    logger.info(f"Running evaluation with {len(queries)} queries for {episodes} episodes")
    
    # Create evaluation engine
    eval_engine = EvaluationEngine(config)
    
    # Convert queries to scenarios
    test_scenarios = create_test_scenarios_from_queries(queries)
    eval_engine.test_scenarios = test_scenarios
    
    # Run evaluation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(
            eval_engine.run_evaluation(num_episodes=episodes, parallel=False)
        )
    finally:
        loop.close()
    
    # Add tool accuracy tracking if enabled
    if track_tool_accuracy:
        results = add_tool_accuracy_metrics(results, queries)
    
    # Save results if output directory specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save full results
        results_path = Path(output_dir) / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary JSON
        summary_path = Path(output_dir) / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary = extract_summary(results)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    return results

def add_tool_accuracy_metrics(results: Dict[str, Any], queries: List[TestQuery]) -> Dict[str, Any]:
    """
    Add tool selection accuracy metrics to results.
    
    Calculates:
    - Exact match rate: Perfect tool selection
    - Partial match rate: Some correct tools selected
    - Tool distance: Average distance from optimal
    """
    
    # Create ground truth mapping
    ground_truth = {f"query_{i}": q.optimal_tools for i, q in enumerate(queries)}
    
    # Process each strategy's results
    for strategy_name, strategy_results in results.get('strategies', {}).items():
        if 'selections' not in strategy_results:
            continue
        
        selections = strategy_results['selections']
        exact_matches = 0
        partial_matches = 0
        total_distance = 0
        
        for i, selected_tools in enumerate(selections):
            query_id = f"query_{i % len(queries)}"
            optimal = set(ground_truth.get(query_id, []))
            selected = set(selected_tools) if selected_tools else set()
            
            # Exact match
            if selected == optimal:
                exact_matches += 1
                partial_matches += 1
            # Partial match
            elif len(selected.intersection(optimal)) > 0:
                partial_matches += 1
            
            # Calculate distance (Jaccard distance)
            if len(selected.union(optimal)) > 0:
                distance = 1 - len(selected.intersection(optimal)) / len(selected.union(optimal))
            else:
                distance = 1.0
            total_distance += distance
        
        # Add metrics to results
        num_episodes = len(selections)
        if num_episodes > 0:
            strategy_results['tool_accuracy_metrics'] = {
                'exact_match_rate': exact_matches / num_episodes,
                'partial_match_rate': partial_matches / num_episodes,
                'average_distance': total_distance / num_episodes
            }
            
            # Update tool_selection_accuracy if not present
            if 'tool_selection_accuracy' not in strategy_results.get('statistics', {}):
                if 'statistics' not in strategy_results:
                    strategy_results['statistics'] = {}
                strategy_results['statistics']['tool_selection_accuracy'] = {
                    'mean': exact_matches / num_episodes,
                    'std': np.std([1 if i < exact_matches else 0 for i in range(num_episodes)])
                }
    
    return results

def extract_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a JSON-serializable summary from results."""
    
    summary = {
        'timestamp': results.get('timestamp', datetime.now().isoformat()),
        'num_episodes': results.get('num_episodes', 0),
        'strategies': {}
    }
    
    for strategy_name, strategy_data in results.get('strategies', {}).items():
        stats = strategy_data.get('statistics', {})
        
        summary['strategies'][strategy_name] = {
            'task_completion_rate': {
                'mean': stats.get('reward', {}).get('mean', 0),
                'std': stats.get('reward', {}).get('std', 0)
            },
            'tool_selection_accuracy': {
                'mean': stats.get('tool_selection_accuracy', {}).get('mean', 0),
                'std': stats.get('tool_selection_accuracy', {}).get('std', 0)
            },
            'convergence': stats.get('convergence', {})
        }
        
        # Add tool accuracy metrics if present
        if 'tool_accuracy_metrics' in strategy_data:
            summary['strategies'][strategy_name]['tool_accuracy_metrics'] = strategy_data['tool_accuracy_metrics']
    
    # Add comparisons if present
    if 'comparisons' in results:
        summary['comparisons'] = results['comparisons']
    
    # Add summary statistics
    if 'summary' in results:
        summary['overall_summary'] = results['summary']
    
    return summary

def compare_configurations(
    standard_config: Dict[str, Any],
    optimized_config: Dict[str, Any],
    episodes: int = 100
) -> Dict[str, Any]:
    """
    Compare standard and tool-optimized configurations.
    
    Returns comparative analysis of both approaches.
    """
    
    logger.info("Running comparative evaluation...")
    
    # Run both evaluations
    standard_results = run_evaluation_with_config(
        standard_config, 
        episodes=episodes,
        track_tool_accuracy=True
    )
    
    optimized_results = run_evaluation_with_config(
        optimized_config,
        episodes=episodes,
        track_tool_accuracy=True
    )
    
    # Compare results
    comparison = {
        'standard': extract_summary(standard_results),
        'optimized': extract_summary(optimized_results),
        'improvements': {}
    }
    
    # Calculate improvements for each strategy
    for strategy in standard_results.get('strategies', {}).keys():
        if strategy in optimized_results.get('strategies', {}):
            std_acc = standard_results['strategies'][strategy].get('statistics', {}).get('tool_selection_accuracy', {}).get('mean', 0)
            opt_acc = optimized_results['strategies'][strategy].get('statistics', {}).get('tool_selection_accuracy', {}).get('mean', 0)
            
            std_task = standard_results['strategies'][strategy].get('statistics', {}).get('reward', {}).get('mean', 0)
            opt_task = optimized_results['strategies'][strategy].get('statistics', {}).get('reward', {}).get('mean', 0)
            
            comparison['improvements'][strategy] = {
                'tool_accuracy_improvement': opt_acc - std_acc,
                'tool_accuracy_percent_change': ((opt_acc - std_acc) / std_acc * 100) if std_acc > 0 else 0,
                'task_completion_change': opt_task - std_task,
                'task_completion_percent_change': ((opt_task - std_task) / std_task * 100) if std_task > 0 else 0
            }
    
    return comparison

if __name__ == "__main__":
    # Test the module
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # Run a quick test
    results = run_evaluation_with_config(
        config,
        query_set='quick_test',
        episodes=10,
        output_dir='test_results',
        track_tool_accuracy=True
    )
    
    print(f"Test completed. Results saved to test_results/")
    print(f"Strategies evaluated: {list(results.get('strategies', {}).keys())}")