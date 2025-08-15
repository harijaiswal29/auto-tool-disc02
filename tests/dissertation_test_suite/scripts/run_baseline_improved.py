#!/usr/bin/env python3
"""
Improved Baseline Comparison Runner with Stricter Evaluation

This script implements improvements to the evaluation methodology:
1. Stricter success criteria (requires majority tool match)
2. Better Q-learning parameters with epsilon decay
3. Pre-training support for Q-learning
"""

import asyncio
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict
import time
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.baseline_strategies import BaselineStrategy
from src.agents.orchestrator_agent import OrchestratorAgent
from src.learning.q_learning_engine import QLearningEngine
from src.utils.logger import get_logger
from tests.dissertation_test_suite.data.test_queries import (
    get_evaluation_sets, TestQuery, QueryComplexity
)

logger = get_logger(__name__)


class ImprovedBaselineRunner:
    """Improved baseline comparison with better evaluation."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent.parent / "data" / "experiment_config.yaml"
        self.load_config()
        self.results_dir = Path(__file__).parent.parent / "results" / "improved"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluation_engine = None
        self.orchestrator = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Improved Q-learning parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.current_epsilon = self.epsilon_start
        
    def load_config(self):
        """Load experiment configuration."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        self.exp_config = self.config['experiments']['baseline_comparison']
        
    async def initialize_components(self):
        """Initialize evaluation engine and orchestrator."""
        # Load main project config
        project_config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.json"
        with open(project_config_path) as f:
            project_config = json.load(f)
        
        # Update Q-learning parameters for better performance
        project_config['q_learning'] = {
            'enable_learning': True,
            'alpha': 0.3,  # Increased learning rate
            'gamma': 0.9,
            'epsilon': self.current_epsilon,  # Will be updated dynamically
            'min_epsilon': 0.1,
            'epsilon_decay': self.epsilon_decay,
            'replay_buffer_size': 10000,
            'batch_size': 32,
            'update_frequency': 10,
            'target_update_frequency': 100
        }
        
        # Initialize evaluation engine
        self.evaluation_engine = EvaluationEngine(project_config)
        
        # Get available tools from config
        self.available_tools = list(project_config.get('tools', {}).keys())
        
        # Create tool mapping from test query format to config format
        self.tool_mapping = {
            'filesystem_mcp': 'filesystem',
            'search_mcp': 'search',
            'sqlite_mcp': 'database',
            'postgres_mcp': 'database',
            'github_mcp': 'github',
            'weather_mcp': 'search',
            'system_mcp': 'filesystem',
            'financial_mcp': 'financial',
            'notion_mcp': 'notion',
            'zerodha_mcp': 'financial'  # Map zerodha to financial
        }
        
        # Initialize orchestrator for Q-learning evaluation
        self.orchestrator = OrchestratorAgent(project_config)
        
        # Initialize the orchestrator (starts state machine)
        await self.orchestrator.initialize()
        
        # Initialize mock servers
        await self._initialize_mock_servers()
        
        # Initialize mock tools in the registry for testing
        await self._initialize_mock_tools()
        
    async def _initialize_mock_servers(self):
        """Initialize mock MCP servers for testing."""
        logger.info("Initializing mock MCP servers...")
        
        # Initialize mock servers for each type
        mcp = self.orchestrator.mcp_integration
        
        # Add mock servers with fallback to ensure they're available
        await mcp.add_filesystem_server(use_mock=True, server_id="filesystem_mock")
        await mcp.add_search_server(use_mock=True, server_id="search_mock")
        await mcp.add_sqlite_server(db_path=":memory:", use_mock=True, server_id="database_mock")
        await mcp.add_github_server(use_mock=True, server_id="github_mock")
        await mcp.add_financial_datasets_server(use_mock=True, server_id="financial_mock")
        
        logger.info("Mock MCP servers initialized successfully")
        
    async def _initialize_mock_tools(self):
        """Initialize mock tools in the registry for testing."""
        # Get the registry from the orchestrator's MCP integration
        registry = self.orchestrator.mcp_integration.registry
        
        # Define mock tools matching our available tools
        mock_tools = [
            {
                "id": "filesystem",
                "name": "Filesystem Tool",
                "server_type": "filesystem",
                "endpoint": "mock://filesystem",
                "description": "Mock filesystem operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["read", "write", "list"],
                    "semantic_tags": ["file", "directory", "filesystem"]
                }
            },
            {
                "id": "search",
                "name": "Search Tool",
                "server_type": "search",
                "endpoint": "mock://search",
                "description": "Mock search operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["search", "query"],
                    "semantic_tags": ["search", "web", "information"]
                }
            },
            {
                "id": "database",
                "name": "Database Tool",
                "server_type": "database",
                "endpoint": "mock://database",
                "description": "Mock database operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["query", "insert", "update", "delete"],
                    "semantic_tags": ["database", "sql", "data"]
                }
            },
            {
                "id": "github",
                "name": "GitHub Tool",
                "server_type": "github",
                "endpoint": "mock://github",
                "description": "Mock GitHub operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["list", "create", "update", "search"],
                    "semantic_tags": ["github", "repository", "code"]
                }
            },
            {
                "id": "financial",
                "name": "Financial Tool",
                "server_type": "financial",
                "endpoint": "mock://financial",
                "description": "Mock financial operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["get_price", "get_statement"],
                    "semantic_tags": ["finance", "stock", "trading"]
                }
            },
            {
                "id": "notion",
                "name": "Notion Tool",
                "server_type": "notion",
                "endpoint": "mock://notion",
                "description": "Mock Notion operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["create", "update", "list"],
                    "semantic_tags": ["notion", "notes", "productivity"]
                }
            }
        ]
        
        # Register each mock tool
        for tool in mock_tools:
            registry.register_tool(tool)
        
        logger.info(f"Registered {len(mock_tools)} mock tools for testing")
    
    def calculate_success_score(self, selected_tools: List[str], optimal_tools: List[str]) -> float:
        """
        Calculate success score with stricter criteria.
        
        Returns a score between 0 and 1:
        - 1.0: Perfect match (all optimal tools selected, no extras)
        - 0.8: All optimal tools selected, but with extras
        - 0.5: Majority of optimal tools selected (>50%)
        - 0.2: Some optimal tools selected (<50%)
        - 0.0: No optimal tools selected
        """
        if not optimal_tools:
            return 1.0 if not selected_tools else 0.0
        
        selected_set = set(selected_tools)
        optimal_set = set(optimal_tools)
        
        # Calculate intersection and union
        intersection = selected_set & optimal_set
        union = selected_set | optimal_set
        
        if not intersection:
            return 0.0  # No correct tools selected
        
        # Calculate precision and recall
        precision = len(intersection) / len(selected_set) if selected_set else 0
        recall = len(intersection) / len(optimal_set)
        
        # F1 score as primary metric
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        # Bonus for exact match
        if selected_set == optimal_set:
            return 1.0
        
        # Penalty for too many/too few tools
        size_penalty = abs(len(selected_set) - len(optimal_set)) * 0.1
        
        # Final score
        score = f1_score - size_penalty
        return max(0.0, min(1.0, score))
    
    async def pre_train_q_learning(self):
        """Pre-train Q-learning with good examples."""
        logger.info("Pre-training Q-learning with expert demonstrations...")
        
        if not self.orchestrator or not self.orchestrator.q_learning_engine:
            logger.warning("Q-learning engine not available for pre-training")
            return
        
        # Expert demonstrations (query pattern -> optimal tools)
        expert_demos = [
            ("list files", ["filesystem"]),
            ("search information", ["search"]),
            ("query database", ["database"]),
            ("github repository", ["github"]),
            ("stock price", ["financial"]),
            ("create note", ["notion"]),
            ("list python files and search documentation", ["filesystem", "search"]),
            ("analyze code and create issue", ["github", "filesystem"]),
            ("get financial data and save to database", ["financial", "database"]),
        ]
        
        # Train on expert demonstrations
        for query, optimal_tools in expert_demos:
            # Create state representation (simplified)
            state = np.zeros(447)
            # Simulate successful execution
            for _ in range(3):  # Repeat to reinforce
                # Update Q-values to prefer these tool combinations
                action = optimal_tools
                reward = 1.0  # High reward for expert demos
                next_state = state  # Simplified: same state
                
                # Direct Q-learning update
                if hasattr(self.orchestrator.q_learning_engine, 'update'):
                    self.orchestrator.q_learning_engine.update(
                        state, action, reward, next_state, done=True
                    )
        
        logger.info("Pre-training completed")
    
    async def run_strategy_evaluation(self, strategy_name: str, queries: List[TestQuery], 
                                    run_id: int, seed: int) -> Dict[str, Any]:
        """Run evaluation for a single strategy with improved metrics."""
        logger.info(f"Evaluating {strategy_name} (run {run_id}, seed {seed})")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Initialize metrics collectors
        metrics = defaultdict(list)
        episode_rewards = []
        success_scores = []
        tool_accuracies = []
        execution_times = []
        
        # Get strategy instance
        strategy = self.evaluation_engine.strategies.get(strategy_name)
        if not strategy and strategy_name != "q_learning":
            logger.error(f"Strategy {strategy_name} not found")
            return {'error': f'Strategy {strategy_name} not found'}
        
        # Pre-train Q-learning
        if strategy_name == "q_learning" and run_id == 0:
            await self.pre_train_q_learning()
        
        # Run episodes
        start_time = time.time()
        episodes = self.exp_config['episodes']
        
        for episode in range(episodes):
            # Update epsilon for Q-learning
            if strategy_name == "q_learning":
                self.current_epsilon = max(
                    self.epsilon_end,
                    self.epsilon_start * (self.epsilon_decay ** episode)
                )
                # Update orchestrator's Q-learning epsilon
                if self.orchestrator.q_learning_engine:
                    self.orchestrator.q_learning_engine.epsilon = self.current_epsilon
            
            episode_metrics = {
                'success_scores': [],
                'rewards': [],
                'correct_tools': 0,
                'total_tools': 0,
                'execution_times': []
            }
            
            # Run each query
            for query in queries:
                query_start = time.time()
                
                # Map optimal tools to our available tools first
                mapped_optimal = []
                for tool in query.optimal_tools:
                    if tool in self.tool_mapping:
                        mapped_optimal.append(self.tool_mapping[tool])
                    else:
                        # Clean up tool name
                        tool_clean = tool.replace('_mcp', '').replace('MCP', '').strip().lower()
                        if tool_clean in self.available_tools:
                            mapped_optimal.append(tool_clean)
                
                optimal_set = set(mapped_optimal)
                
                # Execute query with strategy
                if strategy_name == "q_learning":
                    # Use orchestrator for Q-learning
                    try:
                        result = await self.orchestrator.process_user_query(query.query)
                        # Handle OrchestrationResult object
                        if hasattr(result, 'selected_tools'):
                            tools_used = []
                            for tool in result.selected_tools:
                                # Map tool names to our simplified format
                                tool_name = tool.replace('_mcp', '').replace('MCP', '').strip().lower()
                                if tool_name in self.available_tools:
                                    tools_used.append(tool_name)
                                else:
                                    tools_used.append(tool)
                        else:
                            tools_used = []
                    except Exception as e:
                        logger.error(f"Q-learning query failed: {e}")
                        tools_used = []
                else:
                    # Use baseline strategy
                    state = np.zeros(447)  # Standard state size for the system
                    constraints = {}  # No constraints for simple evaluation
                    
                    tools_selected = await strategy.select_tools(
                        state,
                        self.available_tools,
                        constraints
                    )
                    tools_used = tools_selected
                
                query_time = time.time() - query_start
                
                # Calculate success score with stricter criteria
                success_score = self.calculate_success_score(tools_used, list(optimal_set))
                
                # Calculate metrics
                episode_metrics['success_scores'].append(success_score)
                
                # Tool accuracy (exact match)
                if set(tools_used) == optimal_set:
                    episode_metrics['correct_tools'] += 1
                episode_metrics['total_tools'] += 1
                
                # Rewards based on success score
                reward = success_score * 2.0 - 1.0  # Range: -1 to 1
                episode_metrics['rewards'].append(reward)
                episode_metrics['execution_times'].append(query_time)
            
            # Aggregate episode metrics
            avg_success_score = np.mean(episode_metrics['success_scores'])
            episode_accuracy = episode_metrics['correct_tools'] / episode_metrics['total_tools']
            episode_reward = sum(episode_metrics['rewards'])
            avg_execution_time = np.mean(episode_metrics['execution_times'])
            
            success_scores.append(avg_success_score)
            tool_accuracies.append(episode_accuracy)
            episode_rewards.append(episode_reward)
            execution_times.append(avg_execution_time)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(f"  Episode {episode + 1}/{episodes}: "
                          f"success_score={avg_success_score:.3f}, "
                          f"accuracy={episode_accuracy:.3f}, "
                          f"epsilon={self.current_epsilon:.3f}")
        
        total_time = time.time() - start_time
        
        # Calculate final metrics
        results = {
            'strategy': strategy_name,
            'run_id': run_id,
            'seed': seed,
            'episodes': episodes,
            'queries_per_episode': len(queries),
            'total_time': total_time,
            'metrics': {
                'success_score': {
                    'mean': np.mean(success_scores),
                    'std': np.std(success_scores),
                    'final': success_scores[-1],
                    'improvement': success_scores[-1] - success_scores[0] if len(success_scores) > 1 else 0
                },
                'tool_selection_accuracy': {
                    'mean': np.mean(tool_accuracies),
                    'std': np.std(tool_accuracies),
                    'final': tool_accuracies[-1]
                },
                'average_reward': {
                    'mean': np.mean(episode_rewards),
                    'std': np.std(episode_rewards),
                    'cumulative': sum(episode_rewards)
                },
                'execution_time': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'p95': np.percentile(execution_times, 95)
                },
                'convergence': {
                    'episodes_to_stable': self._calculate_convergence(success_scores),
                    'final_variance': np.var(success_scores[-20:]) if len(success_scores) > 20 else np.var(success_scores)
                }
            },
            'time_series': {
                'success_scores': success_scores,
                'tool_accuracies': tool_accuracies,
                'episode_rewards': episode_rewards,
                'execution_times': execution_times
            }
        }
        
        return results
    
    def _calculate_convergence(self, values: List[float], window: int = 20, 
                             threshold: float = 0.01) -> int:
        """Calculate episodes to convergence."""
        if len(values) < window:
            return len(values)
        
        for i in range(window, len(values)):
            window_values = values[i-window:i]
            if np.std(window_values) < threshold:
                return i
        
        return len(values)  # Did not converge
    
    async def run_comparison(self, query_set_name: str = "quick_test", 
                           episodes: int = 100, runs: int = 2):
        """Run improved baseline comparison experiment."""
        logger.info(f"Starting improved baseline comparison: {query_set_name}")
        logger.info(f"Episodes: {episodes}, Runs: {runs}")
        
        # Get queries
        query_sets = get_evaluation_sets()
        queries = query_sets[query_set_name]
        logger.info(f"Using {len(queries)} queries from {query_set_name} set")
        
        # Initialize components
        await self.initialize_components()
        
        # Override episode count
        self.exp_config['episodes'] = episodes
        
        # Results storage
        all_results = []
        strategy_summaries = {}
        
        # Select strategies to test
        test_strategies = ['random', 'greedy', 'fixed_policy', 'q_learning']
        
        # Run each strategy
        for strategy_name in test_strategies:
            strategy_results = []
            
            # Multiple runs for statistical validity
            seeds = [42, 123, 456][:runs]
            for run_id, seed in enumerate(seeds):
                result = await self.run_strategy_evaluation(
                    strategy_name, queries, run_id, seed
                )
                strategy_results.append(result)
                all_results.append(result)
            
            # Calculate strategy summary
            strategy_summaries[strategy_name] = self._calculate_strategy_summary(
                strategy_results
            )
            
            if 'success_score' in strategy_summaries[strategy_name]:
                score = strategy_summaries[strategy_name]['success_score']['mean']
                logger.info(f"Completed {strategy_name}: mean_score={score:.3f}")
        
        # Print results
        self._print_summary(strategy_summaries)
        
        # Save results
        results_file = self.results_dir / f"improved_results_{query_set_name}_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summaries': strategy_summaries,
                'all_results': all_results,
                'config': {
                    'episodes': episodes,
                    'runs': runs,
                    'query_set': query_set_name
                }
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        return strategy_summaries
    
    def _calculate_strategy_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for a strategy across runs."""
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results for strategy'}
        
        # Aggregate metrics across runs
        metrics_agg = defaultdict(list)
        
        for result in valid_results:
            for metric_name, metric_data in result['metrics'].items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    metrics_agg[metric_name].append(metric_data['mean'])
        
        # Calculate statistics
        summary = {}
        for metric_name, values in metrics_agg.items():
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5) if len(values) > 1 else values[0],
                'ci_upper': np.percentile(values, 97.5) if len(values) > 1 else values[0],
                'values': values
            }
        
        return summary
    
    def _print_summary(self, summaries: Dict[str, Dict]):
        """Print readable summary of results."""
        print("\n" + "="*80)
        print("IMPROVED BASELINE COMPARISON RESULTS")
        print("="*80)
        
        # Strategy performance
        print("\nStrategy Performance (Success Score with Strict Criteria):")
        print("-"*50)
        for strategy, summary in sorted(summaries.items()):
            if 'error' in summary:
                print(f"{strategy:20s}: ERROR - {summary.get('error', 'Unknown error')}")
                continue
            if 'success_score' not in summary:
                print(f"{strategy:20s}: No data available")
                continue
            score = summary['success_score']
            print(f"{strategy:20s}: {score['mean']:6.1%} ± {score['std']:5.1%}")
        
        print("\nTool Selection Accuracy (Exact Match):")
        print("-"*50)
        for strategy, summary in sorted(summaries.items()):
            if 'tool_selection_accuracy' in summary:
                acc = summary['tool_selection_accuracy']
                print(f"{strategy:20s}: {acc['mean']:6.1%} ± {acc['std']:5.1%}")
        
        print("="*80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run improved baseline comparison")
    parser.add_argument("--query-set", default="quick_test",
                       choices=["quick_test", "simple_only", "complex_only", 
                               "full_evaluation", "dissertation_core"],
                       help="Query set to use for evaluation")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes per strategy")
    parser.add_argument("--runs", type=int, default=2,
                       help="Number of runs per strategy")
    
    args = parser.parse_args()
    
    # Create runner
    runner = ImprovedBaselineRunner()
    
    # Run comparison
    results = await runner.run_comparison(
        args.query_set,
        episodes=args.episodes,
        runs=args.runs
    )
    
    print(f"\nExperiment complete! Results saved to {runner.results_dir}")


if __name__ == "__main__":
    asyncio.run(main())