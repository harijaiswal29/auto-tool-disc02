#!/usr/bin/env python3
"""
Baseline Comparison Runner for Dissertation Evaluation

This script orchestrates the comparison of Q-learning against all baseline
strategies, collecting comprehensive metrics and generating statistical results.
"""

import asyncio
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict
import time
import sys
import pickle
import os

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


class CheckpointManager:
    """Manages checkpoint saving and loading for experiment resumption."""
    
    def __init__(self, checkpoint_dir: Path, checkpoint_interval: int = 100):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Episodes between checkpoint saves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.current_checkpoint_file = None
        
    def should_checkpoint(self, episode: int) -> bool:
        """Check if checkpoint should be saved at this episode."""
        return episode > 0 and episode % self.checkpoint_interval == 0
    
    def save_checkpoint(self, state: Dict[str, Any], episode: int, strategy_name: str) -> str:
        """
        Save checkpoint to disk.
        
        Args:
            state: Complete state to save
            episode: Current episode number
            strategy_name: Name of current strategy
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{strategy_name}_ep{episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Create backup if overwriting
        if self.current_checkpoint_file and self.current_checkpoint_file.exists():
            backup_file = self.current_checkpoint_file.with_suffix('.pkl.bak')
            self.current_checkpoint_file.rename(backup_file)
            logger.info(f"Backed up previous checkpoint to {backup_file.name}")
        
        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        self.current_checkpoint_file = checkpoint_file
        logger.info(f"Checkpoint saved to {checkpoint_file.name} (episode {episode})")
        
        # Clean up old backups (keep only last 2)
        backups = sorted(self.checkpoint_dir.glob("*.pkl.bak"))
        if len(backups) > 2:
            for old_backup in backups[:-2]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup.name}")
        
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load checkpoint from disk.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Loaded state dictionary
        """
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path.name}")
        logger.info(f"  Resume from episode: {state.get('episode', 0)}")
        logger.info(f"  Strategy: {state.get('strategy_name', 'unknown')}")
        logger.info(f"  Metrics collected: {len(state.get('metrics', {}).get('completion_rates', []))}")
        
        return state
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoint files."""
        return sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))


class BaselineComparisonRunner:
    """Orchestrates baseline comparison experiments."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent.parent / "data" / "experiment_config.yaml"
        self.load_config()
        self.results_dir = Path(__file__).parent.parent / "results" / "raw_data"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluation_engine = None
        self.orchestrator = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_retries = False  # Default: no retries for experiments
        
        # Checkpoint manager (initialized when needed)
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.resume_state: Optional[Dict[str, Any]] = None
        
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
        
        # Configure retries based on experiment settings
        project_config = self._configure_retries(project_config)
        
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
            'weather_mcp': 'search',  # Weather queries can use search
            'system_mcp': 'filesystem',  # System queries can use filesystem
            'financial_mcp': 'financial',
            'notion_mcp': 'notion'
        }
        
        # Initialize orchestrator for Q-learning evaluation
        self.orchestrator = OrchestratorAgent(project_config)
        
        # Initialize the orchestrator (starts state machine)
        await self.orchestrator.initialize()
        
        # Initialize mock servers before registering tools
        await self._initialize_mock_servers()
        
        # Initialize mock tools in the registry for testing
        await self._initialize_mock_tools()
        
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
                    "operations": ["repository", "issues", "pull_requests"],
                    "semantic_tags": ["git", "github", "repository", "version"]
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
                    "operations": ["data", "analysis", "market"],
                    "semantic_tags": ["finance", "market", "data", "analysis"]
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
                    "operations": ["pages", "blocks", "databases"],
                    "semantic_tags": ["notion", "documentation", "pages"]
                }
            }
        ]
        
        # Register each mock tool
        for tool in mock_tools:
            registry.register_tool(tool)
        
        logger.info(f"Registered {len(mock_tools)} mock tools for testing")
    
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
        # Note: add_notion_server doesn't exist, skip it
        
        logger.info("Mock MCP servers initialized successfully")
        
    async def run_strategy_evaluation(self, strategy_name: str, queries: List[TestQuery], 
                                    run_id: int, seed: int, checkpoint_enabled: bool = False) -> Dict[str, Any]:
        """Run evaluation for a single strategy with optional checkpoint support."""
        logger.info(f"Evaluating {strategy_name} (run {run_id}, seed {seed})")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Check if resuming from checkpoint
        start_episode = 0
        completion_rates = []
        tool_accuracies = []
        episode_rewards = []
        execution_times = []
        
        if self.resume_state and self.resume_state.get('strategy_name') == strategy_name:
            logger.info(f"Resuming {strategy_name} from checkpoint")
            start_episode = self.resume_state.get('episode', 0)
            metrics = self.resume_state.get('metrics', {})
            completion_rates = metrics.get('completion_rates', [])
            tool_accuracies = metrics.get('tool_accuracies', [])
            episode_rewards = metrics.get('episode_rewards', [])
            execution_times = metrics.get('execution_times', [])
            
            # Restore Q-learning state if applicable
            if 'q_learning' in strategy_name and 'q_learning_state' in self.resume_state:
                strategy = self.evaluation_engine.strategies.get(strategy_name)
                if hasattr(strategy, 'load_state'):
                    strategy.load_state(self.resume_state['q_learning_state'])
                    logger.info(f"Restored Q-learning state for {strategy_name}")
            
            logger.info(f"Resuming from episode {start_episode} with {len(completion_rates)} episodes already completed")
            self.resume_state = None  # Clear resume state after using
        
        # Log which Q-learning variant is being used
        if strategy_name == "q_learning_dqn":
            logger.info("Using Deep Q-Network (DQN) for tool selection")
        elif strategy_name == "q_learning_tabular":
            logger.info("Using Tabular Q-learning (Q-table) for tool selection")
        elif strategy_name == "q_learning" and self.orchestrator.q_learning_engine:
            if self.orchestrator.q_learning_engine.use_dqn:
                logger.info("Using Q-learning with DQN (config-based)")
            else:
                logger.info("Using Q-learning with Q-table (config-based)")
        
        # Initialize metrics collectors (skip if resuming)
        if not completion_rates:  # Only initialize if not resuming
            metrics = defaultdict(list)
        
        # Get strategy instance
        strategy = self.evaluation_engine.strategies.get(strategy_name)
        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return {'error': f'Strategy {strategy_name} not found'}
        
        # Run episodes
        start_time = time.time()
        episodes = self.exp_config['episodes']
        
        for episode in range(start_episode, episodes):
            episode_metrics = {
                'completed': 0,
                'total': 0,
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
                        mapped_optimal.append(tool)
                
                optimal_set = set(mapped_optimal)
                
                # Execute query with strategy
                if strategy_name in ["q_learning_tabular", "q_learning_dqn"]:
                    # Use strategy instance for new Q-learning strategies
                    # These strategies have their own select_tools method
                    try:
                        # Create state from query (full 476-dimensional state)
                        state = np.zeros(476)  # Full state size with all features
                        constraints = {}  # No constraints for simple evaluation
                        
                        tools_selected = await strategy.select_tools(
                            state,
                            self.available_tools,
                            constraints
                        )
                        tools_used = tools_selected  # Strategy returns list of tool names
                        
                        # Simulate execution success based on tool match
                        selected_set = set(tools_used)
                        success = len(optimal_set & selected_set) > 0
                        
                        logger.debug(f"{strategy_name} selected tools: {tools_used}")
                    except Exception as e:
                        logger.error(f"{strategy_name} query failed: {e}")
                        tools_used = []
                        success = False
                        
                elif strategy_name == "q_learning":
                    # Legacy Q-learning uses orchestrator (backward compatibility)
                    try:
                        result = await self.orchestrator.process_user_query(query.query)
                        # Handle OrchestrationResult object
                        if hasattr(result, 'selected_tools'):
                            # Extract tool names from selected_tools list
                            tools_used = []
                            for tool in result.selected_tools:
                                # Map tool names to our simplified format
                                tool_name = tool.replace('_mcp', '').replace('MCP', '').strip().lower()
                                if tool_name in self.available_tools:
                                    tools_used.append(tool_name)
                                else:
                                    tools_used.append(tool)
                            
                            # For mock testing, consider it successful if we selected the right tools
                            # even if execution failed due to no active servers
                            selected_set = set(tools_used)
                            success = len(optimal_set & selected_set) > 0
                            
                            # Override with actual success if execution worked
                            if hasattr(result, 'success') and result.success:
                                success = True
                        else:
                            # Fallback 
                            tools_used = []
                            success = False
                    except Exception as e:
                        logger.error(f"Q-learning query failed: {e}")
                        tools_used = []
                        success = False
                else:
                    # Use baseline strategy
                    # Create a dummy state representation for baseline strategies
                    state = np.zeros(476)  # Full state size for consistency
                    constraints = {}  # No constraints for simple evaluation
                    
                    tools_selected = await strategy.select_tools(
                        state,
                        self.available_tools,
                        constraints
                    )
                    tools_used = tools_selected  # Baseline strategies return list of tool names
                    # Simulate execution success based on tool match
                    selected_set = set(tools_used)
                    success = len(optimal_set & selected_set) > 0
                
                query_time = time.time() - query_start
                
                # Calculate metrics
                episode_metrics['total'] += 1
                if success:
                    episode_metrics['completed'] += 1
                
                # Tool accuracy
                if set(tools_used) == optimal_set:
                    episode_metrics['correct_tools'] += 1
                episode_metrics['total_tools'] += 1
                
                # Rewards (simple: 1 for success, -0.1 for failure)
                reward = 1.0 if success else -0.1
                episode_metrics['rewards'].append(reward)
                episode_metrics['execution_times'].append(query_time)
            
            # Aggregate episode metrics
            episode_completion = episode_metrics['completed'] / episode_metrics['total']
            episode_accuracy = episode_metrics['correct_tools'] / episode_metrics['total_tools']
            episode_reward = sum(episode_metrics['rewards'])
            avg_execution_time = np.mean(episode_metrics['execution_times'])
            
            completion_rates.append(episode_completion)
            tool_accuracies.append(episode_accuracy)
            episode_rewards.append(episode_reward)
            execution_times.append(avg_execution_time)
            
            # Log progress
            if (episode + 1) % 100 == 0:
                logger.info(f"  Episode {episode + 1}/{episodes}: "
                          f"completion={episode_completion:.2f}, "
                          f"accuracy={episode_accuracy:.2f}")
            
            # Save checkpoint if enabled
            if checkpoint_enabled and self.checkpoint_manager and self.checkpoint_manager.should_checkpoint(episode + 1):
                checkpoint_state = {
                    'strategy_name': strategy_name,
                    'episode': episode + 1,
                    'run_id': run_id,
                    'seed': seed,
                    'metrics': {
                        'completion_rates': completion_rates,
                        'tool_accuracies': tool_accuracies,
                        'episode_rewards': episode_rewards,
                        'execution_times': execution_times
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save Q-learning state if applicable
                if 'q_learning' in strategy_name and hasattr(strategy, 'get_state'):
                    checkpoint_state['q_learning_state'] = strategy.get_state()
                
                self.checkpoint_manager.save_checkpoint(checkpoint_state, episode + 1, strategy_name)
        
        # Save final checkpoint after training loop completes
        if checkpoint_enabled and self.checkpoint_manager:
            final_episode = episodes
            # Only save if we haven't just saved at this episode
            if not self.checkpoint_manager.should_checkpoint(final_episode):
                final_checkpoint_state = {
                    'strategy_name': strategy_name,
                    'episode': final_episode,
                    'run_id': run_id,
                    'seed': seed,
                    'metrics': {
                        'completion_rates': completion_rates,
                        'tool_accuracies': tool_accuracies,
                        'rewards': episode_rewards,
                        'execution_times': execution_times
                    }
                }
                
                # Save Q-learning state if applicable
                if 'q_learning' in strategy_name and hasattr(strategy, 'get_state'):
                    final_checkpoint_state['q_learning_state'] = strategy.get_state()
                
                # Save with special suffix to indicate final checkpoint
                logger.info(f"Saving final checkpoint for {strategy_name} at episode {final_episode}")
                self.checkpoint_manager.save_checkpoint(final_checkpoint_state, final_episode, f"{strategy_name}_final")
        
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
                'task_completion_rate': {
                    'mean': np.mean(completion_rates),
                    'std': np.std(completion_rates),
                    'final': completion_rates[-1],
                    'improvement': completion_rates[-1] - completion_rates[0] if len(completion_rates) > 1 else 0
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
                    'episodes_to_stable': self._calculate_convergence(completion_rates),
                    'final_variance': np.var(completion_rates[-100:]) if len(completion_rates) > 100 else np.var(completion_rates)
                }
            },
            'time_series': {
                'completion_rates': completion_rates,
                'tool_accuracies': tool_accuracies,
                'episode_rewards': episode_rewards,
                'execution_times': execution_times
            }
        }
        
        return results
    
    def _configure_retries(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure retry settings based on experiment requirements.
        
        Args:
            config: Project configuration dictionary
            
        Returns:
            Modified configuration with retry settings
        """
        if self.enable_retries:
            # Keep default retry settings (usually max_retries=3)
            logger.info("Retries ENABLED for this experiment run")
            
            # Ensure sections exist with default retry values
            if 'orchestration_state_machine' not in config:
                config['orchestration_state_machine'] = {}
            if 'max_retries' not in config['orchestration_state_machine']:
                config['orchestration_state_machine']['max_retries'] = 3
            
            if 'mcp' not in config:
                config['mcp'] = {}
            if 'tool_discovery' not in config['mcp']:
                config['mcp']['tool_discovery'] = {}
            if 'max_retries' not in config['mcp']['tool_discovery']:
                config['mcp']['tool_discovery']['max_retries'] = 3
            
            logger.info(f"  Orchestration max_retries: {config['orchestration_state_machine']['max_retries']}")
            logger.info(f"  Tool discovery max_retries: {config['mcp']['tool_discovery']['max_retries']}")
        else:
            # Disable retries for clean experiment measurements
            logger.info("Retries DISABLED for this experiment run (recommended for research)")
            
            # Create sections if they don't exist and set retries to 0
            if 'orchestration_state_machine' not in config:
                config['orchestration_state_machine'] = {}
            config['orchestration_state_machine']['max_retries'] = 0
            logger.info("  Set orchestration_state_machine.max_retries = 0")
            
            # Disable MCP tool discovery retries
            if 'mcp' not in config:
                config['mcp'] = {}
            if 'tool_discovery' not in config['mcp']:
                config['mcp']['tool_discovery'] = {}
            config['mcp']['tool_discovery']['max_retries'] = 0
            logger.info("  Set mcp.tool_discovery.max_retries = 0")
            
            # Also patch retry decorator to ensure no retries at function level
            try:
                from src.utils import retry
                
                def no_retry_decorator(*args, **kwargs):
                    def decorator(func):
                        return func
                    return decorator
                
                retry.retry_async = no_retry_decorator
                logger.info("  Patched retry decorator to disable function-level retries")
            except ImportError:
                logger.warning("  Could not patch retry decorator (module not found)")
        
        return config
    
    def _calculate_convergence(self, values: List[float], window: int = 100, 
                             threshold: float = 0.01) -> int:
        """Calculate episodes to convergence."""
        if len(values) < window:
            return len(values)
        
        for i in range(window, len(values)):
            window_values = values[i-window:i]
            if np.std(window_values) < threshold:
                return i
        
        return len(values)  # Did not converge
    
    async def run_full_comparison(self, query_set_name: str = "dissertation_core", 
                                 episodes: int = None, runs: int = None,
                                 checkpoint_interval: int = None):
        """Run full baseline comparison experiment with optional checkpointing."""
        logger.info(f"Starting baseline comparison experiment: {query_set_name}")
        
        # Check if checkpointing is enabled
        checkpoint_enabled = checkpoint_interval is not None and checkpoint_interval > 0
        if checkpoint_enabled:
            logger.info(f"Checkpointing enabled: saving every {checkpoint_interval} episodes")
        
        # Get queries
        query_sets = get_evaluation_sets()
        queries = query_sets[query_set_name]
        logger.info(f"Using {len(queries)} queries from {query_set_name} set")
        
        # Set episodes and runs based on query set type if not specified
        if episodes is None:
            episodes = 100 if query_set_name == "quick_test" else self.exp_config['episodes']
        if runs is None:
            runs = 2 if query_set_name == "quick_test" else self.exp_config['runs_per_strategy']
        
        # Override config with command-line parameters
        self.exp_config['episodes'] = episodes
        self.exp_config['runs_per_strategy'] = runs
        
        logger.info(f"Configuration: {episodes} episodes, {runs} runs per strategy")
        
        # Initialize components
        await self.initialize_components()
        
        # Results storage
        all_results = []
        strategy_summaries = {}
        
        # Run each strategy
        for strategy_config in self.exp_config['strategies']:
            strategy_name = strategy_config['name']
            strategy_results = []
            
            # Multiple runs for statistical validity
            seeds = self.exp_config['random_seeds'][:runs]
            for run_id, seed in enumerate(seeds):
                result = await self.run_strategy_evaluation(
                    strategy_name, queries, run_id, seed, 
                    checkpoint_enabled=checkpoint_enabled
                )
                strategy_results.append(result)
                all_results.append(result)
                
                # Save intermediate results
                self._save_intermediate_results(result)
            
            # Calculate strategy summary
            strategy_summaries[strategy_name] = self._calculate_strategy_summary(
                strategy_results
            )
            
            logger.info(f"Completed {strategy_name}: "
                       f"mean_completion={strategy_summaries[strategy_name]['task_completion_rate']['mean']:.3f}")
        
        # Statistical comparison
        comparison_results = self._perform_statistical_comparison(strategy_summaries)
        
        # Save final results
        final_results = {
            'experiment': 'baseline_comparison',
            'timestamp': self.timestamp,
            'query_set': query_set_name,
            'config': self.exp_config,
            'strategy_summaries': strategy_summaries,
            'statistical_comparison': comparison_results,
            'raw_results': all_results
        }
        
        self._save_final_results(final_results)
        
        # Print summary
        self._print_summary(strategy_summaries, comparison_results)
        
        return final_results
    
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
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'values': values
            }
        
        return summary
    
    def _perform_statistical_comparison(self, summaries: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical tests comparing Q-learning to baselines."""
        from scipy import stats
        
        comparison_results = {}
        
        # Get Q-learning values - check both variants
        qlearning_values = []
        if 'q_learning_tabular' in summaries:
            qlearning_values = summaries.get('q_learning_tabular', {}).get('task_completion_rate', {}).get('values', [])
        elif 'q_learning_dqn' in summaries:
            qlearning_values = summaries.get('q_learning_dqn', {}).get('task_completion_rate', {}).get('values', [])
        elif 'q_learning' in summaries:
            qlearning_values = summaries.get('q_learning', {}).get('task_completion_rate', {}).get('values', [])
        
        # Skip if no Q-learning data
        if not qlearning_values:
            return comparison_results
        
        for baseline_name, baseline_summary in summaries.items():
            if baseline_name in ['q_learning', 'q_learning_tabular', 'q_learning_dqn'] or 'error' in baseline_summary:
                continue
            
            if 'task_completion_rate' not in baseline_summary:
                continue
                
            baseline_values = baseline_summary['task_completion_rate']['values']
            
            # T-test
            t_stat, p_value = stats.ttest_ind(qlearning_values, baseline_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(qlearning_values)**2 + np.std(baseline_values)**2) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(qlearning_values) - np.mean(baseline_values)) / pooled_std
            else:
                cohens_d = 0.0  # Handle zero variance case
            
            # Improvement percentage
            baseline_mean = np.mean(baseline_values)
            if baseline_mean > 0:
                improvement = ((np.mean(qlearning_values) - baseline_mean) / baseline_mean * 100)
            else:
                improvement = 0.0  # Handle zero baseline case
            
            comparison_results[baseline_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'improvement_percent': improvement,
                'significant': p_value < 0.05,
                'large_effect': abs(cohens_d) > 0.8
            }
        
        # Bonferroni correction
        n_comparisons = len(comparison_results)
        corrected_alpha = 0.05 / n_comparisons
        
        for result in comparison_results.values():
            result['significant_corrected'] = result['p_value'] < corrected_alpha
        
        return comparison_results
    
    def _save_intermediate_results(self, result: Dict):
        """Save intermediate results during execution."""
        # Skip saving if there was an error
        if 'error' in result:
            return
            
        filename = f"{result['strategy']}_run{result['run_id']}_{self.timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            result_copy = result.copy()
            if 'time_series' in result_copy:
                for key, value in result_copy['time_series'].items():
                    result_copy['time_series'][key] = [float(v) for v in value]
            
            json.dump(result_copy, f, indent=2)
    
    def _save_final_results(self, results: Dict):
        """Save final aggregated results."""
        filename = f"baseline_comparison_final_{self.timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, bool):
                return bool(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        logger.info(f"Final results saved to {filepath}")
    
    def _print_summary(self, summaries: Dict[str, Dict], comparisons: Dict[str, Dict]):
        """Print readable summary of results."""
        print("\n" + "="*80)
        print("BASELINE COMPARISON RESULTS")
        print("="*80)
        
        # Strategy performance
        print("\nStrategy Performance (Task Completion Rate):")
        print("-"*50)
        for strategy, summary in sorted(summaries.items()):
            if 'error' in summary:
                print(f"{strategy:20s}: ERROR - {summary.get('error', 'Unknown error')}")
                continue
            if 'task_completion_rate' not in summary:
                print(f"{strategy:20s}: No data available")
                continue
            tcr = summary['task_completion_rate']
            print(f"{strategy:20s}: {tcr['mean']:6.1%} ± {tcr['std']:5.1%} "
                  f"[{tcr['ci_lower']:5.1%}, {tcr['ci_upper']:5.1%}]")
        
        # Statistical comparisons
        print("\nStatistical Comparisons (Q-Learning vs Baselines):")
        print("-"*50)
        print(f"{'Baseline':20s} {'Improvement':>12s} {'p-value':>10s} {'Cohen\'s d':>10s} {'Significant':>12s}")
        print("-"*50)
        
        for baseline, comp in sorted(comparisons.items()):
            sig_marker = "***" if comp['significant_corrected'] else ("*" if comp['significant'] else "")
            print(f"{baseline:20s} {comp['improvement_percent']:11.1f}% "
                  f"{comp['p_value']:10.4f} {comp['cohens_d']:10.2f} "
                  f"{sig_marker:>12s}")
        
        print("\n*** = significant after Bonferroni correction")
        print("*   = significant at p < 0.05")
        print("="*80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run baseline comparison experiments")
    parser.add_argument("--query-set", default="dissertation_core",
                       choices=["quick_test", "simple_only", "complex_only", 
                               "full_evaluation", "dissertation_core"],
                       help="Query set to use for evaluation")
    parser.add_argument("--config", type=str, help="Path to experiment config")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of episodes per strategy (default: 100 for quick_test, 1000 for others)")
    parser.add_argument("--runs", type=int, default=None,
                       help="Number of runs per strategy (default: 2 for quick_test, 5 for others)")
    parser.add_argument("--enable-retries", action="store_true", default=False,
                       help="Enable retry mechanism (default: False for clean measurements)")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                       help="Save checkpoint every N episodes (default: None - no checkpoints)")
    parser.add_argument("--checkpoint-dir", type=str, default="results/checkpoints",
                       help="Directory to save checkpoints (default: results/checkpoints)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume from a checkpoint file")
    
    args = parser.parse_args()
    
    # Create runner
    runner = BaselineComparisonRunner(args.config)
    
    # Set retry configuration
    runner.enable_retries = args.enable_retries
    
    if args.output_dir:
        runner.results_dir = Path(args.output_dir)
        runner.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up checkpointing if enabled
    if args.checkpoint_interval:
        runner.checkpoint_manager = CheckpointManager(
            Path(args.checkpoint_dir),
            args.checkpoint_interval
        )
        print(f"Checkpointing enabled: saving every {args.checkpoint_interval} episodes to {args.checkpoint_dir}")
    
    # Load checkpoint if resuming
    if args.resume_from:
        if not runner.checkpoint_manager:
            runner.checkpoint_manager = CheckpointManager(Path(args.checkpoint_dir))
        
        try:
            runner.resume_state = runner.checkpoint_manager.load_checkpoint(args.resume_from)
            print(f"Resuming from checkpoint: {args.resume_from}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Run comparison with optional parameters
    try:
        results = await runner.run_full_comparison(
            args.query_set, 
            episodes=args.episodes,
            runs=args.runs,
            checkpoint_interval=args.checkpoint_interval
        )
        
        print(f"\nExperiment complete! Results saved to {runner.results_dir}")
    finally:
        # Ensure proper cleanup
        if runner.orchestrator:
            try:
                await runner.orchestrator.shutdown()
            except Exception as e:
                logger.warning(f"Error during orchestrator shutdown: {e}")


if __name__ == "__main__":
    asyncio.run(main())