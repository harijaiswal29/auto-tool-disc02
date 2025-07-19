"""Evaluation engine for automated baseline comparisons.

This module orchestrates the evaluation process, running multiple strategies
in parallel and collecting comparative metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
import json
import pickle
from pathlib import Path
import logging
from collections import defaultdict
import random
from scipy import stats

from learning.q_learning_engine import QLearningEngine, StateRepresentation
from learning.dqn_agent import DQNAgent
from evaluation.baseline_strategies import (
    BaselineStrategy, RandomSelectionBaseline, MostPopularToolsBaseline,
    FixedPolicyBaseline, GreedySingleToolBaseline, ContextAgnosticQLearningBaseline
)
from evaluation.metrics_collector import MetricsCollector
from models.intent import Intent
from utils.logger import get_logger

logger = get_logger(__name__)


class TestScenario:
    """Represents a test scenario for evaluation."""
    
    def __init__(self, scenario_id: str, intent: Intent, available_tools: List[str],
                 constraints: Dict[str, Any], expected_reward_range: Tuple[float, float],
                 description: str = ""):
        self.scenario_id = scenario_id
        self.intent = intent
        self.available_tools = available_tools
        self.constraints = constraints
        self.expected_reward_range = expected_reward_range
        self.description = description
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'available_tools': self.available_tools,
            'constraints': self.constraints,
            'expected_reward_range': self.expected_reward_range,
            'description': self.description
        }


class EvaluationEngine:
    """Main evaluation engine for comparing strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_representation = StateRepresentation()
        self.metrics_collector = MetricsCollector(config)
        self.strategies = {}
        self.test_scenarios = []
        self.evaluation_results = defaultdict(list)
        self.comparison_results = {}
        
        # Initialize strategies
        self._initialize_strategies()
        
    def _initialize_strategies(self):
        """Initialize all evaluation strategies."""
        eval_config = self.config.get('evaluation', {})
        baselines = eval_config.get('baselines', [
            'random', 'popular', 'fixed_policy', 'greedy', 'context_agnostic'
        ])
        
        # Create baseline strategies
        if 'random' in baselines:
            self.strategies['random'] = RandomSelectionBaseline(self.config)
        if 'popular' in baselines:
            self.strategies['popular'] = MostPopularToolsBaseline(self.config)
        if 'fixed_policy' in baselines:
            self.strategies['fixed_policy'] = FixedPolicyBaseline(self.config)
        if 'greedy' in baselines:
            self.strategies['greedy'] = GreedySingleToolBaseline(self.config)
        if 'context_agnostic' in baselines:
            self.strategies['context_agnostic'] = ContextAgnosticQLearningBaseline(self.config)
            
        # Add main Q-learning strategy
        self.strategies['q_learning'] = QLearningEngine(self.config)
        
        # Add DQN if enabled
        if self.config.get('dqn', {}).get('enabled', False):
            self.strategies['dqn'] = DQNAgent(self.config)
            
        logger.info(f"Initialized {len(self.strategies)} strategies for evaluation")
        
    def generate_test_scenarios(self, num_scenarios: int = 100) -> List[TestScenario]:
        """Generate diverse test scenarios for evaluation."""
        scenarios = []
        
        # Define scenario templates
        scenario_templates = [
            {
                'name': 'file_search',
                'tools': ['filesystem_mcp', 'search_mcp', 'github_mcp'],
                'reward_range': (0.5, 1.0),
                'description': 'File search and discovery tasks'
            },
            {
                'name': 'data_query',
                'tools': ['sqlite_mcp', 'postgres_mcp', 'search_mcp'],
                'reward_range': (0.4, 0.9),
                'description': 'Database query and data retrieval'
            },
            {
                'name': 'multi_tool',
                'tools': ['filesystem_mcp', 'sqlite_mcp', 'search_mcp', 'github_mcp'],
                'reward_range': (0.3, 0.8),
                'description': 'Complex tasks requiring multiple tools'
            },
            {
                'name': 'api_integration',
                'tools': ['weather_mcp', 'search_mcp', 'github_mcp'],
                'reward_range': (0.2, 0.7),
                'description': 'External API integration tasks'
            }
        ]
        
        for i in range(num_scenarios):
            template = random.choice(scenario_templates)
            
            # Create mock intent
            intent = self._create_mock_intent(template['name'])
            
            # Random subset of available tools
            num_tools = random.randint(2, len(template['tools']))
            available_tools = random.sample(template['tools'], num_tools)
            
            # Generate constraints
            constraints = self._generate_constraints(available_tools)
            
            scenario = TestScenario(
                scenario_id=f"{template['name']}_{i}",
                intent=intent,
                available_tools=available_tools,
                constraints=constraints,
                expected_reward_range=template['reward_range'],
                description=template['description']
            )
            scenarios.append(scenario)
            
        self.test_scenarios = scenarios
        return scenarios
    
    def _create_mock_intent(self, intent_type: str) -> Intent:
        """Create a mock intent for testing."""
        # Create a simple mock intent with random embedding
        class MockIntent:
            def __init__(self, intent_type):
                self.type = intent_type
                self.embedding = np.random.randn(384)  # Random embedding
                self.confidence = random.uniform(0.7, 0.95)
                
        return MockIntent(intent_type)
    
    def _generate_constraints(self, tools: List[str]) -> Dict[str, Any]:
        """Generate random constraints for tools."""
        constraints = {'conflicts': {}, 'requires': {}}
        
        # Add some random conflicts (10% chance)
        for tool in tools:
            if random.random() < 0.1:
                other_tools = [t for t in tools if t != tool]
                if other_tools:
                    constraints['conflicts'][tool] = [random.choice(other_tools)]
                    
        return constraints
    
    async def run_evaluation(self, num_episodes: int = 1000, 
                           parallel: bool = True) -> Dict[str, Any]:
        """Run evaluation comparing all strategies.
        
        Args:
            num_episodes: Number of episodes to run
            parallel: Whether to run strategies in parallel
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation with {num_episodes} episodes")
        
        # Generate test scenarios if not already done
        if not self.test_scenarios:
            self.generate_test_scenarios(num_episodes)
            
        # Run evaluation for each strategy
        if parallel:
            tasks = []
            for strategy_name, strategy in self.strategies.items():
                task = self._evaluate_strategy(strategy_name, strategy, num_episodes)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for strategy_name, strategy in self.strategies.items():
                result = await self._evaluate_strategy(strategy_name, strategy, num_episodes)
                results.append(result)
                
        # Perform statistical comparisons
        self.comparison_results = self._perform_comparisons()
        
        # Compile final results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'strategies': {name: self.evaluation_results[name] for name in self.strategies},
            'comparisons': self.comparison_results,
            'summary': self._generate_summary()
        }
        
        return evaluation_results
    
    async def _evaluate_strategy(self, strategy_name: str, strategy: Any, 
                               num_episodes: int) -> Dict[str, Any]:
        """Evaluate a single strategy."""
        logger.info(f"Evaluating strategy: {strategy_name}")
        
        episode_rewards = []
        episode_times = []
        tool_selections = []
        
        for episode in range(num_episodes):
            # Get scenario for this episode
            scenario = self.test_scenarios[episode % len(self.test_scenarios)]
            
            # Create state from scenario
            context = {
                'domain': 'evaluation',
                'session_id': f'eval_{episode}',
                'metrics': {},
                'failure_rates': {},
                'failure_types': {},
                'retry_patterns': {}
            }
            state = self.state_representation.encode_state(
                scenario.intent, context, []
            )
            
            # Time the selection
            start_time = asyncio.get_event_loop().time()
            
            # Select tools
            if hasattr(strategy, 'select_action'):
                # Q-learning/DQN interface
                selected_tools = await strategy.select_action(
                    state, scenario.available_tools, scenario.constraints
                )
            else:
                # Baseline interface
                selected_tools = await strategy.select_tools(
                    state, scenario.available_tools, scenario.constraints
                )
                
            selection_time = asyncio.get_event_loop().time() - start_time
            
            # Simulate reward (in real evaluation, would execute tools)
            reward = self._simulate_reward(selected_tools, scenario)
            
            # Update strategy if it learns
            if hasattr(strategy, 'update'):
                next_state = state  # Simplified - in reality would be different
                strategy.update(state, selected_tools, reward, next_state)
            
            # Collect metrics
            episode_rewards.append(reward)
            episode_times.append(selection_time)
            tool_selections.append(selected_tools)
            
            # Update metrics collector
            self.metrics_collector.record_episode(
                strategy_name, episode, reward, selected_tools, selection_time
            )
            
        # Store results
        self.evaluation_results[strategy_name] = {
            'rewards': episode_rewards,
            'times': episode_times,
            'selections': tool_selections,
            'statistics': self._calculate_statistics(episode_rewards, episode_times)
        }
        
        return self.evaluation_results[strategy_name]
    
    def _simulate_reward(self, selected_tools: List[str], scenario: TestScenario) -> float:
        """Simulate reward for tool selection.
        
        In a real evaluation, this would execute the tools and calculate actual reward.
        For now, we simulate based on heuristics.
        """
        base_reward = random.uniform(*scenario.expected_reward_range)
        
        # Bonus for selecting multiple complementary tools
        if len(selected_tools) > 1:
            base_reward += 0.1
            
        # Penalty for constraint violations
        conflicts = scenario.constraints.get('conflicts', {})
        for tool in selected_tools:
            for other_tool in selected_tools:
                if tool != other_tool and other_tool in conflicts.get(tool, []):
                    base_reward -= 0.3
                    
        # Add some noise
        noise = random.gauss(0, 0.05)
        
        return np.clip(base_reward + noise, -1.0, 1.0)
    
    def _calculate_statistics(self, rewards: List[float], times: List[float]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        return {
            'reward': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards),
                'percentiles': {
                    '25': np.percentile(rewards, 25),
                    '75': np.percentile(rewards, 75),
                    '95': np.percentile(rewards, 95)
                }
            },
            'time': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times)
            },
            'convergence': self._calculate_convergence(rewards)
        }
    
    def _calculate_convergence(self, rewards: List[float], window: int = 100) -> Dict[str, Any]:
        """Calculate convergence metrics."""
        if len(rewards) < window:
            return {'converged': False, 'convergence_episode': None}
            
        # Calculate moving average
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # Check for convergence (std < threshold in last window)
        last_window_std = np.std(rewards[-window:])
        converged = last_window_std < 0.1
        
        # Find convergence point
        convergence_episode = None
        if converged:
            for i in range(window, len(moving_avg)):
                window_std = np.std(rewards[i-window:i])
                if window_std < 0.1:
                    convergence_episode = i
                    break
                    
        return {
            'converged': converged,
            'convergence_episode': convergence_episode,
            'final_performance': np.mean(rewards[-window:])
        }
    
    def _perform_comparisons(self) -> Dict[str, Any]:
        """Perform statistical comparisons between strategies."""
        comparisons = {}
        
        # Get baseline for comparison (random selection)
        baseline_name = 'random'
        if baseline_name not in self.evaluation_results:
            logger.warning("Random baseline not found for comparison")
            return comparisons
            
        baseline_rewards = self.evaluation_results[baseline_name]['rewards']
        
        for strategy_name, results in self.evaluation_results.items():
            if strategy_name == baseline_name:
                continue
                
            strategy_rewards = results['rewards']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(strategy_rewards, baseline_rewards)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.std(strategy_rewards)**2 + np.std(baseline_rewards)**2) / 2
            )
            cohens_d = (np.mean(strategy_rewards) - np.mean(baseline_rewards)) / pooled_std
            
            # Win rate
            wins = sum(1 for s, b in zip(strategy_rewards, baseline_rewards) if s > b)
            win_rate = wins / len(strategy_rewards)
            
            comparisons[strategy_name] = {
                'vs_baseline': baseline_name,
                'improvement': np.mean(strategy_rewards) - np.mean(baseline_rewards),
                'improvement_percent': (
                    (np.mean(strategy_rewards) - np.mean(baseline_rewards)) / 
                    np.mean(baseline_rewards) * 100
                ),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'win_rate': win_rate,
                'effect_size': self._interpret_effect_size(cohens_d)
            }
            
        return comparisons
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(cohens_d) < 0.2:
            return 'negligible'
        elif abs(cohens_d) < 0.5:
            return 'small'
        elif abs(cohens_d) < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        # Rank strategies by mean reward
        rankings = []
        for strategy_name, results in self.evaluation_results.items():
            rankings.append({
                'strategy': strategy_name,
                'mean_reward': results['statistics']['reward']['mean'],
                'convergence': results['statistics']['convergence']['converged']
            })
            
        rankings.sort(key=lambda x: x['mean_reward'], reverse=True)
        
        # Find best strategy
        best_strategy = rankings[0]['strategy']
        
        # Count significant improvements
        significant_improvements = sum(
            1 for comp in self.comparison_results.values()
            if comp['significant'] and comp['improvement'] > 0
        )
        
        return {
            'best_strategy': best_strategy,
            'rankings': rankings,
            'significant_improvements': significant_improvements,
            'total_strategies': len(self.strategies)
        }
    
    async def run_online_evaluation(self, callback: Callable[[str, float], None]):
        """Run online evaluation during normal operation.
        
        Args:
            callback: Function to call with (strategy_name, reward) after each episode
        """
        # This would integrate with the actual system to evaluate strategies
        # in real-time during normal operation
        pass
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        results = {
            'evaluation_results': self.evaluation_results,
            'comparison_results': self.comparison_results,
            'test_scenarios': [s.to_dict() for s in self.test_scenarios],
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
        # Also save JSON summary
        json_path = filepath.replace('.pkl', '_summary.json')
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(),
            'comparisons': self.comparison_results
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Saved evaluation results to {filepath}")