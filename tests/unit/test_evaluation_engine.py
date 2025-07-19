"""Unit tests for evaluation engine."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime

from src.evaluation.evaluation_engine import EvaluationEngine, TestScenario
from src.evaluation.baseline_strategies import RandomSelectionBaseline


class TestTestScenario:
    """Test TestScenario class."""
    
    def test_scenario_creation(self):
        """Test creating a test scenario."""
        mock_intent = MagicMock()
        mock_intent.type = 'file_search'
        mock_intent.embedding = np.random.randn(384)
        
        scenario = TestScenario(
            scenario_id='test_001',
            intent=mock_intent,
            available_tools=['tool1', 'tool2'],
            constraints={'conflicts': {}},
            expected_reward_range=(0.5, 0.9),
            description='Test scenario'
        )
        
        assert scenario.scenario_id == 'test_001'
        assert scenario.intent == mock_intent
        assert scenario.available_tools == ['tool1', 'tool2']
        assert scenario.constraints == {'conflicts': {}}
        assert scenario.expected_reward_range == (0.5, 0.9)
        assert scenario.description == 'Test scenario'
    
    def test_scenario_to_dict(self):
        """Test converting scenario to dictionary."""
        mock_intent = MagicMock()
        scenario = TestScenario(
            scenario_id='test_001',
            intent=mock_intent,
            available_tools=['tool1'],
            constraints={},
            expected_reward_range=(0.5, 0.9)
        )
        
        data = scenario.to_dict()
        
        assert data['scenario_id'] == 'test_001'
        assert data['available_tools'] == ['tool1']
        assert data['constraints'] == {}
        assert data['expected_reward_range'] == (0.5, 0.9)


class TestEvaluationEngine:
    """Test EvaluationEngine class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'evaluation': {
                'baselines': ['random'],
                'evaluation_interval': 10,
                'min_episodes_for_comparison': 5
            },
            'q_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.2
            }
        }
    
    @pytest.fixture
    def engine(self, config):
        """Create evaluation engine."""
        with patch('src.evaluation.evaluation_engine.QLearningEngine'):
            return EvaluationEngine(config)
    
    def test_initialization(self, engine, config):
        """Test engine initialization."""
        assert engine.config == config
        assert 'random' in engine.strategies
        assert 'q_learning' in engine.strategies
        assert len(engine.test_scenarios) == 0
        assert len(engine.evaluation_results) == 0
    
    def test_generate_test_scenarios(self, engine):
        """Test scenario generation."""
        scenarios = engine.generate_test_scenarios(10)
        
        assert len(scenarios) == 10
        assert all(isinstance(s, TestScenario) for s in scenarios)
        assert all(hasattr(s.intent, 'embedding') for s in scenarios)
        assert all(len(s.available_tools) >= 2 for s in scenarios)
    
    def test_create_mock_intent(self, engine):
        """Test mock intent creation."""
        intent = engine._create_mock_intent('file_search')
        
        assert intent.type == 'file_search'
        assert hasattr(intent, 'embedding')
        assert intent.embedding.shape == (384,)
        assert 0.7 <= intent.confidence <= 0.95
    
    def test_generate_constraints(self, engine):
        """Test constraint generation."""
        tools = ['tool1', 'tool2', 'tool3', 'tool4']
        
        # Run multiple times to test randomness
        constraints_list = []
        for _ in range(20):
            constraints = engine._generate_constraints(tools)
            constraints_list.append(constraints)
        
        # Should have basic structure
        assert all('conflicts' in c for c in constraints_list)
        assert all('requires' in c for c in constraints_list)
        
        # Some should have conflicts (probabilistic)
        has_conflicts = any(
            len(c['conflicts']) > 0 for c in constraints_list
        )
        assert has_conflicts  # With 20 runs, very likely to have at least one
    
    @pytest.mark.asyncio
    async def test_evaluate_strategy(self, engine):
        """Test evaluating a single strategy."""
        # Create test scenarios
        engine.test_scenarios = engine.generate_test_scenarios(5)
        
        # Create mock strategy
        mock_strategy = AsyncMock()
        mock_strategy.select_tools.return_value = ['tool1']
        
        # Evaluate strategy
        result = await engine._evaluate_strategy('test_strategy', mock_strategy, 5)
        
        assert 'rewards' in result
        assert 'times' in result
        assert 'selections' in result
        assert 'statistics' in result
        assert len(result['rewards']) == 5
        assert mock_strategy.select_tools.call_count == 5
    
    def test_simulate_reward(self, engine):
        """Test reward simulation."""
        scenario = TestScenario(
            scenario_id='test',
            intent=MagicMock(),
            available_tools=['tool1', 'tool2'],
            constraints={'conflicts': {'tool1': ['tool2']}},
            expected_reward_range=(0.5, 0.9)
        )
        
        # Test normal selection
        reward1 = engine._simulate_reward(['tool1'], scenario)
        assert 0.4 <= reward1 <= 1.0  # With bonus and noise
        
        # Test conflicting selection (should get penalty)
        reward2 = engine._simulate_reward(['tool1', 'tool2'], scenario)
        assert reward2 < reward1  # Should be penalized
    
    def test_calculate_statistics(self, engine):
        """Test statistics calculation."""
        rewards = [0.5, 0.6, 0.7, 0.8, 0.9]
        times = [0.1, 0.2, 0.1, 0.15, 0.25]
        
        stats = engine._calculate_statistics(rewards, times)
        
        assert stats['reward']['mean'] == pytest.approx(0.7, 0.01)
        assert stats['reward']['std'] == pytest.approx(0.158, 0.01)
        assert stats['reward']['min'] == 0.5
        assert stats['reward']['max'] == 0.9
        assert stats['time']['mean'] == pytest.approx(0.16, 0.01)
        assert 'convergence' in stats
    
    def test_calculate_convergence(self, engine):
        """Test convergence calculation."""
        # Non-converged case
        rewards1 = [0.5 + i * 0.1 for i in range(50)]
        conv1 = engine._calculate_convergence(rewards1, window=10)
        assert not conv1['converged']
        
        # Converged case
        rewards2 = [0.8] * 200  # Constant rewards
        conv2 = engine._calculate_convergence(rewards2, window=50)
        assert conv2['converged']
        assert conv2['final_performance'] == 0.8
    
    def test_perform_comparisons(self, engine):
        """Test statistical comparisons."""
        # Set up mock results
        engine.evaluation_results = {
            'random': {
                'rewards': np.random.normal(0.5, 0.1, 100).tolist()
            },
            'strategy1': {
                'rewards': np.random.normal(0.7, 0.1, 100).tolist()
            },
            'strategy2': {
                'rewards': np.random.normal(0.3, 0.1, 100).tolist()
            }
        }
        
        comparisons = engine._perform_comparisons()
        
        assert 'strategy1' in comparisons
        assert 'strategy2' in comparisons
        
        # Strategy1 should show improvement
        assert comparisons['strategy1']['improvement'] > 0
        assert comparisons['strategy1']['win_rate'] > 0.5
        
        # Strategy2 should show negative improvement
        assert comparisons['strategy2']['improvement'] < 0
        assert comparisons['strategy2']['win_rate'] < 0.5
    
    def test_interpret_effect_size(self, engine):
        """Test effect size interpretation."""
        assert engine._interpret_effect_size(0.1) == 'negligible'
        assert engine._interpret_effect_size(0.3) == 'small'
        assert engine._interpret_effect_size(0.6) == 'medium'
        assert engine._interpret_effect_size(1.0) == 'large'
    
    def test_generate_summary(self, engine):
        """Test summary generation."""
        engine.evaluation_results = {
            'strategy1': {
                'statistics': {
                    'reward': {'mean': 0.8},
                    'convergence': {'converged': True}
                }
            },
            'strategy2': {
                'statistics': {
                    'reward': {'mean': 0.6},
                    'convergence': {'converged': False}
                }
            }
        }
        
        engine.comparison_results = {
            'strategy1': {'significant': True, 'improvement': 0.3}
        }
        
        summary = engine._generate_summary()
        
        assert summary['best_strategy'] == 'strategy1'
        assert len(summary['rankings']) == 2
        assert summary['rankings'][0]['strategy'] == 'strategy1'
        assert summary['significant_improvements'] == 1
        assert summary['total_strategies'] == 2
    
    @pytest.mark.asyncio
    async def test_run_evaluation(self, engine):
        """Test full evaluation run."""
        # Mock strategies
        engine.strategies = {
            'random': AsyncMock(),
            'test': AsyncMock()
        }
        
        for strategy in engine.strategies.values():
            strategy.select_tools = AsyncMock(return_value=['tool1'])
        
        # Run evaluation
        results = await engine.run_evaluation(num_episodes=10, parallel=False)
        
        assert 'timestamp' in results
        assert 'num_episodes' in results
        assert 'strategies' in results
        assert 'comparisons' in results
        assert 'summary' in results
        
        # Check strategies were called
        for strategy in engine.strategies.values():
            assert strategy.select_tools.call_count == 10
    
    def test_save_results(self, engine, tmp_path):
        """Test saving results."""
        engine.evaluation_results = {
            'test': {'rewards': [0.5, 0.6, 0.7]}
        }
        engine.comparison_results = {
            'test': {'improvement': 0.2}
        }
        engine.test_scenarios = [
            TestScenario('s1', MagicMock(), ['tool1'], {}, (0.5, 0.9))
        ]
        
        filepath = str(tmp_path / 'test_results.pkl')
        engine.save_results(filepath)
        
        assert os.path.exists(filepath)
        assert os.path.exists(filepath.replace('.pkl', '_summary.json'))
        
        # Check JSON summary
        import json
        with open(filepath.replace('.pkl', '_summary.json'), 'r') as f:
            summary = json.load(f)
        
        assert 'timestamp' in summary
        assert 'summary' in summary
        assert 'comparisons' in summary