"""Unit tests for the enhanced reward calculator."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics


class TestRewardCalculator:
    """Test suite for enhanced reward calculator."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'reward_calculation': {
                'base_weights': {
                    'success': 1.0,
                    'failure': -0.5,
                    'partial_success': 0.3
                },
                'failure_penalties': {
                    'network_timeout': -0.2,
                    'permission_error': -0.8,
                    'rate_limit': -0.3,
                    'connection_error': -0.25,
                    'retryable': -0.1,
                    'non_retryable': -0.7,
                    'unknown': -0.5
                },
                'resource_penalties': {
                    'memory_weight': 0.05,
                    'cpu_weight': 0.05,
                    'api_calls_weight': 0.1,
                    'time_weight': 0.1
                },
                'synergy_bonuses': {
                    'known_good_combo': 0.2,
                    'discovered_combo': 0.3,
                    'complementary_tools': 0.15
                },
                'context_multipliers': {
                    'exploration': 1.2,
                    'production': 0.8,
                    'high_confidence': 1.0,
                    'low_confidence': 1.1,
                    'user_initiated': 1.0,
                    'system_initiated': 0.9
                }
            }
        }
    
    @pytest.fixture
    def calculator(self, config):
        """Create reward calculator instance."""
        return RewardCalculator(config)
    
    def test_successful_execution(self, calculator):
        """Test reward for successful execution."""
        metrics = [
            ExecutionMetrics(
                tool_id='tool1',
                success=True,
                execution_time_ms=100
            )
        ]
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward, breakdown = calculator.calculate_reward(metrics, context)
        
        assert reward > 0
        assert breakdown['base_reward'] == 1.0
        assert 'total' in breakdown
    
    def test_failed_execution(self, calculator):
        """Test reward for failed execution."""
        metrics = [
            ExecutionMetrics(
                tool_id='tool1',
                success=False,
                error_type='network_timeout'
            )
        ]
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward, breakdown = calculator.calculate_reward(metrics, context)
        
        assert reward < 0
        assert breakdown['base_reward'] == -0.5
        assert breakdown['failure_adjustment'] == -0.2  # network_timeout penalty
    
    def test_partial_success(self, calculator):
        """Test reward for partial success."""
        metrics = [
            ExecutionMetrics(
                tool_id='tool1',
                success=False,
                partial_success=True,
                completion_percentage=0.7,
                result_quality=0.8
            )
        ]
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward, breakdown = calculator.calculate_reward(metrics, context)
        
        assert breakdown['partial_success_bonus'] > 0
        # Partial success should mitigate failure penalty
        assert reward > -0.5
    
    def test_resource_efficiency_penalty(self, calculator):
        """Test resource usage penalties."""
        metrics = [
            ExecutionMetrics(
                tool_id='tool1',
                success=True,
                execution_time_ms=5000,  # 5 seconds - should incur time penalty
                resource_usage={
                    'memory_mb': 500,
                    'cpu_percent': 80,
                    'api_calls': 50
                }
            )
        ]
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward, breakdown = calculator.calculate_reward(metrics, context)
        
        assert breakdown['resource_penalty'] < 0
        # Success minus resource penalty should still be positive but reduced
        assert 0 < reward < 1.0
    
    def test_tool_synergy_bonus(self, calculator):
        """Test tool synergy recognition."""
        # Add known synergy
        calculator.known_synergies[frozenset(['tool1', 'tool2'])] = 0.2
        
        metrics = [
            ExecutionMetrics(tool_id='tool1', success=True),
            ExecutionMetrics(tool_id='tool2', success=True)
        ]
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward, breakdown = calculator.calculate_reward(metrics, context)
        
        assert breakdown['synergy_bonus'] > 0
        assert reward > 1.0  # Base success + synergy
    
    def test_failure_type_differentiation(self, calculator):
        """Test different failure types have different penalties."""
        failure_types = ['network_timeout', 'permission_error', 'rate_limit']
        rewards = []
        
        for error_type in failure_types:
            metrics = [
                ExecutionMetrics(
                    tool_id='tool1',
                    success=False,
                    error_type=error_type
                )
            ]
            context = {'mode': 'production', 'intent_confidence': 0.8}
            
            reward, _ = calculator.calculate_reward(metrics, context)
            rewards.append(reward)
        
        # Different error types should have different penalties
        assert len(set(rewards)) == len(failure_types)
        # Permission error should be most severe
        assert rewards[1] < rewards[0]  # permission < network_timeout
        assert rewards[1] < rewards[2]  # permission < rate_limit
    
    def test_context_sensitivity(self, calculator):
        """Test context-based reward adjustment."""
        metrics = [
            ExecutionMetrics(tool_id='tool1', success=True)
        ]
        
        # Test exploration vs production
        exploration_context = {'mode': 'exploration', 'intent_confidence': 0.8}
        production_context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward_exploration, _ = calculator.calculate_reward(metrics, exploration_context)
        reward_production, _ = calculator.calculate_reward(metrics, production_context)
        
        # Exploration should have higher rewards to encourage trying new things
        assert reward_exploration > reward_production
    
    def test_user_satisfaction_adjustment(self, calculator):
        """Test user feedback integration."""
        metrics = [
            ExecutionMetrics(tool_id='tool1', success=True)
        ]
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        # Positive feedback
        positive_feedback = {
            'rating': 5,
            'result_used': True
        }
        reward_positive, breakdown = calculator.calculate_reward(
            metrics, context, positive_feedback
        )
        
        # Negative feedback  
        negative_feedback = {
            'rating': 1,
            'query_reformulated': True,
            'follow_up_time_seconds': 5
        }
        reward_negative, _ = calculator.calculate_reward(
            metrics, context, negative_feedback
        )
        
        assert reward_positive > reward_negative
        assert breakdown['user_satisfaction'] > 0
    
    def test_uncertainty_adjustment(self, calculator):
        """Test uncertainty handling in rewards."""
        # Mixed results - high uncertainty
        metrics = [
            ExecutionMetrics(tool_id='tool1', success=True, result_quality=0.9),
            ExecutionMetrics(tool_id='tool2', success=False),
            ExecutionMetrics(tool_id='tool3', success=True, result_quality=0.3)
        ]
        
        high_confidence_context = {'mode': 'production', 'intent_confidence': 0.9}
        low_confidence_context = {'mode': 'production', 'intent_confidence': 0.4}
        
        reward_high_conf, breakdown_high = calculator.calculate_reward(
            metrics, high_confidence_context
        )
        reward_low_conf, breakdown_low = calculator.calculate_reward(
            metrics, low_confidence_context
        )
        
        # Uncertainty factor should differ
        assert breakdown_high['uncertainty_factor'] != breakdown_low['uncertainty_factor']
    
    def test_retry_learning(self, calculator):
        """Test that retries reduce failure penalties."""
        metrics_no_retry = [
            ExecutionMetrics(
                tool_id='tool1',
                success=False,
                error_type='network_timeout',
                retry_count=0
            )
        ]
        
        metrics_with_retry = [
            ExecutionMetrics(
                tool_id='tool1',
                success=False,
                error_type='network_timeout',
                retry_count=3
            )
        ]
        
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward_no_retry, _ = calculator.calculate_reward(metrics_no_retry, context)
        reward_with_retry, _ = calculator.calculate_reward(metrics_with_retry, context)
        
        # Retry attempts should reduce penalty (system learned to retry)
        assert reward_with_retry > reward_no_retry
    
    def test_empty_results(self, calculator):
        """Test handling of empty execution results."""
        metrics = []
        context = {'mode': 'production', 'intent_confidence': 0.8}
        
        reward, breakdown = calculator.calculate_reward(metrics, context)
        
        assert reward == -0.5  # Default penalty for no results
        assert breakdown['no_results'] == -0.5
    
    def test_update_known_synergies(self, calculator):
        """Test updating tool synergy knowledge."""
        tool_combo = ['tool1', 'tool2', 'tool3']
        
        # Update with good success rate
        calculator.update_known_synergies(tool_combo, success_rate=0.9, occurrences=10)
        
        combo_key = frozenset(tool_combo)
        assert combo_key in calculator.known_synergies
        assert calculator.known_synergies[combo_key] > 0
        
        # Update with poor success rate
        calculator.update_known_synergies(['tool4', 'tool5'], success_rate=0.2, occurrences=8)
        
        poor_combo = frozenset(['tool4', 'tool5'])
        assert poor_combo in calculator.known_synergies
        assert calculator.known_synergies[poor_combo] < 0
    
    def test_reward_clipping(self, calculator):
        """Test that rewards are clipped to reasonable range."""
        # Create scenario with very high theoretical reward
        metrics = [
            ExecutionMetrics(tool_id='tool1', success=True, result_quality=1.0),
            ExecutionMetrics(tool_id='tool2', success=True, result_quality=1.0)
        ]
        
        # Add synergy and positive feedback
        calculator.known_synergies[frozenset(['tool1', 'tool2'])] = 0.5
        
        context = {'mode': 'exploration', 'intent_confidence': 0.95}
        feedback = {'rating': 5, 'result_used': True}
        
        reward, _ = calculator.calculate_reward(metrics, context, feedback)
        
        # Should be clipped to maximum
        assert reward <= 2.0
        
        # Test negative clipping
        failure_metrics = [
            ExecutionMetrics(
                tool_id='tool1',
                success=False,
                error_type='permission_error'
            )
        ]
        
        negative_feedback = {
            'rating': 1,
            'query_reformulated': True,
            'follow_up_time_seconds': 2
        }
        
        negative_reward, _ = calculator.calculate_reward(
            failure_metrics, context, negative_feedback
        )
        
        # Should be clipped to minimum
        assert negative_reward >= -1.0