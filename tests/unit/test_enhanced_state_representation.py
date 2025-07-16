"""Unit tests for enhanced state representation with failure dimensions."""

import pytest
import numpy as np
from unittest.mock import Mock
import json

from src.learning.q_learning_engine import StateRepresentation


class TestEnhancedStateRepresentation:
    """Test suite for enhanced state representation."""
    
    @pytest.fixture
    def state_encoder(self):
        """Create state encoder instance."""
        return StateRepresentation()
    
    @pytest.fixture
    def mock_intent(self):
        """Create mock intent with embedding."""
        intent = Mock()
        intent.embedding = np.random.rand(384).tolist()
        return intent
    
    @pytest.fixture
    def base_context(self):
        """Create base context without failure metrics."""
        return {
            'domain': 'engineering',
            'query_count': 5,
            'session_duration': 1800,  # 30 minutes
            'total_queries': 50,
            'success_rate': 0.8,
            'metrics': {
                'avg_response_time': 2000,
                'success_rate': 0.8,
                'error_rate': 0.2,
                'tools_invoked': 3,
                'cache_hit_rate': 0.6
            }
        }
    
    @pytest.fixture
    def failure_context(self, base_context):
        """Create context with failure metrics."""
        context = base_context.copy()
        context.update({
            'failure_rates': {
                'filesystem_mcp': 0.1,
                'sqlite_mcp': 0.05,
                'search_mcp': 0.2,
                'postgres_mcp': 0.0,
                'github_mcp': 0.15
            },
            'failure_types': {
                'network_timeout': 5,
                'permission_error': 2,
                'rate_limit': 3,
                'connection_error': 1,
                'other': 2
            },
            'retry_patterns': {
                'avg_retry_count': 1.5,
                'retry_success_rate': 0.7,
                'avg_retry_delay_ms': 2500,
                'circuit_breaker_triggers': 2,
                'max_consecutive_failures': 3
            }
        })
        return context
    
    def test_state_dimensions(self, state_encoder):
        """Test that state dimensions are correctly defined."""
        expected_dimensions = {
            'intent_vector': 384,
            'context_features': 10,
            'tool_history': 20,
            'performance_metrics': 5,
            'failure_rates': 10,
            'failure_types': 5,
            'retry_patterns': 5
        }
        
        assert state_encoder.state_dimensions == expected_dimensions
        assert state_encoder.total_dimensions == 439  # Sum of all dimensions
    
    def test_encode_state_without_failures(self, state_encoder, mock_intent, base_context):
        """Test encoding state without failure metrics."""
        history = ['tool1', 'tool2', 'tool3']
        
        state_vector = state_encoder.encode_state(mock_intent, base_context, history)
        
        assert isinstance(state_vector, np.ndarray)
        assert len(state_vector) == state_encoder.total_dimensions
        # Failure dimensions should be zeros
        failure_start = 384 + 10 + 20 + 5  # Start of failure dimensions
        failure_end = failure_start + 10 + 5 + 5
        assert np.all(state_vector[failure_start:failure_end] == 0)
    
    def test_encode_state_with_failures(self, state_encoder, mock_intent, failure_context):
        """Test encoding state with failure metrics."""
        history = ['filesystem_mcp', 'sqlite_mcp', 'search_mcp']
        
        state_vector = state_encoder.encode_state(mock_intent, failure_context, history)
        
        assert isinstance(state_vector, np.ndarray)
        assert len(state_vector) == state_encoder.total_dimensions
        
        # Failure dimensions should have non-zero values
        failure_start = 384 + 10 + 20 + 5
        failure_rates_end = failure_start + 10
        failure_rates = state_vector[failure_start:failure_rates_end]
        
        # Check specific tool failure rates are encoded
        assert failure_rates[0] == 0.1  # filesystem_mcp
        assert failure_rates[1] == 0.05  # sqlite_mcp
        assert failure_rates[2] == 0.2  # search_mcp
    
    def test_encode_failure_rates(self, state_encoder):
        """Test failure rate encoding."""
        failure_rates = {
            'filesystem_mcp': 0.3,
            'unknown_tool': 0.8,
            'search_mcp': 0.1
        }
        
        features = state_encoder._encode_failure_rates(failure_rates)
        
        assert len(features) == 10
        assert features[0] == 0.3  # filesystem_mcp
        assert features[2] == 0.1  # search_mcp
        assert features[6] > 0  # Average failure rate
        assert features[7] == 0.8  # Max failure rate
        assert features[9] == 1  # One tool with >0.5 failure rate
    
    def test_encode_failure_types(self, state_encoder):
        """Test failure type distribution encoding."""
        failure_types = {
            'network_timeout': 10,
            'permission_error': 5,
            'rate_limit': 8,
            'other': 2
        }
        
        features = state_encoder._encode_failure_types(failure_types)
        
        assert len(features) == 5
        total = sum(failure_types.values())
        assert features[0] == pytest.approx(10/total)  # network_timeout
        assert features[1] == pytest.approx(5/total)   # permission_error
        assert features[2] == pytest.approx(8/total)   # rate_limit
        assert features[4] == pytest.approx(2/total)   # other
    
    def test_encode_retry_patterns(self, state_encoder):
        """Test retry pattern encoding."""
        retry_patterns = {
            'avg_retry_count': 2.5,
            'retry_success_rate': 0.8,
            'avg_retry_delay_ms': 5000,
            'circuit_breaker_triggers': 3,
            'max_consecutive_failures': 4
        }
        
        features = state_encoder._encode_retry_patterns(retry_patterns)
        
        assert len(features) == 5
        assert features[0] == 2.5 / 5.0  # Normalized retry count
        assert features[1] == 0.8  # Success rate
        assert features[2] == 0.5  # Normalized delay (5000/10000)
        assert features[3] == 0.3  # Circuit breaker triggers/10
        assert features[4] == 0.8  # Max consecutive failures/5
    
    def test_state_hash_consistency(self, state_encoder, mock_intent, failure_context):
        """Test that same state produces same hash."""
        history = ['tool1', 'tool2']
        
        state1 = state_encoder.encode_state(mock_intent, failure_context, history)
        state2 = state_encoder.encode_state(mock_intent, failure_context, history)
        
        hash1 = state_encoder.encode_to_hash(state1)
        hash2 = state_encoder.encode_to_hash(state2)
        
        assert hash1 == hash2
    
    def test_empty_failure_metrics(self, state_encoder):
        """Test handling of empty failure metrics."""
        empty_rates = {}
        empty_types = {}
        empty_patterns = {}
        
        rate_features = state_encoder._encode_failure_rates(empty_rates)
        type_features = state_encoder._encode_failure_types(empty_types)
        pattern_features = state_encoder._encode_retry_patterns(empty_patterns)
        
        assert np.all(rate_features == 0)
        assert np.all(type_features == 0)
        # Retry success rate should default to 0.5
        assert pattern_features[1] == 0.5
    
    def test_context_encoding_with_time(self, state_encoder):
        """Test that time features are properly encoded."""
        context = {'domain': 'general'}
        
        features = state_encoder._encode_context(context)
        
        # Check cyclic time encoding
        assert -1 <= features[8] <= 1  # sin component
        assert -1 <= features[9] <= 1  # cos component
        # sin^2 + cos^2 should equal 1 (approximately)
        assert pytest.approx(features[8]**2 + features[9]**2, rel=1e-3) == 1.0
    
    def test_tool_history_encoding(self, state_encoder):
        """Test tool history encoding with diversity metrics."""
        # Repetitive history
        repetitive_history = ['tool1'] * 10
        features_rep = state_encoder._encode_history(repetitive_history)
        
        # Diverse history
        diverse_history = [f'tool{i}' for i in range(10)]
        features_div = state_encoder._encode_history(diverse_history)
        
        # Diversity should be different
        assert features_rep[10] < features_div[10]  # Diversity metric
        # Consecutive usage should be high for repetitive
        assert features_rep[12] > features_div[12]
    
    def test_performance_metrics_encoding(self, state_encoder):
        """Test performance metrics normalization."""
        metrics = {
            'avg_response_time': 10000,  # 10 seconds - very slow
            'success_rate': 0.95,
            'error_rate': 0.05,
            'tools_invoked': 10,  # Many tools
            'cache_hit_rate': 0.8
        }
        
        features = state_encoder._encode_metrics(metrics)
        
        assert features[0] > 1.0  # Response time normalized but can exceed 1
        assert features[1] == 0.95  # Success rate
        assert features[2] == 0.05  # Error rate
        assert features[3] == 1.0  # Tools capped at 1.0 (10/5 = 2, min(2,1) = 1)
        assert features[4] == 0.8  # Cache hit rate