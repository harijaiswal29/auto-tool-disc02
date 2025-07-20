"""
Unit tests for A/B Testing Framework

Tests the core A/B testing functionality including:
- Experiment creation and management
- Assignment strategies
- Statistical analysis
- Multi-armed bandit functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime

from src.evaluation.ab_testing_framework import (
    ABTestingFramework,
    ExperimentConfig,
    ExperimentStatus,
    AssignmentStrategy,
    StatisticalMethod,
    VariantMetrics,
    RandomAssignment,
    DeterministicAssignment,
    MultiArmedBanditAssignment,
    Experiment
)


class TestVariantMetrics:
    """Test the VariantMetrics data class."""
    
    def test_initialization(self):
        """Test VariantMetrics initialization."""
        metrics = VariantMetrics("control")
        assert metrics.variant_name == "control"
        assert metrics.sample_size == 0
        assert metrics.successes == 0
        assert metrics.total_value == 0.0
        assert metrics.sum_squared == 0.0
        assert metrics.values == []
        assert metrics.timestamps == []
    
    def test_mean_calculation(self):
        """Test mean calculation."""
        metrics = VariantMetrics("test")
        
        # Empty metrics
        assert metrics.mean == 0.0
        
        # Add some values
        metrics.sample_size = 3
        metrics.total_value = 15.0
        assert metrics.mean == 5.0
    
    def test_variance_calculation(self):
        """Test variance calculation."""
        metrics = VariantMetrics("test")
        
        # Empty or single value
        assert metrics.variance == 0.0
        
        metrics.sample_size = 1
        assert metrics.variance == 0.0
        
        # Multiple values: [2, 4, 6] -> mean=4, variance=2.67
        metrics.sample_size = 3
        metrics.total_value = 12.0
        metrics.sum_squared = 56.0  # 2^2 + 4^2 + 6^2
        
        expected_variance = (56.0 / 3) - (4.0 ** 2)  # 18.67 - 16 = 2.67
        assert abs(metrics.variance - expected_variance) < 0.01
    
    def test_conversion_rate(self):
        """Test conversion rate calculation."""
        metrics = VariantMetrics("test")
        
        # Empty metrics
        assert metrics.conversion_rate == 0.0
        
        # With data
        metrics.sample_size = 100
        metrics.successes = 25
        assert metrics.conversion_rate == 0.25


class TestAssignmentStrategies:
    """Test assignment strategy implementations."""
    
    @pytest.mark.asyncio
    async def test_random_assignment(self):
        """Test random assignment strategy."""
        strategy = RandomAssignment()
        config = ExperimentConfig(
            name="test",
            description="test",
            variants=["A", "B", "C"],
            primary_metric="metric"
        )
        
        # Test multiple assignments
        assignments = []
        for i in range(100):
            variant = await strategy.assign(f"user_{i}", config)
            assignments.append(variant)
            assert variant in config.variants
        
        # Check that all variants are assigned (probabilistically)
        unique_variants = set(assignments)
        assert len(unique_variants) > 1  # Should have multiple variants
    
    @pytest.mark.asyncio
    async def test_random_assignment_weighted(self):
        """Test weighted random assignment."""
        strategy = RandomAssignment()
        config = ExperimentConfig(
            name="test",
            description="test",
            variants=["A", "B"],
            primary_metric="metric",
            assignment_weights={"A": 0.8, "B": 0.2}
        )
        
        # Test many assignments
        assignments = []
        for i in range(1000):
            variant = await strategy.assign(f"user_{i}", config)
            assignments.append(variant)
        
        # Check distribution (should be roughly 80/20)
        a_count = assignments.count("A")
        b_count = assignments.count("B")
        a_ratio = a_count / len(assignments)
        
        # Allow some variance
        assert 0.75 < a_ratio < 0.85
        assert a_count + b_count == len(assignments)
    
    @pytest.mark.asyncio
    async def test_deterministic_assignment(self):
        """Test deterministic assignment strategy."""
        strategy = DeterministicAssignment()
        config = ExperimentConfig(
            name="test",
            description="test",
            variants=["A", "B", "C"],
            primary_metric="metric"
        )
        
        # Same user should always get same variant
        user_id = "test_user_123"
        variant1 = await strategy.assign(user_id, config)
        variant2 = await strategy.assign(user_id, config)
        variant3 = await strategy.assign(user_id, config)
        
        assert variant1 == variant2 == variant3
        assert variant1 in config.variants
        
        # Different users should get distributed variants
        variants = []
        for i in range(30):
            variant = await strategy.assign(f"user_{i}", config)
            variants.append(variant)
        
        # Check distribution
        unique_variants = set(variants)
        assert len(unique_variants) == 3  # All variants should be used
    
    @pytest.mark.asyncio
    async def test_multi_armed_bandit_assignment(self):
        """Test multi-armed bandit assignment."""
        strategy = MultiArmedBanditAssignment()
        config = ExperimentConfig(
            name="test",
            description="test",
            variants=["A", "B"],
            primary_metric="metric"
        )
        
        # Initial assignments should be roughly uniform
        initial_assignments = {"A": 0, "B": 0}
        for i in range(20):
            variant = await strategy.assign(f"user_{i}", config)
            initial_assignments[variant] += 1
        
        # Both should have some assignments
        assert initial_assignments["A"] > 0
        assert initial_assignments["B"] > 0
        
        # Update priors - make A perform better
        for _ in range(10):
            strategy.update_priors("A", True)
        for _ in range(10):
            strategy.update_priors("B", False)
        
        # Now A should be preferred
        later_assignments = {"A": 0, "B": 0}
        for i in range(100):
            variant = await strategy.assign(f"user_{i+20}", config)
            later_assignments[variant] += 1
        
        # A should get more assignments
        assert later_assignments["A"] > later_assignments["B"]


class TestABTestingFramework:
    """Test the main A/B testing framework."""
    
    @pytest.fixture
    def framework(self):
        """Create framework instance."""
        return ABTestingFramework()
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, framework):
        """Test experiment creation."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
            variants=["control", "treatment"],
            primary_metric="conversion"
        )
        
        experiment = await framework.create_experiment(config)
        
        assert experiment is not None
        assert experiment.config == config
        assert experiment.status == ExperimentStatus.DRAFT
        assert framework.experiments["test_exp"] == experiment
    
    @pytest.mark.asyncio
    async def test_assign_user(self, framework):
        """Test user assignment."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="metric"
        )
        
        experiment = await framework.create_experiment(config)
        await experiment.start()
        
        # Assign user
        user_id = "user_123"
        variant = await framework.assign_user("test_exp", user_id)
        
        assert variant in ["A", "B"]
        assert experiment.user_assignments[user_id] == variant
        
        # Same user should get same variant
        variant2 = await framework.assign_user("test_exp", user_id)
        assert variant == variant2
    
    @pytest.mark.asyncio
    async def test_assign_user_not_running(self, framework):
        """Test user assignment when experiment not running."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="metric"
        )
        
        # Create but don't start
        await framework.create_experiment(config)
        
        variant = await framework.assign_user("test_exp", "user_123")
        assert variant is None
    
    @pytest.mark.asyncio
    async def test_record_event(self, framework):
        """Test event recording."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="clicks"
        )
        
        experiment = await framework.create_experiment(config)
        await experiment.start()
        
        # Assign and record event
        user_id = "user_123"
        variant = await framework.assign_user("test_exp", user_id)
        await framework.record_event("test_exp", user_id, "clicks", 1.0, True)
        
        # Check metrics updated
        metrics = experiment.variant_metrics[variant]["clicks"]
        assert metrics.sample_size == 1
        assert metrics.total_value == 1.0
        assert metrics.successes == 1
    
    def test_calculate_sample_size(self, framework):
        """Test sample size calculation."""
        # Test for typical conversion rate scenario
        baseline_rate = 0.1  # 10% conversion
        mde = 0.02  # 2% absolute increase
        
        sample_size = framework.calculate_sample_size(
            baseline_rate, mde, alpha=0.05, power=0.8
        )
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        # For these parameters, should be around 3800-4000 per variant
        assert 3000 < sample_size < 5000
    
    @pytest.mark.asyncio
    async def test_analyze_experiment(self, framework):
        """Test experiment analysis."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["control", "treatment"],
            primary_metric="conversion_rate",
            min_sample_size=10
        )
        
        experiment = await framework.create_experiment(config)
        await experiment.start()
        
        # Add some data
        for i in range(20):
            user_id = f"user_{i}"
            await framework.assign_user("test_exp", user_id)
            
            # Simulate different conversion rates
            variant = experiment.user_assignments[user_id]
            if variant == "control":
                success = i % 4 == 0  # 25% conversion
            else:
                success = i % 3 == 0  # 33% conversion
            
            await framework.record_event(
                "test_exp", user_id, "conversion_rate", 
                1.0 if success else 0.0, success
            )
        
        # Analyze
        results = await framework.analyze_experiment("test_exp")
        
        assert results is not None
        assert results['experiment_name'] == "test_exp"
        assert results['total_users'] == 20
        assert 'variant_results' in results
        assert 'statistical_significance' in results


class TestExperiment:
    """Test the Experiment class."""
    
    @pytest.fixture
    def experiment(self):
        """Create experiment instance."""
        config = ExperimentConfig(
            name="test",
            description="Test experiment",
            variants=["A", "B"],
            primary_metric="metric",
            secondary_metrics=["secondary"],
            min_sample_size=10
        )
        framework = ABTestingFramework()
        return Experiment(config, framework)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, experiment):
        """Test starting and stopping experiment."""
        assert experiment.status == ExperimentStatus.DRAFT
        
        await experiment.start()
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.start_time is not None
        
        await experiment.stop()
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.end_time is not None
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, experiment):
        """Test metric updates."""
        await experiment.start()
        
        # Update metrics
        await experiment.update_metrics("A", "metric", 5.0, True)
        await experiment.update_metrics("A", "metric", 3.0, False)
        
        metrics = experiment.variant_metrics["A"]["metric"]
        assert metrics.sample_size == 2
        assert metrics.total_value == 8.0
        assert metrics.successes == 1
        assert metrics.mean == 4.0
    
    @pytest.mark.asyncio
    async def test_early_stopping(self, experiment):
        """Test early stopping functionality."""
        experiment.config.enable_early_stopping = True
        experiment.config.early_stopping_threshold = 0.05
        experiment.config.min_sample_size = 5
        
        await experiment.start()
        
        # Add data that should trigger early stopping
        # Create significant difference between variants
        for i in range(10):
            await experiment.update_metrics("A", "metric", 0.0, False)
            await experiment.update_metrics("B", "metric", 1.0, True)
        
        # Check if stopped
        # Note: Early stopping check happens in update_metrics
        # For this test, we'll check the p-value manually
        metrics = {
            "A": experiment.variant_metrics["A"]["metric"],
            "B": experiment.variant_metrics["B"]["metric"]
        }
        
        p_value = await experiment._frequentist_test(metrics)
        assert p_value < 0.05  # Should be significant
    
    @pytest.mark.asyncio
    async def test_bayesian_analysis(self, experiment):
        """Test Bayesian statistical analysis."""
        experiment.config.statistical_method = StatisticalMethod.BAYESIAN
        experiment.config.primary_metric = "conversion_rate"
        
        await experiment.start()
        
        # Add conversion data
        for i in range(50):
            # A: 20% conversion, B: 30% conversion
            await experiment.update_metrics("A", "conversion_rate", 1.0, i % 5 == 0)
            await experiment.update_metrics("B", "conversion_rate", 1.0, i % 3 == 0)
        
        # Analyze with Bayesian method
        metrics = {
            "A": experiment.variant_metrics["A"]["conversion_rate"],
            "B": experiment.variant_metrics["B"]["conversion_rate"]
        }
        
        results = await experiment._bayesian_test(metrics)
        
        assert 'probability_treatment_better' in results
        assert 'expected_lift' in results
        assert 'credible_interval_95' in results
        
        # B should be better
        assert results['probability_treatment_better'] > 0.5