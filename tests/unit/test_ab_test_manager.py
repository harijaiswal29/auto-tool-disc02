"""
Unit tests for A/B Test Manager

Tests the experiment lifecycle management including:
- Database persistence
- Experiment monitoring
- Result analysis
- API integration
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.evaluation.ab_test_manager import ABTestManager
from src.evaluation.ab_testing_framework import (
    ExperimentConfig, ExperimentStatus, AssignmentStrategy
)


class TestABTestManager:
    """Test the A/B test manager functionality."""
    
    @pytest.fixture
    async def manager(self):
        """Create manager instance with temp database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'database': {'ab_test_db': os.path.join(temp_dir, 'test_ab.db')},
                'ab_testing': {
                    'enable_monitoring': False,  # Disable for tests
                    'monitoring_interval': 1
                }
            }
            manager = ABTestManager(config)
            await manager.initialize()
            yield manager
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.framework is not None
        assert manager.db_manager is not None
        assert manager.active_experiments == set()
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, manager):
        """Test experiment creation."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
            variants=["control", "treatment"],
            primary_metric="conversion"
        )
        
        experiment_id = await manager.create_experiment(config)
        
        assert experiment_id == "test_exp"
        
        # Check database
        async with manager.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name, status FROM ab_experiments WHERE name = ?",
                ("test_exp",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "test_exp"
            assert row[1] == "draft"
    
    @pytest.mark.asyncio
    async def test_start_stop_experiment(self, manager):
        """Test starting and stopping experiments."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="metric"
        )
        
        # Create and start
        await manager.create_experiment(config)
        await manager.start_experiment("test_exp")
        
        assert "test_exp" in manager.active_experiments
        
        # Check database status
        async with manager.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT status, start_time FROM ab_experiments WHERE name = ?",
                ("test_exp",)
            )
            row = await cursor.fetchone()
            assert row[0] == "running"
            assert row[1] is not None
        
        # Stop experiment
        results = await manager.stop_experiment("test_exp")
        
        assert "test_exp" not in manager.active_experiments
        assert results is not None
        assert results['experiment_name'] == "test_exp"
        assert results['status'] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_variant_for_user(self, manager):
        """Test user variant assignment."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="metric"
        )
        
        await manager.create_experiment(config)
        await manager.start_experiment("test_exp")
        
        # Get variant for new user
        variant = await manager.get_variant_for_user("test_exp", "user_123")
        assert variant in ["A", "B"]
        
        # Check assignment persisted
        async with manager.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT variant FROM ab_assignments WHERE experiment_name = ? AND user_id = ?",
                ("test_exp", "user_123")
            )
            row = await cursor.fetchone()
            assert row[0] == variant
        
        # Same user should get same variant
        variant2 = await manager.get_variant_for_user("test_exp", "user_123")
        assert variant == variant2
    
    @pytest.mark.asyncio
    async def test_record_metric(self, manager):
        """Test metric recording."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="clicks"
        )
        
        await manager.create_experiment(config)
        await manager.start_experiment("test_exp")
        
        # Assign user and record metric
        user_id = "user_123"
        variant = await manager.get_variant_for_user("test_exp", user_id)
        
        await manager.record_metric(
            "test_exp", user_id, "clicks", 5.0, True,
            metadata={"page": "home"}
        )
        
        # Check database
        async with manager.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                """SELECT variant, metric_name, value, success, metadata 
                   FROM ab_metrics 
                   WHERE experiment_name = ? AND user_id = ?""",
                ("test_exp", user_id)
            )
            row = await cursor.fetchone()
            assert row[0] == variant
            assert row[1] == "clicks"
            assert row[2] == 5.0
            assert row[3] == 1  # True stored as 1
            assert json.loads(row[4])["page"] == "home"
    
    @pytest.mark.asyncio
    async def test_experiment_status(self, manager):
        """Test getting experiment status."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="metric"
        )
        
        await manager.create_experiment(config)
        await manager.start_experiment("test_exp")
        
        # Add some activity
        for i in range(10):
            user_id = f"user_{i}"
            await manager.get_variant_for_user("test_exp", user_id)
            await manager.record_metric("test_exp", user_id, "metric", i, True)
        
        # Get status
        status = await manager.get_experiment_status("test_exp")
        
        assert status['name'] == "test_exp"
        assert status['status'] == "running"
        assert status['total_users'] == 10
        assert len(status['variant_metrics']) > 0
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, manager):
        """Test listing experiments."""
        # Create multiple experiments
        for i in range(3):
            config = ExperimentConfig(
                name=f"exp_{i}",
                description=f"Test {i}",
                variants=["A", "B"],
                primary_metric="metric"
            )
            await manager.create_experiment(config)
            if i < 2:
                await manager.start_experiment(f"exp_{i}")
        
        # List all
        all_exps = await manager.list_experiments()
        assert len(all_exps) == 3
        
        # List by status
        running_exps = await manager.list_experiments(status="running")
        assert len(running_exps) == 2
        
        draft_exps = await manager.list_experiments(status="draft")
        assert len(draft_exps) == 1
    
    @pytest.mark.asyncio
    async def test_experiment_results(self, manager):
        """Test getting experiment results."""
        config = ExperimentConfig(
            name="test_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="conversion_rate",
            min_sample_size=5
        )
        
        await manager.create_experiment(config)
        await manager.start_experiment("test_exp")
        
        # Add data and stop
        for i in range(10):
            user_id = f"user_{i}"
            await manager.get_variant_for_user("test_exp", user_id)
            success = i % 3 == 0
            await manager.record_metric(
                "test_exp", user_id, "conversion_rate", 
                1.0 if success else 0.0, success
            )
        
        results = await manager.stop_experiment("test_exp")
        
        # Get results from database
        stored_results = await manager.get_experiment_results("test_exp")
        
        assert stored_results is not None
        assert stored_results['experiment_name'] == "test_exp"
        assert 'winner' in stored_results
        assert 'confidence' in stored_results
    
    @pytest.mark.asyncio
    async def test_load_active_experiments(self):
        """Test loading active experiments on initialization."""
        # Create manager and add running experiment
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'database': {'ab_test_db': os.path.join(temp_dir, 'test_ab.db')},
                'ab_testing': {'enable_monitoring': False}
            }
            
            manager1 = ABTestManager(config)
            await manager1.initialize()
            
            exp_config = ExperimentConfig(
                name="persistent_exp",
                description="Test",
                variants=["A", "B"],
                primary_metric="metric"
            )
            
            await manager1.create_experiment(exp_config)
            await manager1.start_experiment("persistent_exp")
            
            # Add some assignments
            await manager1.get_variant_for_user("persistent_exp", "user_1")
            await manager1.get_variant_for_user("persistent_exp", "user_2")
            
            # Create new manager instance
            manager2 = ABTestManager(config)
            await manager2.initialize()
            
            # Check experiment loaded
            assert "persistent_exp" in manager2.active_experiments
            
            exp = await manager2.framework.get_experiment("persistent_exp")
            assert exp is not None
            assert exp.status == ExperimentStatus.RUNNING
            assert len(exp.user_assignments) == 2
    
    @pytest.mark.asyncio
    async def test_monitoring_task(self):
        """Test experiment monitoring for completion conditions."""
        config = {
            'database': {'ab_test_db': ':memory:'},
            'ab_testing': {
                'enable_monitoring': True,
                'monitoring_interval': 0.1  # Fast for testing
            }
        }
        
        manager = ABTestManager(config)
        await manager.initialize()
        
        # Create experiment with max duration
        exp_config = ExperimentConfig(
            name="timed_exp",
            description="Test",
            variants=["A", "B"],
            primary_metric="metric",
            max_duration_days=0.00001  # Very short for testing
        )
        
        await manager.create_experiment(exp_config)
        await manager.start_experiment("timed_exp")
        
        # Wait for monitoring to stop it
        await asyncio.sleep(0.3)
        
        # Check if stopped
        assert "timed_exp" not in manager.active_experiments
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_determine_winner(self, manager):
        """Test winner determination logic."""
        # No statistical significance
        results1 = {
            'statistical_significance': {'significant': False},
            'variant_results': {
                'A': {'metrics': {'conversion': {'conversion_rate': 0.2}}},
                'B': {'metrics': {'conversion': {'conversion_rate': 0.21}}}
            }
        }
        assert manager._determine_winner(results1) is None
        
        # With significance
        results2 = {
            'statistical_significance': {'significant': True},
            'variant_results': {
                'A': {'metrics': {'conversion': {'conversion_rate': 0.2}}},
                'B': {'metrics': {'conversion': {'conversion_rate': 0.25}}}
            }
        }
        assert manager._determine_winner(results2) == "B"
        
        # Mean values instead of conversion rate
        results3 = {
            'statistical_significance': {'significant': True},
            'variant_results': {
                'A': {'metrics': {'revenue': {'mean': 10.5}}},
                'B': {'metrics': {'revenue': {'mean': 12.3}}}
            }
        }
        assert manager._determine_winner(results3) == "B"