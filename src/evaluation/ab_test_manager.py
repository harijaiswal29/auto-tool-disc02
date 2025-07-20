"""
A/B Test Manager for Experiment Lifecycle Management

This module provides comprehensive experiment management including:
- Experiment creation, monitoring, and conclusion
- Real-time metric collection and aggregation
- Database persistence for experiments
- Integration with orchestrator and reward systems
- Dashboard API support
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from src.evaluation.ab_testing_framework import (
    ABTestingFramework, ExperimentConfig, ExperimentStatus,
    AssignmentStrategy, StatisticalMethod
)
from src.database.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ABTestManager:
    """Manages A/B test experiments lifecycle and persistence."""
    
    def __init__(self, config: Dict[str, Any], db_path: Optional[str] = None):
        self.config = config
        self.framework = ABTestingFramework()
        self.db_path = db_path or config.get('database', {}).get('ab_test_db', 'data/ab_tests.db')
        self.db_manager = DatabaseManager(self.db_path)
        self.active_experiments: Set[str] = set()
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the A/B test manager and database."""
        await self.db_manager.initialize()
        await self._create_database_schema()
        await self._load_active_experiments()
        
        # Start monitoring task
        if self.config.get('ab_testing', {}).get('enable_monitoring', True):
            self._monitoring_task = asyncio.create_task(self._monitor_experiments())
    
    async def _create_database_schema(self):
        """Create database tables for A/B testing."""
        async with self.db_manager.get_connection() as conn:
            # Experiments table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    config JSON NOT NULL,
                    status TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User assignments table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_name) REFERENCES ab_experiments(name),
                    UNIQUE(experiment_name, user_id)
                )
            """)
            
            # Metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    value REAL,
                    success BOOLEAN,
                    user_id TEXT,
                    metadata JSON,
                    FOREIGN KEY (experiment_name) REFERENCES ab_experiments(name)
                )
            """)
            
            # Experiment results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    results JSON NOT NULL,
                    winner TEXT,
                    confidence REAL,
                    FOREIGN KEY (experiment_name) REFERENCES ab_experiments(name)
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_assignments_experiment 
                ON ab_assignments(experiment_name)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_experiment_variant 
                ON ab_metrics(experiment_name, variant)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON ab_metrics(timestamp)
            """)
            
            await conn.commit()
    
    async def _load_active_experiments(self):
        """Load active experiments from database."""
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT name, config FROM ab_experiments 
                WHERE status IN ('running', 'paused')
            """)
            rows = await cursor.fetchall()
            
            for name, config_json in rows:
                config_dict = json.loads(config_json)
                config = self._dict_to_experiment_config(config_dict)
                experiment = await self.framework.create_experiment(config)
                
                # Load user assignments
                assignment_cursor = await conn.execute("""
                    SELECT user_id, variant FROM ab_assignments
                    WHERE experiment_name = ?
                """, (name,))
                assignments = await assignment_cursor.fetchall()
                
                for user_id, variant in assignments:
                    experiment.user_assignments[user_id] = variant
                
                # Set experiment status
                experiment.status = ExperimentStatus.RUNNING
                self.active_experiments.add(name)
                
                logger.info(f"Loaded active experiment: {name}")
    
    def _dict_to_experiment_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        return ExperimentConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            variants=config_dict['variants'],
            primary_metric=config_dict['primary_metric'],
            secondary_metrics=config_dict.get('secondary_metrics', []),
            assignment_strategy=AssignmentStrategy(config_dict.get('assignment_strategy', 'random')),
            assignment_weights=config_dict.get('assignment_weights'),
            statistical_method=StatisticalMethod(config_dict.get('statistical_method', 'frequentist')),
            target_sample_size=config_dict.get('target_sample_size'),
            min_sample_size=config_dict.get('min_sample_size', 100),
            confidence_level=config_dict.get('confidence_level', 0.95),
            power=config_dict.get('power', 0.8),
            minimum_detectable_effect=config_dict.get('minimum_detectable_effect', 0.05),
            max_duration_days=config_dict.get('max_duration_days'),
            enable_early_stopping=config_dict.get('enable_early_stopping', True),
            early_stopping_threshold=config_dict.get('early_stopping_threshold', 0.001),
            enable_multi_armed_bandit=config_dict.get('enable_multi_armed_bandit', False),
            mab_exploration_rate=config_dict.get('mab_exploration_rate', 0.1)
        )
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        # Create experiment in framework
        experiment = await self.framework.create_experiment(config)
        
        # Save to database
        config_dict = {
            'name': config.name,
            'description': config.description,
            'variants': config.variants,
            'primary_metric': config.primary_metric,
            'secondary_metrics': config.secondary_metrics,
            'assignment_strategy': config.assignment_strategy.value,
            'assignment_weights': config.assignment_weights,
            'statistical_method': config.statistical_method.value,
            'target_sample_size': config.target_sample_size,
            'min_sample_size': config.min_sample_size,
            'confidence_level': config.confidence_level,
            'power': config.power,
            'minimum_detectable_effect': config.minimum_detectable_effect,
            'max_duration_days': config.max_duration_days,
            'enable_early_stopping': config.enable_early_stopping,
            'early_stopping_threshold': config.early_stopping_threshold,
            'enable_multi_armed_bandit': config.enable_multi_armed_bandit,
            'mab_exploration_rate': config.mab_exploration_rate
        }
        
        async with self.db_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO ab_experiments (name, description, config, status)
                VALUES (?, ?, ?, ?)
            """, (config.name, config.description, json.dumps(config_dict), 'draft'))
            await conn.commit()
        
        logger.info(f"Created experiment: {config.name}")
        return config.name
    
    async def start_experiment(self, experiment_name: str):
        """Start an experiment."""
        experiment = await self.framework.get_experiment(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        await experiment.start()
        self.active_experiments.add(experiment_name)
        
        # Update database
        async with self.db_manager.get_connection() as conn:
            await conn.execute("""
                UPDATE ab_experiments 
                SET status = ?, start_time = ?, updated_at = ?
                WHERE name = ?
            """, ('running', datetime.now(), datetime.now(), experiment_name))
            await conn.commit()
        
        logger.info(f"Started experiment: {experiment_name}")
    
    async def stop_experiment(self, experiment_name: str):
        """Stop an experiment."""
        experiment = await self.framework.get_experiment(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        await experiment.stop()
        self.active_experiments.discard(experiment_name)
        
        # Analyze results
        results = await experiment.analyze()
        
        # Update database
        async with self.db_manager.get_connection() as conn:
            await conn.execute("""
                UPDATE ab_experiments 
                SET status = ?, end_time = ?, updated_at = ?
                WHERE name = ?
            """, ('completed', datetime.now(), datetime.now(), experiment_name))
            
            # Save results
            winner = self._determine_winner(results)
            confidence = results.get('statistical_significance', {}).get('p_value', 0)
            
            # Convert any non-serializable types in results
            serializable_results = self._make_json_serializable(results)
            
            await conn.execute("""
                INSERT INTO ab_results (experiment_name, results, winner, confidence)
                VALUES (?, ?, ?, ?)
            """, (experiment_name, json.dumps(serializable_results), winner, confidence))
            
            await conn.commit()
        
        logger.info(f"Stopped experiment: {experiment_name}")
        return results
    
    def _determine_winner(self, results: Dict[str, Any]) -> Optional[str]:
        """Determine the winning variant from results."""
        if not results.get('statistical_significance', {}).get('significant', False):
            return None
        
        # Find variant with best primary metric
        best_variant = None
        best_value = float('-inf')
        
        for variant, variant_result in results.get('variant_results', {}).items():
            primary_metric = list(variant_result['metrics'].values())[0]
            value = primary_metric.get('conversion_rate') or primary_metric.get('mean', 0)
            
            if value > best_value:
                best_value = value
                best_variant = variant
        
        return best_variant
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert non-JSON serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, bool):
            return bool(obj)  # Ensure it's a Python bool, not numpy bool
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            # Convert numpy types and other non-serializable types to Python types
            try:
                return json.loads(json.dumps(obj))
            except:
                return str(obj)
    
    async def get_variant_for_user(self, experiment_name: str, user_id: str, 
                                  context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get variant assignment for a user."""
        if experiment_name not in self.active_experiments:
            return None
        
        # Check existing assignment
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT variant FROM ab_assignments
                WHERE experiment_name = ? AND user_id = ?
            """, (experiment_name, user_id))
            row = await cursor.fetchone()
            
            if row:
                return row[0]
        
        # Assign new variant
        variant = await self.framework.assign_user(experiment_name, user_id)
        if variant:
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO ab_assignments (experiment_name, user_id, variant)
                    VALUES (?, ?, ?)
                """, (experiment_name, user_id, variant))
                await conn.commit()
        
        return variant
    
    async def record_metric(self, experiment_name: str, user_id: str,
                           metric_name: str, value: float, success: bool = True,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a metric for an experiment."""
        # Record in framework
        await self.framework.record_event(experiment_name, user_id, metric_name, value, success)
        
        # Get variant for user
        variant = await self.get_variant_for_user(experiment_name, user_id)
        if not variant:
            return
        
        # Save to database
        async with self.db_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO ab_metrics 
                (experiment_name, variant, metric_name, value, success, user_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (experiment_name, variant, metric_name, value, success, user_id, 
                  json.dumps(metadata) if metadata else None))
            await conn.commit()
    
    async def _monitor_experiments(self):
        """Monitor active experiments for completion conditions."""
        while True:
            try:
                for experiment_name in list(self.active_experiments):
                    experiment = await self.framework.get_experiment(experiment_name)
                    if not experiment:
                        continue
                    
                    # Check if experiment should be stopped
                    should_stop = False
                    
                    # Check max duration
                    if experiment.config.max_duration_days and experiment.start_time:
                        duration = datetime.now() - experiment.start_time
                        if duration > timedelta(days=experiment.config.max_duration_days):
                            should_stop = True
                            logger.info(f"Experiment {experiment_name} reached max duration")
                    
                    # Check target sample size
                    if experiment.config.target_sample_size:
                        total_users = len(experiment.user_assignments)
                        if total_users >= experiment.config.target_sample_size:
                            should_stop = True
                            logger.info(f"Experiment {experiment_name} reached target sample size")
                    
                    # Check if already stopped (by early stopping)
                    if experiment.status == ExperimentStatus.COMPLETED:
                        should_stop = True
                    
                    if should_stop:
                        await self.stop_experiment(experiment_name)
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config.get('ab_testing', {}).get('monitoring_interval', 60))
                
            except Exception as e:
                logger.error(f"Error in experiment monitoring: {e}")
                await asyncio.sleep(60)
    
    async def get_experiment_status(self, experiment_name: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        experiment = await self.framework.get_experiment(experiment_name)
        if not experiment:
            return {'error': 'Experiment not found'}
        
        # Get metrics from database
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT variant, COUNT(DISTINCT user_id) as users,
                       COUNT(*) as events, AVG(value) as avg_value
                FROM ab_metrics
                WHERE experiment_name = ?
                GROUP BY variant
            """, (experiment_name,))
            metrics = await cursor.fetchall()
        
        return {
            'name': experiment_name,
            'status': experiment.status.value,
            'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
            'total_users': len(experiment.user_assignments),
            'variant_metrics': [
                {
                    'variant': variant,
                    'users': users,
                    'events': events,
                    'avg_value': avg_value
                }
                for variant, users, events, avg_value in metrics
            ]
        }
    
    async def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments with optional status filter."""
        async with self.db_manager.get_connection() as conn:
            if status:
                cursor = await conn.execute("""
                    SELECT name, description, status, start_time, end_time
                    FROM ab_experiments
                    WHERE status = ?
                    ORDER BY created_at DESC
                """, (status,))
            else:
                cursor = await conn.execute("""
                    SELECT name, description, status, start_time, end_time
                    FROM ab_experiments
                    ORDER BY created_at DESC
                """)
            
            rows = await cursor.fetchall()
        
        return [
            {
                'name': name,
                'description': description,
                'status': status,
                'start_time': start_time,
                'end_time': end_time
            }
            for name, description, status, start_time, end_time in rows
        ]
    
    async def get_experiment_results(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get results for a completed experiment."""
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT results, winner, confidence, analysis_timestamp
                FROM ab_results
                WHERE experiment_name = ?
                ORDER BY analysis_timestamp DESC
                LIMIT 1
            """, (experiment_name,))
            row = await cursor.fetchone()
            
            if row:
                results_json, winner, confidence, timestamp = row
                results = json.loads(results_json)
                results['winner'] = winner
                results['confidence'] = confidence
                results['analysis_timestamp'] = timestamp
                return results
            
            return None
    
    async def cleanup(self):
        """Clean up resources."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass