"""
Comprehensive A/B Testing Framework for Evaluation

This module provides a robust A/B testing framework that supports:
- Multiple assignment strategies (random, deterministic, weighted)
- Power analysis for sample size determination
- Sequential testing with early stopping
- Bayesian and frequentist statistical methods
- Multi-armed bandit integration for adaptive allocation
"""

import asyncio
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import scipy.stats as stats
from scipy.stats import beta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class AssignmentStrategy(Enum):
    """Strategy for assigning users to test groups."""
    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    WEIGHTED = "weighted"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"


class StatisticalMethod(Enum):
    """Statistical method for analysis."""
    FREQUENTIST = "frequentist"
    BAYESIAN = "bayesian"
    SEQUENTIAL = "sequential"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    variants: List[str]  # e.g., ["control", "treatment_a", "treatment_b"]
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.RANDOM
    assignment_weights: Optional[Dict[str, float]] = None
    statistical_method: StatisticalMethod = StatisticalMethod.FREQUENTIST
    target_sample_size: Optional[int] = None
    min_sample_size: int = 100
    confidence_level: float = 0.95
    power: float = 0.8
    minimum_detectable_effect: float = 0.05
    max_duration_days: Optional[int] = None
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.001  # p-value threshold
    enable_multi_armed_bandit: bool = False
    mab_exploration_rate: float = 0.1


@dataclass
class VariantMetrics:
    """Metrics collected for a variant."""
    variant_name: str
    sample_size: int = 0
    successes: int = 0
    total_value: float = 0.0
    sum_squared: float = 0.0
    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        """Calculate mean value."""
        return self.total_value / self.sample_size if self.sample_size > 0 else 0.0
    
    @property
    def variance(self) -> float:
        """Calculate variance."""
        if self.sample_size <= 1:
            return 0.0
        mean_val = self.mean
        return (self.sum_squared / self.sample_size) - (mean_val ** 2)
    
    @property
    def std_dev(self) -> float:
        """Calculate standard deviation."""
        return np.sqrt(self.variance)
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate for binary metrics."""
        return self.successes / self.sample_size if self.sample_size > 0 else 0.0


class AssignmentStrategyBase(ABC):
    """Base class for assignment strategies."""
    
    @abstractmethod
    async def assign(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Assign a user to a variant."""
        pass


class RandomAssignment(AssignmentStrategyBase):
    """Random assignment strategy."""
    
    async def assign(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Randomly assign user to a variant."""
        if experiment_config.assignment_weights:
            variants = list(experiment_config.assignment_weights.keys())
            weights = list(experiment_config.assignment_weights.values())
            return np.random.choice(variants, p=weights)
        else:
            return np.random.choice(experiment_config.variants)


class DeterministicAssignment(AssignmentStrategyBase):
    """Deterministic assignment based on user ID hash."""
    
    async def assign(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Deterministically assign user to a variant based on hash."""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant_index = hash_value % len(experiment_config.variants)
        return experiment_config.variants[variant_index]


class MultiArmedBanditAssignment(AssignmentStrategyBase):
    """Multi-armed bandit assignment using Thompson sampling."""
    
    def __init__(self):
        self.variant_priors: Dict[str, Tuple[float, float]] = {}  # (alpha, beta) for each variant
    
    async def assign(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Assign using Thompson sampling."""
        # Initialize priors if not set
        for variant in experiment_config.variants:
            if variant not in self.variant_priors:
                self.variant_priors[variant] = (1.0, 1.0)  # Uniform prior
        
        # Sample from Beta distributions
        samples = {}
        for variant, (alpha, beta_param) in self.variant_priors.items():
            samples[variant] = beta.rvs(alpha, beta_param)
        
        # Select variant with highest sample
        return max(samples, key=samples.get)
    
    def update_priors(self, variant: str, success: bool):
        """Update priors based on observed outcome."""
        alpha, beta_param = self.variant_priors.get(variant, (1.0, 1.0))
        if success:
            alpha += 1
        else:
            beta_param += 1
        self.variant_priors[variant] = (alpha, beta_param)


class ABTestingFramework:
    """Main A/B testing framework class."""
    
    def __init__(self):
        self.experiments: Dict[str, 'Experiment'] = {}
        self.assignment_strategies = {
            AssignmentStrategy.RANDOM: RandomAssignment(),
            AssignmentStrategy.DETERMINISTIC: DeterministicAssignment(),
            AssignmentStrategy.MULTI_ARMED_BANDIT: MultiArmedBanditAssignment(),
            AssignmentStrategy.WEIGHTED: RandomAssignment(),  # Use random for weighted for now
        }
    
    async def create_experiment(self, config: ExperimentConfig) -> 'Experiment':
        """Create a new experiment."""
        experiment = Experiment(config, self)
        self.experiments[config.name] = experiment
        logger.info(f"Created experiment: {config.name}")
        return experiment
    
    async def get_experiment(self, name: str) -> Optional['Experiment']:
        """Get an experiment by name."""
        return self.experiments.get(name)
    
    async def assign_user(self, experiment_name: str, user_id: str) -> Optional[str]:
        """Assign a user to a variant in an experiment."""
        experiment = await self.get_experiment(experiment_name)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None
        
        # Check if user already assigned
        existing_variant = experiment.user_assignments.get(user_id)
        if existing_variant:
            return existing_variant
        
        # Assign user to variant
        try:
            strategy = self.assignment_strategies[experiment.config.assignment_strategy]
        except KeyError:
            logger.error(f"Assignment strategy not found: {experiment.config.assignment_strategy}")
            logger.error(f"Available strategies: {list(self.assignment_strategies.keys())}")
            # Fallback to random
            strategy = self.assignment_strategies[AssignmentStrategy.RANDOM]
        variant = await strategy.assign(user_id, experiment.config)
        experiment.user_assignments[user_id] = variant
        
        logger.debug(f"Assigned user {user_id} to variant {variant} in experiment {experiment_name}")
        return variant
    
    async def record_event(self, experiment_name: str, user_id: str, 
                          metric_name: str, value: float, success: bool = True):
        """Record an event for a user in an experiment."""
        experiment = await self.get_experiment(experiment_name)
        if not experiment:
            return
        
        variant = experiment.user_assignments.get(user_id)
        if not variant:
            return
        
        # Update metrics
        await experiment.update_metrics(variant, metric_name, value, success)
        
        # Update MAB priors if enabled
        if (experiment.config.enable_multi_armed_bandit and 
            isinstance(self.assignment_strategies[AssignmentStrategy.MULTI_ARMED_BANDIT], 
                      MultiArmedBanditAssignment)):
            mab_strategy = self.assignment_strategies[AssignmentStrategy.MULTI_ARMED_BANDIT]
            mab_strategy.update_priors(variant, success)
    
    def calculate_sample_size(self, baseline_rate: float, mde: float, 
                            alpha: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size for given parameters."""
        effect_size = mde / np.sqrt(baseline_rate * (1 - baseline_rate))
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        return int(np.ceil(n))
    
    async def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze experiment results."""
        experiment = await self.get_experiment(experiment_name)
        if not experiment:
            return {}
        
        return await experiment.analyze()


class Experiment:
    """Represents a single A/B test experiment."""
    
    def __init__(self, config: ExperimentConfig, framework: ABTestingFramework):
        self.config = config
        self.framework = framework
        self.status = ExperimentStatus.DRAFT
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.user_assignments: Dict[str, str] = {}
        self.variant_metrics: Dict[str, Dict[str, VariantMetrics]] = {}
        
        # Initialize metrics for each variant
        for variant in config.variants:
            self.variant_metrics[variant] = {}
            for metric in [config.primary_metric] + config.secondary_metrics:
                self.variant_metrics[variant][metric] = VariantMetrics(variant)
    
    async def start(self):
        """Start the experiment."""
        self.status = ExperimentStatus.RUNNING
        self.start_time = datetime.now()
        logger.info(f"Started experiment: {self.config.name}")
    
    async def stop(self):
        """Stop the experiment."""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()
        logger.info(f"Stopped experiment: {self.config.name}")
    
    async def update_metrics(self, variant: str, metric_name: str, 
                           value: float, success: bool = True):
        """Update metrics for a variant."""
        if variant not in self.variant_metrics:
            return
        
        if metric_name not in self.variant_metrics[variant]:
            self.variant_metrics[variant][metric_name] = VariantMetrics(variant)
        
        metrics = self.variant_metrics[variant][metric_name]
        metrics.sample_size += 1
        metrics.total_value += value
        metrics.sum_squared += value ** 2
        metrics.values.append(value)
        metrics.timestamps.append(datetime.now())
        
        if success:
            metrics.successes += 1
        
        # Check for early stopping
        if self.config.enable_early_stopping:
            await self._check_early_stopping()
    
    async def _check_early_stopping(self):
        """Check if experiment should be stopped early."""
        primary_metrics = {
            variant: self.variant_metrics[variant][self.config.primary_metric]
            for variant in self.config.variants
        }
        
        # Need minimum sample size
        if any(m.sample_size < self.config.min_sample_size for m in primary_metrics.values()):
            return
        
        # Perform statistical test
        if self.config.statistical_method == StatisticalMethod.FREQUENTIST:
            p_value = await self._frequentist_test(primary_metrics)
            if p_value < self.config.early_stopping_threshold:
                await self.stop()
                logger.info(f"Early stopping triggered for experiment {self.config.name} (p={p_value:.4f})")
    
    async def _frequentist_test(self, metrics: Dict[str, VariantMetrics]) -> float:
        """Perform frequentist statistical test."""
        if len(metrics) != 2:
            # For now, only support two-variant tests
            return 1.0
        
        variants = list(metrics.keys())
        m1, m2 = metrics[variants[0]], metrics[variants[1]]
        
        # For binary metrics (conversion rates)
        if self.config.primary_metric.endswith('_rate'):
            # Chi-square test
            observed = [[m1.successes, m1.sample_size - m1.successes],
                       [m2.successes, m2.sample_size - m2.successes]]
            _, p_value, _, _ = stats.chi2_contingency(observed)
            return p_value
        else:
            # T-test for continuous metrics
            _, p_value = stats.ttest_ind_from_stats(
                m1.mean, m1.std_dev, m1.sample_size,
                m2.mean, m2.std_dev, m2.sample_size
            )
            return p_value
    
    async def _bayesian_test(self, metrics: Dict[str, VariantMetrics]) -> Dict[str, Any]:
        """Perform Bayesian statistical test."""
        results = {}
        
        if len(metrics) != 2:
            return results
        
        variants = list(metrics.keys())
        m1, m2 = metrics[variants[0]], metrics[variants[1]]
        
        # For binary metrics
        if self.config.primary_metric.endswith('_rate'):
            # Beta-Binomial model
            alpha1, beta1 = m1.successes + 1, m1.sample_size - m1.successes + 1
            alpha2, beta2 = m2.successes + 1, m2.sample_size - m2.successes + 1
            
            # Monte Carlo simulation
            samples = 10000
            dist1 = beta.rvs(alpha1, beta1, size=samples)
            dist2 = beta.rvs(alpha2, beta2, size=samples)
            
            prob_1_better = np.mean(dist1 > dist2)
            expected_lift = np.mean((dist2 - dist1) / dist1)
            
            results = {
                'probability_treatment_better': prob_1_better,
                'expected_lift': expected_lift,
                'credible_interval_95': (
                    np.percentile(dist2 - dist1, 2.5),
                    np.percentile(dist2 - dist1, 97.5)
                )
            }
        
        return results
    
    async def analyze(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        results = {
            'experiment_name': self.config.name,
            'status': self.status.value,
            'duration_days': None,
            'total_users': len(self.user_assignments),
            'variant_results': {}
        }
        
        if self.start_time:
            end_time = self.end_time or datetime.now()
            results['duration_days'] = (end_time - self.start_time).days
        
        # Analyze each variant
        for variant in self.config.variants:
            variant_result = {
                'users': sum(1 for v in self.user_assignments.values() if v == variant),
                'metrics': {}
            }
            
            for metric_name, metrics in self.variant_metrics[variant].items():
                metric_info = {
                    'sample_size': metrics.sample_size,
                    'mean': metrics.mean,
                    'std_dev': metrics.std_dev,
                }
                
                # Add appropriate rate/value based on metric type
                if metric_name.endswith('_rate'):
                    metric_info['conversion_rate'] = metrics.conversion_rate
                else:
                    # For non-rate metrics (like reward), show the mean as the primary value
                    metric_info['value'] = metrics.mean
                    # Also show success rate if applicable
                    if metrics.sample_size > 0:
                        metric_info['success_rate'] = metrics.conversion_rate
                
                variant_result['metrics'][metric_name] = metric_info
            
            results['variant_results'][variant] = variant_result
        
        # Statistical analysis
        primary_metrics = {
            variant: self.variant_metrics[variant][self.config.primary_metric]
            for variant in self.config.variants
        }
        
        if self.config.statistical_method == StatisticalMethod.FREQUENTIST:
            p_value = await self._frequentist_test(primary_metrics)
            results['statistical_significance'] = {
                'method': 'frequentist',
                'p_value': p_value,
                'significant': p_value < (1 - self.config.confidence_level)
            }
        elif self.config.statistical_method == StatisticalMethod.BAYESIAN:
            bayesian_results = await self._bayesian_test(primary_metrics)
            results['statistical_significance'] = {
                'method': 'bayesian',
                **bayesian_results
            }
        
        return results