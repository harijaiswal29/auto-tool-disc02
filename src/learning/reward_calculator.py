"""Enhanced reward calculator for the Q-learning system.

This module implements sophisticated reward calculation that considers:
- Failure type differentiation
- Partial success handling
- Resource efficiency
- User satisfaction signals
- Tool synergy recognition
- Context sensitivity
- Uncertainty measures
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

from src.utils.logger import get_logger

# Import advanced reward strategies
try:
    from .advanced_rewards.strategy_manager import StrategyManager
    ADVANCED_REWARDS_AVAILABLE = True
except ImportError:
    ADVANCED_REWARDS_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning("Advanced reward strategies not available")

logger = get_logger(__name__)


@dataclass
class ExecutionMetrics:
    """Detailed metrics from tool execution."""
    tool_id: str
    success: bool
    partial_success: bool = False
    completion_percentage: float = 0.0
    execution_time_ms: float = 0.0
    error_type: Optional[str] = None
    retry_count: int = 0
    resource_usage: Dict[str, float] = None
    result_quality: float = 1.0  # 0-1 score for result quality


class RewardCalculator:
    """
    Enhanced reward calculator that provides nuanced rewards based on
    multiple factors including failure types, resource usage, and context.
    """
    
    def __init__(self, config: Dict[str, Any], use_advanced_strategies: bool = True):
        """Initialize reward calculator with configuration.
        
        Args:
            config: Configuration dictionary
            use_advanced_strategies: Whether to use advanced reward strategies
        """
        self.config = config.get('reward_calculation', {})
        self.use_advanced_strategies = use_advanced_strategies and ADVANCED_REWARDS_AVAILABLE
        
        # Initialize advanced strategy manager if enabled
        self.strategy_manager = None
        if self.use_advanced_strategies and config.get('advanced_reward_strategies', {}).get('enabled', True):
            try:
                self.strategy_manager = StrategyManager(config)
                logger.info("Advanced reward strategies enabled")
            except Exception as e:
                logger.error(f"Failed to initialize advanced strategies: {e}")
                self.use_advanced_strategies = False
        
        # Base weights - Enhanced for dense rewards with stricter partial credit
        self.base_weights = self.config.get('base_weights', {
            'success': 25.0,  # Increased full success reward
            'failure': -3.0,  # Slightly higher penalty for complete failure
            'partial_success': 2.0,  # REDUCED from 8.0 - stricter partial credit
            'step_progress': 0.5,  # Reward for each step completed
            'exploration_bonus': 0.2,  # Bonus for trying new combinations
            'learning_bonus': 0.3,  # Bonus when discovering patterns
            'tool_efficiency': 8.0,  # INCREASED - stronger efficiency bonus
            'correct_tool_bonus': 4.0,  # Per correct tool selected
            'wrong_tool_penalty': -2.0,  # INCREASED penalty for wrong tools
            'unnecessary_tool_penalty': -3.0,  # NEW: Penalty for redundant tools
            'tool_order_bonus': 2.0,  # NEW: Bonus for correct tool ordering
            'progress_multiplier': 1.5  # NEW: Multiplier for incremental progress
        })
        
        # Failure penalties by type - Reduced to encourage exploration
        self.failure_penalties = self.config.get('failure_penalties', {
            'network_timeout': -0.1,  # Reduced - not agent's fault
            'permission_error': -0.4,  # Reduced - learn from this
            'rate_limit': -0.15,  # Reduced - temporary issue
            'connection_error': -0.12,  # Reduced - often retryable
            'retryable': -0.05,  # Minimal penalty for retryable errors
            'non_retryable': -0.35,  # Reduced but still significant
            'unknown': -0.25,  # Reduced for unknown errors
            'exploration_failure': -0.02  # NEW: Minimal penalty during exploration
        })
        
        # Resource penalties
        self.resource_penalties = self.config.get('resource_penalties', {
            'memory_weight': 0.05,
            'cpu_weight': 0.05,
            'api_calls_weight': 0.1,
            'time_weight': 0.1
        })
        
        # Synergy bonuses
        self.synergy_bonuses = self.config.get('synergy_bonuses', {
            'known_good_combo': 0.2,
            'discovered_combo': 0.3,
            'complementary_tools': 0.15
        })
        
        # Context multipliers
        self.context_multipliers = self.config.get('context_multipliers', {
            'exploration': 1.2,
            'production': 0.8,
            'high_confidence': 1.0,
            'low_confidence': 1.1,
            'user_initiated': 1.0,
            'system_initiated': 0.9
        })
        
        # Known good tool combinations - Enhanced
        self.known_synergies = {
            frozenset(['filesystem_mcp', 'search_mcp']): 0.5,
            frozenset(['sqlite_mcp', 'postgres_mcp']): -0.1,  # Redundant
            frozenset(['github_mcp', 'filesystem_mcp']): 0.6
        }
        
        # Track discovered combinations for exploration bonus
        self.discovered_combinations = set()
        self.exploration_history = defaultdict(int)
        
        # Episode tracking for context-aware rewards
        self.episode_count = 0
        self.success_streak = 0
        self.recent_successes = deque(maxlen=100)
        
        logger.info("Enhanced reward calculator initialized with dense reward shaping")
    
    def calculate_reward(self, 
                        execution_results: List[ExecutionMetrics],
                        context: Dict[str, Any],
                        user_feedback: Optional[Dict[str, Any]] = None,
                        state: Optional[np.ndarray] = None,
                        action: Optional[List[str]] = None,
                        next_state: Optional[np.ndarray] = None,
                        optimal_tools: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward based on execution results and context.
        
        Args:
            execution_results: Results from tool execution
            context: Execution context
            user_feedback: Optional user feedback
            state: Current state vector (for advanced strategies)
            action: Action taken (for advanced strategies)
            next_state: Next state vector (for advanced strategies)
        
        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        # If advanced strategies are enabled and we have state information, use them
        if (self.use_advanced_strategies and self.strategy_manager and 
            state is not None and action is not None and next_state is not None):
            
            # Convert ExecutionMetrics to format expected by strategy manager
            strategy_results = []
            for metric in execution_results:
                strategy_results.append({
                    'tool_id': metric.tool_id,
                    'success': metric.success,
                    'partial_success': metric.partial_success,
                    'completion_percentage': metric.completion_percentage,
                    'execution_time_ms': metric.execution_time_ms,
                    'error_type': metric.error_type,
                    'retry_count': metric.retry_count,
                    'resource_usage': metric.resource_usage,
                    'result_quality': metric.result_quality
                })
            
            # Calculate using advanced strategies
            advanced_reward, advanced_breakdown = self.strategy_manager.calculate_reward(
                state, action, next_state, strategy_results, context
            )
            
            # Also calculate basic reward for comparison
            basic_reward, basic_breakdown = self._calculate_basic_reward(
                execution_results, context, user_feedback
            )
            
            # Combine advanced and basic rewards
            if self.config.get('blend_basic_advanced', True):
                blend_ratio = self.config.get('advanced_blend_ratio', 0.7)
                total_reward = blend_ratio * advanced_reward + (1 - blend_ratio) * basic_reward
                
                breakdown = {
                    'advanced_reward': advanced_reward,
                    'basic_reward': basic_reward,
                    'blended_total': total_reward,
                    'advanced_components': advanced_breakdown,
                    'basic_components': basic_breakdown
                }
            else:
                # Use only advanced strategies
                total_reward = advanced_reward
                breakdown = advanced_breakdown
            
            return total_reward, breakdown
        
        # Fall back to basic reward calculation
        return self._calculate_basic_reward(execution_results, context, user_feedback)
    
    def _calculate_basic_reward(self, 
                               execution_results: List[ExecutionMetrics],
                               context: Dict[str, Any],
                               user_feedback: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, float]]:
        """Calculate reward using the basic (original) method."""
        if not execution_results:
            return -0.5, {'no_results': -0.5}
        
        # Calculate individual components
        base_reward = self._calculate_base_reward(execution_results)
        failure_adjustment = self._failure_type_adjustment(execution_results)
        partial_success_bonus = self._partial_success_bonus(execution_results)
        resource_penalty = self._resource_efficiency_penalty(execution_results)
        synergy_bonus = self._tool_synergy_bonus(execution_results)
        user_satisfaction = self._user_satisfaction_adjustment(user_feedback)
        
        # NEW: Calculate enhanced reward components
        tool_efficiency_bonus = self._calculate_tool_efficiency_bonus(execution_results)
        correct_tool_bonus = self._calculate_correct_tool_bonus(execution_results, context)
        wrong_tool_penalty = self._calculate_wrong_tool_penalty(execution_results)
        
        # Apply context sensitivity
        context_multiplier = self._get_context_multiplier(context)
        
        # Apply uncertainty factor
        uncertainty_factor = self._uncertainty_adjustment(execution_results, context)
        
        # Combine all components (including new ones)
        raw_reward = (
            base_reward + 
            failure_adjustment + 
            partial_success_bonus + 
            synergy_bonus + 
            user_satisfaction +
            tool_efficiency_bonus +
            correct_tool_bonus +
            wrong_tool_penalty
        ) * context_multiplier * uncertainty_factor - resource_penalty
        
        # Clip to reasonable range - allow larger rewards for better differentiation
        total_reward = np.clip(raw_reward, -5.0, 15.0)
        
        # Create breakdown for analysis
        breakdown = {
            'base_reward': base_reward,
            'failure_adjustment': failure_adjustment,
            'partial_success_bonus': partial_success_bonus,
            'resource_penalty': -resource_penalty,
            'synergy_bonus': synergy_bonus,
            'user_satisfaction': user_satisfaction,
            'tool_efficiency_bonus': tool_efficiency_bonus,
            'correct_tool_bonus': correct_tool_bonus,
            'wrong_tool_penalty': wrong_tool_penalty,
            'context_multiplier': context_multiplier,
            'uncertainty_factor': uncertainty_factor,
            'total': total_reward
        }
        
        logger.debug(f"Reward calculation breakdown: {breakdown}")
        
        return total_reward, breakdown
    
    def _calculate_base_reward(self, results: List[ExecutionMetrics]) -> float:
        """Calculate base reward with dense reward shaping."""
        if not results:
            return self.base_weights['failure']
        
        success_count = sum(1 for r in results if r.success)
        partial_count = sum(1 for r in results if r.partial_success and not r.success)
        total_count = len(results)
        
        # Base success/failure rewards
        if success_count == total_count:
            # Full success - maximum reward
            base_reward = self.base_weights['success']
        elif success_count > 0:
            # Partial success - proportional reward
            success_ratio = success_count / total_count
            base_reward = self.base_weights['success'] * success_ratio
        elif partial_count > 0:
            # Some partial success
            base_reward = self.base_weights['partial_success'] * (partial_count / total_count)
        else:
            # Complete failure but with reduced penalty
            base_reward = self.base_weights['failure']
        
        # Add step progress rewards (dense rewards)
        step_reward = 0.0
        for result in results:
            if result.completion_percentage > 0:
                # Reward proportional to completion
                step_reward += self.base_weights['step_progress'] * result.completion_percentage
            
            # Small reward just for attempting
            step_reward += 0.1
        
        # Exploration bonus for new tool combinations
        tool_combo = frozenset([r.tool_id for r in results])
        if tool_combo not in self.discovered_combinations:
            self.discovered_combinations.add(tool_combo)
            base_reward += self.base_weights['exploration_bonus'] * 2  # Double bonus for first discovery
        elif self.exploration_history[tool_combo] < 5:
            # Still learning this combination
            base_reward += self.base_weights['exploration_bonus']
        
        self.exploration_history[tool_combo] += 1
        
        return base_reward + step_reward
    
    def _failure_type_adjustment(self, results: List[ExecutionMetrics]) -> float:
        """Adjust reward based on failure types with learning incentives."""
        adjustment = 0.0
        
        for result in results:
            if not result.success:
                if result.error_type:
                    # Get specific penalty for error type
                    penalty = self.failure_penalties.get(
                        result.error_type, 
                        self.failure_penalties['unknown']
                    )
                    
                    # Significantly reduce penalty if retry was attempted (shows learning)
                    if result.retry_count > 0:
                        penalty *= 0.3  # 70% reduction for retry attempts
                        # Add learning bonus for retry
                        adjustment += self.base_weights.get('learning_bonus', 0.3) * 0.5
                    
                    # Further reduce penalty during early exploration
                    if hasattr(self, 'episode_count') and self.episode_count < 1000:
                        penalty *= 0.5  # Half penalty during exploration phase
                    
                    adjustment += penalty
                else:
                    # Unknown failure during exploration - minimal penalty
                    adjustment += self.failure_penalties.get('exploration_failure', -0.02)
        
        return adjustment
    
    def _partial_success_bonus(self, results: List[ExecutionMetrics]) -> float:
        """Stricter bonus for partial successes - only reward significant progress."""
        bonus = 0.0
        
        for result in results:
            if result.partial_success:
                # STRICTER: Only give partial credit for >80% completion
                if result.completion_percentage >= 0.8:
                    # Near completion - moderate bonus
                    completion_bonus = self.base_weights['partial_success'] * result.completion_percentage
                    bonus += completion_bonus + 1.0
                elif result.completion_percentage >= 0.6:
                    # Good progress - small bonus
                    bonus += self.base_weights['partial_success'] * 0.5
                elif result.completion_percentage >= 0.4:
                    bonus += 0.5  # Some progress
                else:
                    bonus += 0.2  # Any progress is good
                
                # Extra bonus for high-quality partial results
                if result.result_quality > 0.8:
                    bonus += 0.1
        
        return bonus
    
    def _resource_efficiency_penalty(self, results: List[ExecutionMetrics]) -> float:
        """Penalize inefficient resource usage."""
        total_penalty = 0.0
        
        for result in results:
            if result.resource_usage:
                # Memory penalty (normalized to percentage)
                if 'memory_mb' in result.resource_usage:
                    memory_percent = min(result.resource_usage['memory_mb'] / 1000, 1.0)
                    total_penalty += memory_percent * self.resource_penalties['memory_weight']
                
                # CPU penalty
                if 'cpu_percent' in result.resource_usage:
                    cpu_percent = result.resource_usage['cpu_percent'] / 100
                    total_penalty += cpu_percent * self.resource_penalties['cpu_weight']
                
                # API calls penalty (for rate-limited resources)
                if 'api_calls' in result.resource_usage:
                    api_usage = min(result.resource_usage['api_calls'] / 100, 1.0)
                    total_penalty += api_usage * self.resource_penalties['api_calls_weight']
            
            # Time penalty (logarithmic to avoid harsh penalties)
            if result.execution_time_ms > 0:
                time_penalty = np.log(max(result.execution_time_ms / 1000, 1))
                total_penalty += time_penalty * self.resource_penalties['time_weight']
        
        return total_penalty
    
    def _tool_synergy_bonus(self, results: List[ExecutionMetrics]) -> float:
        """Bonus for using complementary tools effectively."""
        if len(results) < 2:
            return 0.0
        
        bonus = 0.0
        tools_used = [r.tool_id for r in results if r.success or r.partial_success]
        
        if len(tools_used) < 2:
            return 0.0
        
        # Check for known good combinations
        tool_set = frozenset(tools_used)
        for known_combo, combo_bonus in self.known_synergies.items():
            if known_combo.issubset(tool_set):
                bonus += combo_bonus
        
        # Bonus for discovering new successful combinations
        if len(tools_used) > 1 and all(r.success for r in results):
            # Check if this is a new combination (would need history tracking)
            discovery_bonus = self.synergy_bonuses['discovered_combo'] * (
                len(tools_used) / len(results)
            )
            bonus += discovery_bonus
        
        # Penalty for redundant tools
        if self._has_redundant_tools(tools_used):
            bonus -= 0.1
        
        return bonus
    
    def _has_redundant_tools(self, tools: List[str]) -> bool:
        """Check if tool combination has redundancies."""
        # Simple heuristic - tools with similar names/prefixes
        tool_types = defaultdict(int)
        for tool in tools:
            tool_type = tool.split('_')[0]  # Get prefix
            tool_types[tool_type] += 1
        
        # Multiple tools of same type might be redundant
        return any(count > 1 for count in tool_types.values())
    
    def _user_satisfaction_adjustment(self, feedback: Optional[Dict[str, Any]]) -> float:
        """Adjust reward based on user satisfaction signals."""
        if not feedback:
            return 0.0
        
        adjustment = 0.0
        
        # Explicit feedback
        if 'rating' in feedback:
            # Assume rating is 1-5, normalize to -0.5 to 0.5
            adjustment += (feedback['rating'] - 3) * 0.2
        
        # Implicit feedback from follow-up actions
        if 'query_reformulated' in feedback and feedback['query_reformulated']:
            adjustment -= 0.1  # User had to reformulate
        
        if 'follow_up_query' in feedback:
            # Quick follow-up might indicate incomplete results
            if feedback.get('follow_up_time_seconds', float('inf')) < 10:
                adjustment -= 0.15
        
        if 'result_used' in feedback and feedback['result_used']:
            adjustment += 0.2  # User actually used the results
        
        return adjustment
    
    def _get_context_multiplier(self, context: Dict[str, Any]) -> float:
        """Get context-sensitive multiplier."""
        # Mode-based multiplier
        mode = context.get('mode', 'production')
        multiplier = self.context_multipliers.get(mode, 1.0)
        
        # Confidence adjustment
        confidence = context.get('intent_confidence', 0.7)
        if confidence > 0.8:
            multiplier *= self.context_multipliers['high_confidence']
        elif confidence < 0.5:
            multiplier *= self.context_multipliers['low_confidence']
        
        # User vs system initiated
        if context.get('user_initiated', True):
            multiplier *= self.context_multipliers['user_initiated']
        else:
            multiplier *= self.context_multipliers['system_initiated']
        
        return multiplier
    
    def _uncertainty_adjustment(self, results: List[ExecutionMetrics], 
                               context: Dict[str, Any]) -> float:
        """Adjust for uncertainty in results and context."""
        # Base uncertainty from intent confidence
        intent_confidence = context.get('intent_confidence', 0.7)
        
        # Result consistency check
        success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
        
        # High variance in results indicates uncertainty
        result_variance = np.var([r.result_quality for r in results])
        
        # Calculate uncertainty factor (0.5 to 1.5)
        uncertainty = 1.0
        
        # Lower confidence increases uncertainty impact
        if intent_confidence < 0.6:
            uncertainty *= 1.2
        
        # Inconsistent results increase uncertainty
        if 0.3 < success_rate < 0.7:  # Mixed results
            uncertainty *= 1.1
        
        # High variance in quality
        if result_variance > 0.2:
            uncertainty *= 1.1
        
        return np.clip(uncertainty, 0.5, 1.5)
    
    def _calculate_tool_efficiency_bonus(self, results: List[ExecutionMetrics], 
                                        expected_tools: int = 3) -> float:
        """
        Calculate bonus for using fewer tools efficiently.
        Rewards achieving success with minimal tool usage.
        """
        if not results:
            return 0.0
        
        successful_results = [r for r in results if r.success or r.partial_success]
        if not successful_results:
            return 0.0
        
        tools_used = len(results)
        efficiency_ratio = len(successful_results) / tools_used
        
        # Bonus for using fewer tools than expected
        if tools_used < expected_tools and efficiency_ratio > 0.7:
            bonus = self.base_weights.get('tool_efficiency', 5.0) * (1 - tools_used/expected_tools)
        else:
            bonus = 0.0
        
        return bonus
    
    def _calculate_correct_tool_bonus(self, results: List[ExecutionMetrics], 
                                     context: Dict[str, Any]) -> float:
        """
        Calculate bonus for selecting appropriate tools for the task.
        """
        if not results:
            return 0.0
        
        bonus = 0.0
        correct_tool_weight = self.base_weights.get('correct_tool_bonus', 3.0)
        
        # Check each tool selection
        for result in results:
            # Tool is "correct" if it contributed to success
            if result.success or (result.partial_success and result.completion_percentage > 0.5):
                bonus += correct_tool_weight
            elif result.partial_success:
                # Partial credit for partial success
                bonus += correct_tool_weight * result.completion_percentage
        
        # Extra bonus for perfect tool selection (all tools contributed)
        if all(r.success or r.partial_success for r in results):
            bonus *= 1.2
        
        return bonus
    
    def _calculate_wrong_tool_penalty(self, results: List[ExecutionMetrics]) -> float:
        """
        Calculate penalty for selecting inappropriate tools.
        Mild penalty to encourage exploration while discouraging random selection.
        """
        if not results:
            return 0.0
        
        penalty = 0.0
        wrong_tool_weight = self.base_weights.get('wrong_tool_penalty', -1.0)
        
        for result in results:
            # Tool is "wrong" if it completely failed without contributing
            if not result.success and not result.partial_success:
                if result.completion_percentage == 0:
                    penalty += wrong_tool_weight
                else:
                    # Reduced penalty if there was some progress
                    penalty += wrong_tool_weight * (1 - result.completion_percentage)
        
        return penalty
    
    def update_known_synergies(self, tool_combination: List[str], 
                              success_rate: float, occurrences: int):
        """Update known tool synergies based on observed patterns."""
        if len(tool_combination) < 2 or occurrences < 5:
            return
        
        tool_set = frozenset(tool_combination)
        
        # Calculate synergy score based on success rate
        if success_rate > 0.8:
            synergy_score = 0.2 * (success_rate - 0.5)
        elif success_rate < 0.3:
            synergy_score = -0.1  # Mark as poor combination
        else:
            return  # Neutral, don't record
        
        # Update or add to known synergies
        if tool_set in self.known_synergies:
            # Weighted average with existing score
            old_score = self.known_synergies[tool_set]
            self.known_synergies[tool_set] = (old_score + synergy_score) / 2
        else:
            self.known_synergies[tool_set] = synergy_score
        
        logger.info(f"Updated synergy for {tool_combination}: {synergy_score:.3f}")
    
    def set_episode_count(self, episode: int):
        """Set current episode count for context-aware rewards."""
        self.episode_count = episode
    
    def update_success_tracking(self, success: bool):
        """Track recent successes for momentum rewards."""
        self.recent_successes.append(success)
        if success:
            self.success_streak += 1
        else:
            self.success_streak = 0
    
    def calculate_curiosity_bonus(self, state_hash: str, action: List[str]) -> float:
        """Calculate curiosity bonus for exploring new state-action pairs."""
        # Create unique identifier for state-action pair
        action_str = '_'.join(sorted(action))
        sa_pair = f"{state_hash}:{action_str}"
        
        # Check if this is a novel state-action pair
        if not hasattr(self, 'visited_pairs'):
            self.visited_pairs = {}
        
        visit_count = self.visited_pairs.get(sa_pair, 0)
        self.visited_pairs[sa_pair] = visit_count + 1
        
        # Calculate curiosity bonus (decreases with visits)
        if visit_count == 0:
            # First visit - high curiosity bonus
            return 1.0
        elif visit_count < 5:
            # Still exploring
            return 0.5 / (visit_count + 1)
        else:
            # Well-explored
            return 0.0
    
    def calculate_momentum_bonus(self) -> float:
        """Calculate bonus for maintaining success momentum."""
        if self.success_streak > 5:
            return min(self.success_streak * 0.1, 2.0)
        elif self.success_streak > 2:
            return 0.3
        return 0.0