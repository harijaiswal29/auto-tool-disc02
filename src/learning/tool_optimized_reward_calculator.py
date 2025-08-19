"""Tool-Selection-Optimized Reward Calculator.

This variant optimizes for TOOL SELECTION ACCURACY rather than just task completion.
Target: 25-30% tool selection accuracy (8-10x improvement over random baseline).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from src.utils.logger import get_logger

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
    result_quality: float = 1.0


class ToolOptimizedRewardCalculator:
    """
    Reward calculator optimized for tool selection accuracy.
    
    Key differences from standard calculator:
    1. Prioritizes optimal tool selection over task completion
    2. Strong penalties for suboptimal tools
    3. Exact match bonuses for perfect tool selection
    4. Distance-based penalties from optimal tool set
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with tool-optimized weights."""
        self.config = config.get('reward_calculation', {})
        
        # REBALANCED WEIGHTS: Prioritize tool selection accuracy
        self.base_weights = {
            # Task completion (reduced from 25.0)
            'task_success': 10.0,  
            'partial_success': 1.0,
            'task_failure': -2.0,
            
            # Tool selection accuracy (massively increased)
            'optimal_tool_match': 15.0,  # Per optimal tool correctly selected
            'exact_match_bonus': 10.0,   # Bonus for perfect tool selection
            'suboptimal_tool_penalty': -5.0,  # Per non-optimal tool used
            'tool_distance_penalty': -3.0,  # Multiplied by distance from optimal
            'unnecessary_tool_penalty': -8.0,  # For completely wrong tools
            
            # Efficiency rewards (maintained)
            'tool_efficiency': 5.0,
            'execution_speed_bonus': 2.0,
            
            # Learning incentives
            'exploration_bonus': 0.5,  # Small, to encourage optimal learning
            'pattern_discovery': 3.0,  # For finding optimal patterns
        }
        
        # Track optimal tool patterns
        self.optimal_patterns = {}
        self.learned_associations = {}
        
        logger.info("Initialized Tool-Optimized Reward Calculator")
        logger.info(f"Weights prioritize tool accuracy: {self.base_weights}")
    
    def calculate_reward(self, 
                        execution_results: List[ExecutionMetrics],
                        context: Dict[str, Any],
                        optimal_tools: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward with focus on tool selection accuracy.
        
        Args:
            execution_results: Actual execution results
            context: Query context
            optimal_tools: Ground truth optimal tools for this query
            
        Returns:
            (total_reward, breakdown)
        """
        if not execution_results:
            return -5.0, {'no_execution': -5.0}
        
        breakdown = {}
        
        # 1. Task completion component (reduced importance)
        task_reward = self._calculate_task_reward(execution_results)
        breakdown['task_completion'] = task_reward
        
        # 2. Tool selection accuracy (PRIMARY FOCUS)
        if optimal_tools:
            tool_accuracy_reward = self._calculate_tool_accuracy_reward(
                execution_results, optimal_tools
            )
            breakdown.update(tool_accuracy_reward)
        else:
            # Fallback: reward based on tool effectiveness
            tool_effectiveness = self._calculate_tool_effectiveness(execution_results)
            breakdown['tool_effectiveness'] = tool_effectiveness
        
        # 3. Efficiency bonuses
        efficiency_bonus = self._calculate_efficiency_bonus(execution_results)
        breakdown['efficiency'] = efficiency_bonus
        
        # 4. Pattern learning bonus
        pattern_bonus = self._track_pattern_learning(execution_results, context)
        breakdown['pattern_learning'] = pattern_bonus
        
        # Calculate total
        total_reward = sum(breakdown.values())
        
        # Clip to reasonable range
        total_reward = np.clip(total_reward, -20.0, 50.0)
        breakdown['total'] = total_reward
        
        logger.debug(f"Tool-optimized reward: {total_reward:.2f}, breakdown: {breakdown}")
        
        return total_reward, breakdown
    
    def _calculate_task_reward(self, results: List[ExecutionMetrics]) -> float:
        """Calculate basic task completion reward (reduced importance)."""
        success_count = sum(1 for r in results if r.success)
        partial_count = sum(1 for r in results if r.partial_success and not r.success)
        total_count = len(results)
        
        if success_count == total_count:
            return self.base_weights['task_success']
        elif success_count > 0:
            return self.base_weights['task_success'] * (success_count / total_count)
        elif partial_count > 0:
            return self.base_weights['partial_success'] * (partial_count / total_count)
        else:
            return self.base_weights['task_failure']
    
    def _calculate_tool_accuracy_reward(self, 
                                       results: List[ExecutionMetrics],
                                       optimal_tools: List[str]) -> Dict[str, float]:
        """
        Calculate tool selection accuracy reward (PRIMARY COMPONENT).
        
        This is where the magic happens - strong rewards for optimal selection.
        """
        rewards = {}
        
        actual_tools = [r.tool_id for r in results]
        optimal_set = set(optimal_tools)
        actual_set = set(actual_tools)
        
        # 1. Exact match bonus (huge reward for perfect selection)
        if actual_set == optimal_set and len(actual_tools) == len(optimal_tools):
            rewards['exact_match'] = self.base_weights['exact_match_bonus']
            rewards['optimal_tools'] = self.base_weights['optimal_tool_match'] * len(optimal_tools)
            return rewards  # Perfect score, no penalties
        
        # 2. Partial match rewards
        correct_tools = actual_set.intersection(optimal_set)
        rewards['optimal_tools'] = self.base_weights['optimal_tool_match'] * len(correct_tools)
        
        # 3. Penalties for wrong tools
        wrong_tools = actual_set - optimal_set
        rewards['wrong_tools'] = self.base_weights['suboptimal_tool_penalty'] * len(wrong_tools)
        
        # 4. Distance penalty (how far from optimal?)
        missing_tools = optimal_set - actual_set
        extra_tools = len(actual_tools) - len(optimal_tools)
        
        distance = len(missing_tools) + abs(extra_tools)
        rewards['distance_penalty'] = self.base_weights['tool_distance_penalty'] * distance
        
        # 5. Unnecessary tool penalty (tools that don't help at all)
        unnecessary = sum(1 for r in results 
                         if r.tool_id in wrong_tools and not r.success and not r.partial_success)
        rewards['unnecessary_penalty'] = self.base_weights['unnecessary_tool_penalty'] * unnecessary
        
        return rewards
    
    def _calculate_tool_effectiveness(self, results: List[ExecutionMetrics]) -> float:
        """Fallback: reward effective tool use when no ground truth available."""
        effectiveness = 0.0
        
        for result in results:
            if result.success:
                effectiveness += 2.0
            elif result.partial_success:
                effectiveness += 0.5 * result.completion_percentage
            else:
                effectiveness -= 0.5
        
        return effectiveness
    
    def _calculate_efficiency_bonus(self, results: List[ExecutionMetrics]) -> float:
        """Reward efficient execution (speed, resource usage)."""
        bonus = 0.0
        
        # Speed bonus
        avg_time = np.mean([r.execution_time_ms for r in results])
        if avg_time < 100:  # Fast execution
            bonus += self.base_weights['execution_speed_bonus']
        
        # Efficiency ratio (successful tools / total tools)
        success_ratio = sum(1 for r in results if r.success) / len(results)
        if success_ratio > 0.8:
            bonus += self.base_weights['tool_efficiency'] * success_ratio
        
        return bonus
    
    def _track_pattern_learning(self, 
                               results: List[ExecutionMetrics],
                               context: Dict[str, Any]) -> float:
        """Track and reward learning of optimal patterns."""
        # Extract query pattern (simplified)
        query_type = context.get('intent', 'unknown')
        tools_used = tuple(sorted([r.tool_id for r in results]))
        
        # Check if this is a successful pattern
        success_rate = sum(1 for r in results if r.success) / len(results)
        
        if success_rate > 0.8:
            # Record successful pattern
            if query_type not in self.optimal_patterns:
                self.optimal_patterns[query_type] = {}
            
            if tools_used not in self.optimal_patterns[query_type]:
                # New pattern discovered!
                self.optimal_patterns[query_type][tools_used] = 1
                return self.base_weights['pattern_discovery']
            else:
                # Reinforcing known pattern
                self.optimal_patterns[query_type][tools_used] += 1
                return self.base_weights['exploration_bonus']
        
        return 0.0
    
    def get_learned_patterns(self) -> Dict[str, Any]:
        """Return learned optimal patterns for analysis."""
        return {
            'optimal_patterns': self.optimal_patterns,
            'pattern_count': sum(len(p) for p in self.optimal_patterns.values())
        }