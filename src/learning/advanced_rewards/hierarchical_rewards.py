"""Hierarchical goal-based reward calculation.

This module implements multi-level goal structures with rewards
for achieving high-level objectives and sub-goals.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
import logging

from .base_strategy import BaseRewardStrategy, RewardStrategyResult

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger(__name__)


class GoalType(Enum):
    """Types of goals in the hierarchy."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    MILESTONE = "milestone"
    SUBTASK = "subtask"


@dataclass
class Goal:
    """Represents a goal in the hierarchy."""
    id: str
    name: str
    type: GoalType
    parent_id: Optional[str] = None
    required_tools: Set[str] = None
    success_criteria: Dict[str, Any] = None
    weight: float = 1.0
    achieved: bool = False
    progress: float = 0.0


class HierarchicalRewardCalculator(BaseRewardStrategy):
    """Calculates rewards based on hierarchical goal achievement."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hierarchical reward calculator.
        
        Args:
            config: Configuration with keys:
                - goal_weights: Weights for different goal types
                - milestone_bonus: Extra reward for milestone achievement
                - progress_reward: Reward for partial goal progress
                - subtask_completion_threshold: Threshold for subtask rewards
        """
        super().__init__(config)
        
    def _initialize_strategy(self):
        """Initialize hierarchical reward components."""
        self.goal_weights = self.config.get('goal_weights', {
            'primary': 1.0,
            'secondary': 0.5,
            'tertiary': 0.25,
            'milestone': 0.4,
            'subtask': 0.1
        })
        
        self.milestone_bonus = self.config.get('milestone_bonus', 0.5)
        self.progress_reward = self.config.get('progress_reward', True)
        self.subtask_threshold = self.config.get('subtask_completion_threshold', 0.8)
        
        # Goal hierarchy
        self.goals: Dict[str, Goal] = {}
        self.goal_tree: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
        # Task decomposition mappings
        self.task_to_goals: Dict[str, List[str]] = {}
        self.tool_to_goals: Dict[str, List[str]] = {}
        
        # Initialize default goals
        self._initialize_default_goals()
        
        logger.info(f"Initialized HierarchicalRewardCalculator with weights: {self.goal_weights}")
    
    def _initialize_default_goals(self):
        """Initialize default goal hierarchy."""
        # Primary goals
        self.add_goal(Goal(
            id="complete_task",
            name="Complete User Task",
            type=GoalType.PRIMARY,
            weight=1.0
        ))
        
        # Secondary goals
        self.add_goal(Goal(
            id="find_tools",
            name="Find Appropriate Tools",
            type=GoalType.SECONDARY,
            parent_id="complete_task",
            weight=0.5
        ))
        
        self.add_goal(Goal(
            id="execute_efficiently",
            name="Execute Efficiently",
            type=GoalType.SECONDARY,
            parent_id="complete_task",
            weight=0.5
        ))
        
        # Tertiary goals
        self.add_goal(Goal(
            id="minimize_errors",
            name="Minimize Execution Errors",
            type=GoalType.TERTIARY,
            parent_id="execute_efficiently",
            weight=0.25
        ))
        
        self.add_goal(Goal(
            id="optimize_resources",
            name="Optimize Resource Usage",
            type=GoalType.TERTIARY,
            parent_id="execute_efficiently",
            weight=0.25
        ))
    
    def calculate(self, 
                  state: np.ndarray,
                  action: List[str],
                  next_state: np.ndarray,
                  execution_results: List[Any],
                  context: Dict[str, Any]) -> RewardStrategyResult:
        """Calculate hierarchical goal-based rewards."""
        start_time = time.time()
        
        # Identify relevant goals based on context
        relevant_goals = self._identify_relevant_goals(context, action)
        
        # Update goal progress based on execution results
        self._update_goal_progress(relevant_goals, execution_results, action)
        
        # Calculate rewards for each goal level
        primary_reward = self._calculate_goal_rewards(GoalType.PRIMARY, relevant_goals)
        secondary_reward = self._calculate_goal_rewards(GoalType.SECONDARY, relevant_goals)
        tertiary_reward = self._calculate_goal_rewards(GoalType.TERTIARY, relevant_goals)
        milestone_reward = self._calculate_milestone_rewards(relevant_goals)
        subtask_reward = self._calculate_subtask_rewards(relevant_goals)
        
        # Calculate progress rewards
        progress_bonus = self._calculate_progress_bonus(relevant_goals) if self.progress_reward else 0.0
        
        # Combine rewards hierarchically
        final_reward = (
            primary_reward +
            secondary_reward +
            tertiary_reward +
            milestone_reward +
            subtask_reward +
            progress_bonus
        )
        
        # Apply goal completion cascading
        final_reward = self._apply_cascading_bonuses(final_reward, relevant_goals)
        
        # Clip to reasonable range
        final_reward = self._clip_reward(final_reward)
        
        computation_time = (time.time() - start_time) * 1000
        
        return RewardStrategyResult(
            reward=final_reward,
            components={
                'primary_reward': primary_reward,
                'secondary_reward': secondary_reward,
                'tertiary_reward': tertiary_reward,
                'milestone_reward': milestone_reward,
                'subtask_reward': subtask_reward,
                'progress_bonus': progress_bonus
            },
            metadata={
                'relevant_goals': len(relevant_goals),
                'achieved_goals': sum(1 for g in relevant_goals if self.goals[g].achieved),
                'average_progress': np.mean([self.goals[g].progress for g in relevant_goals]) if relevant_goals else 0.0
            },
            computation_time_ms=computation_time
        )
    
    def add_goal(self, goal: Goal):
        """Add a goal to the hierarchy."""
        self.goals[goal.id] = goal
        
        # Update tree structure
        if goal.parent_id:
            if goal.parent_id not in self.goal_tree:
                self.goal_tree[goal.parent_id] = []
            self.goal_tree[goal.parent_id].append(goal.id)
        
        # Update tool mappings
        if goal.required_tools:
            for tool in goal.required_tools:
                if tool not in self.tool_to_goals:
                    self.tool_to_goals[tool] = []
                self.tool_to_goals[tool].append(goal.id)
    
    def _identify_relevant_goals(self, context: Dict[str, Any], action: List[str]) -> List[str]:
        """Identify goals relevant to current context and action."""
        relevant_goals = set()
        
        # Add goals based on task type
        task_type = context.get('task_type', 'general')
        if task_type in self.task_to_goals:
            relevant_goals.update(self.task_to_goals[task_type])
        
        # Add goals based on tools used
        for tool in action:
            if tool in self.tool_to_goals:
                relevant_goals.update(self.tool_to_goals[tool])
        
        # Add parent goals of relevant goals
        for goal_id in list(relevant_goals):
            parent_goals = self._get_parent_goals(goal_id)
            relevant_goals.update(parent_goals)
        
        # Always include primary goals
        primary_goals = [g_id for g_id, g in self.goals.items() if g.type == GoalType.PRIMARY]
        relevant_goals.update(primary_goals)
        
        return list(relevant_goals)
    
    def _get_parent_goals(self, goal_id: str) -> List[str]:
        """Get all parent goals in the hierarchy."""
        parents = []
        current = self.goals.get(goal_id)
        
        while current and current.parent_id:
            parents.append(current.parent_id)
            current = self.goals.get(current.parent_id)
        
        return parents
    
    def _update_goal_progress(self, goal_ids: List[str], execution_results: List[Any], action: List[str]):
        """Update progress for relevant goals."""
        for goal_id in goal_ids:
            goal = self.goals[goal_id]
            
            # Skip if already achieved
            if goal.achieved:
                continue
            
            # Update based on success criteria
            if goal.success_criteria:
                progress = self._evaluate_success_criteria(goal.success_criteria, execution_results, action)
            else:
                # Default progress based on execution success
                if execution_results:
                    success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
                    progress = success_rate
                else:
                    progress = 0.0
            
            # Update goal progress
            goal.progress = max(goal.progress, progress)
            
            # Check if goal is achieved
            if goal.progress >= 1.0:
                goal.achieved = True
                logger.info(f"Goal achieved: {goal.name}")
            
            # Update parent goal progress
            if goal.parent_id:
                self._update_parent_progress(goal.parent_id)
    
    def _evaluate_success_criteria(self, criteria: Dict[str, Any], 
                                  execution_results: List[Any], 
                                  action: List[str]) -> float:
        """Evaluate goal success criteria."""
        progress_scores = []
        
        # Check tool requirements
        if 'required_tools' in criteria:
            required = set(criteria['required_tools'])
            used = set(action)
            tool_progress = len(required.intersection(used)) / len(required) if required else 1.0
            progress_scores.append(tool_progress)
        
        # Check execution success rate
        if 'min_success_rate' in criteria:
            if execution_results:
                success_rate = sum(1 for r in execution_results if hasattr(r, 'success') and r.success) / len(execution_results)
                rate_progress = success_rate / criteria['min_success_rate']
                progress_scores.append(min(1.0, rate_progress))
            else:
                progress_scores.append(0.0)
        
        # Check result quality
        if 'min_quality' in criteria:
            if execution_results:
                avg_quality = np.mean([getattr(r, 'result_quality', 0.5) for r in execution_results])
                quality_progress = avg_quality / criteria['min_quality']
                progress_scores.append(min(1.0, quality_progress))
        
        return np.mean(progress_scores) if progress_scores else 0.0
    
    def _update_parent_progress(self, parent_id: str):
        """Update parent goal progress based on children."""
        if parent_id not in self.goal_tree:
            return
        
        child_ids = self.goal_tree[parent_id]
        if not child_ids:
            return
        
        # Calculate average progress of children
        child_progress = [self.goals[child_id].progress for child_id in child_ids]
        avg_progress = np.mean(child_progress)
        
        parent = self.goals[parent_id]
        parent.progress = max(parent.progress, avg_progress)
        
        if parent.progress >= 1.0:
            parent.achieved = True
    
    def _calculate_goal_rewards(self, goal_type: GoalType, goal_ids: List[str]) -> float:
        """Calculate rewards for goals of a specific type."""
        reward = 0.0
        type_weight = self.goal_weights.get(goal_type.value, 0.1)
        
        for goal_id in goal_ids:
            goal = self.goals[goal_id]
            if goal.type != goal_type:
                continue
            
            if goal.achieved:
                # Full reward for achieved goals
                reward += goal.weight * type_weight
            elif self.progress_reward and goal.progress > 0:
                # Partial reward for progress
                reward += goal.weight * type_weight * goal.progress * 0.5
        
        return reward
    
    def _calculate_milestone_rewards(self, goal_ids: List[str]) -> float:
        """Calculate special milestone rewards."""
        reward = 0.0
        
        for goal_id in goal_ids:
            goal = self.goals[goal_id]
            if goal.type == GoalType.MILESTONE and goal.achieved:
                reward += self.milestone_bonus * goal.weight
        
        return reward
    
    def _calculate_subtask_rewards(self, goal_ids: List[str]) -> float:
        """Calculate rewards for subtask completion."""
        reward = 0.0
        
        for goal_id in goal_ids:
            goal = self.goals[goal_id]
            if goal.type == GoalType.SUBTASK and goal.progress >= self.subtask_threshold:
                reward += self.goal_weights['subtask'] * goal.weight * goal.progress
        
        return reward
    
    def _calculate_progress_bonus(self, goal_ids: List[str]) -> float:
        """Calculate bonus for overall progress."""
        if not goal_ids:
            return 0.0
        
        # Calculate weighted progress
        total_weight = 0.0
        weighted_progress = 0.0
        
        for goal_id in goal_ids:
            goal = self.goals[goal_id]
            total_weight += goal.weight
            weighted_progress += goal.progress * goal.weight
        
        if total_weight > 0:
            avg_progress = weighted_progress / total_weight
            # Non-linear progress bonus
            return 0.2 * (avg_progress ** 2)
        
        return 0.0
    
    def _apply_cascading_bonuses(self, base_reward: float, goal_ids: List[str]) -> float:
        """Apply cascading bonuses for achieving parent-child goal chains."""
        bonus_multiplier = 1.0
        
        # Check for complete chains
        for goal_id in goal_ids:
            if self._is_complete_chain(goal_id):
                bonus_multiplier += 0.1
        
        return base_reward * bonus_multiplier
    
    def _is_complete_chain(self, goal_id: str) -> bool:
        """Check if a goal and all its children are achieved."""
        goal = self.goals[goal_id]
        if not goal.achieved:
            return False
        
        if goal_id in self.goal_tree:
            for child_id in self.goal_tree[goal_id]:
                if not self._is_complete_chain(child_id):
                    return False
        
        return True
    
    def update_parameters(self, feedback: Dict[str, Any]):
        """Update hierarchical parameters based on feedback."""
        # Adjust goal weights based on task success patterns
        if 'task_success_by_type' in feedback:
            for task_type, success_rate in feedback['task_success_by_type'].items():
                if success_rate < 0.5:
                    # Increase weight for struggling goal types
                    self._adjust_goal_type_weight(task_type, 1.1)
                elif success_rate > 0.8:
                    # Slightly decrease weight for mastered goal types
                    self._adjust_goal_type_weight(task_type, 0.95)
        
        # Add new goals based on discovered patterns
        if 'discovered_patterns' in feedback:
            self._create_goals_from_patterns(feedback['discovered_patterns'])
    
    def _adjust_goal_type_weight(self, goal_type: str, multiplier: float):
        """Adjust weight for a goal type."""
        if goal_type in self.goal_weights:
            self.goal_weights[goal_type] = np.clip(
                self.goal_weights[goal_type] * multiplier,
                0.1, 2.0
            )
    
    def _create_goals_from_patterns(self, patterns: List[Dict[str, Any]]):
        """Create new goals from discovered patterns."""
        for pattern in patterns:
            if pattern.get('confidence', 0) > 0.8:
                # Create a new subtask goal for the pattern
                goal = Goal(
                    id=f"pattern_{pattern['id']}",
                    name=f"Execute Pattern: {pattern['name']}",
                    type=GoalType.SUBTASK,
                    required_tools=set(pattern.get('tools', [])),
                    weight=pattern.get('importance', 0.5)
                )
                self.add_goal(goal)
    
    def reset_goal_progress(self):
        """Reset progress for all goals."""
        for goal in self.goals.values():
            goal.progress = 0.0
            goal.achieved = False
    
    def save_state(self) -> Dict[str, Any]:
        """Save strategy state including goal hierarchy."""
        state = super().save_state()
        state.update({
            'goals': {g_id: {
                'name': g.name,
                'type': g.type.value,
                'parent_id': g.parent_id,
                'weight': g.weight,
                'progress': g.progress,
                'achieved': g.achieved
            } for g_id, g in self.goals.items()},
            'goal_weights': self.goal_weights
        })
        return state