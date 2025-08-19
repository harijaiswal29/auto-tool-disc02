"""Baseline strategies for evaluation comparison.

This module implements various baseline strategies to compare against the
Q-learning and DQN approaches. Each baseline represents a different approach
to tool selection without sophisticated learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import logging
import json
import hashlib

from learning.q_learning_engine import StateRepresentation, ActionSpace
from utils.logger import get_logger

logger = get_logger(__name__)


class BaselineStrategy(ABC):
    """Abstract base class for baseline strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.performance_history = []
        self.selection_history = []
        
    @abstractmethod
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select tools based on the baseline strategy.
        
        Args:
            state: Current state representation
            available_tools: List of available tool IDs
            constraints: Tool constraints (conflicts, requirements)
            
        Returns:
            List of selected tool IDs
        """
        pass
    
    def update(self, state: np.ndarray, action: List[str], reward: float, 
               next_state: np.ndarray):
        """Update strategy based on experience (most baselines don't learn).
        
        Args:
            state: Previous state
            action: Tools selected
            reward: Reward received
            next_state: New state after action
        """
        # Store performance for analysis
        self.performance_history.append({
            'action': action,
            'reward': reward
        })
        self.selection_history.extend(action)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the baseline."""
        if not self.performance_history:
            return {'avg_reward': 0, 'total_episodes': 0}
            
        rewards = [h['reward'] for h in self.performance_history]
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'total_episodes': len(self.performance_history),
            'tool_frequency': Counter(self.selection_history)
        }


class RandomSelectionBaseline(BaselineStrategy):
    """Randomly select tools from available options."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RandomSelection", config)
        self.max_tools = config.get('max_tools', 3)
        
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Randomly select 1-3 tools."""
        if not available_tools:
            return []
            
        # Random number of tools to select (1 to max_tools)
        num_tools = random.randint(1, min(len(available_tools), self.max_tools))
        
        # Randomly select tools
        selected = random.sample(available_tools, num_tools)
        
        # Basic constraint checking
        selected = self._apply_constraints(selected, constraints)
        
        return selected
    
    def _apply_constraints(self, selected: List[str], constraints: Dict[str, Any]) -> List[str]:
        """Apply basic constraints to random selection."""
        conflicts = constraints.get('conflicts', {})
        
        # Remove conflicting tools
        final_selection = []
        for tool in selected:
            has_conflict = False
            for existing_tool in final_selection:
                if (tool in conflicts.get(existing_tool, []) or 
                    existing_tool in conflicts.get(tool, [])):
                    has_conflict = True
                    break
            if not has_conflict:
                final_selection.append(tool)
                
        return final_selection


class MostPopularToolsBaseline(BaselineStrategy):
    """Select tools based on historical usage frequency."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MostPopularTools", config)
        self.max_tools = config.get('max_tools', 3)
        self.tool_popularity = defaultdict(int)
        self.min_history = config.get('min_history', 10)
        
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select most frequently used tools."""
        if not available_tools:
            return []
            
        # If not enough history, fall back to random
        if sum(self.tool_popularity.values()) < self.min_history:
            num_tools = random.randint(1, min(len(available_tools), self.max_tools))
            return random.sample(available_tools, num_tools)
        
        # Sort tools by popularity
        tool_scores = [(tool, self.tool_popularity.get(tool, 0)) 
                      for tool in available_tools]
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top tools
        selected = []
        for tool, _ in tool_scores:
            if len(selected) >= self.max_tools:
                break
            if self._check_constraints(tool, selected, constraints):
                selected.append(tool)
                
        # Ensure at least one tool is selected
        if not selected and available_tools:
            selected = [available_tools[0]]
            
        return selected
    
    def _check_constraints(self, tool: str, selected: List[str], 
                          constraints: Dict[str, Any]) -> bool:
        """Check if tool can be added given constraints."""
        conflicts = constraints.get('conflicts', {})
        
        for existing_tool in selected:
            if (tool in conflicts.get(existing_tool, []) or 
                existing_tool in conflicts.get(tool, [])):
                return False
        return True
    
    def update(self, state: np.ndarray, action: List[str], reward: float, 
               next_state: np.ndarray):
        """Update tool popularity based on usage."""
        super().update(state, action, reward, next_state)
        
        # Update popularity counts
        for tool in action:
            self.tool_popularity[tool] += 1


class FixedPolicyBaseline(BaselineStrategy):
    """Pre-defined tool mappings for intent types."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FixedPolicy", config)
        # Disable PCA to maintain full 476 dimensions
        self.state_representation = StateRepresentation(use_pca=False)
        
        # Define fixed policies for different intent patterns
        self.intent_policies = {
            'file_search': ['filesystem_mcp', 'search_mcp'],
            'data_query': ['sqlite_mcp', 'postgres_mcp'],
            'code_analysis': ['github_mcp', 'filesystem_mcp'],
            'web_search': ['search_mcp'],
            'weather_query': ['weather_mcp'],
            'default': ['filesystem_mcp']
        }
        
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select tools based on fixed policy rules."""
        # Extract intent features from state
        intent_type = self._classify_intent(state)
        
        # Get policy for intent type
        policy_tools = self.intent_policies.get(intent_type, 
                                               self.intent_policies['default'])
        
        # Filter to available tools
        selected = [tool for tool in policy_tools if tool in available_tools]
        
        # Apply constraints
        selected = self._apply_constraints(selected, constraints)
        
        # Ensure at least one tool
        if not selected and available_tools:
            selected = [available_tools[0]]
            
        return selected
    
    def _classify_intent(self, state: np.ndarray) -> str:
        """Simple intent classification based on state features."""
        # This is a simplified version - in reality would analyze the intent vector
        # For now, return a random intent type for demonstration
        intent_types = list(self.intent_policies.keys())
        return random.choice(intent_types[:-1])  # Exclude 'default'
    
    def _apply_constraints(self, selected: List[str], constraints: Dict[str, Any]) -> List[str]:
        """Apply constraints to selection."""
        conflicts = constraints.get('conflicts', {})
        final_selection = []
        
        for tool in selected:
            has_conflict = False
            for existing_tool in final_selection:
                if (tool in conflicts.get(existing_tool, []) or 
                    existing_tool in conflicts.get(tool, [])):
                    has_conflict = True
                    break
            if not has_conflict:
                final_selection.append(tool)
                
        return final_selection


class GreedySingleToolBaseline(BaselineStrategy):
    """Always select the single best-performing tool."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("GreedySingleTool", config)
        self.tool_rewards = defaultdict(list)
        self.min_history = config.get('min_history', 5)
        
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select single best tool based on average reward."""
        if not available_tools:
            return []
            
        # Calculate average rewards
        tool_avg_rewards = {}
        for tool in available_tools:
            if tool in self.tool_rewards and len(self.tool_rewards[tool]) >= self.min_history:
                tool_avg_rewards[tool] = np.mean(self.tool_rewards[tool])
        
        # If not enough history, select randomly
        if not tool_avg_rewards:
            return [random.choice(available_tools)]
        
        # Select best tool
        best_tool = max(tool_avg_rewards.items(), key=lambda x: x[1])[0]
        return [best_tool]
    
    def update(self, state: np.ndarray, action: List[str], reward: float, 
               next_state: np.ndarray):
        """Update tool rewards."""
        super().update(state, action, reward, next_state)
        
        # Distribute reward among selected tools
        if action:
            reward_per_tool = reward / len(action)
            for tool in action:
                self.tool_rewards[tool].append(reward_per_tool)


class ContextAgnosticQLearningBaseline(BaselineStrategy):
    """Q-Learning without context features - limited state representation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ContextAgnosticQLearning", config)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.9)
        self.exploration_rate = config.get('exploration_rate', 0.2)
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_space = ActionSpace(max_tools=config.get('max_tools', 3))
        
    async def select_tools(self, state: np.ndarray, available_tools: List[str], 
                          constraints: Dict[str, Any]) -> List[str]:
        """Select tools using simplified Q-learning."""
        if not available_tools:
            return []
            
        # Simplify state to just a hash of available tools
        state_key = self._get_simple_state_key(available_tools)
        
        # Get valid actions
        valid_actions = self.action_space.get_valid_actions(available_tools, constraints)
        if not valid_actions:
            return [available_tools[0]] if available_tools else []
        
        # Epsilon-greedy selection
        if random.random() < self.exploration_rate:
            # Explore: random action
            action = random.choice(valid_actions)
        else:
            # Exploit: best Q-value
            action = self._get_best_action(state_key, valid_actions)
            
        return list(action)
    
    def _get_simple_state_key(self, available_tools: List[str]) -> str:
        """Create simplified state key ignoring context."""
        # Sort tools for consistent hashing
        tools_str = ','.join(sorted(available_tools))
        return hashlib.md5(tools_str.encode()).hexdigest()[:8]
    
    def _get_best_action(self, state_key: str, valid_actions: List[Tuple[str, ...]]) -> Tuple[str, ...]:
        """Get action with highest Q-value."""
        best_action = valid_actions[0]
        best_q_value = self.q_table[state_key][str(best_action)]
        
        for action in valid_actions[1:]:
            q_value = self.q_table[state_key][str(action)]
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
                
        return best_action
    
    def update(self, state: np.ndarray, action: List[str], reward: float, 
               next_state: np.ndarray):
        """Update Q-values."""
        super().update(state, action, reward, next_state)
        
        # Note: This is simplified - in real implementation would need
        # access to available tools for both states
        # For now, just store the experience
        
    def update_q_value(self, state_key: str, action_key: str, reward: float, 
                      next_state_key: str, next_valid_actions: List[Tuple[str, ...]]):
        """Update Q-value for state-action pair."""
        current_q = self.q_table[state_key][action_key]
        
        # Get max Q-value for next state
        if next_valid_actions:
            max_next_q = max(self.q_table[next_state_key][str(a)] 
                           for a in next_valid_actions)
        else:
            max_next_q = 0
            
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action_key] = new_q