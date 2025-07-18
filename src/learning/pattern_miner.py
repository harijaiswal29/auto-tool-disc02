"""Pattern Miner for discovering tool usage patterns.

This module implements pattern mining algorithms to discover frequent tool
combinations and sequences that lead to successful outcomes.
"""

import asyncio
import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional, Any
import hashlib

import aiosqlite
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("PatternMiner")


@dataclass
class Pattern:
    """Represents a discovered pattern."""
    pattern_type: str  # 'sequential', 'combination', 'temporal'
    tool_sequence: List[str]
    support: float
    confidence: float
    lift: float
    contexts: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage."""
        return {
            'pattern_type': self.pattern_type,
            'tool_sequence': json.dumps(self.tool_sequence),
            'support': self.support,
            'confidence': self.confidence,
            'lift': self.lift,
            'contexts': json.dumps(self.contexts),
            'discovered_at': self.discovered_at.isoformat(),
            'usage_count': self.usage_count
        }
    
    def get_hash(self) -> str:
        """Generate unique hash for pattern."""
        pattern_str = f"{self.pattern_type}:{':'.join(sorted(self.tool_sequence))}"
        return hashlib.md5(pattern_str.encode()).hexdigest()


@dataclass
class ExecutionSequence:
    """Represents a sequence of tool executions."""
    execution_id: str
    tools: List[str]
    success: bool
    reward: float
    context: Dict[str, Any]
    timestamp: datetime


class PatternMiner:
    """Discovers and mines patterns in tool usage."""
    
    def __init__(self, config: Any, min_support: float = 0.1, min_confidence: float = 0.8):
        """Initialize pattern miner.
        
        Args:
            config: System configuration
            min_support: Minimum support threshold for patterns
            min_confidence: Minimum confidence threshold for patterns
        """
        self.config = config
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = 1.0  # Only consider patterns with positive correlation
        
        # Pattern storage
        self.discovered_patterns: Dict[str, Pattern] = {}
        self.pattern_cache: Dict[str, float] = {}  # Cache for quick lookup
        
        # Database path
        self.db_path = config.get('database', {}).get('path', './data/registry/tool_registry.db')
        
        # Mining configuration
        self.max_pattern_length = config.get('pattern_mining', {}).get('max_pattern_length', 5)
        self.time_window = timedelta(days=config.get('pattern_mining', {}).get('time_window_days', 30))
        
        logger.info(f"PatternMiner initialized with support={min_support}, confidence={min_confidence}")
    
    async def extract_sequences(self, time_window: Optional[timedelta] = None) -> List[ExecutionSequence]:
        """Extract execution sequences from database.
        
        Args:
            time_window: Time window to consider (default: self.time_window)
            
        Returns:
            List of execution sequences
        """
        if time_window is None:
            time_window = self.time_window
            
        cutoff_time = datetime.now() - time_window
        sequences = []
        
        async with aiosqlite.connect(self.db_path) as db:
            query = """
                SELECT id, query, intent, tools_used, success, reward, created_at
                FROM execution_history
                WHERE created_at > ? AND tools_used IS NOT NULL
                ORDER BY created_at DESC
            """
            
            async with db.execute(query, (cutoff_time.isoformat(),)) as cursor:
                async for row in cursor:
                    try:
                        tools = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                        intent = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                        
                        sequence = ExecutionSequence(
                            execution_id=row[0],
                            tools=tools if isinstance(tools, list) else [tools],
                            success=bool(row[4]),
                            reward=float(row[5]) if row[5] is not None else 0.0,
                            context={'query': row[1], 'intent': intent},
                            timestamp=datetime.fromisoformat(row[6])
                        )
                        sequences.append(sequence)
                    except Exception as e:
                        logger.warning(f"Error parsing execution sequence {row[0]}: {e}")
                        continue
        
        logger.info(f"Extracted {len(sequences)} execution sequences from last {time_window.days} days")
        return sequences
    
    def _generate_subsequences(self, sequence: List[str], max_length: int) -> List[List[str]]:
        """Generate all subsequences up to max_length."""
        subsequences = []
        seq_len = len(sequence)
        
        for length in range(1, min(max_length + 1, seq_len + 1)):
            for start in range(seq_len - length + 1):
                subsequences.append(sequence[start:start + length])
        
        return subsequences
    
    def _calculate_support(self, pattern: List[str], sequences: List[ExecutionSequence]) -> float:
        """Calculate support for a pattern."""
        if not sequences:
            return 0.0
            
        count = 0
        for seq in sequences:
            # Check if pattern is a subsequence of seq.tools
            if self._is_subsequence(pattern, seq.tools):
                count += 1
        
        return count / len(sequences)
    
    def _is_subsequence(self, pattern: List[str], sequence: List[str]) -> bool:
        """Check if pattern is a subsequence of sequence."""
        if len(pattern) > len(sequence):
            return False
            
        # For sequential patterns, order matters
        p_idx = 0
        for tool in sequence:
            if p_idx < len(pattern) and tool == pattern[p_idx]:
                p_idx += 1
        
        return p_idx == len(pattern)
    
    def _calculate_confidence(self, pattern: List[str], sequences: List[ExecutionSequence]) -> float:
        """Calculate confidence for a pattern."""
        if len(pattern) < 2:
            return 1.0
            
        # Confidence = P(success | pattern) = successful_with_pattern / total_with_pattern
        total_with_pattern = 0
        successful_with_pattern = 0
        
        for seq in sequences:
            if self._is_subsequence(pattern, seq.tools):
                total_with_pattern += 1
                if seq.success or seq.reward > 0:
                    successful_with_pattern += 1
        
        if total_with_pattern == 0:
            return 0.0
            
        return successful_with_pattern / total_with_pattern
    
    def _calculate_lift(self, pattern: List[str], sequences: List[ExecutionSequence]) -> float:
        """Calculate lift for a pattern."""
        if len(pattern) < 2:
            return 1.0
            
        # Lift = P(B|A) / P(B) where A is first part, B is last tool
        # This measures how much more likely B is to occur given A
        
        total_sequences = len(sequences)
        if total_sequences == 0:
            return 1.0
            
        # Count occurrences
        pattern_count = sum(1 for seq in sequences if self._is_subsequence(pattern, seq.tools))
        last_tool_count = sum(1 for seq in sequences if pattern[-1] in seq.tools)
        prefix_count = sum(1 for seq in sequences if self._is_subsequence(pattern[:-1], seq.tools))
        
        if prefix_count == 0 or last_tool_count == 0:
            return 1.0
            
        # P(pattern) / (P(prefix) * P(last_tool))
        observed_prob = pattern_count / total_sequences
        expected_prob = (prefix_count / total_sequences) * (last_tool_count / total_sequences)
        
        if expected_prob == 0:
            return 1.0
            
        return observed_prob / expected_prob
    
    async def mine_sequential_patterns(self, sequences: List[ExecutionSequence]) -> List[Pattern]:
        """Mine sequential patterns using PrefixSpan algorithm.
        
        Args:
            sequences: List of execution sequences
            
        Returns:
            List of discovered patterns
        """
        logger.info("Starting sequential pattern mining...")
        
        # Filter successful sequences for pattern mining
        successful_sequences = [seq for seq in sequences if seq.success or seq.reward > 0]
        
        if not successful_sequences:
            logger.warning("No successful sequences found for pattern mining")
            return []
        
        # Count single item frequencies
        item_counts = Counter()
        for seq in successful_sequences:
            for tool in seq.tools:
                item_counts[tool] += 1
        
        # Find frequent single items
        total_sequences = len(successful_sequences)
        frequent_items = {
            item: count/total_sequences 
            for item, count in item_counts.items() 
            if count/total_sequences >= self.min_support
        }
        
        logger.info(f"Found {len(frequent_items)} frequent single items")
        
        # Mine patterns using simplified PrefixSpan
        patterns = []
        
        # Start with single items
        for item, support in frequent_items.items():
            pattern = Pattern(
                pattern_type='sequential',
                tool_sequence=[item],
                support=support,
                confidence=1.0,  # Single items have confidence 1.0
                lift=1.0
            )
            patterns.append(pattern)
        
        # Mine longer patterns
        for length in range(2, self.max_pattern_length + 1):
            logger.info(f"Mining patterns of length {length}")
            
            # Generate candidate patterns
            candidates = set()
            for seq in successful_sequences:
                if len(seq.tools) >= length:
                    subsequences = self._generate_subsequences(seq.tools, length)
                    for subseq in subsequences:
                        # Only consider if all items are frequent
                        if all(item in frequent_items for item in subseq):
                            candidates.add(tuple(subseq))
            
            # Calculate metrics for candidates
            for candidate in candidates:
                pattern_list = list(candidate)
                support = self._calculate_support(pattern_list, sequences)
                
                if support >= self.min_support:
                    confidence = self._calculate_confidence(pattern_list, sequences)
                    
                    if confidence >= self.min_confidence:
                        lift = self._calculate_lift(pattern_list, sequences)
                        
                        if lift >= self.min_lift:
                            # Extract contexts where this pattern appears
                            contexts = []
                            for seq in successful_sequences:
                                if self._is_subsequence(pattern_list, seq.tools):
                                    intent_type = seq.context.get('intent', {}).get('type', 'unknown')
                                    if intent_type not in contexts:
                                        contexts.append(intent_type)
                            
                            pattern = Pattern(
                                pattern_type='sequential',
                                tool_sequence=pattern_list,
                                support=support,
                                confidence=confidence,
                                lift=lift,
                                contexts=contexts[:5]  # Limit context storage
                            )
                            patterns.append(pattern)
            
            logger.info(f"Found {len([p for p in patterns if len(p.tool_sequence) == length])} patterns of length {length}")
        
        logger.info(f"Total patterns discovered: {len(patterns)}")
        return patterns
    
    async def mine_combination_patterns(self, sequences: List[ExecutionSequence]) -> List[Pattern]:
        """Mine combination patterns (order doesn't matter).
        
        Args:
            sequences: List of execution sequences
            
        Returns:
            List of discovered combination patterns
        """
        logger.info("Starting combination pattern mining...")
        
        # For combinations, we care about co-occurrence, not order
        successful_sequences = [seq for seq in sequences if seq.success or seq.reward > 0]
        
        if not successful_sequences:
            return []
        
        patterns = []
        
        # Mine combinations of different sizes
        for size in range(2, min(self.max_pattern_length + 1, 4)):  # Limit combinations to 3 tools
            logger.info(f"Mining combinations of size {size}")
            
            # Count occurrences of combinations
            combination_counts = Counter()
            total_sequences = len(sequences)
            
            for seq in successful_sequences:
                # Get unique tools in sequence
                unique_tools = list(set(seq.tools))
                
                if len(unique_tools) >= size:
                    # Generate all combinations of given size
                    from itertools import combinations
                    for combo in combinations(sorted(unique_tools), size):
                        combination_counts[combo] += 1
            
            # Create patterns for frequent combinations
            for combo, count in combination_counts.items():
                support = count / total_sequences
                
                if support >= self.min_support:
                    # Calculate confidence (success rate with this combination)
                    success_count = count  # We already filtered successful sequences
                    total_with_combo = sum(
                        1 for seq in sequences 
                        if all(tool in seq.tools for tool in combo)
                    )
                    
                    confidence = success_count / total_with_combo if total_with_combo > 0 else 0
                    
                    if confidence >= self.min_confidence:
                        # Calculate lift
                        individual_probs = []
                        for tool in combo:
                            tool_count = sum(1 for seq in sequences if tool in seq.tools)
                            individual_probs.append(tool_count / total_sequences)
                        
                        expected_prob = np.prod(individual_probs)
                        lift = support / expected_prob if expected_prob > 0 else 1.0
                        
                        if lift >= self.min_lift:
                            pattern = Pattern(
                                pattern_type='combination',
                                tool_sequence=list(combo),
                                support=support,
                                confidence=confidence,
                                lift=lift
                            )
                            patterns.append(pattern)
        
        logger.info(f"Discovered {len(patterns)} combination patterns")
        return patterns
    
    async def store_patterns(self, patterns: List[Pattern]) -> None:
        """Store discovered patterns in database.
        
        Args:
            patterns: List of patterns to store
        """
        if not patterns:
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            # Clear old patterns
            await db.execute("DELETE FROM discovered_patterns WHERE discovered_at < datetime('now', '-90 days')")
            
            # Insert new patterns
            for pattern in patterns:
                pattern_data = pattern.to_dict()
                
                await db.execute("""
                    INSERT OR REPLACE INTO discovered_patterns 
                    (pattern_type, tool_sequence, support, confidence, lift, contexts, 
                     discovered_at, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_data['pattern_type'],
                    pattern_data['tool_sequence'],
                    pattern_data['support'],
                    pattern_data['confidence'],
                    pattern_data['lift'],
                    pattern_data['contexts'],
                    pattern_data['discovered_at'],
                    pattern_data['usage_count']
                ))
            
            await db.commit()
            logger.info(f"Stored {len(patterns)} patterns in database")
    
    async def load_patterns(self) -> None:
        """Load patterns from database into memory."""
        self.discovered_patterns.clear()
        
        async with aiosqlite.connect(self.db_path) as db:
            query = """
                SELECT pattern_type, tool_sequence, support, confidence, lift, 
                       contexts, discovered_at, usage_count
                FROM discovered_patterns
                WHERE support >= ? AND confidence >= ?
                ORDER BY lift DESC, support DESC
            """
            
            async with db.execute(query, (self.min_support, self.min_confidence)) as cursor:
                async for row in cursor:
                    try:
                        pattern = Pattern(
                            pattern_type=row[0],
                            tool_sequence=json.loads(row[1]),
                            support=row[2],
                            confidence=row[3],
                            lift=row[4],
                            contexts=json.loads(row[5]) if row[5] else [],
                            discovered_at=datetime.fromisoformat(row[6]),
                            usage_count=row[7]
                        )
                        
                        pattern_hash = pattern.get_hash()
                        self.discovered_patterns[pattern_hash] = pattern
                        
                        # Cache pattern score for quick lookup
                        self.pattern_cache[pattern_hash] = pattern.confidence * pattern.lift
                        
                    except Exception as e:
                        logger.warning(f"Error loading pattern: {e}")
                        continue
        
        logger.info(f"Loaded {len(self.discovered_patterns)} patterns from database")
    
    def get_matching_patterns(self, tools: List[str], context: Optional[Dict] = None) -> List[Pattern]:
        """Get patterns that match the given tools.
        
        Args:
            tools: List of tools to match
            context: Optional context for filtering
            
        Returns:
            List of matching patterns sorted by score
        """
        matching_patterns = []
        
        for pattern_hash, pattern in self.discovered_patterns.items():
            # Check if pattern matches tools
            if pattern.pattern_type == 'sequential':
                # For sequential, check if pattern is subsequence
                if self._is_subsequence(pattern.tool_sequence, tools):
                    matching_patterns.append(pattern)
            elif pattern.pattern_type == 'combination':
                # For combination, check if all pattern tools are in tools
                if all(tool in tools for tool in pattern.tool_sequence):
                    matching_patterns.append(pattern)
        
        # Sort by score (confidence * lift)
        matching_patterns.sort(
            key=lambda p: p.confidence * p.lift,
            reverse=True
        )
        
        return matching_patterns
    
    def suggest_next_tools(self, current_tools: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """Suggest next tools based on patterns.
        
        Args:
            current_tools: Tools already selected/used
            k: Number of suggestions to return
            
        Returns:
            List of (tool, score) tuples
        """
        suggestions = Counter()
        
        for pattern in self.discovered_patterns.values():
            if pattern.pattern_type != 'sequential':
                continue
                
            # Check if current_tools is a prefix of pattern
            if len(pattern.tool_sequence) > len(current_tools):
                is_prefix = all(
                    pattern.tool_sequence[i] == current_tools[i] 
                    for i in range(len(current_tools))
                )
                
                if is_prefix:
                    # Suggest the next tool in pattern
                    next_tool = pattern.tool_sequence[len(current_tools)]
                    score = pattern.confidence * pattern.lift
                    suggestions[next_tool] = max(suggestions[next_tool], score)
        
        # Return top k suggestions
        return suggestions.most_common(k)
    
    async def mine_patterns(self) -> Dict[str, List[Pattern]]:
        """Run complete pattern mining process.
        
        Returns:
            Dictionary with pattern types as keys and patterns as values
        """
        logger.info("Starting pattern mining process...")
        
        # Extract sequences
        sequences = await self.extract_sequences()
        
        if not sequences:
            logger.warning("No sequences found for mining")
            return {}
        
        # Mine different types of patterns
        all_patterns = {
            'sequential': [],
            'combination': []
        }
        
        # Mine sequential patterns
        sequential_patterns = await self.mine_sequential_patterns(sequences)
        all_patterns['sequential'] = sequential_patterns
        
        # Mine combination patterns
        combination_patterns = await self.mine_combination_patterns(sequences)
        all_patterns['combination'] = combination_patterns
        
        # Store all patterns
        all_pattern_list = sequential_patterns + combination_patterns
        await self.store_patterns(all_pattern_list)
        
        # Update in-memory patterns
        await self.load_patterns()
        
        logger.info(f"Pattern mining complete. Found {len(all_pattern_list)} total patterns")
        
        return all_patterns
    
    async def update_pattern_usage(self, pattern_hash: str) -> None:
        """Update usage count for a pattern.
        
        Args:
            pattern_hash: Hash of the pattern to update
        """
        if pattern_hash in self.discovered_patterns:
            pattern = self.discovered_patterns[pattern_hash]
            pattern.usage_count += 1
            
            # Update in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE discovered_patterns 
                    SET usage_count = usage_count + 1
                    WHERE pattern_type = ? AND tool_sequence = ?
                """, (pattern.pattern_type, json.dumps(pattern.tool_sequence)))
                await db.commit()