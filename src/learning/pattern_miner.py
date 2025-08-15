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
import statistics
from scipy import signal
from sklearn.cluster import DBSCAN

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
    # Temporal pattern fields
    time_intervals: Optional[List[float]] = None  # Intervals between tool executions (seconds)
    periodic_info: Optional[Dict[str, Any]] = None  # Periodicity info (period, strength, type)
    duration_stats: Optional[Dict[str, float]] = None  # Execution duration statistics
    temporal_metadata: Optional[Dict[str, Any]] = None  # Additional temporal metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage."""
        result = {
            'pattern_type': self.pattern_type,
            'tool_sequence': json.dumps(self.tool_sequence),
            'support': self.support,
            'confidence': self.confidence,
            'lift': self.lift,
            'contexts': json.dumps(self.contexts),
            'discovered_at': self.discovered_at.isoformat(),
            'usage_count': self.usage_count
        }
        
        # Add temporal fields if present
        if self.pattern_type == 'temporal' and self.temporal_metadata:
            result['temporal_metadata'] = json.dumps(self.temporal_metadata)
        
        return result
    
    def get_hash(self) -> str:
        """Generate unique hash for pattern."""
        if self.pattern_type == 'sequential':
            # For sequential patterns, order matters
            pattern_str = f"{self.pattern_type}:{':'.join(self.tool_sequence)}"
        else:
            # For combination patterns, order doesn't matter
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
    # Temporal fields
    tool_timestamps: Optional[List[datetime]] = None  # Individual tool start times
    tool_durations: Optional[List[float]] = None  # Individual tool execution times (ms)
    total_duration: Optional[float] = None  # Total execution time (ms)
    # Context-aware fields
    user_expertise: str = 'intermediate'  # novice, intermediate, expert
    domain: str = 'general'  # general, engineering, data_science, web_dev, devops


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
    
    def _get_connection(self):
        """Get a database connection with proper settings."""
        return aiosqlite.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None
        )
    
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
        
        async with self._get_connection() as db:
            # Check if context columns exist
            cursor = await db.execute("PRAGMA table_info(execution_history)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            has_context = 'user_expertise' in column_names and 'domain' in column_names
            
            if has_context:
                query = """
                    SELECT id, query, intent, tools_used, success, reward, created_at,
                           user_expertise, domain
                    FROM execution_history
                    WHERE created_at > ? AND tools_used IS NOT NULL
                    ORDER BY created_at DESC
                """
            else:
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
                            timestamp=datetime.fromisoformat(row[6]),
                            user_expertise=row[7] if has_context and len(row) > 7 else 'intermediate',
                            domain=row[8] if has_context and len(row) > 8 else 'general'
                        )
                        sequences.append(sequence)
                    except Exception as e:
                        logger.warning(f"Error parsing execution sequence {row[0]}: {e}")
                        continue
        
        logger.info(f"Extracted {len(sequences)} execution sequences from last {time_window.days} days")
        return sequences
    
    async def get_pattern_mining_metadata(self, key: str) -> Optional[str]:
        """Get metadata value for pattern mining.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT value FROM pattern_mining_metadata WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None
    
    async def set_pattern_mining_metadata(self, key: str, value: str) -> None:
        """Set metadata value for pattern mining.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        async with self._get_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO pattern_mining_metadata (key, value, last_updated)
                VALUES (?, ?, datetime('now'))
                """,
                (key, value)
            )
            await db.commit()
    
    async def extract_new_sequences(self, batch_size: int = 1000) -> List[ExecutionSequence]:
        """Extract only new execution sequences since last update.
        
        Args:
            batch_size: Maximum number of sequences to process at once
            
        Returns:
            List of new execution sequences
        """
        # Get last processed execution ID
        last_processed_id = await self.get_pattern_mining_metadata('last_processed_execution_id')
        last_update_timestamp = await self.get_pattern_mining_metadata('last_update_timestamp')
        
        sequences = []
        last_id = None
        
        async with self._get_connection() as db:
            # Check if context columns exist
            cursor = await db.execute("PRAGMA table_info(execution_history)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            has_context = 'user_expertise' in column_names and 'domain' in column_names
            
            # Build query based on what metadata we have
            if has_context:
                select_cols = "id, query, intent, tools_used, success, reward, created_at, user_expertise, domain"
            else:
                select_cols = "id, query, intent, tools_used, success, reward, created_at"
                
            if last_processed_id:
                query = f"""
                    SELECT {select_cols}
                    FROM execution_history
                    WHERE id > ? AND tools_used IS NOT NULL
                    ORDER BY id ASC
                    LIMIT ?
                """
                params = (last_processed_id, batch_size)
            elif last_update_timestamp:
                query = f"""
                    SELECT {select_cols}
                    FROM execution_history
                    WHERE created_at > ? AND tools_used IS NOT NULL
                    ORDER BY created_at ASC, id ASC
                    LIMIT ?
                """
                params = (last_update_timestamp, batch_size)
            else:
                # First time - process recent data only
                cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()
                query = f"""
                    SELECT {select_cols}
                    FROM execution_history
                    WHERE created_at > ? AND tools_used IS NOT NULL
                    ORDER BY created_at ASC, id ASC
                    LIMIT ?
                """
                params = (cutoff_time, batch_size)
            
            async with db.execute(query, params) as cursor:
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
                            timestamp=datetime.fromisoformat(row[6]),
                            user_expertise=row[7] if has_context and len(row) > 7 else 'intermediate',
                            domain=row[8] if has_context and len(row) > 8 else 'general'
                        )
                        sequences.append(sequence)
                        last_id = row[0]
                    except Exception as e:
                        logger.warning(f"Error parsing execution sequence {row[0]}: {e}")
                        continue
        
        # Update metadata with last processed ID
        if last_id:
            await self.set_pattern_mining_metadata('last_processed_execution_id', last_id)
            await self.set_pattern_mining_metadata('last_update_timestamp', datetime.now().isoformat())
        
        logger.info(f"Extracted {len(sequences)} new execution sequences for incremental update")
        return sequences
    
    async def load_pattern_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Load pattern statistics from database.
        
        Returns:
            Dictionary mapping pattern hash to statistics
        """
        stats = {}
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT pattern_hash, pattern_type, tool_sequence, occurrence_count,
                       success_count, total_support, total_confidence, last_seen
                FROM pattern_statistics
            """)
            
            async for row in cursor:
                stats[row[0]] = {
                    'pattern_type': row[1],
                    'tool_sequence': json.loads(row[2]),
                    'occurrence_count': row[3],
                    'success_count': row[4],
                    'total_support': row[5],
                    'total_confidence': row[6],
                    'last_seen': datetime.fromisoformat(row[7]) if row[7] else None
                }
        
        return stats
    
    async def update_pattern_statistics(self, pattern_hash: str, pattern_type: str,
                                      tool_sequence: List[str], occurrence_delta: int = 1,
                                      success_delta: int = 0, support_delta: float = 0.0,
                                      confidence_delta: float = 0.0) -> None:
        """Update pattern statistics incrementally.
        
        Args:
            pattern_hash: Hash of the pattern
            pattern_type: Type of pattern (sequential, combination, temporal)
            tool_sequence: Tools in the pattern
            occurrence_delta: Change in occurrence count
            success_delta: Change in success count
            support_delta: Change in support
            confidence_delta: Change in confidence
        """
        async with self._get_connection() as db:
            # Try to update existing record
            cursor = await db.execute("""
                UPDATE pattern_statistics
                SET occurrence_count = occurrence_count + ?,
                    success_count = success_count + ?,
                    total_support = total_support + ?,
                    total_confidence = total_confidence + ?,
                    last_seen = datetime('now')
                WHERE pattern_hash = ?
            """, (occurrence_delta, success_delta, support_delta, 
                  confidence_delta, pattern_hash))
            
            if cursor.rowcount == 0:
                # Insert new record if doesn't exist
                await db.execute("""
                    INSERT INTO pattern_statistics
                    (pattern_hash, pattern_type, tool_sequence, occurrence_count,
                     success_count, total_support, total_confidence, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (pattern_hash, pattern_type, json.dumps(tool_sequence),
                      occurrence_delta, success_delta, support_delta, confidence_delta))
            
            await db.commit()
    
    def calculate_incremental_metrics(self, pattern: List[str], new_sequences: List[ExecutionSequence],
                                    existing_stats: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate metrics for a pattern incrementally.
        
        Args:
            pattern: The pattern to calculate metrics for
            new_sequences: New sequences to process
            existing_stats: Existing statistics for the pattern
            
        Returns:
            Dictionary with updated metrics
        """
        # Count occurrences in new sequences
        new_occurrences = 0
        new_successes = 0
        
        for seq in new_sequences:
            if self._is_subsequence(pattern, seq.tools):
                new_occurrences += 1
                if seq.success or seq.reward > 0:
                    new_successes += 1
        
        if existing_stats:
            # Update existing statistics
            total_occurrences = existing_stats['occurrence_count'] + new_occurrences
            total_successes = existing_stats['success_count'] + new_successes
            
            # Calculate new support (moving average)
            new_support = new_occurrences / len(new_sequences) if new_sequences else 0
            avg_support = (existing_stats['total_support'] + new_support) / 2
            
            # Calculate new confidence
            confidence = total_successes / total_occurrences if total_occurrences > 0 else 0
        else:
            # New pattern
            total_occurrences = new_occurrences
            total_successes = new_successes
            avg_support = new_occurrences / len(new_sequences) if new_sequences else 0
            confidence = total_successes / total_occurrences if total_occurrences > 0 else 0
        
        return {
            'occurrence_count': total_occurrences,
            'success_count': total_successes,
            'support': avg_support,
            'confidence': confidence,
            'new_occurrences': new_occurrences,
            'new_successes': new_successes
        }
    
    async def merge_patterns(self, new_patterns: List[Pattern], 
                           decay_factor: float = 0.95) -> List[Pattern]:
        """Merge new patterns with existing ones.
        
        Args:
            new_patterns: Newly discovered patterns
            decay_factor: Factor to decay old pattern scores
            
        Returns:
            Merged list of patterns
        """
        # Load existing patterns
        await self.load_patterns()
        
        # Apply decay to existing patterns
        for pattern_hash, pattern in self.discovered_patterns.items():
            pattern.support *= decay_factor
            pattern.confidence *= decay_factor
            
            # Remove patterns below threshold after decay
            if pattern.support < self.min_support * 0.5:
                del self.discovered_patterns[pattern_hash]
        
        # Merge new patterns
        for new_pattern in new_patterns:
            pattern_hash = new_pattern.get_hash()
            
            if pattern_hash in self.discovered_patterns:
                # Update existing pattern
                existing = self.discovered_patterns[pattern_hash]
                existing.support = (existing.support + new_pattern.support) / 2
                existing.confidence = (existing.confidence + new_pattern.confidence) / 2
                existing.lift = (existing.lift + new_pattern.lift) / 2
                existing.usage_count += new_pattern.usage_count
                
                # Update contexts
                for context in new_pattern.contexts:
                    if context not in existing.contexts:
                        existing.contexts.append(context)
                existing.contexts = existing.contexts[-10:]  # Keep last 10 contexts
            else:
                # Add new pattern
                self.discovered_patterns[pattern_hash] = new_pattern
        
        # Return all patterns as list
        return list(self.discovered_patterns.values())
    
    async def mine_incremental_sequential_patterns(self, new_sequences: List[ExecutionSequence],
                                                  existing_stats: Dict[str, Dict[str, Any]]) -> List[Pattern]:
        """Mine sequential patterns incrementally from new sequences.
        
        Args:
            new_sequences: New execution sequences
            existing_stats: Existing pattern statistics
            
        Returns:
            List of discovered/updated patterns
        """
        logger.info(f"Mining incremental sequential patterns from {len(new_sequences)} new sequences")
        
        # Get successful sequences
        successful_sequences = [seq for seq in new_sequences if seq.success or seq.reward > 0]
        
        if not successful_sequences:
            return []
        
        patterns = []
        
        # Extract all tools from new sequences
        all_tools = set()
        for seq in new_sequences:
            all_tools.update(seq.tools)
        
        # Check existing patterns in new sequences
        for pattern_hash, stats in existing_stats.items():
            if stats['pattern_type'] == 'sequential':
                pattern_seq = stats['tool_sequence']
                metrics = self.calculate_incremental_metrics(
                    pattern_seq, new_sequences, stats
                )
                
                if metrics['new_occurrences'] > 0:
                    # Update statistics
                    await self.update_pattern_statistics(
                        pattern_hash, 'sequential', pattern_seq,
                        metrics['new_occurrences'], metrics['new_successes'],
                        metrics['support'] - stats.get('total_support', 0),
                        metrics['confidence'] - stats.get('total_confidence', 0)
                    )
                    
                    # Create pattern object if meets thresholds
                    if metrics['support'] >= self.min_support and metrics['confidence'] >= self.min_confidence:
                        pattern = Pattern(
                            pattern_type='sequential',
                            tool_sequence=pattern_seq,
                            support=metrics['support'],
                            confidence=metrics['confidence'],
                            lift=1.0  # Will be calculated later
                        )
                        patterns.append(pattern)
        
        # Mine new patterns from new sequences
        # Use a simplified approach for incremental mining
        candidate_patterns = set()
        
        for seq in successful_sequences:
            # Generate subsequences
            for length in range(1, min(len(seq.tools) + 1, self.max_pattern_length + 1)):
                for i in range(len(seq.tools) - length + 1):
                    subsequence = tuple(seq.tools[i:i+length])
                    candidate_patterns.add(subsequence)
        
        # Evaluate new candidates
        for candidate in candidate_patterns:
            pattern_list = list(candidate)
            pattern_str = f"sequential:{':'.join(sorted(pattern_list))}"
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()
            
            # Skip if already processed
            if pattern_hash in existing_stats:
                continue
            
            metrics = self.calculate_incremental_metrics(pattern_list, new_sequences)
            
            if metrics['support'] >= self.min_support and metrics['confidence'] >= self.min_confidence:
                # Calculate lift
                lift = self._calculate_lift(pattern_list, new_sequences)
                
                if lift >= self.min_lift:
                    # Store new pattern statistics
                    await self.update_pattern_statistics(
                        pattern_hash, 'sequential', pattern_list,
                        metrics['occurrence_count'], metrics['success_count'],
                        metrics['support'], metrics['confidence']
                    )
                    
                    pattern = Pattern(
                        pattern_type='sequential',
                        tool_sequence=pattern_list,
                        support=metrics['support'],
                        confidence=metrics['confidence'],
                        lift=lift
                    )
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} sequential patterns in incremental update")
        return patterns
    
    async def mine_incremental_combination_patterns(self, new_sequences: List[ExecutionSequence],
                                                   existing_stats: Dict[str, Dict[str, Any]]) -> List[Pattern]:
        """Mine combination patterns incrementally from new sequences.
        
        Args:
            new_sequences: New execution sequences
            existing_stats: Existing pattern statistics
            
        Returns:
            List of discovered/updated patterns
        """
        logger.info(f"Mining incremental combination patterns from {len(new_sequences)} new sequences")
        
        successful_sequences = [seq for seq in new_sequences if seq.success or seq.reward > 0]
        
        if not successful_sequences:
            return []
        
        patterns = []
        
        # Check existing combination patterns
        for pattern_hash, stats in existing_stats.items():
            if stats['pattern_type'] == 'combination':
                pattern_tools = stats['tool_sequence']
                
                # Count occurrences in new sequences
                occurrences = 0
                successes = 0
                
                for seq in new_sequences:
                    if all(tool in seq.tools for tool in pattern_tools):
                        occurrences += 1
                        if seq in successful_sequences:
                            successes += 1
                
                if occurrences > 0:
                    # Update statistics
                    new_support = occurrences / len(new_sequences)
                    new_confidence = successes / occurrences if occurrences > 0 else 0
                    
                    await self.update_pattern_statistics(
                        pattern_hash, 'combination', pattern_tools,
                        occurrences, successes,
                        new_support, new_confidence
                    )
                    
                    # Create pattern if meets thresholds
                    avg_support = (stats.get('total_support', 0) + new_support) / 2
                    if avg_support >= self.min_support and new_confidence >= self.min_confidence:
                        pattern = Pattern(
                            pattern_type='combination',
                            tool_sequence=pattern_tools,
                            support=avg_support,
                            confidence=new_confidence,
                            lift=1.0
                        )
                        patterns.append(pattern)
        
        # Mine new combinations
        from itertools import combinations
        
        # Get unique tools from new sequences
        tool_counts = Counter()
        for seq in successful_sequences:
            unique_tools = set(seq.tools)
            tool_counts.update(unique_tools)
        
        # Generate new combinations
        frequent_tools = [tool for tool, count in tool_counts.items() 
                         if count / len(new_sequences) >= self.min_support * 0.5]
        
        for size in range(2, min(len(frequent_tools) + 1, 4)):  # Limit to size 3
            for combo in combinations(sorted(frequent_tools), size):
                pattern_str = f"combination:{':'.join(sorted(combo))}"
                pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()
                
                # Skip if already exists
                if pattern_hash in existing_stats:
                    continue
                
                # Count occurrences
                occurrences = 0
                successes = 0
                
                for seq in new_sequences:
                    if all(tool in seq.tools for tool in combo):
                        occurrences += 1
                        if seq in successful_sequences:
                            successes += 1
                
                support = occurrences / len(new_sequences)
                confidence = successes / occurrences if occurrences > 0 else 0
                
                if support >= self.min_support and confidence >= self.min_confidence:
                    # Calculate lift
                    individual_probs = []
                    for tool in combo:
                        tool_count = sum(1 for seq in new_sequences if tool in seq.tools)
                        individual_probs.append(tool_count / len(new_sequences))
                    
                    expected_prob = np.prod(individual_probs)
                    lift = support / expected_prob if expected_prob > 0 else 1.0
                    
                    if lift >= self.min_lift:
                        await self.update_pattern_statistics(
                            pattern_hash, 'combination', list(combo),
                            occurrences, successes, support, confidence
                        )
                        
                        pattern = Pattern(
                            pattern_type='combination',
                            tool_sequence=list(combo),
                            support=support,
                            confidence=confidence,
                            lift=lift
                        )
                        patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} combination patterns in incremental update")
        return patterns
    
    async def mine_incremental_temporal_patterns(self, new_sequences: List[ExecutionSequence],
                                                existing_stats: Dict[str, Dict[str, Any]]) -> List[Pattern]:
        """Mine temporal patterns incrementally from new sequences.
        
        Args:
            new_sequences: New execution sequences
            existing_stats: Existing pattern statistics
            
        Returns:
            List of discovered/updated patterns
        """
        logger.info(f"Mining incremental temporal patterns from {len(new_sequences)} new sequences")
        
        if len(new_sequences) < 5:  # Need minimum sequences for temporal patterns
            return []
        
        patterns = []
        
        # Extract temporal features from new sequences
        temporal_features = self._extract_temporal_features(new_sequences)
        
        # Update hourly patterns
        for (tool, hour), count in temporal_features['hourly_distribution'].items():
            pattern_seq = [tool]
            pattern_str = f"temporal:hourly:{tool}:{hour}"
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()
            
            support = count / len(new_sequences)
            
            if support >= self.min_support * 0.5:  # Lower threshold for temporal
                # Update or create statistics
                if pattern_hash in existing_stats:
                    stats = existing_stats[pattern_hash]
                    await self.update_pattern_statistics(
                        pattern_hash, 'temporal', pattern_seq,
                        count, count,  # Assume all temporal patterns are successful
                        support, 0.8  # High confidence for temporal patterns
                    )
                else:
                    await self.update_pattern_statistics(
                        pattern_hash, 'temporal', pattern_seq,
                        count, count, support, 0.8
                    )
                
                pattern = Pattern(
                    pattern_type='temporal',
                    tool_sequence=pattern_seq,
                    support=support,
                    confidence=0.8,
                    lift=1.0,
                    temporal_metadata={
                        'pattern_subtype': 'hourly',
                        'hour': hour,
                        'occurrences': count
                    }
                )
                patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} temporal patterns in incremental update")
        return patterns
    
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
    
    def _extract_temporal_features(self, sequences: List[ExecutionSequence]) -> Dict[str, Any]:
        """Extract temporal features from execution sequences.
        
        Args:
            sequences: List of execution sequences
            
        Returns:
            Dictionary containing temporal features
        """
        features = {
            'hourly_distribution': defaultdict(int),
            'daily_distribution': defaultdict(int),
            'weekday_distribution': defaultdict(int),
            'execution_durations': defaultdict(list),
            'inter_execution_intervals': [],
            'tool_sequences_by_time': defaultdict(list)
        }
        
        # Sort sequences by timestamp
        sorted_sequences = sorted(sequences, key=lambda x: x.timestamp)
        
        for i, seq in enumerate(sorted_sequences):
            # Extract hour of day, day of week
            hour = seq.timestamp.hour
            weekday = seq.timestamp.weekday()
            
            # Count tool usage by hour and weekday
            for tool in seq.tools:
                features['hourly_distribution'][(tool, hour)] += 1
                features['weekday_distribution'][(tool, weekday)] += 1
            
            # Store execution durations if available
            if seq.total_duration:
                features['execution_durations'][tuple(seq.tools)].append(seq.total_duration)
            
            # Calculate inter-execution intervals
            if i > 0:
                interval = (seq.timestamp - sorted_sequences[i-1].timestamp).total_seconds()
                features['inter_execution_intervals'].append(interval)
            
            # Group sequences by time window (hourly)
            time_key = seq.timestamp.strftime('%H')
            features['tool_sequences_by_time'][time_key].append(seq.tools)
        
        return features
    
    def _detect_periodicity(self, time_series: List[float], sampling_interval: float = 3600) -> Dict[str, Any]:
        """Detect periodic patterns in time series data.
        
        Args:
            time_series: List of time values
            sampling_interval: Expected interval between samples (seconds)
            
        Returns:
            Dictionary with periodicity information
        """
        if len(time_series) < 10:
            return {'has_periodicity': False, 'reason': 'insufficient_data'}
        
        try:
            # Convert to numpy array
            ts = np.array(time_series)
            
            # Remove trend using differencing
            detrended = np.diff(ts)
            
            # Compute autocorrelation
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(autocorr, height=0.3, distance=5)
            
            if len(peaks) > 0:
                # Estimate period from first significant peak
                period_samples = peaks[0]
                period_seconds = period_samples * sampling_interval
                
                # Classify periodicity type
                period_hours = period_seconds / 3600
                if 0.9 < period_hours < 1.1:
                    period_type = 'hourly'
                elif 23 < period_hours < 25:
                    period_type = 'daily'
                elif 167 < period_hours < 169:
                    period_type = 'weekly'
                else:
                    period_type = 'custom'
                
                return {
                    'has_periodicity': True,
                    'period_seconds': period_seconds,
                    'period_type': period_type,
                    'strength': float(properties['peak_heights'][0]),
                    'confidence': min(1.0, float(properties['peak_heights'][0]) + 0.2)
                }
            
        except Exception as e:
            logger.warning(f"Error in periodicity detection: {e}")
        
        return {'has_periodicity': False, 'reason': 'no_significant_peaks'}
    
    def _analyze_time_intervals(self, intervals: List[float]) -> Dict[str, float]:
        """Analyze time intervals between executions.
        
        Args:
            intervals: List of time intervals in seconds
            
        Returns:
            Statistics about intervals
        """
        if not intervals:
            return {}
        
        return {
            'mean': statistics.mean(intervals),
            'median': statistics.median(intervals),
            'std': statistics.stdev(intervals) if len(intervals) > 1 else 0,
            'min': min(intervals),
            'max': max(intervals),
            'q1': np.percentile(intervals, 25),
            'q3': np.percentile(intervals, 75)
        }
    
    def _cluster_time_patterns(self, timestamps: List[datetime], eps: float = 3600) -> List[List[int]]:
        """Cluster timestamps to find temporal groups.
        
        Args:
            timestamps: List of datetime objects
            eps: Maximum distance between samples in same cluster (seconds)
            
        Returns:
            List of cluster indices
        """
        if len(timestamps) < 2:
            return [[0]] if timestamps else []
        
        # Convert to seconds since first timestamp
        base_time = timestamps[0]
        time_values = [(ts - base_time).total_seconds() for ts in timestamps]
        
        # Reshape for DBSCAN
        X = np.array(time_values).reshape(-1, 1)
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=2).fit(X)
        
        # Group indices by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            clusters[label].append(idx)
        
        # Return clusters (excluding noise points with label -1)
        return [indices for label, indices in clusters.items() if label != -1]
    
    async def mine_temporal_patterns(self, sequences: List[ExecutionSequence]) -> List[Pattern]:
        """Mine temporal patterns from execution sequences.
        
        Args:
            sequences: List of execution sequences
            
        Returns:
            List of discovered temporal patterns
        """
        logger.info("Starting temporal pattern mining...")
        
        if len(sequences) < 10:
            logger.warning("Insufficient sequences for temporal pattern mining")
            return []
        
        patterns = []
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(sequences)
        
        # 1. Mine hourly patterns
        hourly_patterns = self._mine_hourly_patterns(temporal_features['hourly_distribution'], len(sequences))
        patterns.extend(hourly_patterns)
        
        # 2. Mine periodic patterns
        if temporal_features['inter_execution_intervals']:
            periodicity = self._detect_periodicity(temporal_features['inter_execution_intervals'])
            if periodicity['has_periodicity']:
                # Create pattern for periodic execution
                periodic_tools = self._get_most_common_tools_in_period(sequences, periodicity)
                if periodic_tools:
                    pattern = Pattern(
                        pattern_type='temporal',
                        tool_sequence=periodic_tools,
                        support=len(periodic_tools) / len(sequences),
                        confidence=periodicity['confidence'],
                        lift=1.0,  # Will be calculated based on temporal context
                        periodic_info=periodicity,
                        temporal_metadata={
                            'pattern_subtype': 'periodic',
                            'period': periodicity['period_type'],
                            'period_seconds': periodicity['period_seconds']
                        }
                    )
                    patterns.append(pattern)
        
        # 3. Mine duration patterns
        duration_patterns = self._mine_duration_patterns(temporal_features['execution_durations'], len(sequences))
        patterns.extend(duration_patterns)
        
        # 4. Mine time-clustered patterns
        time_clustered_patterns = self._mine_time_clustered_patterns(sequences)
        patterns.extend(time_clustered_patterns)
        
        logger.info(f"Discovered {len(patterns)} temporal patterns")
        return patterns
    
    def _mine_hourly_patterns(self, hourly_dist: Dict[Tuple[str, int], int], total_sequences: int) -> List[Pattern]:
        """Mine patterns based on hourly distribution.
        
        Args:
            hourly_dist: Distribution of tool usage by hour
            total_sequences: Total number of sequences
            
        Returns:
            List of hourly patterns
        """
        patterns = []
        
        # Group by hour to find tools commonly used together at specific times
        hourly_tools = defaultdict(list)
        for (tool, hour), count in hourly_dist.items():
            if count >= 2:  # Minimum occurrences
                hourly_tools[hour].append((tool, count))
        
        # Create patterns for significant hourly concentrations
        for hour, tool_counts in hourly_tools.items():
            if len(tool_counts) >= 2:
                # Sort by count and take top tools
                tool_counts.sort(key=lambda x: x[1], reverse=True)
                tools = [tc[0] for tc in tool_counts[:3]]  # Top 3 tools
                total_count = sum(tc[1] for tc in tool_counts)
                
                support = total_count / total_sequences
                if support >= self.min_support * 0.5:  # Lower threshold for temporal patterns
                    pattern = Pattern(
                        pattern_type='temporal',
                        tool_sequence=tools,
                        support=support,
                        confidence=0.8,  # High confidence for consistent hourly patterns
                        lift=1.0,
                        temporal_metadata={
                            'pattern_subtype': 'hourly',
                            'hour': hour,
                            'occurrences': total_count
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _mine_duration_patterns(self, duration_dist: Dict[Tuple[str], List[float]], total_sequences: int) -> List[Pattern]:
        """Mine patterns based on execution duration.
        
        Args:
            duration_dist: Distribution of execution durations by tool combination
            total_sequences: Total number of sequences
            
        Returns:
            List of duration-based patterns
        """
        patterns = []
        
        for tools, durations in duration_dist.items():
            if len(durations) >= 3:  # Minimum samples
                duration_stats = self._analyze_time_intervals(durations)
                
                # Check for consistent duration (low variance)
                if duration_stats['std'] / duration_stats['mean'] < 0.3:  # CV < 30%
                    support = len(durations) / total_sequences
                    if support >= self.min_support * 0.5:
                        pattern = Pattern(
                            pattern_type='temporal',
                            tool_sequence=list(tools),
                            support=support,
                            confidence=0.9,  # High confidence for consistent durations
                            lift=1.0,
                            duration_stats=duration_stats,
                            temporal_metadata={
                                'pattern_subtype': 'duration',
                                'avg_duration_ms': duration_stats['mean'],
                                'duration_consistency': 1 - (duration_stats['std'] / duration_stats['mean'])
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _mine_time_clustered_patterns(self, sequences: List[ExecutionSequence]) -> List[Pattern]:
        """Mine patterns from time-clustered sequences.
        
        Args:
            sequences: List of execution sequences
            
        Returns:
            List of time-clustered patterns
        """
        patterns = []
        
        if len(sequences) < 5:
            return patterns
        
        # Sort sequences by timestamp
        sorted_sequences = sorted(sequences, key=lambda x: x.timestamp)
        timestamps = [seq.timestamp for seq in sorted_sequences]
        
        # Find time clusters (sequences that occur close together)
        clusters = self._cluster_time_patterns(timestamps, eps=1800)  # 30-minute window
        
        for cluster_indices in clusters:
            if len(cluster_indices) >= 3:
                # Get tools from clustered sequences
                cluster_tools = []
                for idx in cluster_indices:
                    cluster_tools.extend(sorted_sequences[idx].tools)
                
                # Find most common tools in cluster
                tool_counts = Counter(cluster_tools)
                common_tools = [tool for tool, count in tool_counts.most_common(3)]
                
                support = len(cluster_indices) / len(sequences)
                if support >= self.min_support * 0.5:
                    # Calculate time span of cluster
                    cluster_start = timestamps[cluster_indices[0]]
                    cluster_end = timestamps[cluster_indices[-1]]
                    duration_seconds = (cluster_end - cluster_start).total_seconds()
                    
                    pattern = Pattern(
                        pattern_type='temporal',
                        tool_sequence=common_tools,
                        support=support,
                        confidence=0.85,
                        lift=1.0,
                        temporal_metadata={
                            'pattern_subtype': 'time_cluster',
                            'cluster_size': len(cluster_indices),
                            'cluster_duration_seconds': duration_seconds,
                            'avg_time': cluster_start.strftime('%H:%M')
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _get_most_common_tools_in_period(self, sequences: List[ExecutionSequence], periodicity: Dict[str, Any]) -> List[str]:
        """Get most common tools used in periodic pattern.
        
        Args:
            sequences: List of execution sequences
            periodicity: Periodicity information
            
        Returns:
            List of most common tools
        """
        period_seconds = periodicity['period_seconds']
        tool_counts = Counter()
        
        # Group sequences by period
        for seq in sequences:
            # Simple grouping by period
            tool_counts.update(seq.tools)
        
        # Return top 3 tools
        return [tool for tool, _ in tool_counts.most_common(3)]
    
    async def store_patterns(self, patterns: List[Pattern]) -> None:
        """Store discovered patterns in database.
        
        Args:
            patterns: List of patterns to store
        """
        if not patterns:
            return
            
        async with self._get_connection() as db:
            # Clear old patterns
            await db.execute("DELETE FROM discovered_patterns WHERE discovered_at < datetime('now', '-90 days')")
            
            # Insert new patterns
            for pattern in patterns:
                pattern_data = pattern.to_dict()
                
                await db.execute("""
                    INSERT OR REPLACE INTO discovered_patterns 
                    (pattern_type, tool_sequence, support, confidence, lift, contexts, 
                     discovered_at, usage_count, temporal_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_data['pattern_type'],
                    pattern_data['tool_sequence'],
                    pattern_data['support'],
                    pattern_data['confidence'],
                    pattern_data['lift'],
                    pattern_data['contexts'],
                    pattern_data['discovered_at'],
                    pattern_data['usage_count'],
                    pattern_data.get('temporal_metadata', None)
                ))
            
            await db.commit()
            logger.info(f"Stored {len(patterns)} patterns in database")
    
    async def load_patterns(self) -> None:
        """Load patterns from database into memory."""
        self.discovered_patterns.clear()
        
        async with self._get_connection() as db:
            query = """
                SELECT pattern_type, tool_sequence, support, confidence, lift, 
                       contexts, discovered_at, usage_count, temporal_metadata
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
                            usage_count=row[7],
                            temporal_metadata=json.loads(row[8]) if row[8] else None
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
        current_time = datetime.now()
        current_hour = current_time.hour
        
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
            elif pattern.pattern_type == 'temporal':
                # For temporal patterns, check if tools match and time context is relevant
                if self._matches_temporal_pattern(pattern, tools, current_hour):
                    matching_patterns.append(pattern)
        
        # Sort by score (confidence * lift), with temporal boost for time-relevant patterns
        matching_patterns.sort(
            key=lambda p: self._calculate_pattern_score(p, current_hour),
            reverse=True
        )
        
        return matching_patterns
    
    def _matches_temporal_pattern(self, pattern: Pattern, tools: List[str], current_hour: int) -> bool:
        """Check if temporal pattern matches current context.
        
        Args:
            pattern: Temporal pattern to check
            tools: Current tools
            current_hour: Current hour of day
            
        Returns:
            True if pattern matches
        """
        # Check if tools are relevant
        tools_match = any(tool in tools for tool in pattern.tool_sequence)
        if not tools_match:
            return False
        
        # Check temporal relevance
        if pattern.temporal_metadata:
            subtype = pattern.temporal_metadata.get('pattern_subtype')
            if subtype == 'hourly':
                # Check if current hour matches pattern hour
                pattern_hour = pattern.temporal_metadata.get('hour')
                return abs(current_hour - pattern_hour) <= 1  # Within 1 hour window
            elif subtype == 'periodic':
                # Always relevant for periodic patterns
                return True
            elif subtype == 'time_cluster':
                # Check if current time is near cluster time
                avg_time = pattern.temporal_metadata.get('avg_time', '')
                if avg_time:
                    cluster_hour = int(avg_time.split(':')[0])
                    return abs(current_hour - cluster_hour) <= 2  # Within 2 hour window
        
        return True
    
    def _calculate_pattern_score(self, pattern: Pattern, current_hour: int) -> float:
        """Calculate pattern score with temporal boost.
        
        Args:
            pattern: Pattern to score
            current_hour: Current hour of day
            
        Returns:
            Pattern score
        """
        base_score = pattern.confidence * pattern.lift
        
        # Apply temporal boost for time-relevant patterns
        if pattern.pattern_type == 'temporal' and pattern.temporal_metadata:
            subtype = pattern.temporal_metadata.get('pattern_subtype')
            if subtype == 'hourly':
                pattern_hour = pattern.temporal_metadata.get('hour', -1)
                if abs(current_hour - pattern_hour) <= 1:
                    base_score *= 1.5  # 50% boost for time-relevant patterns
            elif subtype == 'duration':
                # Boost consistent duration patterns
                consistency = pattern.temporal_metadata.get('duration_consistency', 0)
                base_score *= (1 + consistency * 0.3)  # Up to 30% boost
        
        return base_score
    
    def suggest_next_tools(self, current_tools: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """Suggest next tools based on patterns.
        
        Args:
            current_tools: Tools already selected/used
            k: Number of suggestions to return
            
        Returns:
            List of (tool, score) tuples
        """
        suggestions = Counter()
        current_time = datetime.now()
        current_hour = current_time.hour
        
        for pattern in self.discovered_patterns.values():
            if pattern.pattern_type == 'sequential':
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
            
            elif pattern.pattern_type == 'temporal':
                # For temporal patterns, suggest tools based on time context
                if self._is_temporal_pattern_relevant(pattern, current_hour):
                    for tool in pattern.tool_sequence:
                        if tool not in current_tools:
                            # Calculate score with temporal boost
                            score = self._calculate_pattern_score(pattern, current_hour)
                            suggestions[tool] = max(suggestions[tool], score)
        
        # Return top k suggestions
        return suggestions.most_common(k)
    
    def _is_temporal_pattern_relevant(self, pattern: Pattern, current_hour: int) -> bool:
        """Check if temporal pattern is relevant at current time.
        
        Args:
            pattern: Temporal pattern
            current_hour: Current hour of day
            
        Returns:
            True if pattern is relevant
        """
        if not pattern.temporal_metadata:
            return False
        
        subtype = pattern.temporal_metadata.get('pattern_subtype')
        if subtype == 'hourly':
            pattern_hour = pattern.temporal_metadata.get('hour', -1)
            return abs(current_hour - pattern_hour) <= 1
        elif subtype == 'periodic':
            # Periodic patterns are always potentially relevant
            return True
        elif subtype == 'time_cluster':
            avg_time = pattern.temporal_metadata.get('avg_time', '')
            if avg_time:
                cluster_hour = int(avg_time.split(':')[0])
                return abs(current_hour - cluster_hour) <= 2
        
        return True
    
    async def mine_context_aware_patterns(self, sequences: List[ExecutionSequence]) -> Dict[str, List[Pattern]]:
        """Mine patterns grouped by context (expertise and domain).
        
        Args:
            sequences: List of execution sequences
            
        Returns:
            Dictionary with context keys and patterns as values
        """
        logger.info("Starting context-aware pattern mining...")
        
        # Group sequences by context
        context_groups = defaultdict(list)
        for seq in sequences:
            context_key = f"{seq.user_expertise}:{seq.domain}"
            context_groups[context_key].append(seq)
        
        # Mine patterns for each context group
        context_patterns = {}
        for context_key, context_sequences in context_groups.items():
            logger.info(f"Mining patterns for context: {context_key} ({len(context_sequences)} sequences)")
            
            if len(context_sequences) < 5:  # Skip if too few sequences
                continue
            
            patterns = {
                'sequential': await self.mine_sequential_patterns(context_sequences),
                'combination': await self.mine_combination_patterns(context_sequences),
                'temporal': await self.mine_temporal_patterns(context_sequences)
            }
            
            # Add context information to patterns
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if context_key not in pattern.contexts:
                        pattern.contexts.append(context_key)
            
            context_patterns[context_key] = patterns
        
        return context_patterns
    
    def get_context_matching_patterns(self, tools: List[str], user_expertise: str, 
                                    domain: str, context: Optional[Dict] = None) -> List[Pattern]:
        """Get patterns that match the given tools and context.
        
        Args:
            tools: List of tools to match
            user_expertise: User expertise level
            domain: Domain context
            context: Optional additional context
            
        Returns:
            List of matching patterns sorted by context-aware score
        """
        matching_patterns = []
        current_time = datetime.now()
        current_hour = current_time.hour
        context_key = f"{user_expertise}:{domain}"
        
        for pattern_hash, pattern in self.discovered_patterns.items():
            # Check if pattern matches tools
            pattern_matches = False
            if pattern.pattern_type == 'sequential':
                pattern_matches = self._is_subsequence(pattern.tool_sequence, tools)
            elif pattern.pattern_type == 'combination':
                pattern_matches = all(tool in tools for tool in pattern.tool_sequence)
            elif pattern.pattern_type == 'temporal':
                pattern_matches = self._matches_temporal_pattern(pattern, tools, current_hour)
            
            if pattern_matches:
                # Check if pattern is relevant to context
                context_relevance = self._calculate_context_relevance(pattern, context_key)
                if context_relevance > 0:
                    matching_patterns.append((pattern, context_relevance))
        
        # Sort by context-aware score
        matching_patterns.sort(
            key=lambda x: self._calculate_context_aware_score(x[0], x[1], current_hour),
            reverse=True
        )
        
        return [p[0] for p in matching_patterns]
    
    def _calculate_context_relevance(self, pattern: Pattern, context_key: str) -> float:
        """Calculate how relevant a pattern is to the given context.
        
        Args:
            pattern: Pattern to evaluate
            context_key: Context key (expertise:domain)
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not pattern.contexts:
            return 0.5  # Default relevance for patterns without context
        
        # Direct match
        if context_key in pattern.contexts:
            return 1.0
        
        # Partial match (same expertise or domain)
        user_expertise, domain = context_key.split(':')
        partial_match_score = 0.0
        
        for ctx in pattern.contexts:
            # Handle both old and new context formats
            if ':' in ctx:
                ctx_expertise, ctx_domain = ctx.split(':', 1)
                if ctx_expertise == user_expertise:
                    partial_match_score = max(partial_match_score, 0.7)
                if ctx_domain == domain:
                    partial_match_score = max(partial_match_score, 0.6)
            else:
                # Legacy format or single context value
                if ctx == user_expertise or ctx == domain:
                    partial_match_score = max(partial_match_score, 0.5)
        
        return partial_match_score
    
    def _calculate_context_aware_score(self, pattern: Pattern, context_relevance: float, 
                                     current_hour: int) -> float:
        """Calculate pattern score with context awareness.
        
        Args:
            pattern: Pattern to score
            context_relevance: Context relevance score
            current_hour: Current hour of day
            
        Returns:
            Context-aware pattern score
        """
        base_score = pattern.confidence * pattern.lift
        
        # Apply context relevance multiplier
        context_multiplier = 0.5 + (0.5 * context_relevance)  # Range: 0.5 to 1.0
        
        # Apply temporal boost if applicable
        temporal_boost = 1.0
        if pattern.pattern_type == 'temporal' and pattern.temporal_metadata:
            subtype = pattern.temporal_metadata.get('pattern_subtype')
            if subtype == 'hourly':
                pattern_hour = pattern.temporal_metadata.get('hour', -1)
                if abs(current_hour - pattern_hour) <= 1:
                    temporal_boost = 1.5
            elif subtype == 'duration':
                consistency = pattern.temporal_metadata.get('duration_consistency', 0)
                temporal_boost = 1 + consistency * 0.3
        
        return base_score * context_multiplier * temporal_boost
    
    async def mine_patterns(self, use_context_aware: bool = True) -> Dict[str, List[Pattern]]:
        """Run complete pattern mining process.
        
        Args:
            use_context_aware: Whether to use context-aware mining
            
        Returns:
            Dictionary with pattern types as keys and patterns as values
        """
        logger.info(f"Starting pattern mining process (context-aware: {use_context_aware})...")
        
        # Extract sequences
        sequences = await self.extract_sequences()
        
        if not sequences:
            logger.warning("No sequences found for mining")
            return {}
        
        all_patterns = {
            'sequential': [],
            'combination': [],
            'temporal': []
        }
        
        if use_context_aware:
            # Mine patterns grouped by context
            context_patterns = await self.mine_context_aware_patterns(sequences)
            
            # Aggregate patterns from all contexts
            for context_key, patterns_dict in context_patterns.items():
                for pattern_type, pattern_list in patterns_dict.items():
                    all_patterns[pattern_type].extend(pattern_list)
            
            logger.info(f"Mined patterns for {len(context_patterns)} different contexts")
        else:
            # Traditional mining without context grouping
            all_patterns['sequential'] = await self.mine_sequential_patterns(sequences)
            all_patterns['combination'] = await self.mine_combination_patterns(sequences)
            all_patterns['temporal'] = await self.mine_temporal_patterns(sequences)
        
        # Store all patterns
        all_pattern_list = (all_patterns['sequential'] + 
                           all_patterns['combination'] + 
                           all_patterns['temporal'])
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
            async with self._get_connection() as db:
                await db.execute("""
                    UPDATE discovered_patterns 
                    SET usage_count = usage_count + 1
                    WHERE pattern_type = ? AND tool_sequence = ?
                """, (pattern.pattern_type, json.dumps(pattern.tool_sequence)))
                await db.commit()
    
    async def incremental_update(self, batch_size: int = 1000, decay_factor: float = 0.95) -> Dict[str, List[Pattern]]:
        """Perform incremental pattern mining update.
        
        This is the main entry point for incremental pattern discovery. It:
        1. Extracts only new execution sequences since last update
        2. Loads existing pattern statistics
        3. Mines patterns incrementally from new data
        4. Merges new patterns with existing ones
        5. Stores updated patterns
        
        Args:
            batch_size: Maximum number of new sequences to process
            decay_factor: Factor to decay old pattern scores
            
        Returns:
            Dictionary with pattern types as keys and patterns as values
        """
        logger.info("Starting incremental pattern mining update...")
        
        # Extract new sequences since last update
        new_sequences = await self.extract_new_sequences(batch_size)
        
        if not new_sequences:
            logger.info("No new sequences to process")
            return {'sequential': [], 'combination': [], 'temporal': []}
        
        logger.info(f"Processing {len(new_sequences)} new sequences for incremental update")
        
        # Load existing pattern statistics
        existing_stats = await self.load_pattern_statistics()
        logger.info(f"Loaded {len(existing_stats)} existing pattern statistics")
        
        # Mine patterns incrementally
        all_patterns = {
            'sequential': [],
            'combination': [],
            'temporal': []
        }
        
        # Mine sequential patterns incrementally
        try:
            sequential_patterns = await self.mine_incremental_sequential_patterns(
                new_sequences, existing_stats
            )
            all_patterns['sequential'] = sequential_patterns
        except Exception as e:
            logger.error(f"Error mining sequential patterns: {e}")
        
        # Mine combination patterns incrementally
        try:
            combination_patterns = await self.mine_incremental_combination_patterns(
                new_sequences, existing_stats
            )
            all_patterns['combination'] = combination_patterns
        except Exception as e:
            logger.error(f"Error mining combination patterns: {e}")
        
        # Mine temporal patterns incrementally
        try:
            temporal_patterns = await self.mine_incremental_temporal_patterns(
                new_sequences, existing_stats
            )
            all_patterns['temporal'] = temporal_patterns
        except Exception as e:
            logger.error(f"Error mining temporal patterns: {e}")
        
        # Merge all new patterns with existing ones
        all_pattern_list = (
            all_patterns['sequential'] + 
            all_patterns['combination'] + 
            all_patterns['temporal']
        )
        
        # Merge patterns with decay
        merged_patterns = await self.merge_patterns(all_pattern_list, decay_factor)
        
        # Store updated patterns in database
        await self.store_patterns(merged_patterns)
        
        # Reload patterns into memory
        await self.load_patterns()
        
        # Prune outdated patterns
        await self.prune_outdated_patterns()
        
        logger.info(f"Incremental update complete. Total patterns: {len(self.discovered_patterns)}")
        logger.info(f"New/updated patterns: {len(all_pattern_list)}")
        
        return all_patterns
    
    async def prune_outdated_patterns(self, min_support_threshold: float = None,
                                    max_age_days: int = 90) -> int:
        """Remove patterns that are below thresholds or too old.
        
        Args:
            min_support_threshold: Minimum support to keep pattern (default: min_support * 0.5)
            max_age_days: Maximum age in days for patterns
            
        Returns:
            Number of patterns pruned
        """
        if min_support_threshold is None:
            min_support_threshold = self.min_support * 0.5
        
        pruned_count = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        async with self._get_connection() as db:
            # Remove old patterns
            cursor = await db.execute("""
                DELETE FROM discovered_patterns
                WHERE support < ? OR discovered_at < ?
            """, (min_support_threshold, cutoff_date.isoformat()))
            
            pruned_count += cursor.rowcount
            
            # Remove outdated pattern statistics
            cursor = await db.execute("""
                DELETE FROM pattern_statistics
                WHERE last_seen < ? OR total_support < ?
            """, (cutoff_date.isoformat(), min_support_threshold))
            
            pruned_count += cursor.rowcount
            
            await db.commit()
        
        logger.info(f"Pruned {pruned_count} outdated patterns")
        return pruned_count