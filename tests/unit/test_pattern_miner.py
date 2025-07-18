"""Unit tests for PatternMiner class."""

import pytest
import asyncio
import json
import tempfile
import aiosqlite
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.learning.pattern_miner import PatternMiner, Pattern, ExecutionSequence


class TestPatternMiner:
    """Test cases for PatternMiner."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create temporary database with test data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
            
        # Create tables
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE execution_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    intent JSON NOT NULL,
                    tools_used JSON NOT NULL,
                    execution_time_ms INTEGER,
                    success BOOLEAN,
                    reward REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE discovered_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    tool_sequence JSON NOT NULL,
                    support REAL NOT NULL,
                    confidence REAL NOT NULL,
                    lift REAL,
                    contexts JSON,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            # Insert test execution data
            test_executions = [
                # Successful pattern: filesystem -> sqlite
                ("exec1", "Find files and analyze", '{"type": "query.search"}', 
                 '["filesystem_mcp", "sqlite_mcp"]', True, 1.0),
                ("exec2", "Search and store", '{"type": "query.search"}', 
                 '["filesystem_mcp", "sqlite_mcp"]', True, 0.9),
                ("exec3", "Find and process", '{"type": "query.search"}', 
                 '["filesystem_mcp", "sqlite_mcp"]', True, 0.8),
                
                # Another pattern: search -> github
                ("exec4", "Search code", '{"type": "query.search"}', 
                 '["search_mcp", "github_mcp"]', True, 0.9),
                ("exec5", "Find in repos", '{"type": "query.search"}', 
                 '["search_mcp", "github_mcp"]', True, 0.85),
                
                # Failed sequences
                ("exec6", "Bad query", '{"type": "query.search"}', 
                 '["filesystem_mcp", "weather_mcp"]', False, -0.5),
                ("exec7", "Wrong tools", '{"type": "action.create"}', 
                 '["weather_mcp"]', False, -0.3),
                
                # Single tool success
                ("exec8", "Simple search", '{"type": "query.search"}', 
                 '["search_mcp"]', True, 0.5),
                ("exec9", "Quick find", '{"type": "query.search"}', 
                 '["filesystem_mcp"]', True, 0.6),
            ]
            
            for exec_data in test_executions:
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (*exec_data, datetime.now().isoformat()))
            
            await db.commit()
            
        yield db_path
        
        # Cleanup
        import os
        os.unlink(db_path)
    
    @pytest.fixture
    def mock_config(self, temp_db):
        """Create mock configuration."""
        config = Mock()
        config.get.return_value = {
            'path': temp_db,
            'max_pattern_length': 3,
            'time_window_days': 30
        }
        return config
    
    @pytest.fixture
    async def pattern_miner(self, mock_config, temp_db):
        """Create PatternMiner instance."""
        miner = PatternMiner(mock_config, min_support=0.2, min_confidence=0.6)
        miner.db_path = temp_db
        return miner
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test PatternMiner initialization."""
        miner = PatternMiner(mock_config, min_support=0.15, min_confidence=0.75)
        
        assert miner.min_support == 0.15
        assert miner.min_confidence == 0.75
        assert miner.min_lift == 1.0
        assert miner.max_pattern_length == 3
        assert isinstance(miner.discovered_patterns, dict)
        assert isinstance(miner.pattern_cache, dict)
    
    @pytest.mark.asyncio
    async def test_extract_sequences(self, pattern_miner):
        """Test extraction of execution sequences."""
        sequences = await pattern_miner.extract_sequences()
        
        assert len(sequences) == 9
        assert all(isinstance(seq, ExecutionSequence) for seq in sequences)
        
        # Check first sequence
        seq = sequences[0]
        assert seq.execution_id == "exec9"  # Most recent first
        assert seq.tools == ["filesystem_mcp"]
        assert seq.success is True
        assert seq.reward == 0.6
    
    @pytest.mark.asyncio
    async def test_extract_sequences_with_time_window(self, pattern_miner):
        """Test extraction with time window."""
        # Very short time window should return no results
        sequences = await pattern_miner.extract_sequences(timedelta(seconds=1))
        assert len(sequences) == 0
        
        # Long time window should return all results
        sequences = await pattern_miner.extract_sequences(timedelta(days=365))
        assert len(sequences) == 9
    
    def test_generate_subsequences(self, pattern_miner):
        """Test subsequence generation."""
        sequence = ["A", "B", "C"]
        
        # Max length 2
        subseqs = pattern_miner._generate_subsequences(sequence, 2)
        expected = [["A"], ["B"], ["C"], ["A", "B"], ["B", "C"]]
        assert sorted(subseqs) == sorted(expected)
        
        # Max length 3
        subseqs = pattern_miner._generate_subsequences(sequence, 3)
        expected.append(["A", "B", "C"])
        assert sorted(subseqs) == sorted(expected)
    
    def test_is_subsequence(self, pattern_miner):
        """Test subsequence checking."""
        # Positive cases
        assert pattern_miner._is_subsequence(["A"], ["A", "B", "C"]) is True
        assert pattern_miner._is_subsequence(["A", "C"], ["A", "B", "C"]) is True
        assert pattern_miner._is_subsequence(["B", "C"], ["A", "B", "C"]) is True
        
        # Negative cases
        assert pattern_miner._is_subsequence(["C", "A"], ["A", "B", "C"]) is False
        assert pattern_miner._is_subsequence(["D"], ["A", "B", "C"]) is False
        assert pattern_miner._is_subsequence(["A", "B", "C", "D"], ["A", "B", "C"]) is False
    
    @pytest.mark.asyncio
    async def test_calculate_support(self, pattern_miner):
        """Test support calculation."""
        sequences = await pattern_miner.extract_sequences()
        
        # Pattern that appears 3 times
        support = pattern_miner._calculate_support(["filesystem_mcp", "sqlite_mcp"], sequences)
        assert support == pytest.approx(3/9, rel=1e-3)
        
        # Pattern that appears 2 times
        support = pattern_miner._calculate_support(["search_mcp", "github_mcp"], sequences)
        assert support == pytest.approx(2/9, rel=1e-3)
        
        # Pattern that doesn't exist
        support = pattern_miner._calculate_support(["nonexistent_mcp"], sequences)
        assert support == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, pattern_miner):
        """Test confidence calculation."""
        sequences = await pattern_miner.extract_sequences()
        
        # Pattern with 100% success rate (3 successes out of 3)
        confidence = pattern_miner._calculate_confidence(["filesystem_mcp", "sqlite_mcp"], sequences)
        assert confidence == 1.0
        
        # Single tool confidence
        confidence = pattern_miner._calculate_confidence(["filesystem_mcp"], sequences)
        assert confidence > 0.5  # Should be successful
    
    @pytest.mark.asyncio
    async def test_calculate_lift(self, pattern_miner):
        """Test lift calculation."""
        sequences = await pattern_miner.extract_sequences()
        
        # Calculate lift for a pattern
        lift = pattern_miner._calculate_lift(["filesystem_mcp", "sqlite_mcp"], sequences)
        assert lift > 1.0  # Should show positive correlation
        
        # Single item should have lift of 1.0
        lift = pattern_miner._calculate_lift(["filesystem_mcp"], sequences)
        assert lift == 1.0
    
    @pytest.mark.asyncio
    async def test_mine_sequential_patterns(self, pattern_miner):
        """Test sequential pattern mining."""
        sequences = await pattern_miner.extract_sequences()
        patterns = await pattern_miner.mine_sequential_patterns(sequences)
        
        assert len(patterns) > 0
        
        # Check pattern properties
        for pattern in patterns:
            assert pattern.pattern_type == 'sequential'
            assert pattern.support >= pattern_miner.min_support
            assert pattern.confidence >= pattern_miner.min_confidence
            assert pattern.lift >= pattern_miner.min_lift
            assert isinstance(pattern.tool_sequence, list)
            assert len(pattern.tool_sequence) <= pattern_miner.max_pattern_length
        
        # Should find the filesystem->sqlite pattern
        fs_sqlite_patterns = [
            p for p in patterns 
            if p.tool_sequence == ["filesystem_mcp", "sqlite_mcp"]
        ]
        assert len(fs_sqlite_patterns) == 1
        assert fs_sqlite_patterns[0].confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_mine_combination_patterns(self, pattern_miner):
        """Test combination pattern mining."""
        sequences = await pattern_miner.extract_sequences()
        patterns = await pattern_miner.mine_combination_patterns(sequences)
        
        # Check pattern properties
        for pattern in patterns:
            assert pattern.pattern_type == 'combination'
            assert pattern.support >= pattern_miner.min_support
            assert pattern.confidence >= pattern_miner.min_confidence
            assert pattern.lift >= pattern_miner.min_lift
            assert len(pattern.tool_sequence) >= 2
    
    @pytest.mark.asyncio
    async def test_store_and_load_patterns(self, pattern_miner):
        """Test pattern storage and loading."""
        # Create test patterns
        patterns = [
            Pattern(
                pattern_type='sequential',
                tool_sequence=['tool1', 'tool2'],
                support=0.5,
                confidence=0.8,
                lift=1.5,
                contexts=['context1']
            ),
            Pattern(
                pattern_type='combination',
                tool_sequence=['tool3', 'tool4'],
                support=0.3,
                confidence=0.7,
                lift=1.2
            )
        ]
        
        # Store patterns
        await pattern_miner.store_patterns(patterns)
        
        # Load patterns
        await pattern_miner.load_patterns()
        
        assert len(pattern_miner.discovered_patterns) == 2
        
        # Check pattern content
        loaded_patterns = list(pattern_miner.discovered_patterns.values())
        assert any(p.tool_sequence == ['tool1', 'tool2'] for p in loaded_patterns)
        assert any(p.tool_sequence == ['tool3', 'tool4'] for p in loaded_patterns)
    
    def test_get_matching_patterns(self, pattern_miner):
        """Test pattern matching."""
        # Add test patterns
        pattern1 = Pattern(
            pattern_type='sequential',
            tool_sequence=['A', 'B'],
            support=0.5,
            confidence=0.8,
            lift=1.5
        )
        pattern2 = Pattern(
            pattern_type='combination',
            tool_sequence=['B', 'C'],
            support=0.3,
            confidence=0.7,
            lift=1.2
        )
        
        pattern_miner.discovered_patterns[pattern1.get_hash()] = pattern1
        pattern_miner.discovered_patterns[pattern2.get_hash()] = pattern2
        
        # Test sequential matching
        matches = pattern_miner.get_matching_patterns(['A', 'B', 'C'])
        assert len(matches) == 2
        
        # Test combination matching
        matches = pattern_miner.get_matching_patterns(['C', 'B', 'D'])
        assert len(matches) == 1
        assert matches[0].pattern_type == 'combination'
    
    def test_suggest_next_tools(self, pattern_miner):
        """Test tool suggestion."""
        # Add sequential patterns
        patterns = [
            Pattern(
                pattern_type='sequential',
                tool_sequence=['A', 'B', 'C'],
                support=0.5,
                confidence=0.9,
                lift=1.8
            ),
            Pattern(
                pattern_type='sequential',
                tool_sequence=['A', 'B', 'D'],
                support=0.3,
                confidence=0.7,
                lift=1.4
            ),
            Pattern(
                pattern_type='sequential',
                tool_sequence=['A', 'E'],
                support=0.2,
                confidence=0.6,
                lift=1.2
            )
        ]
        
        for pattern in patterns:
            pattern_miner.discovered_patterns[pattern.get_hash()] = pattern
        
        # Test suggestions
        suggestions = pattern_miner.suggest_next_tools(['A', 'B'], k=2)
        assert len(suggestions) == 2
        assert suggestions[0][0] == 'C'  # Higher score
        assert suggestions[1][0] == 'D'  # Lower score
        assert suggestions[0][1] > suggestions[1][1]  # Scores are ordered
    
    @pytest.mark.asyncio
    async def test_mine_patterns_integration(self, pattern_miner):
        """Test complete pattern mining process."""
        results = await pattern_miner.mine_patterns()
        
        assert 'sequential' in results
        assert 'combination' in results
        assert len(results['sequential']) > 0
        
        # Check that patterns are stored in memory
        assert len(pattern_miner.discovered_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_update_pattern_usage(self, pattern_miner):
        """Test pattern usage update."""
        # Create and store a pattern
        pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['test1', 'test2'],
            support=0.5,
            confidence=0.8,
            lift=1.5
        )
        pattern_hash = pattern.get_hash()
        pattern_miner.discovered_patterns[pattern_hash] = pattern
        
        # Store in database
        await pattern_miner.store_patterns([pattern])
        
        # Update usage
        await pattern_miner.update_pattern_usage(pattern_hash)
        
        # Check update
        assert pattern_miner.discovered_patterns[pattern_hash].usage_count == 1
    
    def test_pattern_hash(self):
        """Test pattern hashing."""
        pattern1 = Pattern(
            pattern_type='sequential',
            tool_sequence=['A', 'B'],
            support=0.5,
            confidence=0.8,
            lift=1.5
        )
        
        pattern2 = Pattern(
            pattern_type='sequential',
            tool_sequence=['B', 'A'],  # Different order
            support=0.5,
            confidence=0.8,
            lift=1.5
        )
        
        # Sequential patterns with different order should have different hashes
        assert pattern1.get_hash() != pattern2.get_hash()
        
        # Same pattern should have same hash
        pattern3 = Pattern(
            pattern_type='sequential',
            tool_sequence=['A', 'B'],
            support=0.3,  # Different metrics
            confidence=0.6,
            lift=1.2
        )
        assert pattern1.get_hash() == pattern3.get_hash()
    
    def test_pattern_to_dict(self):
        """Test pattern serialization."""
        pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['tool1', 'tool2'],
            support=0.5,
            confidence=0.8,
            lift=1.5,
            contexts=['ctx1', 'ctx2'],
            usage_count=10
        )
        
        pattern_dict = pattern.to_dict()
        
        assert pattern_dict['pattern_type'] == 'sequential'
        assert json.loads(pattern_dict['tool_sequence']) == ['tool1', 'tool2']
        assert pattern_dict['support'] == 0.5
        assert pattern_dict['confidence'] == 0.8
        assert pattern_dict['lift'] == 1.5
        assert json.loads(pattern_dict['contexts']) == ['ctx1', 'ctx2']
        assert pattern_dict['usage_count'] == 10
        assert 'discovered_at' in pattern_dict