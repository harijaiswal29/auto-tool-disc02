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
                    usage_count INTEGER DEFAULT 0,
                    temporal_metadata JSON
                )
            """)
            
            await db.execute("""
                CREATE TABLE pattern_statistics (
                    pattern_hash TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    tool_sequence JSON NOT NULL,
                    occurrence_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    total_support REAL DEFAULT 0.0,
                    total_confidence REAL DEFAULT 0.0,
                    last_seen TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE pattern_mining_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                # Use a timestamp 7 days ago for test data
                created_at = (datetime.now() - timedelta(days=7)).isoformat()
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (*exec_data, created_at))
            
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
        assert 'temporal' in results
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
    
    def test_temporal_pattern_creation(self):
        """Test temporal pattern with additional fields."""
        pattern = Pattern(
            pattern_type='temporal',
            tool_sequence=['tool1', 'tool2'],
            support=0.5,
            confidence=0.8,
            lift=1.5,
            time_intervals=[60.0, 120.0, 90.0],
            periodic_info={'has_periodicity': True, 'period_type': 'hourly'},
            duration_stats={'mean': 100.0, 'std': 20.0},
            temporal_metadata={'pattern_subtype': 'hourly', 'hour': 14}
        )
        
        assert pattern.pattern_type == 'temporal'
        assert pattern.time_intervals == [60.0, 120.0, 90.0]
        assert pattern.periodic_info['period_type'] == 'hourly'
        assert pattern.duration_stats['mean'] == 100.0
        assert pattern.temporal_metadata['hour'] == 14
    
    @pytest.mark.asyncio
    async def test_mine_temporal_patterns(self, pattern_miner, temp_db):
        """Test temporal pattern mining."""
        # Create sequences with temporal information
        now = datetime.now()
        sequences = []
        
        # Create hourly pattern (tools A and B used together at hour 14)
        for i in range(5):
            timestamp = now.replace(hour=14, minute=i*10)
            sequences.append(ExecutionSequence(
                execution_id=f'hourly_{i}',
                tools=['A', 'B'],
                success=True,
                reward=1.0,
                context={'intent': {'type': 'test'}},
                timestamp=timestamp,
                total_duration=1000.0
            ))
        
        # Create periodic pattern (tools C and D every hour)
        for hour in [9, 10, 11, 12, 13]:
            timestamp = now.replace(hour=hour, minute=30)
            sequences.append(ExecutionSequence(
                execution_id=f'periodic_{hour}',
                tools=['C', 'D'],
                success=True,
                reward=1.0,
                context={'intent': {'type': 'test'}},
                timestamp=timestamp,
                total_duration=2000.0
            ))
        
        # Create time-clustered pattern (tools E, F, G used together)
        base_time = now.replace(hour=16, minute=0)
        for i in range(4):
            timestamp = base_time + timedelta(minutes=i*5)
            sequences.append(ExecutionSequence(
                execution_id=f'cluster_{i}',
                tools=['E', 'F', 'G'],
                success=True,
                reward=1.0,
                context={'intent': {'type': 'test'}},
                timestamp=timestamp,
                total_duration=500.0
            ))
        
        # Mine temporal patterns
        temporal_patterns = await pattern_miner.mine_temporal_patterns(sequences)
        
        assert len(temporal_patterns) > 0
        
        # Check for different temporal pattern types
        pattern_subtypes = {p.temporal_metadata.get('pattern_subtype') for p in temporal_patterns if p.temporal_metadata}
        assert 'hourly' in pattern_subtypes or 'time_cluster' in pattern_subtypes
    
    def test_temporal_pattern_matching(self, pattern_miner):
        """Test temporal pattern matching with time context."""
        # Create temporal patterns
        hourly_pattern = Pattern(
            pattern_type='temporal',
            tool_sequence=['A', 'B'],
            support=0.5,
            confidence=0.8,
            lift=1.5,
            temporal_metadata={'pattern_subtype': 'hourly', 'hour': 14}
        )
        
        duration_pattern = Pattern(
            pattern_type='temporal',
            tool_sequence=['C', 'D'],
            support=0.4,
            confidence=0.9,
            lift=1.8,
            duration_stats={'mean': 1000.0, 'std': 100.0},
            temporal_metadata={'pattern_subtype': 'duration', 'duration_consistency': 0.9}
        )
        
        pattern_miner.discovered_patterns = {
            hourly_pattern.get_hash(): hourly_pattern,
            duration_pattern.get_hash(): duration_pattern
        }
        
        # Test matching at relevant hour
        with patch('src.learning.pattern_miner.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 14, 30)  # 2:30 PM
            
            matches = pattern_miner.get_matching_patterns(['A', 'B', 'C'])
            # Should match hourly pattern due to time relevance
            assert any(p.pattern_type == 'temporal' and p.temporal_metadata.get('hour') == 14 for p in matches)
    
    def test_temporal_tool_suggestions(self, pattern_miner):
        """Test tool suggestions with temporal context."""
        # Add temporal patterns
        morning_pattern = Pattern(
            pattern_type='temporal',
            tool_sequence=['morning_backup', 'morning_report'],
            support=0.6,
            confidence=0.9,
            lift=1.5,
            temporal_metadata={'pattern_subtype': 'hourly', 'hour': 8}
        )
        
        pattern_miner.discovered_patterns[morning_pattern.get_hash()] = morning_pattern
        
        # Test suggestions at relevant time
        with patch('src.learning.pattern_miner.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 8, 30)  # 8:30 AM
            
            suggestions = pattern_miner.suggest_next_tools([], k=3)
            
            # Should suggest morning tools
            suggested_tools = [tool for tool, score in suggestions]
            assert 'morning_backup' in suggested_tools or 'morning_report' in suggested_tools
    
    def test_periodicity_detection(self, pattern_miner):
        """Test periodicity detection in time series."""
        # For now, just verify the method exists and returns expected structure
        time_series = [i * 3600 for i in range(24)]  # Hourly data
        
        periodicity = pattern_miner._detect_periodicity(time_series, sampling_interval=3600)
        
        # Check structure
        assert 'has_periodicity' in periodicity
        assert isinstance(periodicity['has_periodicity'], bool)
        
        # If periodicity is detected, check fields
        if periodicity['has_periodicity']:
            assert 'period_seconds' in periodicity
            assert 'period_type' in periodicity
            assert periodicity['period_type'] in ['hourly', 'daily', 'weekly', 'custom']
    
    def test_time_interval_analysis(self, pattern_miner):
        """Test time interval statistics calculation."""
        intervals = [60.0, 120.0, 90.0, 100.0, 80.0]
        
        stats = pattern_miner._analyze_time_intervals(intervals)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert stats['mean'] == 90.0
        assert stats['median'] == 90.0
        assert stats['min'] == 60.0
        assert stats['max'] == 120.0
    
    # Context-Aware Pattern Mining Tests
    
    @pytest.mark.asyncio
    async def test_extract_sequences_with_context(self, pattern_miner, temp_db):
        """Test extraction of sequences includes context information."""
        # Add execution data with context columns
        async with aiosqlite.connect(temp_db) as db:
            # Add context columns if they don't exist
            try:
                await db.execute("ALTER TABLE execution_history ADD COLUMN user_expertise TEXT DEFAULT 'intermediate'")
                await db.execute("ALTER TABLE execution_history ADD COLUMN domain TEXT DEFAULT 'general'")
                await db.commit()
            except:
                pass  # Columns might already exist
            
            # Insert test data with context
            await db.execute("""
                INSERT INTO execution_history 
                (id, query, intent, tools_used, success, reward, user_expertise, domain, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("ctx_exec1", "Debug Python code", '{"type": "action.modify"}', 
                  '["filesystem_mcp", "git_mcp"]', True, 0.9, "expert", "engineering",
                  datetime.now().isoformat()))
            
            await db.execute("""
                INSERT INTO execution_history 
                (id, query, intent, tools_used, success, reward, user_expertise, domain, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("ctx_exec2", "What is a variable?", '{"type": "query.search"}', 
                  '["search_mcp"]', True, 0.5, "novice", "general",
                  datetime.now().isoformat()))
            
            await db.commit()
        
        # Extract sequences
        sequences = await pattern_miner.extract_sequences()
        
        # Find our context-aware sequences
        expert_seq = next((s for s in sequences if s.execution_id == "ctx_exec1"), None)
        novice_seq = next((s for s in sequences if s.execution_id == "ctx_exec2"), None)
        
        assert expert_seq is not None
        assert expert_seq.user_expertise == "expert"
        assert expert_seq.domain == "engineering"
        
        assert novice_seq is not None
        assert novice_seq.user_expertise == "novice"
        assert novice_seq.domain == "general"
    
    @pytest.mark.asyncio
    async def test_mine_context_aware_patterns(self, pattern_miner):
        """Test mining patterns grouped by context."""
        # Create sequences with different contexts
        sequences = [
            # Expert engineering patterns
            ExecutionSequence(
                execution_id='exp_eng_1',
                tools=['git_mcp', 'filesystem_mcp', 'postgres_mcp'],
                success=True,
                reward=1.0,
                context={'intent': {'type': 'action.modify'}},
                timestamp=datetime.now(),
                user_expertise='expert',
                domain='engineering'
            ),
            ExecutionSequence(
                execution_id='exp_eng_2',
                tools=['git_mcp', 'filesystem_mcp'],
                success=True,
                reward=0.9,
                context={'intent': {'type': 'action.modify'}},
                timestamp=datetime.now(),
                user_expertise='expert',
                domain='engineering'
            ),
            ExecutionSequence(
                execution_id='exp_eng_3',
                tools=['git_mcp', 'filesystem_mcp'],
                success=True,
                reward=0.95,
                context={'intent': {'type': 'action.create'}},
                timestamp=datetime.now(),
                user_expertise='expert',
                domain='engineering'
            ),
            ExecutionSequence(
                execution_id='exp_eng_4',
                tools=['git_mcp', 'postgres_mcp'],
                success=True,
                reward=0.85,
                context={'intent': {'type': 'action.modify'}},
                timestamp=datetime.now(),
                user_expertise='expert',
                domain='engineering'
            ),
            ExecutionSequence(
                execution_id='exp_eng_5',
                tools=['filesystem_mcp', 'postgres_mcp'],
                success=True,
                reward=0.9,
                context={'intent': {'type': 'action.modify'}},
                timestamp=datetime.now(),
                user_expertise='expert',
                domain='engineering'
            ),
            
            # Novice general patterns
            ExecutionSequence(
                execution_id='nov_gen_1',
                tools=['filesystem_mcp'],
                success=True,
                reward=0.6,
                context={'intent': {'type': 'query.search'}},
                timestamp=datetime.now(),
                user_expertise='novice',
                domain='general'
            ),
            ExecutionSequence(
                execution_id='nov_gen_2',
                tools=['search_mcp'],
                success=True,
                reward=0.5,
                context={'intent': {'type': 'query.search'}},
                timestamp=datetime.now(),
                user_expertise='novice',
                domain='general'
            ),
            ExecutionSequence(
                execution_id='nov_gen_3',
                tools=['filesystem_mcp'],
                success=True,
                reward=0.55,
                context={'intent': {'type': 'query.retrieve'}},
                timestamp=datetime.now(),
                user_expertise='novice',
                domain='general'
            ),
            ExecutionSequence(
                execution_id='nov_gen_4',
                tools=['search_mcp'],
                success=True,
                reward=0.6,
                context={'intent': {'type': 'query.search'}},
                timestamp=datetime.now(),
                user_expertise='novice',
                domain='general'
            ),
            ExecutionSequence(
                execution_id='nov_gen_5',
                tools=['filesystem_mcp'],
                success=True,
                reward=0.65,
                context={'intent': {'type': 'query.search'}},
                timestamp=datetime.now(),
                user_expertise='novice',
                domain='general'
            ),
            
            # Intermediate data science patterns
            ExecutionSequence(
                execution_id='int_ds_1',
                tools=['filesystem_mcp', 'sqlite_mcp'],
                success=True,
                reward=0.8,
                context={'intent': {'type': 'query.analyze'}},
                timestamp=datetime.now(),
                user_expertise='intermediate',
                domain='data_science'
            ),
            ExecutionSequence(
                execution_id='int_ds_2',
                tools=['filesystem_mcp', 'sqlite_mcp'],
                success=True,
                reward=0.85,
                context={'intent': {'type': 'query.analyze'}},
                timestamp=datetime.now(),
                user_expertise='intermediate',
                domain='data_science'
            ),
            ExecutionSequence(
                execution_id='int_ds_3',
                tools=['sqlite_mcp'],
                success=True,
                reward=0.75,
                context={'intent': {'type': 'query.analyze'}},
                timestamp=datetime.now(),
                user_expertise='intermediate',
                domain='data_science'
            ),
            ExecutionSequence(
                execution_id='int_ds_4',
                tools=['filesystem_mcp', 'sqlite_mcp'],
                success=True,
                reward=0.82,
                context={'intent': {'type': 'query.analyze'}},
                timestamp=datetime.now(),
                user_expertise='intermediate',
                domain='data_science'
            ),
            ExecutionSequence(
                execution_id='int_ds_5',
                tools=['sqlite_mcp', 'postgres_mcp'],
                success=True,
                reward=0.88,
                context={'intent': {'type': 'query.analyze'}},
                timestamp=datetime.now(),
                user_expertise='intermediate',
                domain='data_science'
            ),
        ]
        
        # Mine context-aware patterns
        context_patterns = await pattern_miner.mine_context_aware_patterns(sequences)
        
        # Verify we have patterns for different contexts
        assert 'expert:engineering' in context_patterns
        assert 'novice:general' in context_patterns
        assert 'intermediate:data_science' in context_patterns
        
        # Check expert engineering patterns
        exp_eng_patterns = context_patterns['expert:engineering']
        assert 'sequential' in exp_eng_patterns
        # Should find some patterns for expert engineering context
        assert len(exp_eng_patterns['sequential']) > 0
        
        # Check novice patterns are simpler
        nov_patterns = context_patterns['novice:general']
        assert 'sequential' in nov_patterns
        # Novice patterns should be mostly single tools
        single_tool_patterns = [p for p in nov_patterns['sequential'] 
                               if len(p.tool_sequence) == 1]
        assert len(single_tool_patterns) > 0
    
    def test_get_context_matching_patterns(self, pattern_miner):
        """Test pattern matching with context relevance."""
        # Add patterns with different contexts
        expert_pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['git_mcp', 'postgres_mcp'],
            support=0.5,
            confidence=0.9,
            lift=1.8,
            contexts=['expert:engineering', 'expert:devops']
        )
        
        intermediate_pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['filesystem_mcp', 'sqlite_mcp'],
            support=0.4,
            confidence=0.8,
            lift=1.5,
            contexts=['intermediate:data_science', 'intermediate:general']
        )
        
        novice_pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['filesystem_mcp'],
            support=0.6,
            confidence=0.7,
            lift=1.0,
            contexts=['novice:general']
        )
        
        pattern_miner.discovered_patterns = {
            expert_pattern.get_hash(): expert_pattern,
            intermediate_pattern.get_hash(): intermediate_pattern,
            novice_pattern.get_hash(): novice_pattern
        }
        
        # Test expert engineering context
        matches = pattern_miner.get_context_matching_patterns(
            ['git_mcp', 'postgres_mcp', 'filesystem_mcp'],
            user_expertise='expert',
            domain='engineering'
        )
        
        # Should match expert pattern with high relevance
        assert len(matches) > 0
        expert_match = next((p for p in matches if 'git_mcp' in p.tool_sequence), None)
        assert expert_match is not None
        # Patterns should be sorted by context-aware score
        
        # Test partial context match
        matches = pattern_miner.get_context_matching_patterns(
            ['filesystem_mcp', 'sqlite_mcp'],
            user_expertise='intermediate',
            domain='engineering'  # Different domain
        )
        
        # Should still match but with lower relevance
        intermediate_match = next((p for p in matches if 'sqlite_mcp' in p.tool_sequence), None)
        assert intermediate_match is not None
    
    def test_calculate_context_relevance(self, pattern_miner):
        """Test context relevance scoring."""
        pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['tool1', 'tool2'],
            support=0.5,
            confidence=0.8,
            lift=1.5,
            contexts=['expert:engineering', 'intermediate:engineering']
        )
        
        # Direct match
        relevance = pattern_miner._calculate_context_relevance(pattern, 'expert:engineering')
        assert relevance == 1.0
        
        # Same expertise, different domain
        relevance = pattern_miner._calculate_context_relevance(pattern, 'expert:data_science')
        assert relevance == 0.7  # Partial match on expertise
        
        # Same domain, different expertise
        relevance = pattern_miner._calculate_context_relevance(pattern, 'novice:engineering')
        assert relevance == 0.6  # Partial match on domain
        
        # No match
        relevance = pattern_miner._calculate_context_relevance(pattern, 'novice:devops')
        assert relevance == 0.0
        
        # Pattern with no contexts
        pattern_no_context = Pattern(
            pattern_type='sequential',
            tool_sequence=['tool3'],
            support=0.5,
            confidence=0.8,
            lift=1.5
        )
        relevance = pattern_miner._calculate_context_relevance(pattern_no_context, 'any:context')
        assert relevance == 0.5  # Default relevance
    
    def test_calculate_context_aware_score(self, pattern_miner):
        """Test combined scoring with context relevance."""
        pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['tool1', 'tool2'],
            support=0.5,
            confidence=0.8,
            lift=1.5
        )
        
        # High context relevance should boost score
        score = pattern_miner._calculate_context_aware_score(
            pattern, 
            context_relevance=1.0,
            current_hour=14
        )
        base_score = pattern_miner._calculate_context_aware_score(
            pattern,
            context_relevance=0.5,
            current_hour=14
        )
        assert score > base_score
        
        # Zero context relevance should significantly reduce score
        low_score = pattern_miner._calculate_context_aware_score(
            pattern,
            context_relevance=0.0,
            current_hour=14
        )
        assert low_score < base_score
    
    @pytest.mark.asyncio
    async def test_context_aware_pattern_suggestions(self, pattern_miner):
        """Test tool suggestions consider user context."""
        # Add context-aware patterns
        expert_eng_pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['git_mcp', 'postgres_mcp', 'github_mcp'],
            support=0.6,
            confidence=0.9,
            lift=2.0,
            contexts=['expert:engineering']
        )
        
        novice_pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['filesystem_mcp', 'search_mcp'],
            support=0.7,
            confidence=0.8,
            lift=1.3,
            contexts=['novice:general']
        )
        
        pattern_miner.discovered_patterns = {
            expert_eng_pattern.get_hash(): expert_eng_pattern,
            novice_pattern.get_hash(): novice_pattern
        }
        
        # Test suggestions for expert user
        with patch.object(pattern_miner, 'get_context_matching_patterns') as mock_match:
            mock_match.return_value = [expert_eng_pattern]
            
            suggestions = pattern_miner.suggest_next_tools(
                ['git_mcp'],
                k=3
            )
            
            # Should suggest tools from expert pattern
            suggested_tools = [tool for tool, _ in suggestions]
            assert 'postgres_mcp' in suggested_tools or 'github_mcp' in suggested_tools
    
    @pytest.mark.asyncio
    async def test_mine_patterns_with_context_aware_flag(self, pattern_miner):
        """Test mine_patterns with use_context_aware flag."""
        # Create diverse sequences
        sequences = [
            ExecutionSequence(
                execution_id=f'seq_{i}',
                tools=['filesystem_mcp', 'sqlite_mcp'] if i % 2 == 0 else ['search_mcp'],
                success=True,
                reward=0.8,
                context={'intent': {'type': 'query.search'}},
                timestamp=datetime.now(),
                user_expertise='expert' if i < 5 else 'novice',
                domain='engineering' if i < 5 else 'general'
            )
            for i in range(10)
        ]
        
        # Mine with context awareness
        results = await pattern_miner.mine_patterns(
            use_context_aware=True
        )
        
        # Should have patterns in results
        assert 'sequential' in results
        assert 'combination' in results
        
        # Patterns should have context information
        has_context_patterns = any(
            p.contexts is not None and len(p.contexts) > 0
            for p in results['sequential']
        )
        assert has_context_patterns
    
    def test_pattern_context_persistence(self, pattern_miner):
        """Test that pattern contexts are properly serialized."""
        pattern = Pattern(
            pattern_type='sequential',
            tool_sequence=['tool1', 'tool2'],
            support=0.5,
            confidence=0.8,
            lift=1.5,
            contexts=['expert:engineering', 'intermediate:data_science']
        )
        
        # Convert to dict
        pattern_dict = pattern.to_dict()
        
        # Contexts should be serialized as JSON
        assert 'contexts' in pattern_dict
        contexts_json = pattern_dict['contexts']
        contexts_list = json.loads(contexts_json)
        assert 'expert:engineering' in contexts_list
        assert 'intermediate:data_science' in contexts_list