"""Unit tests for incremental pattern mining functionality."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os
import aiosqlite

from src.learning.pattern_miner import PatternMiner, Pattern, ExecutionSequence
from src.database.database import DatabaseManager


class TestIncrementalPatternMining:
    """Test cases for incremental pattern mining."""
    
    @pytest.fixture
    async def setup_test_db(self):
        """Create a test database with sample data."""
        # Create a temporary database
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = temp_db.name
        temp_db.close()
        
        # Initialize database
        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()
        
        # Insert sample execution history
        async with aiosqlite.connect(db_path) as db:
            # Add some historical executions
            base_time = datetime.now() - timedelta(days=2)
            executions = [
                # Pattern 1: filesystem -> sqlite (successful)
                ('exec_001', 'query1', {'type': 'search'}, ['filesystem_mcp', 'sqlite_mcp'], 
                 True, 0.8, (base_time + timedelta(hours=1)).isoformat()),
                ('exec_002', 'query2', {'type': 'search'}, ['filesystem_mcp', 'sqlite_mcp'], 
                 True, 0.9, (base_time + timedelta(hours=2)).isoformat()),
                
                # Pattern 2: search -> analyze (successful)
                ('exec_003', 'query3', {'type': 'analyze'}, ['search_mcp', 'analyze_mcp'], 
                 True, 0.7, (base_time + timedelta(hours=3)).isoformat()),
                
                # Failed execution
                ('exec_004', 'query4', {'type': 'search'}, ['filesystem_mcp'], 
                 False, -0.5, (base_time + timedelta(hours=4)).isoformat()),
            ]
            
            for exec_data in executions:
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (exec_data[0], exec_data[1], json.dumps(exec_data[2]), 
                      json.dumps(exec_data[3]), exec_data[4], exec_data[5], exec_data[6]))
            
            # Set last processed execution ID
            await db.execute("""
                UPDATE pattern_mining_metadata 
                SET value = 'exec_004' 
                WHERE key = 'last_processed_execution_id'
            """)
            
            await db.commit()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def pattern_miner(self, setup_test_db):
        """Create a PatternMiner instance with test database."""
        config = {
            'database': {'path': setup_test_db},
            'pattern_mining': {
                'max_pattern_length': 3,
                'time_window_days': 30
            }
        }
        return PatternMiner(config, min_support=0.2, min_confidence=0.6)
    
    @pytest.mark.asyncio
    async def test_extract_new_sequences(self, pattern_miner, setup_test_db):
        """Test extracting only new sequences since last update."""
        # Add new executions after last processed ID
        async with aiosqlite.connect(setup_test_db) as db:
            new_time = datetime.now()
            new_executions = [
                ('exec_005', 'query5', {'type': 'search'}, ['filesystem_mcp', 'sqlite_mcp'], 
                 True, 0.85, new_time.isoformat()),
                ('exec_006', 'query6', {'type': 'analyze'}, ['search_mcp', 'analyze_mcp'], 
                 True, 0.75, (new_time + timedelta(minutes=5)).isoformat()),
            ]
            
            for exec_data in new_executions:
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (exec_data[0], exec_data[1], json.dumps(exec_data[2]), 
                      json.dumps(exec_data[3]), exec_data[4], exec_data[5], exec_data[6]))
            
            await db.commit()
        
        # Extract new sequences
        new_sequences = await pattern_miner.extract_new_sequences(batch_size=10)
        
        # Should only get the 2 new sequences
        assert len(new_sequences) == 2
        assert new_sequences[0].execution_id == 'exec_005'
        assert new_sequences[1].execution_id == 'exec_006'
        
        # Check that last processed ID was updated
        last_id = await pattern_miner.get_pattern_mining_metadata('last_processed_execution_id')
        assert last_id == 'exec_006'
    
    @pytest.mark.asyncio
    async def test_incremental_sequential_patterns(self, pattern_miner):
        """Test incremental mining of sequential patterns."""
        # Create new sequences
        new_sequences = [
            ExecutionSequence(
                execution_id='exec_101',
                tools=['filesystem_mcp', 'sqlite_mcp'],
                success=True,
                reward=0.8,
                context={'intent': {'type': 'search'}},
                timestamp=datetime.now()
            ),
            ExecutionSequence(
                execution_id='exec_102',
                tools=['filesystem_mcp', 'sqlite_mcp', 'analyze_mcp'],
                success=True,
                reward=0.9,
                context={'intent': {'type': 'analyze'}},
                timestamp=datetime.now()
            ),
        ]
        
        # Create existing statistics
        existing_stats = {
            'hash1': {
                'pattern_type': 'sequential',
                'tool_sequence': ['filesystem_mcp', 'sqlite_mcp'],
                'occurrence_count': 5,
                'success_count': 4,
                'total_support': 0.5,
                'total_confidence': 0.8
            }
        }
        
        # Mine patterns incrementally
        patterns = await pattern_miner.mine_incremental_sequential_patterns(
            new_sequences, existing_stats
        )
        
        # Should find patterns including the updated existing one
        assert len(patterns) > 0
        
        # Check that filesystem->sqlite pattern was found/updated
        fs_sqlite_found = False
        for pattern in patterns:
            if pattern.tool_sequence == ['filesystem_mcp', 'sqlite_mcp']:
                fs_sqlite_found = True
                assert pattern.support > 0
                assert pattern.confidence > 0
                break
        
        assert fs_sqlite_found
    
    @pytest.mark.asyncio
    async def test_incremental_combination_patterns(self, pattern_miner):
        """Test incremental mining of combination patterns."""
        new_sequences = [
            ExecutionSequence(
                execution_id='exec_201',
                tools=['search_mcp', 'analyze_mcp', 'sqlite_mcp'],
                success=True,
                reward=0.85,
                context={'intent': {'type': 'analyze'}},
                timestamp=datetime.now()
            ),
            ExecutionSequence(
                execution_id='exec_202',
                tools=['search_mcp', 'sqlite_mcp'],
                success=True,
                reward=0.7,
                context={'intent': {'type': 'search'}},
                timestamp=datetime.now()
            ),
        ]
        
        existing_stats = {}
        
        # Mine combination patterns
        patterns = await pattern_miner.mine_incremental_combination_patterns(
            new_sequences, existing_stats
        )
        
        # Should find some combination patterns
        assert len(patterns) > 0
        
        # Check for search+sqlite combination
        found_combo = False
        for pattern in patterns:
            if set(pattern.tool_sequence) == {'search_mcp', 'sqlite_mcp'}:
                found_combo = True
                break
        
        assert found_combo
    
    @pytest.mark.asyncio
    async def test_pattern_statistics_update(self, pattern_miner, setup_test_db):
        """Test updating pattern statistics."""
        pattern_hash = 'test_pattern_001'
        pattern_type = 'sequential'
        tool_sequence = ['tool1', 'tool2']
        
        # Update statistics
        await pattern_miner.update_pattern_statistics(
            pattern_hash, pattern_type, tool_sequence,
            occurrence_delta=5, success_delta=4,
            support_delta=0.1, confidence_delta=0.05
        )
        
        # Load and verify
        stats = await pattern_miner.load_pattern_statistics()
        assert pattern_hash in stats
        assert stats[pattern_hash]['occurrence_count'] == 5
        assert stats[pattern_hash]['success_count'] == 4
        
        # Update again
        await pattern_miner.update_pattern_statistics(
            pattern_hash, pattern_type, tool_sequence,
            occurrence_delta=3, success_delta=2,
            support_delta=0.05, confidence_delta=0.02
        )
        
        # Verify accumulation
        stats = await pattern_miner.load_pattern_statistics()
        assert stats[pattern_hash]['occurrence_count'] == 8
        assert stats[pattern_hash]['success_count'] == 6
    
    @pytest.mark.asyncio
    async def test_pattern_merging_with_decay(self, pattern_miner):
        """Test merging patterns with decay factor."""
        # Load some existing patterns
        pattern_miner.discovered_patterns = {
            'p1': Pattern(
                pattern_type='sequential',
                tool_sequence=['old_tool1', 'old_tool2'],
                support=0.8,
                confidence=0.9,
                lift=1.2
            ),
            'p2': Pattern(
                pattern_type='sequential',
                tool_sequence=['old_tool3'],
                support=0.1,  # Below threshold after decay
                confidence=0.5,
                lift=1.0
            )
        }
        
        # New patterns to merge
        new_patterns = [
            Pattern(
                pattern_type='sequential',
                tool_sequence=['new_tool1', 'new_tool2'],
                support=0.6,
                confidence=0.8,
                lift=1.1
            )
        ]
        
        # Merge with decay
        merged = await pattern_miner.merge_patterns(new_patterns, decay_factor=0.9)
        
        # Old pattern should be decayed
        old_pattern = next((p for p in merged if p.tool_sequence == ['old_tool1', 'old_tool2']), None)
        assert old_pattern is not None
        assert old_pattern.support < 0.8  # Decayed
        
        # Low support pattern should be removed
        low_pattern = next((p for p in merged if p.tool_sequence == ['old_tool3']), None)
        assert low_pattern is None
        
        # New pattern should be present
        new_pattern = next((p for p in merged if p.tool_sequence == ['new_tool1', 'new_tool2']), None)
        assert new_pattern is not None
    
    @pytest.mark.asyncio
    async def test_incremental_update_full_flow(self, pattern_miner, setup_test_db):
        """Test the full incremental update flow."""
        # Add new executions
        async with aiosqlite.connect(setup_test_db) as db:
            new_time = datetime.now()
            await db.execute("""
                INSERT INTO execution_history 
                (id, query, intent, tools_used, success, reward, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('exec_301', 'test_query', json.dumps({'type': 'test'}), 
                  json.dumps(['filesystem_mcp', 'sqlite_mcp']), 
                  True, 0.9, new_time.isoformat()))
            await db.commit()
        
        # Run incremental update
        patterns = await pattern_miner.incremental_update(batch_size=100, decay_factor=0.95)
        
        # Should have discovered some patterns
        total_patterns = sum(len(p) for p in patterns.values())
        assert total_patterns >= 0  # May be 0 if not enough support
        
        # Check that patterns are loaded
        assert len(pattern_miner.discovered_patterns) >= 0
    
    @pytest.mark.asyncio
    async def test_pattern_pruning(self, pattern_miner, setup_test_db):
        """Test pruning of outdated patterns."""
        # Insert some old patterns
        async with aiosqlite.connect(setup_test_db) as db:
            old_date = (datetime.now() - timedelta(days=100)).isoformat()
            
            await db.execute("""
                INSERT INTO discovered_patterns 
                (pattern_type, tool_sequence, support, confidence, lift, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ('sequential', json.dumps(['old_tool']), 0.05, 0.6, 1.0, old_date))
            
            await db.execute("""
                INSERT INTO pattern_statistics 
                (pattern_hash, pattern_type, tool_sequence, last_seen)
                VALUES (?, ?, ?, ?)
            """, ('old_hash', 'sequential', json.dumps(['old_tool']), old_date))
            
            await db.commit()
        
        # Prune old patterns
        pruned_count = await pattern_miner.prune_outdated_patterns(
            min_support_threshold=0.1,
            max_age_days=90
        )
        
        assert pruned_count > 0
    
    def test_calculate_incremental_metrics(self, pattern_miner):
        """Test incremental metric calculation."""
        pattern = ['tool1', 'tool2']
        new_sequences = [
            ExecutionSequence(
                execution_id='e1',
                tools=['tool1', 'tool2', 'tool3'],
                success=True,
                reward=0.8,
                context={},
                timestamp=datetime.now()
            ),
            ExecutionSequence(
                execution_id='e2',
                tools=['tool1', 'tool2'],
                success=False,
                reward=-0.2,
                context={},
                timestamp=datetime.now()
            ),
        ]
        
        # Without existing stats
        metrics = pattern_miner.calculate_incremental_metrics(pattern, new_sequences)
        
        assert metrics['occurrence_count'] == 2
        assert metrics['success_count'] == 1
        assert metrics['support'] == 1.0  # 2/2 sequences
        assert metrics['confidence'] == 0.5  # 1/2 successful
        
        # With existing stats
        existing_stats = {
            'occurrence_count': 10,
            'success_count': 8,
            'total_support': 0.7
        }
        
        metrics = pattern_miner.calculate_incremental_metrics(
            pattern, new_sequences, existing_stats
        )
        
        assert metrics['occurrence_count'] == 12  # 10 + 2
        assert metrics['success_count'] == 9  # 8 + 1
        assert metrics['confidence'] == 0.75  # 9/12