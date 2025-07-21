"""Integration tests for context-aware pattern mining workflow."""

import pytest
import asyncio
import json
import tempfile
import aiosqlite
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.learning.context_extractor import ContextExtractor, UserContext
from src.learning.pattern_miner import PatternMiner, Pattern, ExecutionSequence
from src.learning.q_learning_engine import QLearningEngine


class TestContextAwarePatternMiningIntegration:
    """Integration tests for the complete context-aware pattern mining workflow."""
    
    @pytest.fixture
    async def temp_db_with_context(self):
        """Create temporary database with context-aware execution history."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
            
        # Create tables with context columns
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
                    user_expertise TEXT DEFAULT 'intermediate',
                    domain TEXT DEFAULT 'general',
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
            
            # Insert diverse test data
            test_executions = [
                # Expert engineering patterns
                ("exp_eng_1", "user1", "session1", "Refactor the authentication module", 
                 '{"type": "action.modify"}', '["git_mcp", "filesystem_mcp", "postgres_mcp"]',
                 1200, True, 0.95, "expert", "engineering"),
                 
                ("exp_eng_2", "user1", "session1", "Debug async pipeline issues",
                 '{"type": "action.modify"}', '["git_mcp", "filesystem_mcp"]',
                 800, True, 0.9, "expert", "engineering"),
                 
                ("exp_eng_3", "user1", "session2", "Optimize database queries",
                 '{"type": "action.modify"}', '["postgres_mcp", "filesystem_mcp"]',
                 1500, True, 0.85, "expert", "engineering"),
                 
                # Intermediate data science patterns
                ("int_ds_1", "user2", "session3", "Analyze sales data trends",
                 '{"type": "query.analyze"}', '["filesystem_mcp", "sqlite_mcp"]',
                 600, True, 0.8, "intermediate", "data_science"),
                 
                ("int_ds_2", "user2", "session3", "Create visualization for metrics",
                 '{"type": "query.analyze"}', '["sqlite_mcp", "filesystem_mcp"]',
                 700, True, 0.75, "intermediate", "data_science"),
                 
                ("int_ds_3", "user2", "session4", "Process CSV dataset",
                 '{"type": "query.analyze"}', '["filesystem_mcp", "sqlite_mcp"]',
                 550, True, 0.82, "intermediate", "data_science"),
                 
                # Novice general patterns
                ("nov_gen_1", "user3", "session5", "How to find files?",
                 '{"type": "query.search"}', '["filesystem_mcp"]',
                 200, True, 0.6, "novice", "general"),
                 
                ("nov_gen_2", "user3", "session5", "What is Python?",
                 '{"type": "query.search"}', '["search_mcp"]',
                 150, True, 0.5, "novice", "general"),
                 
                ("nov_gen_3", "user3", "session6", "Show me examples",
                 '{"type": "query.retrieve"}', '["filesystem_mcp"]',
                 180, True, 0.55, "novice", "general"),
                 
                # DevOps patterns
                ("exp_dev_1", "user4", "session7", "Deploy to Kubernetes cluster",
                 '{"type": "action.deploy"}', '["github_mcp", "filesystem_mcp"]',
                 2000, True, 0.9, "expert", "devops"),
                 
                ("exp_dev_2", "user4", "session7", "Monitor container logs",
                 '{"type": "system.monitor"}', '["filesystem_mcp"]',
                 300, True, 0.85, "expert", "devops"),
            ]
            
            # Insert with timestamps spread over time
            base_time = datetime.now() - timedelta(days=7)
            for i, exec_data in enumerate(test_executions):
                timestamp = (base_time + timedelta(hours=i*4)).isoformat()
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, user_id, session_id, query, intent, tools_used, 
                     execution_time_ms, success, reward, user_expertise, domain, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (*exec_data, timestamp))
            
            await db.commit()
            
        yield db_path
        
        # Cleanup
        import os
        os.unlink(db_path)
    
    @pytest.fixture
    def mock_config(self, temp_db_with_context):
        """Create mock configuration for testing."""
        config = Mock()
        config.get.return_value = {
            'path': temp_db_with_context,
            'max_pattern_length': 3,
            'time_window_days': 30,
            'min_support': 0.1,
            'min_confidence': 0.6,
            'use_context_aware_patterns': True
        }
        return config
    
    @pytest.fixture
    def context_extractor(self):
        """Create ContextExtractor instance."""
        return ContextExtractor()
    
    @pytest.fixture
    async def pattern_miner(self, mock_config, temp_db_with_context):
        """Create PatternMiner instance."""
        miner = PatternMiner(mock_config, min_support=0.1, min_confidence=0.6)
        miner.db_path = temp_db_with_context
        return miner
    
    @pytest.mark.asyncio
    async def test_context_extraction_and_pattern_mining_workflow(
        self, context_extractor, pattern_miner
    ):
        """Test the complete workflow from context extraction to pattern mining."""
        # Step 1: Extract context from various queries
        queries_with_context = []
        
        # Expert engineering query
        query1 = "Refactor the async pipeline to optimize performance"
        user_stats1 = {'success_rate': 0.9, 'query_count': 100, 'avg_tools_used': 3.0}
        context1 = context_extractor.extract_context(query1, user_stats1, 'action.modify')
        queries_with_context.append((query1, context1))
        
        # Intermediate data science query
        query2 = "Analyze the dataset and create visualizations"
        user_stats2 = {'success_rate': 0.75, 'query_count': 40, 'avg_tools_used': 2.0}
        context2 = context_extractor.extract_context(query2, user_stats2, 'query.analyze')
        queries_with_context.append((query2, context2))
        
        # Novice general query
        query3 = "How to find files?"
        user_stats3 = {'success_rate': 0.5, 'query_count': 5, 'avg_tools_used': 1.0}
        context3 = context_extractor.extract_context(query3, user_stats3, 'query.search')
        queries_with_context.append((query3, context3))
        
        # Verify contexts are extracted correctly
        assert context1.user_expertise == 'expert'
        assert context1.domain == 'engineering'
        
        assert context2.user_expertise == 'intermediate'
        assert context2.domain == 'data_science'
        
        assert context3.user_expertise == 'novice'
        assert context3.domain == 'general'
        
        # Step 2: Mine patterns from database
        patterns_by_type = await pattern_miner.mine_patterns(use_context_aware=True)
        
        # Verify patterns are discovered
        assert 'sequential' in patterns_by_type
        assert 'combination' in patterns_by_type
        assert len(patterns_by_type['sequential']) > 0
        
        # Step 3: Get context-specific pattern recommendations
        # For expert engineering user
        expert_patterns = pattern_miner.get_context_matching_patterns(
            ['git_mcp'],
            user_expertise='expert',
            domain='engineering'
        )
        
        # Should find patterns relevant to expert engineering
        assert len(expert_patterns) > 0
        expert_tools = set()
        for pattern in expert_patterns:
            expert_tools.update(pattern.tool_sequence)
        # Expert patterns should include advanced tools
        assert any(tool in expert_tools for tool in ['postgres_mcp', 'git_mcp'])
        
        # For novice user
        novice_patterns = pattern_miner.get_context_matching_patterns(
            [],
            user_expertise='novice', 
            domain='general'
        )
        
        # Novice patterns should be simpler
        novice_tools = set()
        for pattern in novice_patterns:
            novice_tools.update(pattern.tool_sequence)
        # Novice patterns should include basic tools
        assert 'filesystem_mcp' in novice_tools or 'search_mcp' in novice_tools
    
    @pytest.mark.asyncio
    async def test_pattern_discovery_across_contexts(self, pattern_miner):
        """Test that patterns are discovered separately for different contexts."""
        # Mine all patterns with context awareness
        context_patterns = await pattern_miner.mine_context_aware_patterns(
            await pattern_miner.extract_sequences()
        )
        
        # Verify we have patterns for different context combinations
        assert len(context_patterns) > 0
        
        # Check specific context patterns
        if 'expert:engineering' in context_patterns:
            exp_eng_patterns = context_patterns['expert:engineering']
            # Expert engineering should have multi-tool patterns
            assert any(
                len(p.tool_sequence) > 1 
                for p in exp_eng_patterns.get('sequential', [])
            )
        
        if 'novice:general' in context_patterns:
            nov_gen_patterns = context_patterns['novice:general']
            # Novice patterns should be simpler
            assert any(
                len(p.tool_sequence) == 1
                for p in nov_gen_patterns.get('sequential', [])
            )
        
        if 'intermediate:data_science' in context_patterns:
            int_ds_patterns = context_patterns['intermediate:data_science']
            # Data science patterns should include specific tools
            tools_in_patterns = set()
            for p in int_ds_patterns.get('sequential', []):
                tools_in_patterns.update(p.tool_sequence)
            assert 'sqlite_mcp' in tools_in_patterns or 'filesystem_mcp' in tools_in_patterns
    
    @pytest.mark.asyncio
    async def test_context_aware_tool_suggestions(self, pattern_miner, context_extractor):
        """Test that tool suggestions adapt based on user context."""
        # Mine patterns first
        await pattern_miner.mine_patterns(use_context_aware=True)
        
        # Test 1: Expert user working on engineering task
        expert_query = "Optimize the database connection pooling"
        expert_context = context_extractor.extract_context(
            expert_query,
            {'success_rate': 0.9, 'query_count': 150, 'avg_tools_used': 3.5},
            'action.modify'
        )
        
        # Get suggestions for expert
        expert_suggestions = []
        patterns = pattern_miner.get_context_matching_patterns(
            ['postgres_mcp'],
            expert_context.user_expertise,
            expert_context.domain
        )
        
        for pattern in patterns[:3]:  # Top 3 patterns
            remaining_tools = [t for t in pattern.tool_sequence if t != 'postgres_mcp']
            expert_suggestions.extend(remaining_tools)
        
        # Expert suggestions should include advanced tools
        assert len(expert_suggestions) > 0
        
        # Test 2: Novice user asking basic question
        novice_query = "How to find files?"
        novice_context = context_extractor.extract_context(
            novice_query,
            {'success_rate': 0.5, 'query_count': 3, 'avg_tools_used': 1.0}
        )
        
        # Get suggestions for novice
        novice_suggestions = []
        patterns = pattern_miner.get_context_matching_patterns(
            [],
            novice_context.user_expertise,
            novice_context.domain
        )
        
        for pattern in patterns[:3]:
            novice_suggestions.extend(pattern.tool_sequence)
        
        # Novice suggestions should be basic tools
        assert 'filesystem_mcp' in novice_suggestions or 'search_mcp' in novice_suggestions
        
        # Expert and novice should get different suggestions
        assert set(expert_suggestions) != set(novice_suggestions)
    
    @pytest.mark.asyncio
    async def test_pattern_persistence_with_context(self, pattern_miner):
        """Test that context information is properly persisted with patterns."""
        # Create patterns with context
        test_patterns = [
            Pattern(
                pattern_type='sequential',
                tool_sequence=['git_mcp', 'postgres_mcp'],
                support=0.5,
                confidence=0.9,
                lift=1.8,
                contexts=['expert:engineering', 'expert:devops']
            ),
            Pattern(
                pattern_type='combination',
                tool_sequence=['filesystem_mcp', 'sqlite_mcp'],
                support=0.4,
                confidence=0.8,
                lift=1.5,
                contexts=['intermediate:data_science']
            )
        ]
        
        # Store patterns
        await pattern_miner.store_patterns(test_patterns)
        
        # Clear in-memory patterns
        pattern_miner.discovered_patterns.clear()
        
        # Load patterns back
        await pattern_miner.load_patterns()
        
        # Verify patterns are loaded with context
        assert len(pattern_miner.discovered_patterns) >= 2
        
        # Find our test patterns
        loaded_patterns = list(pattern_miner.discovered_patterns.values())
        
        # Check that contexts are preserved
        git_postgres_pattern = next(
            (p for p in loaded_patterns if 'git_mcp' in p.tool_sequence and 'postgres_mcp' in p.tool_sequence),
            None
        )
        assert git_postgres_pattern is not None
        assert 'expert:engineering' in git_postgres_pattern.contexts
        assert 'expert:devops' in git_postgres_pattern.contexts
        
        fs_sqlite_pattern = next(
            (p for p in loaded_patterns if 'filesystem_mcp' in p.tool_sequence and 'sqlite_mcp' in p.tool_sequence),
            None
        )
        assert fs_sqlite_pattern is not None
        assert 'intermediate:data_science' in fs_sqlite_pattern.contexts
    
    @pytest.mark.asyncio
    async def test_q_learning_integration_with_context(self, mock_config, pattern_miner, context_extractor):
        """Test Q-learning engine integration with context-aware patterns."""
        # Configure Q-learning with context awareness
        q_config = mock_config.get.return_value
        q_config.update({
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 0.2,
            'use_patterns': True,
            'use_context_aware_patterns': True,
            'pattern_weight': 0.3
        })
        
        # Mock Q-learning engine behavior
        with patch('src.learning.q_learning_engine.QLearningEngine') as MockQLearning:
            mock_q_engine = AsyncMock()
            MockQLearning.return_value = mock_q_engine
            
            # Simulate state encoding with context
            query = "Refactor the codebase"
            context = context_extractor.extract_context(
                query,
                {'success_rate': 0.85, 'query_count': 80, 'avg_tools_used': 2.8}
            )
            
            # Create mock state that includes context vector
            context_vector = context_extractor.get_context_vector(context)
            assert len(context_vector) == 8  # 3 for expertise + 5 for domain
            
            # The state should include the context vector
            mock_state = {
                'intent_vector': [0.1] * 384,
                'context_features': [0.2] * 10,
                'tool_history': [0.0] * 20,
                'performance_metrics': [0.5] * 5,
                'failure_rates': [0.0] * 10,
                'failure_types': [0.0] * 5,
                'retry_patterns': [0.0] * 5,
                'user_expertise': context_vector[:3],
                'domain_context': context_vector[3:]
            }
            
            # Total state dimensions should be 447
            total_dims = sum(len(v) if isinstance(v, list) else 1 for v in mock_state.values())
            assert total_dims == 447
    
    @pytest.mark.asyncio 
    async def test_cross_context_pattern_analysis(self, pattern_miner):
        """Test analysis of patterns across different contexts."""
        # Mine all patterns
        await pattern_miner.mine_patterns(use_context_aware=True)
        
        # Analyze tool usage across contexts
        tool_usage_by_context = {}
        
        for pattern_hash, pattern in pattern_miner.discovered_patterns.items():
            if pattern.contexts:
                for context in pattern.contexts:
                    if context not in tool_usage_by_context:
                        tool_usage_by_context[context] = set()
                    tool_usage_by_context[context].update(pattern.tool_sequence)
        
        # Verify different contexts use different tool sets
        if 'expert:engineering' in tool_usage_by_context and 'novice:general' in tool_usage_by_context:
            expert_tools = tool_usage_by_context['expert:engineering']
            novice_tools = tool_usage_by_context['novice:general']
            
            # Expert should have more diverse tools
            assert len(expert_tools) >= len(novice_tools)
            
            # Some tools might be expert-only
            expert_only = expert_tools - novice_tools
            # Advanced tools like git_mcp, postgres_mcp should be in expert patterns
            assert any(tool in expert_only for tool in ['git_mcp', 'postgres_mcp', 'github_mcp'])
    
    @pytest.mark.asyncio
    async def test_context_transition_patterns(self, pattern_miner, temp_db_with_context):
        """Test patterns when user expertise evolves over time."""
        # Add execution history showing user progression
        async with aiosqlite.connect(temp_db_with_context) as db:
            base_time = datetime.now() - timedelta(days=30)
            
            # User starts as novice
            for i in range(5):
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, user_expertise, domain, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"evolve_nov_{i}", "Find files", '{"type": "query.search"}',
                      '["filesystem_mcp"]', True, 0.5 + i*0.05, "novice", "general",
                      (base_time + timedelta(days=i)).isoformat()))
            
            # User progresses to intermediate
            for i in range(5):
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, user_expertise, domain, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"evolve_int_{i}", "Search and analyze files", '{"type": "query.analyze"}',
                      '["filesystem_mcp", "sqlite_mcp"]', True, 0.7 + i*0.03, "intermediate", "general",
                      (base_time + timedelta(days=10+i)).isoformat()))
            
            # User becomes expert
            for i in range(5):
                await db.execute("""
                    INSERT INTO execution_history 
                    (id, query, intent, tools_used, success, reward, user_expertise, domain, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"evolve_exp_{i}", "Optimize system architecture", '{"type": "action.modify"}',
                      '["git_mcp", "filesystem_mcp", "postgres_mcp"]', True, 0.85 + i*0.02, "expert", "general",
                      (base_time + timedelta(days=20+i)).isoformat()))
            
            await db.commit()
        
        # Mine patterns
        sequences = await pattern_miner.extract_sequences()
        context_patterns = await pattern_miner.mine_context_aware_patterns(sequences)
        
        # Verify we have patterns for each expertise level
        assert any('novice:general' in key for key in context_patterns.keys())
        assert any('intermediate:general' in key for key in context_patterns.keys())
        assert any('expert:general' in key for key in context_patterns.keys())
        
        # Verify pattern complexity increases with expertise
        pattern_complexity = {}
        for context_key, patterns_dict in context_patterns.items():
            if 'general' in context_key:  # Same domain
                avg_length = 0
                count = 0
                for pattern_list in patterns_dict.values():
                    for pattern in pattern_list:
                        avg_length += len(pattern.tool_sequence)
                        count += 1
                if count > 0:
                    pattern_complexity[context_key] = avg_length / count
        
        # Expert patterns should be more complex than novice
        if 'novice:general' in pattern_complexity and 'expert:general' in pattern_complexity:
            assert pattern_complexity['expert:general'] > pattern_complexity['novice:general']