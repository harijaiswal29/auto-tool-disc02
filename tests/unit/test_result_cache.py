"""
Unit tests for Result Cache.

Tests the caching functionality including:
- LRU eviction
- TTL expiration
- Cache key generation
- Metrics tracking
- Persistence
- Cache warming
"""

import pytest
import time
import os
import tempfile
from unittest.mock import Mock, patch
from dataclasses import dataclass

from src.agents.result_cache import ResultCache, CacheEntry


@dataclass
class MockResult:
    """Mock result for testing."""
    query: str
    success: bool
    data: str


class TestResultCache:
    """Test cases for ResultCache class."""
    
    @pytest.fixture
    def default_config(self):
        """Default configuration for testing."""
        return {
            'enabled': True,
            'max_size': 3,  # Small size for testing eviction
            'ttl_seconds': 2,  # Short TTL for testing expiration
            'cache_successful_only': True,
            'consider_context': True,
            'enable_persistence': False  # Disable for most tests
        }
    
    @pytest.fixture
    def cache(self, default_config):
        """Create a ResultCache instance."""
        return ResultCache(default_config)
    
    def test_initialization(self, default_config):
        """Test cache initialization."""
        cache = ResultCache(default_config)
        
        assert cache.enabled is True
        assert cache.max_size == 3
        assert cache.ttl_seconds == 2
        assert cache.cache_successful_only is True
        assert len(cache.cache) == 0
        assert cache.metrics['hits'] == 0
        assert cache.metrics['misses'] == 0
    
    def test_generate_key_basic(self, cache):
        """Test basic cache key generation."""
        query = "Find all Python files"
        key1 = cache.generate_key(query)
        key2 = cache.generate_key(query)
        
        # Same query should generate same key
        assert key1 == key2
        assert len(key1) == 32  # MD5 hex digest length
        
        # Different query should generate different key
        key3 = cache.generate_key("Find all Java files")
        assert key1 != key3
    
    def test_generate_key_with_context(self, cache):
        """Test cache key generation with context."""
        query = "Find files"
        
        # Without context
        key1 = cache.generate_key(query)
        
        # With domain context
        context1 = {'domain': 'development'}
        key2 = cache.generate_key(query, context1)
        assert key1 != key2
        
        # With full context
        context2 = {
            'domain': 'development',
            'user_expertise': 'expert',
            'intent_type': 'query.search',
            'intent_confidence': 0.85
        }
        key3 = cache.generate_key(query, context2)
        assert key2 != key3
        
        # Same context should generate same key
        key4 = cache.generate_key(query, context2)
        assert key3 == key4
    
    def test_put_and_get(self, cache):
        """Test basic put and get operations."""
        query = "Test query"
        result = MockResult(query=query, success=True, data="Test data")
        
        key = cache.generate_key(query)
        
        # Put result
        success = cache.put(key, result)
        assert success is True
        assert len(cache.cache) == 1
        
        # Get result
        cached_result = cache.get(key)
        assert cached_result is not None
        assert cached_result.query == query
        assert cached_result.data == "Test data"
        
        # Verify metrics
        assert cache.metrics['hits'] == 1
        assert cache.metrics['misses'] == 0
    
    def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        key = cache.generate_key("Non-existent query")
        result = cache.get(key)
        
        assert result is None
        assert cache.metrics['hits'] == 0
        assert cache.metrics['misses'] == 1
    
    def test_cache_successful_only(self, cache):
        """Test that only successful results are cached."""
        # Successful result should be cached
        success_result = MockResult(query="Success", success=True, data="Good")
        key1 = cache.generate_key("Success")
        assert cache.put(key1, success_result) is True
        
        # Failed result should not be cached
        failed_result = MockResult(query="Failed", success=False, data="Bad")
        key2 = cache.generate_key("Failed")
        assert cache.put(key2, failed_result) is False
        
        # Verify only successful result is in cache
        assert len(cache.cache) == 1
        assert cache.get(key1) is not None
        assert cache.get(key2) is None
    
    def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        results = []
        keys = []
        for i in range(3):
            result = MockResult(query=f"Query {i}", success=True, data=f"Data {i}")
            key = cache.generate_key(f"Query {i}")
            cache.put(key, result)
            results.append(result)
            keys.append(key)
        
        assert len(cache.cache) == 3
        
        # Access middle item to make it more recent
        cache.get(keys[1])
        
        # Add new item - should evict oldest (Query 0)
        new_result = MockResult(query="New Query", success=True, data="New Data")
        new_key = cache.generate_key("New Query")
        cache.put(new_key, new_result)
        
        # Check eviction
        assert len(cache.cache) == 3
        assert cache.get(keys[0]) is None  # Query 0 evicted
        assert cache.get(keys[1]) is not None  # Query 1 kept (accessed)
        assert cache.get(keys[2]) is not None  # Query 2 kept
        assert cache.get(new_key) is not None  # New query added
        
        # Verify eviction metric
        assert cache.metrics['evictions'] == 1
    
    def test_ttl_expiration(self, cache):
        """Test TTL expiration of cache entries."""
        result = MockResult(query="Expiring", success=True, data="Will expire")
        key = cache.generate_key("Expiring")
        
        # Put result
        cache.put(key, result)
        
        # Should be available immediately
        assert cache.get(key) is not None
        assert cache.metrics['hits'] == 1
        
        # Wait for expiration
        time.sleep(2.5)
        
        # Should be expired now
        assert cache.get(key) is None
        assert cache.metrics['expirations'] == 1
        assert cache.metrics['misses'] == 1
    
    def test_metrics_tracking(self, cache):
        """Test comprehensive metrics tracking."""
        # Generate some cache activity
        for i in range(5):
            result = MockResult(query=f"Q{i}", success=True, data=f"D{i}")
            key = cache.generate_key(f"Q{i}")
            cache.put(key, result)
        
        # Access some entries - Q0 and Q1 were evicted, so they'll be misses
        cache.get(cache.generate_key("Q0"))  # Miss (evicted)
        cache.get(cache.generate_key("Q1"))  # Miss (evicted)
        cache.get(cache.generate_key("Q2"))  # Hit (still in cache)
        cache.get(cache.generate_key("Q3"))  # Hit (still in cache)
        cache.get(cache.generate_key("Q99"))  # Miss (never existed)
        
        metrics = cache.get_metrics()
        
        assert metrics['hits'] == 2
        assert metrics['misses'] == 3
        assert metrics['hit_rate'] == 2/5  # 2 hits out of 5 total queries
        assert metrics['current_size'] == 3  # Max size reached
        assert metrics['evictions'] == 2  # 2 evictions happened (5 items, max 3)
        assert 'avg_retrieval_time_ms' in metrics
        assert metrics['avg_retrieval_time_ms'] >= 0
    
    def test_cache_warming(self, cache):
        """Test cache warming from execution history."""
        execution_history = [
            {
                'query': 'Query 1',
                'success': True,
                'result': MockResult(query='Query 1', success=True, data='Data 1'),
                'context': {'domain': 'test'}
            },
            {
                'query': 'Query 2',
                'success': False,  # Should not be warmed
                'result': MockResult(query='Query 2', success=False, data='Data 2')
            },
            {
                'query': 'Query 3',
                'success': True,
                'result': MockResult(query='Query 3', success=True, data='Data 3')
            }
        ]
        
        cache.warm_cache(execution_history)
        
        # Only successful entries should be warmed
        assert len(cache.cache) == 2
        
        # Verify warmed entries
        key1 = cache.generate_key('Query 1', {'domain': 'test'})
        key3 = cache.generate_key('Query 3')
        
        assert cache.get(key1) is not None
        assert cache.get(key3) is not None
    
    def test_invalidation(self, cache):
        """Test cache invalidation."""
        # Add some entries
        for i in range(3):
            result = MockResult(query=f"tool_{i}_query", success=True, data=f"Data {i}")
            key = cache.generate_key(f"tool_{i}_query")
            cache.put(key, result)
        
        assert len(cache.cache) == 3
        
        # Invalidate specific key
        key_to_invalidate = cache.generate_key("tool_1_query")
        cache.invalidate(key=key_to_invalidate)
        assert len(cache.cache) == 2
        assert cache.get(key_to_invalidate) is None
        
        # Invalidate by pattern
        cache.invalidate(pattern="tool_")
        assert len(cache.cache) == 0  # All had "tool_" pattern
    
    def test_clear_cache(self, cache):
        """Test clearing entire cache."""
        # Add entries
        for i in range(3):
            result = MockResult(query=f"Query {i}", success=True, data=f"Data {i}")
            key = cache.generate_key(f"Query {i}")
            cache.put(key, result)
        
        assert len(cache.cache) == 3
        
        # Clear cache
        cache.clear()
        assert len(cache.cache) == 0
    
    def test_hit_count_tracking(self, cache):
        """Test that hit counts are tracked per entry."""
        result = MockResult(query="Popular", success=True, data="Frequently accessed")
        key = cache.generate_key("Popular")
        
        cache.put(key, result)
        
        # Access multiple times
        for _ in range(5):
            cache.get(key)
        
        # Check hit count
        entry = cache.cache[key]
        assert entry.hit_count == 5
        
        # Verify in metrics
        metrics = cache.get_metrics()
        assert key in [k for k, _ in metrics.get('top_accessed', [])]
    
    @pytest.mark.parametrize("enable_persistence", [True, False])
    def test_persistence(self, default_config, enable_persistence):
        """Test cache persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.pkl")
            default_config['enable_persistence'] = enable_persistence
            default_config['cache_file'] = cache_file
            
            # Create cache and add data
            cache1 = ResultCache(default_config)
            result = MockResult(query="Persist me", success=True, data="Important")
            key = cache1.generate_key("Persist me")
            cache1.put(key, result)
            
            # Save cache
            cache1.save_cache()
            
            if enable_persistence:
                assert os.path.exists(cache_file)
                
                # Create new cache instance - should load from disk
                cache2 = ResultCache(default_config)
                key2 = cache2.generate_key("Persist me")
                loaded_result = cache2.get(key2)
                
                assert loaded_result is not None
                assert loaded_result.data == "Important"
            else:
                assert not os.path.exists(cache_file)
    
    def test_disabled_cache(self):
        """Test that disabled cache doesn't store or retrieve."""
        config = {'enabled': False}
        cache = ResultCache(config)
        
        result = MockResult(query="Test", success=True, data="Data")
        key = cache.generate_key("Test")
        
        # Put should return False
        assert cache.put(key, result) is False
        
        # Get should return None
        assert cache.get(key) is None
        
        # Metrics should not be updated
        assert cache.metrics['hits'] == 0
        assert cache.metrics['misses'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])