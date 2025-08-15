"""
Result Caching for Orchestrator Agent.

This module provides caching functionality for query results to improve
response times for frequently repeated queries and demonstrate learning
capabilities as required by dissertation goals.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Callable
import pickle
import os

from src.utils.logger import get_logger


@dataclass
class CacheEntry:
    """Represents a cached result entry."""
    result: Any
    timestamp: float
    hit_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


class ResultCache:
    """
    LRU Cache for orchestration results with TTL support.
    
    This cache:
    - Uses LRU eviction when size limit is reached
    - Supports TTL for automatic expiration
    - Tracks cache metrics for analysis
    - Generates deterministic cache keys
    - Supports persistence for resilience
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the result cache."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        if config is None:
            config = self._get_default_config()
        self.config = config
        
        # Cache settings
        self.enabled = config.get('enabled', True)
        self.max_size = config.get('max_size', 1000)
        self.ttl_seconds = config.get('ttl_seconds', 3600)  # 1 hour default
        self.cache_successful_only = config.get('cache_successful_only', True)
        self.consider_context = config.get('consider_context', True)
        
        # Cache storage
        self.cache = OrderedDict()
        
        # Metrics tracking
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
            'total_queries': 0,
            'cache_size_bytes': 0,
            'avg_retrieval_time_ms': 0.0,
            'retrieval_times': [],  # Keep last 100 for rolling average
            'query_patterns': {},  # Track patterns for analysis
            'metric_timestamps': []  # Timestamped metric events
        }
        
        # Pattern tracking
        self.pattern_extractor: Optional[Callable[[str], str]] = None
        self.track_patterns = config.get('track_patterns', True)
        self.max_metric_events = config.get('max_metric_events', 1000)
        
        # Persistence settings
        self.persistence_enabled = config.get('enable_persistence', True)
        self.cache_file = config.get('cache_file', 'data/cache/result_cache.pkl')
        
        # Load cache from disk if persistence is enabled
        if self.persistence_enabled and self.enabled:
            self._load_cache()
        
        self.logger.info(f"Result cache initialized - Enabled: {self.enabled}, "
                        f"Max size: {self.max_size}, TTL: {self.ttl_seconds}s")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default cache configuration."""
        return {
            'enabled': True,
            'max_size': 1000,
            'ttl_seconds': 3600,
            'cache_strategy': 'lru',
            'cache_successful_only': True,
            'consider_context': True,
            'enable_persistence': True,
            'cache_file': 'data/cache/result_cache.pkl'
        }
    
    def generate_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a deterministic cache key from query and context.
        
        Args:
            query: The user query
            context: Optional context information
            
        Returns:
            Hash key for cache lookup
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Build key components
        key_parts = [normalized_query]
        
        if self.consider_context and context:
            # Add relevant context to key
            if 'domain' in context:
                key_parts.append(f"domain:{context['domain']}")
            if 'user_expertise' in context:
                key_parts.append(f"expertise:{context['user_expertise']}")
            if 'intent_type' in context:
                key_parts.append(f"intent:{context['intent_type']}")
            if 'intent_confidence' in context:
                # Round confidence to avoid tiny differences
                confidence = round(context['intent_confidence'], 1)
                key_parts.append(f"conf:{confidence}")
        
        # Create hash
        key_string = "|".join(key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        self.logger.debug(f"Generated cache key: {cache_key} for query: '{query}'")
        return cache_key
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached result if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None
        
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Check if key exists
        if key not in self.cache:
            self.metrics['misses'] += 1
            retrieval_time = time.time() - start_time
            self._update_retrieval_time(retrieval_time)
            self._track_metric_event('miss', key, retrieval_time)
            return None
        
        # Get entry and check TTL
        entry = self.cache[key]
        current_time = time.time()
        
        # Check if expired
        if current_time - entry.timestamp > self.ttl_seconds:
            self.logger.debug(f"Cache entry expired for key: {key}")
            del self.cache[key]
            self.metrics['expirations'] += 1
            self.metrics['misses'] += 1
            retrieval_time = time.time() - start_time
            self._update_retrieval_time(retrieval_time)
            self._track_metric_event('expiration', key, retrieval_time)
            return None
        
        # Move to end (LRU behavior)
        self.cache.move_to_end(key)
        
        # Update metrics
        entry.hit_count += 1
        entry.last_accessed = current_time
        self.metrics['hits'] += 1
        
        retrieval_time = time.time() - start_time
        self._update_retrieval_time(retrieval_time)
        
        self.logger.debug(f"Cache hit for key: {key} (hit #{entry.hit_count})")
        
        # Track metric event
        self._track_metric_event('hit', key, retrieval_time)
        
        return entry.result
    
    def put(self, key: str, result: Any) -> bool:
        """
        Store a result in the cache.
        
        Args:
            key: Cache key
            result: Result to cache
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        # Check if we should cache this result
        if self.cache_successful_only:
            # Check if result has success attribute and it's True
            if hasattr(result, 'success') and not result.success:
                self.logger.debug(f"Not caching unsuccessful result for key: {key}")
                return False
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.metrics['evictions'] += 1
            self.logger.debug(f"Evicted oldest cache entry: {oldest_key}")
            self._track_metric_event('eviction', oldest_key, 0)
        
        # Check if result is serializable before caching
        if self.persistence_enabled:
            try:
                # Test if we can pickle the result
                pickle.dumps(result)
            except (pickle.PicklingError, TypeError) as e:
                self.logger.warning(f"Result not serializable, skipping cache: {e}")
                return False
        
        # Create and store entry
        entry = CacheEntry(
            result=result,
            timestamp=time.time()
        )
        self.cache[key] = entry
        
        # Update size metric (approximate)
        try:
            self.metrics['cache_size_bytes'] = len(pickle.dumps(self.cache))
        except:
            # If pickle fails, just estimate
            self.metrics['cache_size_bytes'] = len(self.cache) * 1024
        
        self.logger.debug(f"Cached result for key: {key} (cache size: {len(self.cache)})")
        return True
    
    def invalidate(self, key: Optional[str] = None, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate
            pattern: Pattern to match in cached results for invalidation
        """
        if key:
            if key in self.cache:
                del self.cache[key]
                self.logger.info(f"Invalidated cache key: {key}")
        
        if pattern:
            # Find and remove matching entries based on result content
            keys_to_remove = []
            for cache_key, entry in self.cache.items():
                # Check if pattern appears in various parts of the cached result
                if hasattr(entry.result, 'query') and pattern in entry.result.query:
                    keys_to_remove.append(cache_key)
                elif hasattr(entry.result, 'selected_tools'):
                    # Check if pattern matches any selected tools
                    for tool in entry.result.selected_tools:
                        if pattern in tool:
                            keys_to_remove.append(cache_key)
                            break
            
            for key in keys_to_remove:
                del self.cache[key]
            
            if keys_to_remove:
                self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
    
    def clear(self):
        """Clear all cache entries."""
        size = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cleared cache ({size} entries)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary of cache metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate hit rate
        total = metrics.get('hits', 0) + metrics.get('misses', 0)
        metrics['hit_rate'] = metrics['hits'] / total if total > 0 else 0.0
        
        # Add current cache info
        metrics['current_size'] = len(self.cache)
        metrics['max_size'] = self.max_size
        
        # Add top accessed entries
        if self.cache:
            top_entries = sorted(
                [(k, v.hit_count) for k, v in self.cache.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            metrics['top_accessed'] = top_entries
        
        return metrics
    
    def warm_cache(self, execution_history: list):
        """
        Warm the cache from execution history.
        
        Args:
            execution_history: List of past executions
        """
        if not self.enabled:
            return
        
        warmed = 0
        for execution in execution_history:
            # Only warm successful executions
            if execution.get('success', False):
                query = execution.get('query', '')
                context = execution.get('context', {})
                
                # Generate key
                key = self.generate_key(query, context)
                
                # Check if we have the result
                if 'result' in execution and key not in self.cache:
                    # Create a minimal result object
                    result = execution['result']
                    if self.put(key, result):
                        warmed += 1
        
        if warmed > 0:
            self.logger.info(f"Warmed cache with {warmed} entries from execution history")
    
    def _update_retrieval_time(self, time_ms: float):
        """Update retrieval time metrics."""
        time_ms = time_ms * 1000  # Convert to milliseconds
        self.metrics['retrieval_times'].append(time_ms)
        
        # Keep only last 100
        if len(self.metrics['retrieval_times']) > 100:
            self.metrics['retrieval_times'] = self.metrics['retrieval_times'][-100:]
        
        # Update average
        if self.metrics['retrieval_times']:
            self.metrics['avg_retrieval_time_ms'] = sum(self.metrics['retrieval_times']) / len(self.metrics['retrieval_times'])
    
    def _load_cache(self):
        """Load cache from disk."""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                saved_data = pickle.load(f)
                
            # Validate and restore cache
            if isinstance(saved_data, dict) and 'cache' in saved_data:
                self.cache = OrderedDict(saved_data['cache'])
                self.metrics = saved_data.get('metrics', self.metrics)
                
                # Clean expired entries
                current_time = time.time()
                expired_keys = []
                for key, entry in self.cache.items():
                    if current_time - entry.timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                
                self.logger.info(f"Loaded cache from disk ({len(self.cache)} entries, {len(expired_keys)} expired)")
        except Exception as e:
            self.logger.warning(f"Failed to load cache from disk: {e}")
    
    def save_cache(self):
        """Save cache to disk."""
        if not self.persistence_enabled or not self.enabled:
            return
        
        try:
            # Store references to built-ins in case we're called during shutdown
            _open = open
            _makedirs = os.makedirs
            _dirname = os.path.dirname
            _time = time.time
            _pickle_dump = pickle.dump
            
            # Ensure directory exists
            _makedirs(_dirname(self.cache_file), exist_ok=True)
            
            # Save cache and metrics
            save_data = {
                'cache': dict(self.cache),
                'metrics': self.metrics,
                'timestamp': _time()
            }
            
            with _open(self.cache_file, 'wb') as f:
                _pickle_dump(save_data, f)
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Saved cache to disk ({len(self.cache)} entries)")
        except (NameError, AttributeError) as e:
            # Built-ins may not be available during Python shutdown
            pass
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Failed to save cache to disk: {e}")
    
    def set_pattern_extractor(self, extractor: Callable[[str], str]):
        """Set a function to extract patterns from queries."""
        self.pattern_extractor = extractor
    
    def _track_metric_event(self, event_type: str, key: str, retrieval_time: float):
        """Track a timestamped metric event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'key': key,
            'retrieval_time_ms': retrieval_time * 1000
        }
        
        # Add to timestamped events (limited size)
        if 'metric_timestamps' not in self.metrics:
            self.metrics['metric_timestamps'] = []
        
        self.metrics['metric_timestamps'].append(event)
        if len(self.metrics['metric_timestamps']) > self.max_metric_events:
            self.metrics['metric_timestamps'] = self.metrics['metric_timestamps'][-self.max_metric_events:]
        
        # Track patterns if enabled
        if self.track_patterns and self.pattern_extractor:
            try:
                pattern = self.pattern_extractor(key)
                if pattern not in self.metrics['query_patterns']:
                    self.metrics['query_patterns'][pattern] = {
                        'count': 0,
                        'hits': 0,
                        'misses': 0,
                        'avg_retrieval_time_ms': 0.0
                    }
                
                pattern_metrics = self.metrics['query_patterns'][pattern]
                pattern_metrics['count'] += 1
                
                if event_type == 'hit':
                    pattern_metrics['hits'] += 1
                elif event_type in ['miss', 'expiration']:
                    pattern_metrics['misses'] += 1
                
                # Update moving average of retrieval time
                alpha = 0.1
                pattern_metrics['avg_retrieval_time_ms'] = (
                    alpha * (retrieval_time * 1000) + 
                    (1 - alpha) * pattern_metrics['avg_retrieval_time_ms']
                )
            except Exception as e:
                self.logger.debug(f"Pattern extraction failed: {e}")
    
    def get_pattern_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics organized by query patterns."""
        pattern_metrics = {}
        
        for pattern, metrics in self.metrics.get('query_patterns', {}).items():
            total = metrics['hits'] + metrics['misses']
            pattern_metrics[pattern] = {
                **metrics,
                'hit_rate': metrics['hits'] / total if total > 0 else 0.0
            }
        
        return pattern_metrics
    
    def get_recent_events(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent metric events."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        cutoff_iso = cutoff_time.isoformat()
        
        return [
            event for event in self.metrics.get('metric_timestamps', [])
            if event['timestamp'] >= cutoff_iso
        ]
    
    def __del__(self):
        """Save cache on cleanup."""
        try:
            # Store references to built-ins before they potentially become unavailable
            if hasattr(self, 'persistence_enabled') and self.persistence_enabled:
                # Check if save_cache method is still available and callable
                if hasattr(self, 'save_cache') and callable(getattr(self, 'save_cache', None)):
                    # Try to save cache, but catch all exceptions including NameError
                    try:
                        self.save_cache()
                    except (NameError, AttributeError, RuntimeError) as e:
                        # Built-ins or event loop may not be available during shutdown
                        pass
        except Exception:
            # During interpreter shutdown, some globals might not be available
            pass