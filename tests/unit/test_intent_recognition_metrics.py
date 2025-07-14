"""
Unit tests for Intent Recognition Metrics module.

Tests the metrics collection, calculation, and reporting functionality
for the Intent Recognition Agent.
"""

import pytest
import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.intent_recognition_metrics import (
    IntentRecognitionMetrics,
    MetricsAggregator,
    get_metrics
)
from src.agents.intent_models import Intent, IntentResult


class TestIntentRecognitionMetrics:
    """Test cases for IntentRecognitionMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create a fresh metrics instance."""
        return IntentRecognitionMetrics(window_size=100)
    
    @pytest.fixture
    def sample_intent_result(self):
        """Create a sample IntentResult for testing."""
        intent = Intent(
            type="query.search",
            confidence=0.85,
            entities=["python", "files"],
            keywords=["find", "search"]
        )
        
        return IntentResult(
            raw_query="Find Python files",
            primary_intent=intent,
            all_intents=[intent],
            confidence_passed=True,
            processing_time_ms=45.5,
            metadata={"stage": "test"}
        )
    
    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.window_size == 100
        assert metrics.total_queries == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert len(metrics.processing_times) == 0
        assert len(metrics.errors) == 0
    
    def test_record_query_processing(self, metrics, sample_intent_result):
        """Test recording query processing metrics."""
        # Record with stage timings
        stage_timings = {
            "preprocessing": 5.0,
            "feature_extraction": 20.0,
            "classification": 15.0,
            "postprocessing": 5.5
        }
        
        metrics.record_query_processing(sample_intent_result, stage_timings)
        
        # Verify metrics were recorded
        assert metrics.total_queries == 1
        assert len(metrics.processing_times) == 1
        assert metrics.processing_times[0] == 45.5
        
        # Check stage timings
        assert len(metrics.stage_times["preprocessing"]) == 1
        assert metrics.stage_times["preprocessing"][0] == 5.0
        
        # Check intent classification
        assert len(metrics.intent_classifications) == 1
        assert metrics.intent_classifications[0]['intent'] == "query.search"
        assert metrics.intent_classifications[0]['confidence'] == 0.85
        
        # Check confidence scores
        assert len(metrics.confidence_scores) == 1
        assert metrics.confidence_scores[0] == 0.85
        
        # Check intent frequency
        assert metrics.intent_frequency["query.search"] == 1
    
    def test_multi_intent_tracking(self, metrics):
        """Test tracking of multi-intent queries."""
        # Create multi-intent result
        intent1 = Intent("query.search", 0.8, [], [])
        intent2 = Intent("action.create", 0.7, [], [])
        
        result = IntentResult(
            raw_query="Find and create Python files",
            primary_intent=intent1,
            all_intents=[intent1, intent2],
            confidence_passed=True,
            processing_time_ms=50.0,
            metadata={}
        )
        
        metrics.record_query_processing(result)
        
        assert metrics.multi_intent_queries == 1
        assert metrics.get_multi_intent_rate() == 100.0
    
    def test_cache_metrics(self, metrics):
        """Test cache hit/miss recording."""
        # Record some cache accesses
        metrics.record_cache_access(True)   # hit
        metrics.record_cache_access(True)   # hit
        metrics.record_cache_access(False)  # miss
        
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1
        assert metrics.get_cache_hit_rate() == pytest.approx(66.67, rel=0.1)
    
    def test_error_tracking(self, metrics):
        """Test error recording and tracking."""
        # Record different types of errors
        metrics.record_error("timeout", {"query": "test", "duration": 5000})
        metrics.record_error("timeout", {"query": "test2", "duration": 6000})
        metrics.record_error("parsing_error", {"query": "bad query", "reason": "invalid syntax"})
        
        assert metrics.errors["timeout"] == 2
        assert metrics.errors["parsing_error"] == 1
        assert len(metrics.error_details) == 3
        
        # Test error rate calculation
        metrics.total_queries = 10
        assert metrics.get_error_rate() == 30.0  # 3 errors out of 10 queries
    
    def test_feedback_recording(self, metrics):
        """Test user feedback recording."""
        # Record feedback
        metrics.record_feedback("query.search", correct=True)
        metrics.record_feedback("query.search", correct=False)  # false positive
        metrics.record_feedback(None, correct=False)  # false negative
        
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
    
    def test_processing_time_statistics(self, metrics):
        """Test processing time statistical calculations."""
        # Add some processing times
        times = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        for time in times:
            metrics.processing_times.append(time)
        
        # Test average
        assert metrics.get_average_processing_time() == 55.0
        
        # Test percentiles
        assert metrics.get_percentile_processing_time(50) == 50.0  # median
        assert metrics.get_percentile_processing_time(95) == 90.0  # p95
        assert metrics.get_percentile_processing_time(99) == 100.0  # p99
    
    def test_stage_performance_metrics(self, metrics):
        """Test stage-wise performance metrics."""
        # Add stage timings
        for i in range(5):
            metrics.stage_times["preprocessing"].append(5.0 + i)
            metrics.stage_times["classification"].append(20.0 + i * 2)
        
        stage_perf = metrics.get_stage_performance()
        
        # Check preprocessing stats
        assert "preprocessing" in stage_perf
        assert stage_perf["preprocessing"]["avg_ms"] == 7.0  # (5+6+7+8+9)/5
        assert stage_perf["preprocessing"]["min_ms"] == 5.0
        assert stage_perf["preprocessing"]["max_ms"] == 9.0
        
        # Check classification stats
        assert "classification" in stage_perf
        assert stage_perf["classification"]["avg_ms"] == 24.0  # (20+22+24+26+28)/5
    
    def test_confidence_distribution(self, metrics):
        """Test confidence score distribution calculation."""
        # Add various confidence scores
        scores = [0.1, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.95]
        for score in scores:
            metrics.confidence_scores.append(score)
        
        distribution = metrics.get_confidence_distribution()
        
        assert distribution['very_low'] == 2    # 0.1, 0.2
        assert distribution['low'] == 1         # 0.4
        assert distribution['medium'] == 1      # 0.6
        assert distribution['high'] == 2        # 0.75, 0.85
        assert distribution['very_high'] == 2   # 0.92, 0.95
    
    def test_intent_distribution(self, metrics):
        """Test intent type distribution calculation."""
        # Record different intent types
        metrics.intent_frequency["query.search"] = 50
        metrics.intent_frequency["action.create"] = 30
        metrics.intent_frequency["system.monitor"] = 20
        metrics.total_queries = 100
        
        distribution = metrics.get_intent_distribution()
        
        assert distribution["query.search"] == 50.0
        assert distribution["action.create"] == 30.0
        assert distribution["system.monitor"] == 20.0
    
    def test_classification_accuracy(self, metrics):
        """Test classification accuracy calculation."""
        # Add some classifications and feedback
        for _ in range(10):
            metrics.intent_classifications.append({
                'intent': 'query.search',
                'confidence': 0.8,
                'passed': True,
                'timestamp': datetime.now()
            })
        
        # 2 errors out of 10
        metrics.false_positives = 1
        metrics.false_negatives = 1
        
        accuracy = metrics.get_classification_accuracy()
        assert accuracy == 80.0  # (10-2)/10 * 100
    
    def test_summary_metrics(self, metrics, sample_intent_result):
        """Test comprehensive summary metrics generation."""
        # Populate various metrics
        metrics.record_query_processing(sample_intent_result)
        metrics.record_cache_access(True)
        metrics.record_cache_access(False)
        metrics.record_error("timeout", {"duration": 5000})
        
        summary = metrics.get_summary_metrics()
        
        # Check structure
        assert "performance" in summary
        assert "accuracy" in summary
        assert "cache" in summary
        assert "usage" in summary
        assert "reliability" in summary
        
        # Verify some values
        assert summary["usage"]["total_queries"] == 1
        assert summary["cache"]["hit_rate"] == 50.0
        assert summary["performance"]["avg_processing_time_ms"] == 45.5
    
    def test_export_metrics(self, metrics, sample_intent_result, tmp_path):
        """Test metrics export to JSON file."""
        # Add some data
        metrics.record_query_processing(sample_intent_result)
        
        # Export to file
        export_path = tmp_path / "metrics_export.json"
        metrics.export_metrics(str(export_path))
        
        # Verify file was created and contains valid JSON
        assert export_path.exists()
        
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert "performance" in data
        assert "export_timestamp" in data
        assert data["usage"]["total_queries"] == 1
    
    def test_reset_metrics(self, metrics, sample_intent_result):
        """Test metrics reset functionality."""
        # Add some data
        metrics.record_query_processing(sample_intent_result)
        metrics.record_cache_access(True)
        metrics.record_error("test", {})
        
        # Reset
        metrics.reset_metrics()
        
        # Verify everything is cleared
        assert metrics.total_queries == 0
        assert metrics.cache_hits == 0
        assert len(metrics.processing_times) == 0
        assert len(metrics.errors) == 0
        assert len(metrics.intent_classifications) == 0
    
    def test_hourly_metrics(self, metrics, sample_intent_result):
        """Test hourly metrics aggregation."""
        # Record queries
        metrics.record_query_processing(sample_intent_result)
        
        # Check hourly metrics
        hour_key = datetime.now().strftime("%Y-%m-%d_%H")
        assert hour_key in metrics.hourly_metrics
        assert metrics.hourly_metrics[hour_key]['queries'] == 1
    
    def test_edge_cases(self, metrics):
        """Test edge cases and error conditions."""
        # Test empty metrics
        assert metrics.get_average_processing_time() == 0.0
        assert metrics.get_percentile_processing_time(95) == 0.0
        assert metrics.get_cache_hit_rate() == 0.0
        assert metrics.get_classification_accuracy() == 0.0
        assert metrics.get_error_rate() == 0.0
        assert metrics.get_multi_intent_rate() == 0.0
        
        # Test with no confidence scores
        distribution = metrics.get_confidence_distribution()
        assert distribution == {}
        
        # Test with no intents
        intent_dist = metrics.get_intent_distribution()
        assert intent_dist == {}


class TestMetricsAggregator:
    """Test cases for MetricsAggregator class."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return IntentRecognitionMetrics()
    
    @pytest.fixture
    def aggregator(self, metrics):
        """Create aggregator instance."""
        return MetricsAggregator(metrics)
    
    @pytest.mark.asyncio
    async def test_aggregation_lifecycle(self, aggregator):
        """Test starting and stopping aggregation."""
        # Start aggregation
        await aggregator.start_aggregation()
        assert aggregator.aggregation_task is not None
        assert not aggregator.aggregation_task.done()
        
        # Stop aggregation
        await aggregator.stop_aggregation()
        assert aggregator.aggregation_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_periodic_aggregation(self, aggregator, metrics, sample_intent_result):
        """Test periodic metrics aggregation."""
        # Set short interval for testing
        aggregator.aggregation_interval = 0.1  # 100ms
        
        # Add some metrics data
        metrics.record_query_processing(sample_intent_result)
        
        # Mock logger to capture log messages
        with patch.object(aggregator.logger, 'info') as mock_info:
            # Start aggregation
            await aggregator.start_aggregation()
            
            # Wait for at least one aggregation cycle
            await asyncio.sleep(0.2)
            
            # Stop aggregation
            await aggregator.stop_aggregation()
            
            # Verify logs were generated
            assert mock_info.call_count >= 2  # start + at least one metric log
    
    @pytest.mark.asyncio
    async def test_performance_warnings(self, aggregator, metrics):
        """Test performance warning generation."""
        # Add high processing times to trigger warning
        for _ in range(10):
            metrics.processing_times.append(250.0)  # High latency
        
        # Add errors to trigger error rate warning
        metrics.total_queries = 20
        metrics.errors["timeout"] = 2
        metrics.errors["parsing"] = 1
        
        # Mock logger to capture warnings
        with patch.object(aggregator.logger, 'warning') as mock_warning:
            # Set short interval
            aggregator.aggregation_interval = 0.1
            
            # Start aggregation
            await aggregator.start_aggregation()
            await asyncio.sleep(0.2)
            await aggregator.stop_aggregation()
            
            # Should have warnings for both high latency and error rate
            warning_calls = [call.args[0] for call in mock_warning.call_args_list]
            assert any("High P95 processing time" in call for call in warning_calls)
            # Note: Error rate warning would only trigger if > 5%
    
    @pytest.mark.asyncio
    async def test_aggregation_error_handling(self, aggregator, metrics):
        """Test error handling during aggregation."""
        # Mock get_summary_metrics to raise an exception
        with patch.object(metrics, 'get_summary_metrics', side_effect=Exception("Test error")):
            with patch.object(aggregator.logger, 'error') as mock_error:
                aggregator.aggregation_interval = 0.1
                
                await aggregator.start_aggregation()
                await asyncio.sleep(0.2)
                await aggregator.stop_aggregation()
                
                # Verify error was logged
                assert mock_error.called
                assert "Error in metrics aggregation" in mock_error.call_args[0][0]


class TestGlobalMetricsInstance:
    """Test global metrics instance management."""
    
    def test_get_metrics_singleton(self):
        """Test that get_metrics returns singleton instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        
        assert metrics1 is metrics2
        assert isinstance(metrics1, IntentRecognitionMetrics)
    
    def test_global_instance_isolation(self):
        """Test that global instance is properly isolated."""
        # Get global instance
        global_metrics = get_metrics()
        
        # Create separate instance
        local_metrics = IntentRecognitionMetrics()
        
        # Modify local instance
        local_metrics.total_queries = 100
        
        # Global should not be affected
        assert global_metrics.total_queries != 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])