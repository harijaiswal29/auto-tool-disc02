#!/usr/bin/env python3
"""
Focused test runner for Intent Recognition Agent - Positive cases only.
Tests core functionality that validates research hypothesis.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.agents.intent_models import Intent, IntentResult
import time


class IntentTestRunner:
    """Run focused positive test cases for intent recognition."""
    
    def __init__(self):
        self.passed = 0
        self.total = 0
        self.results = []
    
    async def setup_agent(self):
        """Create an intent recognition agent for testing."""
        config = {
            'model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'enable_state_tracking': True,
            'enable_persistence': False,
            'enable_persistence_service': False
        }
        return IntentRecognitionAgent(config)
    
    async def test_single_intent_recognition(self, agent):
        """Test recognition of single intents - Core functionality."""
        print("\n1. Testing Single Intent Recognition (Core Functionality)")
        print("=" * 60)
        
        test_cases = [
            {
                'query': "Find all Python files in the src directory",
                'expected_intent': "query.search",
                'scenario': "Search query"
            },
            {
                'query': "Create a new configuration file for the project",
                'expected_intent': "action.create",
                'scenario': "Create action"
            },
            {
                'query': "Delete all temporary log files",
                'expected_intent': "action.delete",
                'scenario': "Delete action"
            },
            {
                'query': "Show me the current system status",
                'expected_intent': "system.monitor",
                'scenario': "Monitor query"
            }
        ]
        
        for test_case in test_cases:
            self.total += 1
            try:
                start_time = time.time()
                result = await agent.process_query(test_case['query'])
                processing_time = (time.time() - start_time) * 1000
                
                success = (
                    result.primary_intent.type == test_case['expected_intent'] and
                    result.primary_intent.confidence >= 0.7 and
                    result.confidence_threshold_met
                )
                
                if success:
                    self.passed += 1
                    print(f"✓ {test_case['scenario']}: {test_case['expected_intent']} "
                          f"(confidence: {result.primary_intent.confidence:.2f}, "
                          f"time: {processing_time:.1f}ms)")
                else:
                    print(f"✗ {test_case['scenario']}: Expected {test_case['expected_intent']}, "
                          f"got {result.primary_intent.type}")
                
                self.results.append({
                    'test': 'single_intent',
                    'scenario': test_case['scenario'],
                    'passed': success,
                    'time_ms': processing_time
                })
                
            except Exception as e:
                print(f"✗ {test_case['scenario']}: Error - {str(e)}")
                self.results.append({
                    'test': 'single_intent',
                    'scenario': test_case['scenario'],
                    'passed': False,
                    'error': str(e)
                })
    
    async def test_multi_intent_recognition(self, agent):
        """Test recognition of multi-intent queries."""
        print("\n2. Testing Multi-Intent Recognition")
        print("=" * 60)
        
        self.total += 1
        try:
            query = "Search for Python files and then analyze their complexity"
            result = await agent.process_query(query)
            
            # Check if multi-intent was detected
            success = len(result.all_intents) >= 2
            
            if success:
                self.passed += 1
                intent_types = [intent.type for intent in result.all_intents]
                print(f"✓ Multi-intent detected: {intent_types}")
            else:
                print(f"✗ Expected multiple intents, got {len(result.all_intents)}")
            
            self.results.append({
                'test': 'multi_intent',
                'scenario': 'Multi-intent query',
                'passed': success,
                'intents_count': len(result.all_intents)
            })
            
        except Exception as e:
            print(f"✗ Multi-intent test: Error - {str(e)}")
            self.results.append({
                'test': 'multi_intent',
                'scenario': 'Multi-intent query',
                'passed': False,
                'error': str(e)
            })
    
    async def test_context_enrichment(self, agent):
        """Test context enrichment functionality."""
        print("\n3. Testing Context Enrichment")
        print("=" * 60)
        
        self.total += 1
        try:
            context = {
                'session_id': 'test_session_123',
                'domain': 'software_development',
                'user_id': 'test_user'
            }
            
            result = await agent.process_query("Find configuration files", context)
            
            features = result.metadata.get('features', {})
            success = 'context_score' in features and features['context_score'] >= 0.5
            
            if success:
                self.passed += 1
                print(f"✓ Context enrichment working (score: {features.get('context_score', 0):.2f})")
            else:
                print(f"✗ Context enrichment failed")
            
            self.results.append({
                'test': 'context_enrichment',
                'scenario': 'Context-aware query',
                'passed': success,
                'context_score': features.get('context_score', 0)
            })
            
        except Exception as e:
            print(f"✗ Context enrichment test: Error - {str(e)}")
            self.results.append({
                'test': 'context_enrichment',
                'scenario': 'Context-aware query',
                'passed': False,
                'error': str(e)
            })
    
    async def test_performance_benchmarks(self, agent):
        """Test that processing meets performance requirements."""
        print("\n4. Testing Performance Benchmarks")
        print("=" * 60)
        
        queries = [
            "Find all Python files",
            "Create a new configuration",
            "Delete temporary files",
            "Update the database schema",
            "Monitor system performance"
        ]
        
        processing_times = []
        self.total += 1
        
        try:
            for query in queries:
                start_time = time.time()
                result = await agent.process_query(query)
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
            
            # Calculate metrics
            avg_time = sum(processing_times) / len(processing_times)
            sorted_times = sorted(processing_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            
            # Check performance requirements
            success = avg_time < 100 and p95_time < 200
            
            if success:
                self.passed += 1
                print(f"✓ Performance meets requirements:")
            else:
                print(f"✗ Performance below requirements:")
            
            print(f"  - Average: {avg_time:.1f}ms (target: <100ms)")
            print(f"  - P95: {p95_time:.1f}ms (target: <200ms)")
            print(f"  - Min: {min(processing_times):.1f}ms")
            print(f"  - Max: {max(processing_times):.1f}ms")
            
            self.results.append({
                'test': 'performance',
                'scenario': 'Performance benchmarks',
                'passed': success,
                'avg_time_ms': avg_time,
                'p95_time_ms': p95_time
            })
            
        except Exception as e:
            print(f"✗ Performance test: Error - {str(e)}")
            self.results.append({
                'test': 'performance',
                'scenario': 'Performance benchmarks',
                'passed': False,
                'error': str(e)
            })
    
    async def test_feature_extraction(self, agent):
        """Test that all features are properly extracted."""
        print("\n5. Testing Feature Extraction")
        print("=" * 60)
        
        self.total += 1
        try:
            result = await agent.process_query("What files were modified today?")
            
            features = result.metadata.get('features', {})
            required_features = ['tokens', 'keywords', 'semantic_scores', 
                               'keyword_scores', 'word_count', 'has_question']
            
            missing_features = [f for f in required_features if f not in features]
            success = len(missing_features) == 0 and features.get('has_question') == True
            
            if success:
                self.passed += 1
                print(f"✓ All features extracted correctly")
                print(f"  - Word count: {features.get('word_count', 0)}")
                print(f"  - Question detected: {features.get('has_question', False)}")
                print(f"  - Keywords found: {len(features.get('keywords', []))}")
            else:
                print(f"✗ Feature extraction incomplete")
                if missing_features:
                    print(f"  Missing: {missing_features}")
            
            self.results.append({
                'test': 'feature_extraction',
                'scenario': 'Feature extraction',
                'passed': success,
                'features_count': len(features)
            })
            
        except Exception as e:
            print(f"✗ Feature extraction test: Error - {str(e)}")
            self.results.append({
                'test': 'feature_extraction',
                'scenario': 'Feature extraction',
                'passed': False,
                'error': str(e)
            })
    
    async def test_state_management(self, agent):
        """Test conversation state management - validates core state transitions."""
        print("\n6. Testing State Management")
        print("=" * 60)
        
        self.total += 1
        try:
            # Initial state should be IDLE
            initial_state = agent.get_current_state()
            
            # Process a query
            await agent.process_query("Find Python files")
            
            # Get state history
            history = agent.get_state_history(limit=5)
            
            # Verify transitions occurred
            success = (
                initial_state == 'IDLE' and
                len(history) >= 1 and
                history[0]['from_state'] == 'IDLE' and
                history[0]['to_state'] == 'QUERY_RECEIVED'
            )
            
            if success:
                self.passed += 1
                print(f"✓ State transitions working correctly")
                print(f"  - Initial: {initial_state}")
                print(f"  - Transition: {history[0]['from_state']} → {history[0]['to_state']}")
            else:
                print(f"✗ State management issues detected")
            
            self.results.append({
                'test': 'state_management',
                'scenario': 'State transitions',
                'passed': success,
                'transitions': len(history)
            })
            
        except Exception as e:
            print(f"✗ State management test: Error - {str(e)}")
            self.results.append({
                'test': 'state_management',
                'scenario': 'State transitions',
                'passed': False,
                'error': str(e)
            })
    
    async def run_all_tests(self):
        """Run all positive test cases."""
        print("\nIntent Recognition Agent - Positive Test Cases")
        print("Testing core functionality for dissertation validation")
        print("=" * 60)
        
        # Setup agent
        agent = await self.setup_agent()
        print("Agent initialized successfully")
        
        # Run tests
        await self.test_single_intent_recognition(agent)
        await self.test_multi_intent_recognition(agent)
        await self.test_context_enrichment(agent)
        await self.test_performance_benchmarks(agent)
        await self.test_feature_extraction(agent)
        await self.test_state_management(agent)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {self.total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.total - self.passed}")
        print(f"Success rate: {(self.passed/self.total)*100:.1f}%")
        
        # List of skipped edge cases
        print("\nSKIPPED EDGE CASES (not relevant to core research):")
        print("- Empty query handling")
        print("- Very long query handling") 
        print("- Special characters in queries")
        print("- Unicode character handling")
        print("- Concurrent query processing")
        print("- Error state recovery")
        print("- Low confidence handling")
        
        return self.passed == self.total


async def main():
    """Main entry point."""
    runner = IntentTestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())