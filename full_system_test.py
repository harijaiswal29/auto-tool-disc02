#!/usr/bin/env python3
"""
Full System Test Script for Autonomous Tool Discovery
Tests the system with various queries to verify end-to-end functionality
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.mcp_integration import MCPIntegration

class SystemTester:
    def __init__(self):
        self.orchestrator = None
        self.test_results = []
        
    async def initialize(self):
        """Initialize the orchestrator and components"""
        print("=" * 60)
        print("Full System Test - Autonomous Tool Discovery")
        print("=" * 60)
        print("\nInitializing components...")
        
        self.orchestrator = OrchestratorAgent()
        await self.orchestrator.initialize()
        
        # Setup default MCP servers
        print("\nSetting up default MCP servers...")
        mcp_integration = self.orchestrator.mcp_integration
        
        # Add SQLite server
        success = await mcp_integration.add_sqlite_server(
            db_path="data/test.db",
            server_id="sqlite_main",
            use_mock=True
        )
        if success:
            print("✓ SQLite server added")
        
        # Add filesystem server
        success = await mcp_integration.add_filesystem_server(
            base_path=".",
            server_id="filesystem_main",
            use_mock=True
        )
        if success:
            print("✓ Filesystem server added")
        
        # Add web search server
        success = await mcp_integration.add_search_server(
            server_id="search_main",
            use_mock=True
        )
        if success:
            print("✓ Search server added")
        
        # Add weather server
        success = await mcp_integration.add_weather_server(
            server_id="weather_main",
            use_mock=True
        )
        if success:
            print("✓ Weather server added")
        
        print("\nSystem initialized successfully!")
        
    async def run_test_query(self, query: str, test_name: str):
        """Run a single test query"""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Query: {query}")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Process the query
            result = await self.orchestrator.process_user_query(query)
            
            execution_time = time.time() - start_time
            
            # Extract key information from OrchestrationResult
            test_result = {
                "test_name": test_name,
                "query": query,
                "execution_time": execution_time,
                "status": "success" if result.success else "failed",
                "tools_discovered": result.discovered_tools,
                "tools_selected": result.selected_tools,
                "execution_results": [{
                    "tool_name": er.tool_name,
                    "success": er.success,
                    "error": er.error,
                    "execution_time_ms": er.execution_time_ms
                } for er in result.execution_results],
                "summary": result.summary,
                "error": None
            }
            
            # Print summary
            print(f"\nExecution Time: {execution_time:.2f}s")
            print(f"Tools Discovered: {len(test_result['tools_discovered'])}")
            print(f"Tools Selected: {test_result['tools_selected']}")
            
            if test_result['execution_results']:
                print("\nExecution Results:")
                for idx, res in enumerate(test_result['execution_results']):
                    status = "success" if res.get('success') else "failed"
                    print(f"  {res.get('tool_name', 'unknown')}: {status}")
                    if res.get('error'):
                        print(f"    Error: {res['error']}")
            
            print(f"\nSummary: {result.summary}")
                        
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = {
                "test_name": test_name,
                "query": query,
                "execution_time": execution_time,
                "status": "failed",
                "error": str(e)
            }
            print(f"\nTest Failed: {str(e)}")
            
        self.test_results.append(test_result)
        return test_result
        
    async def run_all_tests(self):
        """Run all test queries"""
        test_queries = [
            ("Database Analysis", "I need to analyze sales data from a database"),
            ("Web Search & File Save", "Search for Python tutorials and save them to a file"),
            ("Weather & Database", "What's the weather in New York and log it to the database")
        ]
        
        for test_name, query in test_queries:
            await self.run_test_query(query, test_name)
            # Small delay between tests
            await asyncio.sleep(2)
            
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*60)
        print("FULL SYSTEM TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['status'] == 'success')
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            print(f"\n- {result['test_name']}: {result['status'].upper()}")
            print(f"  Execution Time: {result['execution_time']:.2f}s")
            if result['status'] == 'success':
                print(f"  Tools Discovered: {len(result.get('tools_discovered', []))}")
                print(f"  Tools Selected: {result.get('tools_selected', [])}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
                
        # Check for dissertation goal impacts
        print("\n" + "="*60)
        print("DISSERTATION GOAL ASSESSMENT")
        print("="*60)
        
        issues = []
        
        # Check if system can discover tools
        discovery_working = any(r['status'] == 'success' and r.get('tools_discovered') 
                              for r in self.test_results)
        if not discovery_working:
            issues.append("Tool discovery not functioning - Critical for H1")
            
        # Check if system can select appropriate tools
        selection_working = any(r['status'] == 'success' and r.get('tools_selected') 
                              for r in self.test_results)
        if not selection_working:
            issues.append("Tool selection not functioning - Critical for H2")
            
        # Check if learning is happening (based on execution results)
        learning_working = any(r['status'] == 'success' and len(r.get('execution_results', [])) > 0
                             for r in self.test_results)
        if not learning_working:
            issues.append("No successful tool executions - May impact H3 (learning)")
            
        # Check execution times
        avg_time = sum(r['execution_time'] for r in self.test_results) / len(self.test_results)
        if avg_time > 10:
            issues.append(f"High average execution time ({avg_time:.1f}s) - May impact H5")
            
        if issues:
            print("\nPotential Issues Affecting Dissertation Goals:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("\nNo critical issues detected. System appears ready for dissertation evaluation.")
            
        # Save results to file
        report_path = Path("full_system_test_results.json")
        with open(report_path, "w") as f:
            json.dump({
                "test_results": self.test_results,
                "summary": {
                    "total_tests": total_tests,
                    "successful": successful_tests,
                    "success_rate": (successful_tests/total_tests)*100,
                    "avg_execution_time": avg_time,
                    "issues": issues
                }
            }, f, indent=2)
            
        print(f"\nDetailed results saved to: {report_path}")
        
    async def cleanup(self):
        """Clean up resources"""
        if self.orchestrator and hasattr(self.orchestrator, 'mcp_integration'):
            # Shutdown all MCP servers
            await self.orchestrator.mcp_integration.shutdown_all()
        # No explicit cleanup method on orchestrator, resources will be cleaned up by garbage collection

async def main():
    """Main test function"""
    tester = SystemTester()
    
    try:
        await tester.initialize()
        await tester.run_all_tests()
        tester.generate_report()
    except Exception as e:
        print(f"\nCritical error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())