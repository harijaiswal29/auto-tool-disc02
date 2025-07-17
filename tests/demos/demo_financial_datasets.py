"""
Demo: Financial Datasets MCP Integration

This script demonstrates real-world financial queries using the Financial Datasets MCP
integration with the Autonomous Tool Discovery System.

Features demonstrated:
- Stock price queries
- Financial statement analysis
- Company news retrieval
- Cryptocurrency data
- Intent recognition for financial queries
- Tool discovery and selection
- Q-learning optimization
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.mcp_integration import MCPIntegration
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FinancialDatasetsDemo:
    """Demonstration of Financial Datasets MCP integration."""
    
    def __init__(self):
        self.orchestrator = None
        self.mcp_integration = None
        
    async def setup(self):
        """Initialize the system with Financial Datasets MCP."""
        logger.info("="*80)
        logger.info("Financial Datasets MCP Integration Demo")
        logger.info("="*80)
        logger.info("Initializing Autonomous Tool Discovery System...\n")
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent()
        await self.orchestrator.initialize()
        
        # Get MCP integration
        self.mcp_integration = self.orchestrator.mcp_integration
        
        # Add Financial Datasets server (using mock)
        logger.info("Adding Financial Datasets MCP server (mock mode)...")
        success = await self.mcp_integration.add_financial_datasets_server(
            server_id="financial_demo",
            use_mock=False
        )
        
        if success:
            logger.info("✓ Financial Datasets server added successfully")
        else:
            logger.error("✗ Failed to add Financial Datasets server")
            return False
            
        # Also add filesystem for comparison
        await self.mcp_integration.add_filesystem_server(
            "/tmp",
            server_id="fs_demo",
            use_mock=True
        )
        
        logger.info("✓ System initialized\n")
        return True
    
    async def run_query(self, query: str, query_num: int):
        """Run a single query and display results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Query {query_num}: {query}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # Process query through orchestrator
            result = await self.orchestrator.process_user_query(query)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Display intent recognition
            logger.info("\n📊 Intent Recognition:")
            logger.info(f"  - Intent Type: {result.intent.type}")
            logger.info(f"  - Confidence: {result.intent.confidence:.2f}")
            logger.info(f"  - Keywords: {', '.join(result.intent.keywords[:5])}")
            
            # Display discovered tools
            logger.info("\n🔧 Discovered Tools:")
            for tool in result.discovered_tools[:3]:
                logger.info(f"  - {tool['name']} (Score: {tool.get('score', 'N/A')})")
            
            # Display selected tools
            logger.info("\n✅ Selected Tools:")
            for tool in result.selected_tools:
                logger.info(f"  - {tool['name']}")
            
            # Display results
            logger.info("\n📈 Results:")
            if result.success:
                # Format financial data nicely
                for tool_result in result.tool_results:
                    if tool_result.success:
                        self._display_financial_result(tool_result.tool_name, tool_result.result)
                    else:
                        logger.warning(f"  ❌ {tool_result.tool_name} failed: {tool_result.error}")
            else:
                logger.error(f"Query failed: {result.error}")
            
            # Display metrics
            logger.info(f"\n⏱️  Execution Time: {duration:.2f} seconds")
            
            # Show Q-learning info if enabled
            if hasattr(self.orchestrator, 'q_learning_enabled') and self.orchestrator.q_learning_enabled:
                logger.info("\n🧠 Q-Learning:")
                logger.info(f"  - Exploration Rate: {self.orchestrator.q_engine.exploration_rate:.3f}")
                logger.info(f"  - Episode Count: {self.orchestrator.q_engine.episode_count}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
    def _display_financial_result(self, tool_name: str, result: dict):
        """Display financial results in a formatted way."""
        if "get_stock_price" in tool_name:
            logger.info(f"\n  💹 Stock Price - {result.get('symbol', 'N/A')}:")
            logger.info(f"     Price: ${result.get('price', 'N/A')}")
            logger.info(f"     Change: {result.get('change', 'N/A')} ({result.get('change_percent', 'N/A')}%)")
            logger.info(f"     Volume: {result.get('volume', 'N/A'):,}")
            
        elif "income_statement" in tool_name:
            logger.info(f"\n  📊 Income Statement - {result.get('symbol', 'N/A')}:")
            logger.info(f"     Revenue: ${result.get('revenue', 0):,.0f}")
            logger.info(f"     Gross Profit: ${result.get('gross_profit', 0):,.0f}")
            logger.info(f"     Net Income: ${result.get('net_income', 0):,.0f}")
            logger.info(f"     EPS: ${result.get('eps', 'N/A')}")
            logger.info(f"     Period: {result.get('period', 'N/A')}")
            
        elif "balance_sheet" in tool_name:
            logger.info(f"\n  📋 Balance Sheet - {result.get('symbol', 'N/A')}:")
            logger.info(f"     Total Assets: ${result.get('total_assets', 0):,.0f}")
            logger.info(f"     Total Liabilities: ${result.get('total_liabilities', 0):,.0f}")
            logger.info(f"     Total Equity: ${result.get('total_equity', 0):,.0f}")
            logger.info(f"     Cash: ${result.get('cash', 0):,.0f}")
            
        elif "cash_flow" in tool_name:
            logger.info(f"\n  💰 Cash Flow - {result.get('symbol', 'N/A')}:")
            logger.info(f"     Operating Cash Flow: ${result.get('operating_cash_flow', 0):,.0f}")
            logger.info(f"     Free Cash Flow: ${result.get('free_cash_flow', 0):,.0f}")
            logger.info(f"     Period: {result.get('period', 'N/A')}")
            
        elif "company_news" in tool_name:
            logger.info(f"\n  📰 Company News - {result.get('symbol', 'N/A')}:")
            news_items = result.get('news', [])[:3]  # Show top 3
            for i, item in enumerate(news_items, 1):
                logger.info(f"     {i}. {item.get('title', 'N/A')}")
                logger.info(f"        Source: {item.get('source', 'N/A')} | Sentiment: {item.get('sentiment', 'N/A')}")
                
        elif "crypto_price" in tool_name:
            logger.info(f"\n  🪙 Crypto Price - {result.get('symbol', 'N/A')}:")
            logger.info(f"     Price: {result.get('price', 'N/A')} {result.get('currency', 'USD')}")
            logger.info(f"     24h Change: {result.get('change_24h', 'N/A')}%")
            
        elif "search_companies" in tool_name:
            logger.info(f"\n  🔍 Company Search Results:")
            companies = result.get('companies', [])[:5]  # Show top 5
            for company in companies:
                logger.info(f"     • {company.get('name', 'N/A')} ({company.get('symbol', 'N/A')}) - {company.get('exchange', 'N/A')}")
        else:
            # Generic display
            logger.info(f"\n  📄 {tool_name} Result:")
            logger.info(f"     {json.dumps(result, indent=6)}")
    
    async def run_demo(self):
        """Run the complete demonstration."""
        # Initialize system
        if not await self.setup():
            return
        
        # Define real-world financial queries
        queries = [
            # Stock Market Analysis
            "What is Apple's current stock price?",
            "Show me Tesla's stock performance with trading volume",
            "Get Microsoft stock price",
            
            # Financial Statements
            "Show me Amazon's income statement",
            "What is Apple's revenue and profit?",
            "Get Microsoft's balance sheet",
            
            # Company Financial Health
            "Analyze Google's cash flow",
            "Show me Apple's financial health including assets and liabilities",
            
            # Market News
            "Get latest news about Tesla",
            "What's happening with Apple stock?",
            
            # Cryptocurrency
            "What's the current Bitcoin price?",
            "Show me Ethereum price in EUR",
            
            # Company Search
            "Find technology companies",
            "Search for companies like Tesla",
            
            # Complex Queries
            "Compare Apple and Microsoft stock prices",
            "Show me Apple's complete financial overview including stock price, revenue, and cash flow"
        ]
        
        # Run each query
        for i, query in enumerate(queries, 1):
            await self.run_query(query, i)
            await asyncio.sleep(0.5)  # Small delay between queries
        
        # Display summary
        await self.display_summary()
    
    async def display_summary(self):
        """Display demo summary and statistics."""
        logger.info("\n" + "="*80)
        logger.info("Demo Summary")
        logger.info("="*80)
        
        # Get registry stats
        tools = self.mcp_integration.registry.get_all_tools()
        financial_tools = [t for t in tools if "financial" in t.get("server_id", "")]
        
        logger.info(f"\n📊 Tool Registry Statistics:")
        logger.info(f"  - Total tools registered: {len(tools)}")
        logger.info(f"  - Financial tools: {len(financial_tools)}")
        
        # List financial tools
        logger.info("\n💰 Available Financial Tools:")
        for tool in financial_tools:
            logger.info(f"  - {tool['name']}: {tool['capabilities'].get('description', 'N/A')}")
        
        # Performance stats
        if hasattr(self.orchestrator.intent_agent, 'get_metrics_summary'):
            metrics = self.orchestrator.intent_agent.get_metrics_summary()
            logger.info(f"\n⚡ Performance Metrics:")
            logger.info(f"  - Avg Intent Recognition Time: {metrics['performance']['avg_processing_time_ms']:.2f}ms")
            logger.info(f"  - Cache Hit Rate: {metrics['cache']['hit_rate']:.1f}%")
        
        logger.info("\n✅ Demo completed successfully!")
        logger.info("This demonstration showcased:")
        logger.info("  • Intent recognition for financial queries")
        logger.info("  • Tool discovery optimized for finance domain")
        logger.info("  • Seamless integration with existing system")
        logger.info("  • Real-world financial data retrieval")
        logger.info("  • Q-learning optimization for tool selection")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()


async def main():
    """Run the financial datasets demonstration."""
    demo = FinancialDatasetsDemo()
    
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())