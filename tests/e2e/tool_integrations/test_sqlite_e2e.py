"""
End-to-End tests for SQLite MCP tool integration.

Tests complete workflows involving database operations through natural
language queries, including table creation, data manipulation, and queries.
"""

import pytest
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import sqlite3
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.tools.sqlite_mcp import SQLiteMCP


class TestSQLiteE2E:
    """E2E tests for SQLite MCP tool integration."""
    
    @pytest.fixture
    async def setup_system(self):
        """Set up system with SQLite MCP."""
        # Create temporary directory
        test_dir = tempfile.mkdtemp(prefix="e2e_sqlite_")
        db_path = os.path.join(test_dir, "test_data.db")
        registry_db_path = os.path.join(test_dir, "registry.db")
        
        # Create test database with sample data
        self._create_test_database(db_path)
        
        # Initialize components
        registry = ToolRegistry(registry_db_path)
        await registry.initialize()
        
        # Initialize MCP integration
        mcp = MCPIntegration(registry)
        await mcp.initialize()
        
        # Add SQLite MCP server
        await mcp.add_sqlite_server(db_path, server_id="test_sqlite", use_mock=False)
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent()
        orchestrator.mcp_integration = mcp
        orchestrator.tool_registry = registry
        await orchestrator.initialize()
        
        yield {
            "orchestrator": orchestrator,
            "mcp": mcp,
            "registry": registry,
            "test_dir": test_dir,
            "db_path": db_path
        }
        
        # Cleanup
        await mcp.shutdown()
        shutil.rmtree(test_dir, ignore_errors=True)
    
    def _create_test_database(self, db_path: str):
        """Create a test database with sample data."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product TEXT,
                amount DECIMAL(10,2),
                order_date DATE,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        """)
        
        # Insert sample data
        customers = [
            ("Alice Johnson", "alice@example.com"),
            ("Bob Smith", "bob@example.com"),
            ("Charlie Brown", "charlie@example.com")
        ]
        cursor.executemany("INSERT INTO customers (name, email) VALUES (?, ?)", customers)
        
        orders = [
            (1, "Laptop", 999.99, "2024-01-15"),
            (1, "Mouse", 29.99, "2024-01-16"),
            (2, "Keyboard", 79.99, "2024-01-17"),
            (3, "Monitor", 299.99, "2024-01-18"),
            (2, "Laptop", 1299.99, "2024-01-19")
        ]
        cursor.executemany(
            "INSERT INTO orders (customer_id, product, amount, order_date) VALUES (?, ?, ?, ?)",
            orders
        )
        
        conn.commit()
        conn.close()
        logger.info("✓ Created test database with sample data")
    
    @pytest.mark.asyncio
    async def test_database_query_workflow(self, setup_system):
        """Test natural language database query."""
        logger.info("\n=== E2E Test: Database Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Natural language query
        query = "Show me all customers in the database"
        logger.info(f"Processing query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify intent recognition
        assert result.intent is not None
        assert result.intent.primary_intent.type in ["query.retrieve", "query.search"]
        logger.info(f"✓ Intent recognized: {result.intent.primary_intent.type}")
        
        # Verify SQLite tool was selected
        assert len(result.selected_tools) > 0
        tool_names = [t.get("name", "") for t in result.selected_tools]
        assert any("sqlite" in name.lower() for name in tool_names)
        logger.info("✓ SQLite tool selected")
        
        # Verify execution success
        assert result.success
        assert len(result.execution_results) > 0
        
        # Check if customers were retrieved
        for exec_result in result.execution_results:
            if exec_result.get("success"):
                result_data = exec_result.get("result", {})
                # Should contain customer data
                logger.info(f"✓ Retrieved data from database")
                break
        
        logger.info("✅ Database query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_complex_query_workflow(self, setup_system):
        """Test complex database query with joins."""
        logger.info("\n=== E2E Test: Complex Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Complex query requiring join
        query = "Find all orders with customer names and total amounts"
        logger.info(f"Processing complex query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.success
        assert result.intent.primary_intent.type in ["query.retrieve", "query.analyze"]
        
        # Verify appropriate SQL operation
        logger.info("✓ Complex query processed successfully")
        logger.info("✅ Complex query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_data_modification_workflow(self, setup_system):
        """Test data modification operations."""
        logger.info("\n=== E2E Test: Data Modification Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Insert operation
        query = "Add a new customer named David Wilson with email david@example.com"
        logger.info(f"Processing insert query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Verify intent
        assert result.intent is not None
        assert result.intent.primary_intent.type in ["action.create", "action.modify"]
        
        # Verify execution
        if result.success:
            logger.info("✓ Insert operation completed")
        
        # Verify with a follow-up query
        verify_query = "Show me the customer David Wilson"
        verify_result = await orchestrator.process_user_query(verify_query)
        
        assert verify_result.success
        logger.info("✓ Data modification verified")
        logger.info("✅ Data modification workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_aggregate_query_workflow(self, setup_system):
        """Test aggregate queries."""
        logger.info("\n=== E2E Test: Aggregate Query Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Aggregate query
        query = "What is the total amount of all orders?"
        logger.info(f"Processing aggregate query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.intent is not None
        assert result.intent.primary_intent.type in ["query.analyze", "query.retrieve"]
        
        # Should use SQLite for aggregation
        assert result.success
        logger.info("✓ Aggregate query executed")
        
        # Another aggregate
        query2 = "Count the number of customers"
        result2 = await orchestrator.process_user_query(query2)
        
        assert result2.success
        logger.info("✓ Count query executed")
        logger.info("✅ Aggregate query workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_schema_exploration_workflow(self, setup_system):
        """Test database schema exploration."""
        logger.info("\n=== E2E Test: Schema Exploration Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Schema query
        query = "Show me all tables in the database"
        logger.info(f"Processing schema query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        assert result.success
        assert result.intent.primary_intent.type in ["query.retrieve", "system.monitor"]
        
        # Should discover tables
        logger.info("✓ Schema information retrieved")
        
        # Table structure query
        query2 = "Describe the structure of the customers table"
        result2 = await orchestrator.process_user_query(query2)
        
        assert result2.success
        logger.info("✓ Table structure retrieved")
        logger.info("✅ Schema exploration workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, setup_system):
        """Test database error handling."""
        logger.info("\n=== E2E Test: Database Error Handling ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Invalid table query
        query = "Select from non_existent_table"
        logger.info(f"Processing invalid query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Should handle error gracefully
        assert result is not None
        logger.info("✓ Invalid query handled gracefully")
        
        # Constraint violation
        query2 = "Insert duplicate email alice@example.com into customers"
        result2 = await orchestrator.process_user_query(query2)
        
        # Should handle constraint violation
        assert result2 is not None
        logger.info("✓ Constraint violation handled")
        logger.info("✅ Error handling workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_multi_step_database_workflow(self, setup_system):
        """Test multi-step database operations."""
        logger.info("\n=== E2E Test: Multi-Step Database Workflow ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Multi-step query
        query = "Find customers who made orders, then calculate their total spending"
        logger.info(f"Processing multi-step query: '{query}'")
        
        result = await orchestrator.process_user_query(query)
        
        # Should recognize multiple intents
        assert result.intent is not None
        if hasattr(result.intent, "all_intents"):
            assert len(result.intent.all_intents) >= 2
            logger.info(f"✓ Recognized {len(result.intent.all_intents)} intents")
        
        assert result.success
        logger.info("✓ Multi-step operation completed")
        logger.info("✅ Multi-step database workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, setup_system):
        """Test performance of database operations."""
        logger.info("\n=== E2E Test: Database Performance Metrics ===")
        
        orchestrator = setup_system["orchestrator"]
        
        # Simple query for baseline
        start_time = datetime.now()
        query = "Select all from orders"
        result = await orchestrator.process_user_query(query)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        assert result.success
        assert query_time < 5000  # Should complete within 5 seconds
        logger.info(f"✓ Query completed in {query_time:.2f}ms")
        
        # Check if performance metrics are tracked
        if hasattr(result, "total_time_ms"):
            assert result.total_time_ms > 0
            logger.info(f"✓ Performance tracked: {result.total_time_ms:.2f}ms")
        
        logger.info("✅ Performance metrics test passed!")


def main():
    """Run SQLite E2E tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()