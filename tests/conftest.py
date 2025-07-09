"""
Pytest configuration and shared fixtures for all tests.
"""
import asyncio
import pytest
import sys
from pathlib import Path
import tempfile
import shutil
from typing import Generator, AsyncGenerator

# Add project root to path for imports (go up one level from tests/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_test_db(temp_dir: Path) -> Path:
    """Create a sample SQLite database for testing."""
    db_path = temp_dir / "test.db"
    import sqlite3
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE tools (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            endpoint TEXT NOT NULL
        )
    """)
    
    # Insert sample data
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", 
                   ("Test User", "test@example.com"))
    cursor.execute("INSERT INTO tools (id, name, type, endpoint) VALUES (?, ?, ?, ?)",
                   ("tool1", "Test Tool", "mcp", "stdio://test"))
    
    conn.commit()
    conn.close()
    
    return db_path


@pytest.fixture
async def mock_mcp_client():
    """Create a mock MCP client for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.list_tools = AsyncMock(return_value=[
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ])
    client.call_tool = AsyncMock(return_value={"result": "success"})
    
    return client


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry for testing."""
    from unittest.mock import AsyncMock
    
    registry = AsyncMock()
    registry.register_tool = AsyncMock()
    registry.get_tool = AsyncMock()
    registry.list_tools = AsyncMock(return_value=[])
    registry.update_relationships = AsyncMock()
    
    return registry


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration for tests."""
    return {
        "mcp_servers": {
            "test_server": {
                "command": ["python", "-m", "test_server"],
                "timeout": 30
            }
        },
        "learning": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.2
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
async def cleanup_async_tasks():
    """Ensure all async tasks are cleaned up after tests."""
    yield
    
    # Cancel any remaining tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    # Wait for all tasks to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# Markers for test categories
# These are dynamically created by pytest, no need to explicitly define them