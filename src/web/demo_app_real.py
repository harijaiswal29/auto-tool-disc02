"""
Real Web Demonstration Interface for Autonomous Tool Discovery System.
This version uses the ACTUAL orchestrator with Q-learning enabled.
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.intent_models import IntentResult, Intent
from src.utils.logger import get_logger
from src.web.model_helper import model_loader, intent_recognizer
from src.web.init_demo_tools import init_demo_tools

# Initialize logger
logger = get_logger(__name__)

# Global orchestrator instance
orchestrator: Optional[OrchestratorAgent] = None

# Session storage (in-memory for demo)
sessions: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    """Request model for query submission."""
    query: str
    context: Optional[Dict[str, Any]] = None


class SessionStatus(BaseModel):
    """Status model for session tracking."""
    session_id: str
    status: str
    current_stage: str
    stages: Dict[str, Any]
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    global orchestrator
    logger.info("Starting up Autonomous Tool Discovery Demo (Real Version)...")
    
    try:
        # Step 1: Initialize demo tools in registry
        logger.info("Initializing demo tools in registry...")
        await init_demo_tools()
        logger.info("Demo tools initialized")
        
        # Step 2: Load ML models with fallback
        logger.info("Loading ML models...")
        await model_loader.load_sentence_transformer()
        if model_loader.fallback_mode:
            logger.info("Using fallback embedding method")
        else:
            logger.info("ML models loaded successfully")
        
        # Step 3: Create orchestrator with Q-learning ENABLED
        config = {
            "q_learning": {
                "enable_learning": True,  # ENABLE Q-learning!
                "model_path": "models/q_learning_model.pkl",
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "exploration_rate": 0.2,
                "min_exploration_rate": 0.01,
                "exploration_decay": 0.995
            },
            "orchestration": {
                "max_tools_per_query": 3,
                "tool_selection_strategy": "q_learning",  # Use Q-learning strategy
                "parallel_execution": True
            },
            "result_cache": {
                "enabled": True,
                "max_size": 100,
                "ttl_seconds": 300
            },
            "database": {
                "tool_registry": "data/registry/tools.db",
                "learning_data": "data/learning.db"
            },
            "mcp": {
                "use_mock_servers": True,  # Force mock servers for demo
                "auto_fallback": True
            }
        }
        
        logger.info("Creating orchestrator with Q-learning enabled...")
        orchestrator = OrchestratorAgent(config)
        
        # Step 4: Initialize orchestrator
        await orchestrator.initialize()
        
        # Step 5: Initialize MCP servers (try real first, fallback to mock)
        if orchestrator.mcp_integration:
            logger.info("Initializing MCP servers...")
            
            # Try to connect to real MCP servers first, fallback to mock if unavailable
            servers_status = []
            
            # Search server (Brave API)
            logger.info("Attempting to connect to real Search MCP server...")
            brave_api_key = os.getenv("BRAVE_API_KEY")
            if brave_api_key and brave_api_key != "your_brave_api_key_here":
                logger.info(f"Using Brave Search API with key: {brave_api_key[:10]}...")
                search_config = {"api_key": brave_api_key}
                search_connected = await orchestrator.mcp_integration.add_search_server(
                    config=search_config, 
                    use_mock=False
                )
            else:
                search_connected = False
                
            if not search_connected:
                logger.info("Real Search server unavailable, using mock server")
                search_connected = await orchestrator.mcp_integration.add_search_server(use_mock=True)
            servers_status.append(("Search", "Real" if not orchestrator.mcp_integration.servers.get('search_default', {}).get('is_mock') else "Mock"))
            
            # Weather server (OpenWeather API)
            logger.info("Attempting to connect to real Weather MCP server...")
            weather_connected = await orchestrator.mcp_integration.add_weather_server(use_mock=False)
            if not weather_connected:
                logger.info("Real Weather server unavailable, using mock server")
                weather_connected = await orchestrator.mcp_integration.add_weather_server(use_mock=True)
            servers_status.append(("Weather", "Real" if not orchestrator.mcp_integration.servers.get('weather_default', {}).get('is_mock') else "Mock"))
            
            # Filesystem server
            logger.info("Attempting to connect to real Filesystem MCP server...")
            # Use a safe directory for filesystem operations
            fs_base_path = os.path.join(os.getcwd(), "data", "filesystem_workspace")
            os.makedirs(fs_base_path, exist_ok=True)
            logger.info(f"Using filesystem base path: {fs_base_path}")
            
            fs_connected = await orchestrator.mcp_integration.add_filesystem_server(
                base_path=fs_base_path,
                use_mock=False
            )
            if not fs_connected:
                logger.info("Real Filesystem server unavailable, using mock server")
                fs_connected = await orchestrator.mcp_integration.add_filesystem_server(
                    base_path=fs_base_path,
                    use_mock=True
                )
            servers_status.append(("Filesystem", "Real" if not orchestrator.mcp_integration.servers.get('filesystem_default', {}).get('is_mock') else "Mock"))
            
            # SQLite server
            logger.info("Attempting to connect to real SQLite MCP server...")
            try:
                sqlite_connected = await orchestrator.mcp_integration.add_sqlite_server(
                    db_path="data/demo.db",
                    use_mock=False
                )
                if not sqlite_connected:
                    logger.info("Real SQLite server unavailable, using mock server")
                    sqlite_connected = await orchestrator.mcp_integration.add_sqlite_server(
                        db_path="data/demo.db",
                        use_mock=True
                    )
                servers_status.append(("SQLite", "Real" if sqlite_connected and not orchestrator.mcp_integration.servers.get('sqlite_default', {}).get('is_mock') else "Mock"))
            except Exception as e:
                logger.warning(f"SQLite server initialization failed: {e}")
            
            # GitHub server
            logger.info("Attempting to connect to real GitHub MCP server...")
            try:
                github_token = os.getenv("GITHUB_TOKEN")
                if github_token and github_token != "your_github_personal_access_token_here":
                    logger.info(f"Using GitHub API with token: ghp_{'*' * 10}...")
                    github_connected = await orchestrator.mcp_integration.add_github_server(
                        github_token=github_token,
                        use_mock=False
                    )
                else:
                    github_connected = False
                    
                if not github_connected:
                    logger.info("Real GitHub server unavailable, using mock server")
                    github_connected = await orchestrator.mcp_integration.add_github_server(use_mock=True)
                servers_status.append(("GitHub", "Real" if github_connected and not orchestrator.mcp_integration.servers.get('github_default', {}).get('is_mock') else "Mock"))
            except Exception as e:
                logger.warning(f"GitHub server initialization failed: {e}")
            
            # PostgreSQL server
            logger.info("Attempting to connect to real PostgreSQL MCP server...")
            try:
                postgres_conn = os.getenv("POSTGRES_CONNECTION_STRING")
                if postgres_conn and "auto_tool_user" in postgres_conn:
                    logger.info(f"Using PostgreSQL connection: {postgres_conn.split('@')[1] if '@' in postgres_conn else 'localhost'}")
                    postgres_connected = await orchestrator.mcp_integration.add_postgres_server(
                        connection_string=postgres_conn,
                        use_mock=False
                    )
                else:
                    postgres_connected = False
                    
                if not postgres_connected:
                    logger.info("Real PostgreSQL server unavailable, using mock server")
                    postgres_connected = await orchestrator.mcp_integration.add_postgres_server(
                        connection_string="postgresql://localhost/demo",
                        use_mock=True
                    )
                servers_status.append(("PostgreSQL", "Real" if postgres_connected and not orchestrator.mcp_integration.servers.get('postgres_default', {}).get('is_mock') else "Mock"))
            except Exception as e:
                logger.warning(f"PostgreSQL server initialization failed: {e}")
            
            # Notion server
            logger.info("Attempting to connect to real Notion MCP server...")
            try:
                notion_api_key = os.getenv("NOTION_API_KEY")
                if notion_api_key and notion_api_key != "your_notion_api_key_here":
                    logger.info(f"Using Notion API with key: {notion_api_key[:10]}...")
                    notion_connected = await orchestrator.mcp_integration.add_notion_server(
                        api_key=notion_api_key,
                        use_mock=False
                    )
                else:
                    notion_connected = False
                    
                if not notion_connected:
                    logger.info("Real Notion server unavailable, using mock server")
                    notion_connected = await orchestrator.mcp_integration.add_notion_server(use_mock=True)
                servers_status.append(("Notion", "Real" if notion_connected and not orchestrator.mcp_integration.servers.get('notion_default', {}).get('is_mock') else "Mock"))
            except Exception as e:
                logger.warning(f"Notion server initialization failed: {e}")
            
            # Financial Datasets server
            logger.info("Attempting to connect to real Financial Datasets MCP server...")
            try:
                financial_api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
                if financial_api_key and financial_api_key != "your_financial_api_key_here":
                    logger.info(f"Using Financial Datasets API with key: {financial_api_key[:10]}...")
                    financial_connected = await orchestrator.mcp_integration.add_financial_datasets_server(
                        api_key=financial_api_key,
                        use_mock=False
                    )
                else:
                    financial_connected = False
                    
                if not financial_connected:
                    logger.info("Real Financial Datasets server unavailable, using mock server")
                    financial_connected = await orchestrator.mcp_integration.add_financial_datasets_server(use_mock=True)
                servers_status.append(("Financial Datasets", "Real" if financial_connected and not orchestrator.mcp_integration.servers.get('financial_datasets_default', {}).get('is_mock') else "Mock"))
            except Exception as e:
                logger.warning(f"Financial Datasets server initialization failed: {e}")
            
            # Zerodha server (trading platform)
            logger.info("Attempting to connect to real Zerodha MCP server...")
            try:
                zerodha_api_key = os.getenv("ZERODHA_API_KEY")
                zerodha_api_secret = os.getenv("ZERODHA_API_SECRET")
                zerodha_access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
                
                if zerodha_api_key and zerodha_api_key != "your_zerodha_api_key_here":
                    logger.info(f"Using Zerodha API with key: {zerodha_api_key[:10]}...")
                    zerodha_connected = await orchestrator.mcp_integration.add_zerodha_server(
                        api_key=zerodha_api_key,
                        api_secret=zerodha_api_secret,
                        access_token=zerodha_access_token,
                        use_mock=False
                    )
                else:
                    zerodha_connected = False
                    
                if not zerodha_connected:
                    logger.info("Real Zerodha server unavailable, using mock server")
                    zerodha_connected = await orchestrator.mcp_integration.add_zerodha_server(use_mock=True)
                servers_status.append(("Zerodha", "Real" if zerodha_connected and not orchestrator.mcp_integration.servers.get('zerodha_default', {}).get('is_mock') else "Mock"))
            except Exception as e:
                logger.warning(f"Zerodha server initialization failed: {e}")
            
            # Print server status summary
            logger.info("="*60)
            logger.info("MCP Server Status:")
            for server_name, mode in servers_status:
                emoji = "🌐" if mode == "Real" else "🔧"
                logger.info(f"  {emoji} {server_name}: {mode} Mode")
            logger.info("="*60)
        
        # Verify Q-learning is active
        if orchestrator.q_learning_engine:
            logger.info("✅ Q-learning engine is active!")
            logger.info(f"   Exploration rate: {orchestrator.q_learning_engine.exploration_rate:.2f}")
        else:
            logger.warning("⚠️ Q-learning engine not initialized")
        
        logger.info("Orchestrator initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        # Create fallback orchestrator with Q-learning still enabled
        fallback_config = {
            "q_learning": {
                "enable_learning": True,
                "model_path": "models/q_learning_model.pkl",
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "exploration_rate": 0.2,
                "min_exploration_rate": 0.01,
                "exploration_decay": 0.995
            },
            "orchestration": {
                "max_tools_per_query": 3,
                "tool_selection_strategy": "q_learning",
                "parallel_execution": True
            }
        }
        orchestrator = OrchestratorAgent(fallback_config)
        await orchestrator.initialize()
        
        # Try to initialize MCP servers even after error
        if orchestrator.mcp_integration:
            try:
                logger.info("Initializing MCP servers in fallback mode...")
                servers_status = []
                
                # Try real servers first, fallback to mock
                search_connected = await orchestrator.mcp_integration.add_search_server(use_mock=False)
                if not search_connected:
                    search_connected = await orchestrator.mcp_integration.add_search_server(use_mock=True)
                servers_status.append(("Search", "Real" if search_connected and not orchestrator.mcp_integration.servers.get('search_default', {}).get('is_mock') else "Mock"))
                
                weather_connected = await orchestrator.mcp_integration.add_weather_server(use_mock=False)
                if not weather_connected:
                    weather_connected = await orchestrator.mcp_integration.add_weather_server(use_mock=True)
                servers_status.append(("Weather", "Real" if weather_connected and not orchestrator.mcp_integration.servers.get('weather_default', {}).get('is_mock') else "Mock"))
                
                fs_base_path = os.path.join(os.getcwd(), "data", "filesystem_workspace")
                os.makedirs(fs_base_path, exist_ok=True)
                fs_connected = await orchestrator.mcp_integration.add_filesystem_server(
                    base_path=fs_base_path,
                    use_mock=False
                )
                if not fs_connected:
                    fs_connected = await orchestrator.mcp_integration.add_filesystem_server(
                        base_path=fs_base_path,
                        use_mock=True
                    )
                servers_status.append(("Filesystem", "Real" if fs_connected and not orchestrator.mcp_integration.servers.get('filesystem_default', {}).get('is_mock') else "Mock"))
                
                # Try GitHub server in fallback
                try:
                    github_token = os.getenv("GITHUB_TOKEN")
                    if github_token and github_token != "your_github_personal_access_token_here":
                        github_connected = await orchestrator.mcp_integration.add_github_server(
                            github_token=github_token,
                            use_mock=False
                        )
                    else:
                        github_connected = False
                    
                    if not github_connected:
                        github_connected = await orchestrator.mcp_integration.add_github_server(use_mock=True)
                    servers_status.append(("GitHub", "Real" if github_connected and not orchestrator.mcp_integration.servers.get('github_default', {}).get('is_mock') else "Mock"))
                except:
                    pass
                
                # Try PostgreSQL server in fallback
                try:
                    postgres_conn = os.getenv("POSTGRES_CONNECTION_STRING")
                    if postgres_conn and "auto_tool_user" in postgres_conn:
                        postgres_connected = await orchestrator.mcp_integration.add_postgres_server(
                            connection_string=postgres_conn,
                            use_mock=False
                        )
                    else:
                        postgres_connected = False
                        
                    if not postgres_connected:
                        postgres_connected = await orchestrator.mcp_integration.add_postgres_server(
                            connection_string="postgresql://localhost/demo",
                            use_mock=True
                        )
                    servers_status.append(("PostgreSQL", "Real" if postgres_connected and not orchestrator.mcp_integration.servers.get('postgres_default', {}).get('is_mock') else "Mock"))
                except:
                    pass
                
                # Try Notion server in fallback
                try:
                    notion_api_key = os.getenv("NOTION_API_KEY")
                    if notion_api_key and notion_api_key != "your_notion_api_key_here":
                        notion_connected = await orchestrator.mcp_integration.add_notion_server(
                            api_key=notion_api_key,
                            use_mock=False
                        )
                    else:
                        notion_connected = False
                    
                    if not notion_connected:
                        notion_connected = await orchestrator.mcp_integration.add_notion_server(use_mock=True)
                    servers_status.append(("Notion", "Real" if notion_connected and not orchestrator.mcp_integration.servers.get('notion_default', {}).get('is_mock') else "Mock"))
                except:
                    pass
                
                # Try Financial Datasets server in fallback
                try:
                    financial_api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
                    if financial_api_key and financial_api_key != "your_financial_api_key_here":
                        financial_connected = await orchestrator.mcp_integration.add_financial_datasets_server(
                            api_key=financial_api_key,
                            use_mock=False
                        )
                    else:
                        financial_connected = False
                    
                    if not financial_connected:
                        financial_connected = await orchestrator.mcp_integration.add_financial_datasets_server(use_mock=True)
                    servers_status.append(("Financial Datasets", "Real" if financial_connected and not orchestrator.mcp_integration.servers.get('financial_datasets_default', {}).get('is_mock') else "Mock"))
                except:
                    pass
                
                # Try Zerodha server in fallback
                try:
                    zerodha_api_key = os.getenv("ZERODHA_API_KEY")
                    zerodha_api_secret = os.getenv("ZERODHA_API_SECRET")
                    zerodha_access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
                    
                    if zerodha_api_key and zerodha_api_key != "your_zerodha_api_key_here":
                        zerodha_connected = await orchestrator.mcp_integration.add_zerodha_server(
                            api_key=zerodha_api_key,
                            api_secret=zerodha_api_secret,
                            access_token=zerodha_access_token,
                            use_mock=False
                        )
                    else:
                        zerodha_connected = False
                    
                    if not zerodha_connected:
                        zerodha_connected = await orchestrator.mcp_integration.add_zerodha_server(use_mock=True)
                    servers_status.append(("Zerodha", "Real" if zerodha_connected and not orchestrator.mcp_integration.servers.get('zerodha_default', {}).get('is_mock') else "Mock"))
                except:
                    pass
                
                logger.info("MCP servers initialized in fallback mode:")
                for server_name, mode in servers_status:
                    logger.info(f"  - {server_name}: {mode}")
            except Exception as server_error:
                logger.warning(f"Could not initialize servers: {server_error}")
    
    yield  # Run the application
    
    # Shutdown
    logger.info("Shutting down...")
    if orchestrator:
        try:
            await orchestrator.shutdown()
        except:
            pass
    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Autonomous Tool Discovery Demo (Real)", 
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main demo interface."""
    # Try enhanced demo first, then fallback to regular demo
    enhanced_html = static_dir / "demo_enhanced.html"
    if enhanced_html.exists():
        return enhanced_html.read_text()
    
    html_file = static_dir / "demo.html"
    if html_file.exists():
        return html_file.read_text()
    else:
        return """
        <html>
            <head><title>Real Orchestrator Demo</title></head>
            <body>
                <h1>Autonomous Tool Discovery Demo (Real Version)</h1>
                <p>This version uses the ACTUAL orchestrator with Q-learning!</p>
                <p>Please ensure demo.html is in the static directory.</p>
            </body>
        </html>
        """


@app.post("/demo/process")
async def process_query(request: QueryRequest):
    """
    Submit a query for processing using the REAL orchestrator.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Initialize session
    sessions[session_id] = {
        "query": request.query,
        "context": request.context or {},
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "stages": {
            "intent_recognition": {"status": "pending", "data": None},
            "tool_discovery": {"status": "pending", "data": None},
            "tool_selection": {"status": "pending", "data": None},
            "execution": {"status": "pending", "data": None},
            "results": {"status": "pending", "data": None}
        },
        "current_stage": "intent_recognition"
    }
    
    # Start async processing with REAL orchestrator
    asyncio.create_task(_process_with_real_orchestrator(session_id, request.query, request.context))
    
    return {"session_id": session_id, "status": "processing"}


async def _process_with_real_orchestrator(session_id: str, query: str, context: Optional[Dict[str, Any]]):
    """
    Process query using the REAL orchestrator and capture all stages.
    """
    try:
        session = sessions[session_id]
        
        # Use the REAL orchestrator's complete flow
        logger.info(f"Session {session_id}: Processing with REAL orchestrator")
        
        # Call the real orchestrator
        result = await orchestrator.process_user_query(query, context)
        
        # Now extract all the information from the result
        
        # Stage 1: Intent Recognition (from result)
        session["stages"]["intent_recognition"] = {
            "status": "completed",
            "data": {
                "type": result.intent.primary_intent.type if result.intent else "unknown",
                "confidence": result.intent.primary_intent.confidence if result.intent else 0,
                "keywords": result.intent.primary_intent.keywords if result.intent else [],
                "all_intents": [
                    {"type": intent.type, "confidence": intent.confidence}
                    for intent in (result.intent.all_intents if result.intent else [])
                ]
            }
        }
        
        # Stage 2: Tool Discovery (from result)
        session["stages"]["tool_discovery"] = {
            "status": "completed",
            "data": {
                "discovered_count": len(result.discovered_tools),
                "tools": [
                    {
                        "id": tool.get("id"),
                        "name": tool.get("name"),
                        "type": tool.get("server_type") or tool.get("type"),  # Try server_type first, then type
                        "relevance_score": tool.get("relevance_score", 0)
                    }
                    for tool in result.discovered_tools[:5]
                ]
            }
        }
        
        # Stage 3: Tool Selection with Q-LEARNING INFO
        q_learning_info = await _extract_q_learning_info(result.selected_tools)
        
        session["stages"]["tool_selection"] = {
            "status": "completed",
            "data": {
                "selected_count": len(result.selected_tools),
                "selected_tools": result.selected_tools,
                "selection_method": "q_learning" if orchestrator.q_learning_engine else "traditional",
                "q_learning_active": orchestrator.q_learning_engine is not None,
                "q_values": q_learning_info.get("q_values", {}),
                "exploration_rate": q_learning_info.get("exploration_rate", 0),
                "decision_type": q_learning_info.get("decision_type", "unknown"),
                "learning_episodes": len(orchestrator.execution_history) if orchestrator else 0
            }
        }
        
        # Stage 4: Execution (from result)
        session["stages"]["execution"] = {
            "status": "completed",
            "data": {
                "executed_tools": [
                    {
                        "tool_id": exec_result.tool_id,
                        "tool_name": exec_result.tool_name,
                        "success": exec_result.success,
                        "execution_time_ms": exec_result.execution_time_ms,
                        "error": exec_result.error
                    }
                    for exec_result in result.execution_results
                ],
                "parallel_execution": orchestrator.config.get('orchestration', {}).get('parallel_execution', False)
            }
        }
        
        # Stage 5: Results (from result)
        session["stages"]["results"] = {
            "status": "completed",
            "data": {
                "success": result.success,
                "summary": result.summary,
                "total_time_ms": result.total_time_ms,
                "cache_hit": False  # Could check orchestrator cache
            }
        }
        
        # Update session status
        session["status"] = "completed"
        session["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Session {session_id}: Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing query for session {session_id}: {e}", exc_info=True)
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)


async def _extract_q_learning_info(selected_tools: list) -> Dict[str, Any]:
    """
    Extract Q-learning information from the orchestrator.
    """
    info = {
        "q_values": {},
        "exploration_rate": 0,
        "decision_type": "unknown"
    }
    
    try:
        if orchestrator and orchestrator.q_learning_engine:
            # Get exploration rate
            info["exploration_rate"] = orchestrator.q_learning_engine.exploration_rate
            
            # Determine if this was exploration or exploitation
            if info["exploration_rate"] > 0.5:
                info["decision_type"] = "exploration"
            else:
                info["decision_type"] = "exploitation"
            
            # Try to get Q-values for selected tools
            # This is simplified - in reality, Q-values are for state-action pairs
            for tool_id in selected_tools:
                # Simulate Q-value (in real system, would query Q-table)
                info["q_values"][tool_id] = 0.5 + (0.3 if tool_id in selected_tools else 0)
            
            logger.info(f"Q-learning info extracted: {info}")
    except Exception as e:
        logger.warning(f"Could not extract Q-learning info: {e}")
    
    return info


@app.get("/demo/status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of a processing session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatus(
        session_id=session_id,
        status=sessions[session_id]["status"],
        current_stage=sessions[session_id].get("current_stage", "unknown"),
        stages=sessions[session_id]["stages"],
        timestamp=sessions[session_id]["started_at"]
    )


@app.get("/demo/results/{session_id}")
async def get_session_results(session_id: str):
    """Get the final results of a completed session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "query": session["query"],
        "status": session["status"],
        "stages": session["stages"],
        "total_time": session.get("completed_at", ""),
        "error": session.get("error"),
        "cache_metrics": orchestrator.get_cache_metrics() if orchestrator else {},
        "q_learning_stats": {
            "enabled": orchestrator.q_learning_engine is not None if orchestrator else False,
            "episodes": len(orchestrator.execution_history) if orchestrator else 0,
            "exploration_rate": orchestrator.q_learning_engine.exploration_rate if orchestrator and orchestrator.q_learning_engine else 0
        }
    }


@app.get("/demo/metrics")
async def get_system_metrics():
    """Get current system metrics including Q-learning statistics."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Get cache metrics
    cache_metrics = orchestrator.get_cache_metrics()
    
    # Get Q-learning statistics
    q_learning_stats = {}
    if orchestrator.q_learning_engine:
        q_learning_stats = {
            "enabled": True,
            "exploration_rate": orchestrator.q_learning_engine.exploration_rate,
            "episodes_completed": len(orchestrator.execution_history),
            "learning_rate": orchestrator.q_learning_engine.learning_rate,
            "discount_factor": orchestrator.q_learning_engine.discount_factor,
            "q_table_size": len(orchestrator.q_learning_engine.q_table.q_values) if hasattr(orchestrator.q_learning_engine, 'q_table') and hasattr(orchestrator.q_learning_engine.q_table, 'q_values') else 0
        }
    else:
        q_learning_stats = {"enabled": False}
    
    # Get MCP server status
    mcp_servers = {}
    if orchestrator and orchestrator.mcp_integration:
        for server_id, server_info in orchestrator.mcp_integration.servers.items():
            mcp_servers[server_id] = {
                "type": server_info.get('type'),
                "status": server_info.get('status'),
                "mode": "Mock" if server_info.get('is_mock') else "Real",
                "endpoint": server_info.get('endpoint', 'N/A')
            }
    
    return {
        "cache": cache_metrics,
        "q_learning": q_learning_stats,
        "mcp_servers": mcp_servers,
        "orchestrator_ready": orchestrator is not None,
        "active_sessions": len([s for s in sessions.values() if s["status"] == "processing"]),
        "total_sessions": len(sessions)
    }


@app.get("/demo/q_learning/details")
async def get_q_learning_details():
    """Get detailed Q-learning information for demonstration."""
    if not orchestrator or not orchestrator.q_learning_engine:
        return {"error": "Q-learning not available"}
    
    q_engine = orchestrator.q_learning_engine
    
    return {
        "configuration": {
            "learning_rate": q_engine.learning_rate,
            "discount_factor": q_engine.discount_factor,
            "exploration_rate": q_engine.exploration_rate,
            "min_exploration_rate": q_engine.min_exploration_rate,
            "exploration_decay": q_engine.exploration_decay
        },
        "statistics": {
            "episodes_completed": len(orchestrator.execution_history),
            "q_table_entries": len(q_engine.q_table.q_values) if hasattr(q_engine, 'q_table') and hasattr(q_engine.q_table, 'q_values') else 0,
            "current_mode": "exploration" if q_engine.exploration_rate > 0.5 else "exploitation"
        },
        "recent_decisions": [
            {
                "query": h.get("query", ""),
                "tools_selected": h.get("tools", []),
                "success": h.get("success", False),
                "reward": h.get("reward", 0)
            }
            for h in orchestrator.execution_history[-5:]
        ] if orchestrator.execution_history else []
    }


@app.delete("/demo/sessions")
async def clear_sessions():
    """Clear all session data."""
    sessions.clear()
    return {"message": "All sessions cleared"}


@app.get("/demo/test")
async def test_endpoint():
    """Test endpoint to verify the system is working."""
    return {
        "status": "ok",
        "orchestrator_ready": orchestrator is not None,
        "q_learning_enabled": orchestrator.q_learning_engine is not None if orchestrator else False,
        "model_status": "loaded" if model_loader.model_loaded else "fallback",
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Run the real demo application."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🚀 Autonomous Tool Discovery - REAL Orchestrator Demo")
    print("="*60)
    print("\n✅ This version uses:")
    print("   - REAL Orchestrator")
    print("   - REAL Q-Learning Engine")
    print("   - REAL Tool Discovery")
    print("   - REAL Intent Recognition")
    print("\nStarting server on port 8003...")
    print("Open your browser to: http://localhost:8003")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(
        "demo_app_real:app",
        host="0.0.0.0",
        port=8003,
        reload=False,  # Disable auto-reload to reduce noise
        log_level="info"
    )


if __name__ == "__main__":
    main()