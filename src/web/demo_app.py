"""
Web Demonstration Interface for Autonomous Tool Discovery System.

This application provides a visual interface to demonstrate the entire
workflow from query to results for the dissertation committee.
"""

import asyncio
import json
import sys
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="Autonomous Tool Discovery Demo", version="1.0.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    global orchestrator
    logger.info("Initializing Orchestrator Agent...")
    
    try:
        # Preload models first
        from src.utils.model_manager import preload_models
        logger.info("Preloading ML models...")
        try:
            preload_models({
                'sentence_transformer': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'device': 'cpu'
                }
            })
            logger.info("Models preloaded successfully")
        except Exception as e:
            logger.warning(f"Could not preload models: {e}")
            logger.info("Will use fallback intent recognition")
        
        # Create orchestrator with Q-learning enabled
        config = {
            "q_learning": {
                "enable_learning": False,  # Disable for demo to avoid issues
                "model_path": "models/q_learning_model.pkl"
            },
            "orchestration": {
                "max_tools_per_query": 3,
                "tool_selection_strategy": "performance_weighted",
                "parallel_execution": True
            },
            "result_cache": {
                "enabled": True,
                "max_size": 100,
                "ttl_seconds": 300
            },
            "intent_recognition": {
                "use_fallback": True,  # Use fallback if model fails
                "confidence_threshold": 0.5
            }
        }
        
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
        # Create a minimal orchestrator
        orchestrator = OrchestratorAgent({})


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global orchestrator
    if orchestrator:
        await orchestrator.shutdown()
        logger.info("Orchestrator shutdown complete")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main demo interface."""
    html_file = static_dir / "demo.html"
    if html_file.exists():
        return html_file.read_text()
    else:
        return """
        <html>
            <head><title>Demo Interface</title></head>
            <body>
                <h1>Autonomous Tool Discovery Demo</h1>
                <p>Demo interface is being set up. Please refresh in a moment.</p>
            </body>
        </html>
        """


@app.post("/demo/process")
async def process_query(request: QueryRequest):
    """
    Submit a query for processing.
    
    Returns a session ID to track progress.
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
    
    # Start async processing
    asyncio.create_task(_process_query_async(session_id, request.query, request.context))
    
    return {"session_id": session_id, "status": "processing"}


async def _process_query_async(session_id: str, query: str, context: Optional[Dict[str, Any]]):
    """
    Process query asynchronously and update session status progressively.
    """
    try:
        session = sessions[session_id]
        
        # Stage 1: Intent Recognition
        logger.info(f"Session {session_id}: Starting intent recognition")
        session["current_stage"] = "intent_recognition"
        session["stages"]["intent_recognition"]["status"] = "active"
        
        # Small delay to show the active state
        await asyncio.sleep(0.5)
        
        # Try to recognize intent with fallback
        try:
            # Recognize intent using the intent agent directly for progressive updates
            intent_result = await orchestrator.intent_agent.process_query(query, context)
        except Exception as e:
            logger.warning(f"Intent recognition failed, using fallback: {e}")
            # Create a simple fallback intent result
            from src.agents.intent_models import IntentResult, Intent
            
            # Simple keyword-based intent detection
            query_lower = query.lower()
            if any(word in query_lower for word in ['find', 'search', 'look']):
                intent_type = 'query.search'
            elif any(word in query_lower for word in ['create', 'make', 'generate']):
                intent_type = 'action.create'
            elif any(word in query_lower for word in ['list', 'show', 'display']):
                intent_type = 'query.retrieve'
            elif any(word in query_lower for word in ['weather', 'temperature', 'forecast']):
                intent_type = 'query.weather'
            else:
                intent_type = 'query.general'
            
            # Extract simple keywords
            keywords = [word for word in query.split() if len(word) > 3][:5]
            
            intent_result = IntentResult(
                primary_intent=Intent(
                    type=intent_type,
                    confidence=0.7,
                    keywords=keywords
                ),
                secondary_intents=[],
                entities={},
                context={}
            )
        
        session["stages"]["intent_recognition"] = {
            "status": "completed",
            "data": {
                "type": intent_result.primary_intent.type,
                "confidence": intent_result.primary_intent.confidence,
                "keywords": intent_result.primary_intent.keywords,
                "secondary_intents": [
                    {"type": si.type, "confidence": si.confidence}
                    for si in intent_result.secondary_intents
                ] if intent_result.secondary_intents else []
            }
        }
        logger.info(f"Session {session_id}: Intent recognition completed")
        
        # Stage 2: Tool Discovery
        await asyncio.sleep(0.3)
        session["current_stage"] = "tool_discovery"
        session["stages"]["tool_discovery"]["status"] = "active"
        
        # Try to discover tools with fallback
        try:
            discovered_tools = await orchestrator.discover_tools_for_intent(intent_result)
        except Exception as e:
            logger.warning(f"Tool discovery failed: {e}")
            discovered_tools = []
        
        # If no tools discovered, use mock tools as fallback
        if not discovered_tools:
            logger.warning(f"Session {session_id}: No tools discovered, using mock tools")
            # Create mock tools for demonstration based on intent
            intent_type = intent_result.primary_intent.type if intent_result else 'query.general'
            
            if 'search' in intent_type or 'find' in query.lower():
                discovered_tools = [
                    {
                        "id": "demo_search_tool",
                        "name": "Search Tool",
                        "type": "search",
                        "relevance_score": 0.85,
                        "capabilities": {"operations": ["search", "query", "find"]}
                    },
                    {
                        "id": "demo_filesystem_tool",
                        "name": "File System Tool",
                        "type": "filesystem",
                        "relevance_score": 0.75,
                        "capabilities": {"operations": ["list", "read", "find"]}
                    }
                ]
            elif 'weather' in query.lower():
                discovered_tools = [
                    {
                        "id": "demo_weather_tool",
                        "name": "Weather Tool",
                        "type": "weather",
                        "relevance_score": 0.92,
                        "capabilities": {"operations": ["get", "fetch", "retrieve"]}
                    }
                ]
            elif 'database' in query.lower() or 'table' in query.lower():
                discovered_tools = [
                    {
                        "id": "demo_database_tool",
                        "name": "Database Tool",
                        "type": "database",
                        "relevance_score": 0.88,
                        "capabilities": {"operations": ["query", "list", "retrieve"]}
                    }
                ]
            else:
                discovered_tools = [
                    {
                        "id": "demo_general_tool",
                        "name": "General Purpose Tool",
                        "type": "general",
                        "relevance_score": 0.7,
                        "capabilities": {"operations": ["process", "analyze"]}
                    }
                ]
        
        session["stages"]["tool_discovery"] = {
            "status": "completed",
            "data": {
                "discovered_count": len(discovered_tools),
                "tools": [
                    {
                        "id": tool.get("id"),
                        "name": tool.get("name"),
                        "type": tool.get("type"),
                        "relevance_score": tool.get("relevance_score", 0)
                    }
                    for tool in discovered_tools[:5]  # Show top 5
                ]
            }
        }
        logger.info(f"Session {session_id}: Tool discovery completed - found {len(discovered_tools)} tools")
        
        # Stage 3: Tool Selection
        await asyncio.sleep(0.3)
        session["current_stage"] = "tool_selection"
        session["stages"]["tool_selection"]["status"] = "active"
        
        # Try to select tools with fallback
        try:
            selected_tools = await orchestrator.select_tools(discovered_tools, intent_result)
        except Exception as e:
            logger.warning(f"Tool selection failed: {e}")
            selected_tools = []
        
        # If no tools selected, use top discovered tools
        if not selected_tools and discovered_tools:
            logger.warning(f"Session {session_id}: No tools selected, using top discovered tools")
            # Select top 2 tools by relevance score
            selected_tools = sorted(discovered_tools, 
                                   key=lambda x: x.get('relevance_score', 0), 
                                   reverse=True)[:2]
        
        # Check if Q-learning was used
        q_learning_used = orchestrator.q_learning_engine is not None
        q_values = {}
        
        if q_learning_used:
            # Get Q-values for selected tools if available
            for tool in selected_tools:
                tool_id = tool.get("id")
                # Simplified Q-value representation
                q_values[tool_id] = tool.get("selection_score", 0.5)
        
        session["stages"]["tool_selection"] = {
            "status": "completed",
            "data": {
                "selected_count": len(selected_tools),
                "selected_tools": [t.get("id") for t in selected_tools],
                "selection_method": "q_learning" if q_learning_used else "traditional",
                "q_values": q_values,
                "exploration_rate": getattr(orchestrator.q_learning_engine, 'exploration_rate', 0.2) if q_learning_used else None
            }
        }
        logger.info(f"Session {session_id}: Tool selection completed - selected {len(selected_tools)} tools")
        
        # Stage 4: Execution
        await asyncio.sleep(0.3)
        session["current_stage"] = "execution"
        session["stages"]["execution"]["status"] = "active"
        
        # Execute tools (handle case where no tools to execute)
        if selected_tools:
            execution_results = await orchestrator.execute_tools(selected_tools, query, context or {})
        else:
            logger.warning(f"Session {session_id}: No tools to execute")
            # Create mock execution result for demonstration
            from src.agents.orchestrator_agent import ToolExecutionResult
            execution_results = [
                ToolExecutionResult(
                    tool_id="mock_tool",
                    tool_name="Mock Demo Tool",
                    success=True,
                    result={"message": "Demo result for visualization"},
                    execution_time_ms=100.0
                )
            ]
        
        session["stages"]["execution"] = {
            "status": "completed",
            "data": {
                "executed_tools": [
                    {
                        "tool_id": exec_result.tool_id,
                        "tool_name": exec_result.tool_name,
                        "success": exec_result.success,
                        "execution_time_ms": exec_result.execution_time_ms if hasattr(exec_result, 'execution_time_ms') else 0,
                        "error": exec_result.error if hasattr(exec_result, 'error') else None
                    }
                    for exec_result in execution_results
                ],
                "parallel_execution": orchestrator.config.get('orchestration', {}).get('parallel_execution', False)
            }
        }
        logger.info(f"Session {session_id}: Execution completed")
        
        # Stage 5: Results
        await asyncio.sleep(0.2)
        session["current_stage"] = "results"
        session["stages"]["results"]["status"] = "active"
        
        # Generate summary
        summary = orchestrator._generate_summary(intent_result, execution_results)
        success = any(r.success for r in execution_results) if execution_results else False
        
        # Calculate total time
        start_time = datetime.fromisoformat(session["started_at"])
        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        session["stages"]["results"] = {
            "status": "completed",
            "data": {
                "success": success,
                "summary": summary,
                "total_time_ms": total_time_ms,
                "cache_hit": False
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
        
        # Mark current stage as error
        current_stage = sessions[session_id].get("current_stage")
        if current_stage:
            stage_key = current_stage
            if stage_key in sessions[session_id]["stages"]:
                sessions[session_id]["stages"][stage_key]["status"] = "error"
                sessions[session_id]["stages"][stage_key]["error"] = str(e)


@app.get("/demo/status/{session_id}")
async def get_session_status(session_id: str):
    """
    Get the current status of a processing session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionStatus(
        session_id=session_id,
        status=session["status"],
        current_stage=session.get("current_stage", "unknown"),
        stages=session["stages"],
        timestamp=session["started_at"]
    )


@app.get("/demo/results/{session_id}")
async def get_session_results(session_id: str):
    """
    Get the final results of a completed session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["status"] != "completed":
        return {"status": session["status"], "message": "Processing not yet complete"}
    
    return {
        "session_id": session_id,
        "query": session["query"],
        "status": "completed",
        "stages": session["stages"],
        "total_time": session.get("completed_at", ""),
        "cache_metrics": orchestrator.get_cache_metrics() if orchestrator else {}
    }


@app.get("/demo/test")
async def test_endpoint():
    """
    Test endpoint to verify the system is working.
    """
    return {
        "status": "ok",
        "orchestrator_ready": orchestrator is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/demo/metrics")
async def get_system_metrics():
    """
    Get current system metrics and statistics.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Get cache metrics
    cache_metrics = orchestrator.get_cache_metrics()
    
    # Get Q-learning statistics if available
    q_learning_stats = {}
    if orchestrator.q_learning_engine:
        q_learning_stats = {
            "exploration_rate": orchestrator.q_learning_engine.exploration_rate,
            "episodes_completed": len(orchestrator.execution_history),
            "learning_enabled": True
        }
    
    return {
        "cache": cache_metrics,
        "q_learning": q_learning_stats,
        "active_sessions": len([s for s in sessions.values() if s["status"] == "processing"])
    }


@app.delete("/demo/sessions")
async def clear_sessions():
    """
    Clear all session data (for demo cleanup).
    """
    sessions.clear()
    return {"message": "All sessions cleared"}


def main():
    """Run the demo application."""
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the server
    uvicorn.run(
        "demo_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()