"""
Simplified Web Demo for Autonomous Tool Discovery System.
This version works with minimal dependencies and uses fallbacks.
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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.intent_models import IntentResult, Intent
from src.agents.orchestrator_agent import ToolExecutionResult
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="Autonomous Tool Discovery Demo (Simplified)", version="1.0.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Session storage
sessions: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    """Request model for query submission."""
    query: str
    context: Optional[Dict[str, Any]] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main demo interface."""
    html_file = static_dir / "demo.html"
    if html_file.exists():
        return html_file.read_text()
    else:
        return "<h1>Demo Interface</h1><p>Loading...</p>"


@app.post("/demo/process")
async def process_query(request: QueryRequest):
    """Submit a query for processing with simplified logic."""
    session_id = str(uuid.uuid4())
    
    # Initialize session
    sessions[session_id] = {
        "query": request.query,
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "stages": {
            "intent_recognition": {"status": "pending", "data": None},
            "tool_discovery": {"status": "pending", "data": None},
            "tool_selection": {"status": "pending", "data": None},
            "execution": {"status": "pending", "data": None},
            "results": {"status": "pending", "data": None}
        }
    }
    
    # Start async processing
    asyncio.create_task(_process_query_simple(session_id, request.query))
    
    return {"session_id": session_id, "status": "processing"}


async def _process_query_simple(session_id: str, query: str):
    """Simplified query processing for demonstration."""
    try:
        session = sessions[session_id]
        query_lower = query.lower()
        
        # Stage 1: Intent Recognition (simplified)
        await asyncio.sleep(0.5)
        session["stages"]["intent_recognition"]["status"] = "active"
        await asyncio.sleep(0.8)
        
        # Determine intent based on keywords
        if any(word in query_lower for word in ['find', 'search', 'look']):
            intent_type = 'query.search'
        elif any(word in query_lower for word in ['weather', 'temperature']):
            intent_type = 'query.weather'
        elif any(word in query_lower for word in ['database', 'table', 'sql']):
            intent_type = 'query.database'
        elif any(word in query_lower for word in ['file', 'directory', 'folder']):
            intent_type = 'query.filesystem'
        else:
            intent_type = 'query.general'
        
        keywords = [word for word in query.split() if len(word) > 3][:5]
        
        session["stages"]["intent_recognition"] = {
            "status": "completed",
            "data": {
                "type": intent_type,
                "confidence": 0.85,
                "keywords": keywords
            }
        }
        
        # Stage 2: Tool Discovery
        await asyncio.sleep(0.5)
        session["stages"]["tool_discovery"]["status"] = "active"
        await asyncio.sleep(0.8)
        
        # Discover tools based on intent
        if 'search' in intent_type:
            tools = [
                {"id": "search_tool", "name": "Web Search Tool", "type": "search", "relevance_score": 0.9},
                {"id": "file_tool", "name": "File Search Tool", "type": "filesystem", "relevance_score": 0.7}
            ]
        elif 'weather' in intent_type:
            tools = [
                {"id": "weather_tool", "name": "Weather API Tool", "type": "weather", "relevance_score": 0.95},
                {"id": "forecast_tool", "name": "Forecast Tool", "type": "weather", "relevance_score": 0.8}
            ]
        elif 'database' in intent_type:
            tools = [
                {"id": "sql_tool", "name": "SQL Query Tool", "type": "database", "relevance_score": 0.92},
                {"id": "db_tool", "name": "Database Manager", "type": "database", "relevance_score": 0.85}
            ]
        elif 'filesystem' in intent_type:
            tools = [
                {"id": "fs_tool", "name": "File System Tool", "type": "filesystem", "relevance_score": 0.88},
                {"id": "dir_tool", "name": "Directory Scanner", "type": "filesystem", "relevance_score": 0.75}
            ]
        else:
            tools = [
                {"id": "general_tool", "name": "General Purpose Tool", "type": "general", "relevance_score": 0.7}
            ]
        
        session["stages"]["tool_discovery"] = {
            "status": "completed",
            "data": {
                "discovered_count": len(tools),
                "tools": tools
            }
        }
        
        # Stage 3: Tool Selection
        await asyncio.sleep(0.5)
        session["stages"]["tool_selection"]["status"] = "active"
        await asyncio.sleep(0.8)
        
        # Select top tools
        selected = tools[:2] if len(tools) > 1 else tools
        
        session["stages"]["tool_selection"] = {
            "status": "completed",
            "data": {
                "selected_count": len(selected),
                "selected_tools": [t["id"] for t in selected],
                "selection_method": "q_learning",
                "q_values": {t["id"]: t["relevance_score"] for t in selected},
                "exploration_rate": 0.15
            }
        }
        
        # Stage 4: Execution
        await asyncio.sleep(0.5)
        session["stages"]["execution"]["status"] = "active"
        await asyncio.sleep(1.2)
        
        session["stages"]["execution"] = {
            "status": "completed",
            "data": {
                "executed_tools": [
                    {
                        "tool_id": tool["id"],
                        "tool_name": tool["name"],
                        "success": True,
                        "execution_time_ms": 150 + (i * 50),
                        "error": None
                    }
                    for i, tool in enumerate(selected)
                ],
                "parallel_execution": len(selected) > 1
            }
        }
        
        # Stage 5: Results
        await asyncio.sleep(0.3)
        session["stages"]["results"]["status"] = "active"
        await asyncio.sleep(0.5)
        
        # Generate result based on query type
        if 'weather' in query_lower:
            summary = "Weather information retrieved: Sunny, 22°C with light clouds"
        elif 'file' in query_lower or 'python' in query_lower:
            summary = "Found 42 Python files in the project directory"
        elif 'database' in query_lower:
            summary = "Listed 5 database tables: users, tools, executions, metrics, cache"
        elif 'search' in query_lower:
            summary = "Search completed: Found 10 relevant results"
        else:
            summary = f"Query processed successfully using {len(selected)} tools"
        
        total_time = (datetime.now() - datetime.fromisoformat(session["started_at"])).total_seconds() * 1000
        
        session["stages"]["results"] = {
            "status": "completed",
            "data": {
                "success": True,
                "summary": summary,
                "total_time_ms": total_time,
                "cache_hit": False
            }
        }
        
        session["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Error in simplified processing: {e}")
        session["status"] = "error"
        session["error"] = str(e)


@app.get("/demo/status/{session_id}")
async def get_session_status(session_id: str):
    """Get session status."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]


@app.get("/demo/metrics")
async def get_metrics():
    """Get simplified metrics."""
    return {
        "cache": {
            "hit_rate": 0.35,
            "total_queries": 42
        },
        "q_learning": {
            "exploration_rate": 0.15,
            "episodes_completed": 127,
            "learning_enabled": True
        },
        "active_sessions": len([s for s in sessions.values() if s["status"] == "processing"])
    }


@app.delete("/demo/sessions")
async def clear_sessions():
    """Clear all sessions."""
    sessions.clear()
    return {"message": "Sessions cleared"}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Simplified Demo Server")
    print("="*60)
    print("\nThis simplified version works without ML models")
    print("Perfect for demonstration purposes!")
    print("\nStarting server at http://localhost:8002")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")