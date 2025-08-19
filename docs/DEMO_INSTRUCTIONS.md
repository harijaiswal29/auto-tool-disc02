# Web Demo Launch Instructions

## Quick Start (Manual)

Since you're in WSL and port 8000 was busy, here's how to run the demo manually:

### Option 1: Using the improved launcher (Recommended)
```bash
# This will find an available port automatically
python launch_demo.py
```

### Option 2: Direct uvicorn command
```bash
# Navigate to the web directory
cd src/web

# Run on port 8001 (or any available port)
python -m uvicorn demo_app:app --host 0.0.0.0 --port 8001 --reload

# Or try port 8080
python -m uvicorn demo_app:app --host 0.0.0.0 --port 8080 --reload
```

### Option 3: Using the shell script
```bash
# This will check for available ports
./run_demo.sh
```

## Accessing the Demo

After starting the server, open your **Windows browser** (not WSL) and navigate to:
- http://localhost:8001 (or whatever port is shown in the console)

## If Port is Still Busy

Check and kill existing processes:
```bash
# Check what's using the port
ps aux | grep uvicorn

# Kill the process (replace PID with actual process ID)
kill <PID>
```

## Features to Try

1. **Sample Queries** - Use the quick buttons:
   - "Find weather information for London"
   - "Search for machine learning papers"
   - "List all database tables"
   - "Find Python files in current directory"

2. **Watch the Workflow** - See each stage progress:
   - Intent Recognition (with confidence scores)
   - Tool Discovery (with relevance scores)
   - Tool Selection (Q-learning visualization)
   - Execution (parallel processing)
   - Results (with timing metrics)

3. **Metrics Dashboard** - Bottom of page shows:
   - Cache performance
   - Q-learning statistics
   - System performance

## Troubleshooting

### "Address already in use" error
- Another process is using the port
- Try a different port (8001, 8080, 8888, etc.)
- Or kill the existing process

### Page doesn't load
- Make sure you're using a Windows browser (not WSL terminal browser)
- Check the console for the actual port number being used
- Ensure the server is running (you should see log messages)

### No tools discovered
- This is normal! The demo includes mock tools for visualization
- The system will show demo tools even without real MCP servers

## For Your Dissertation Presentation

1. Start the server before your presentation
2. Test with a few queries to warm up the cache
3. Use the sample queries for consistent demonstrations
4. The visual workflow is the key feature - emphasize how each stage is visible
5. Run the same query twice to show Q-learning improvement