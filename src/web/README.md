# Web Demonstration Interface

## Overview
This web interface provides a visual demonstration of the Autonomous Tool Discovery System's complete workflow, designed specifically for dissertation committee presentations.

## Features

### 🎯 Visual Workflow Display
- **5-Stage Pipeline Visualization**: Real-time display of each processing stage
  1. Intent Recognition - Shows detected intent, confidence, and keywords
  2. Tool Discovery - Displays discovered tools with relevance scores
  3. Tool Selection - Highlights Q-learning decisions and selected tools
  4. Execution - Progress tracking for tool execution
  5. Results - Final aggregated results with performance metrics

### 🤖 Q-Learning Visualization
- Display of Q-values for tool selection
- Exploration vs exploitation indicators
- Learning rate and episode tracking

### 📊 Performance Metrics Dashboard
- Cache performance (hit rate, total queries)
- Q-learning statistics (exploration rate, episodes)
- System performance (response time, active sessions)

### 🎨 Professional UI Design
- Clean, modern interface with gradient styling
- Color-coded stage status (pending, active, completed, error)
- Smooth animations for stage transitions
- Responsive design for different screen sizes

## Quick Start

### Installation
```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Launch the Demo
```bash
# Simple one-command launch
python launch_demo.py
```

Or manually:
```bash
# Navigate to web directory
cd src/web

# Start the server
uvicorn demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser to: http://localhost:8000

## Demo Scenarios

### 1. Basic Query Processing
- **Query**: "Find weather information for London"
- **Demonstrates**: Simple tool discovery and execution flow

### 2. Q-Learning in Action
- **Query**: "Search for machine learning papers"
- **Run twice** to show:
  - First run: Exploration behavior
  - Second run: Exploitation with learned preferences

### 3. Cache Performance
- **Query**: "List all database tables"
- **Run twice** to demonstrate:
  - First run: Full processing
  - Second run: Instant cache hit

### 4. Complex Multi-Tool Orchestration
- **Query**: "Find Python files in current directory"
- **Shows**: Parallel tool execution and result aggregation

## Presentation Tips

### 1. Pre-Demo Setup
- Clear all sessions using the "Clear All Sessions" button
- Ensure mock servers are running (handled automatically)
- Have sample queries ready

### 2. During Presentation
- Use sample query buttons for quick demonstrations
- Click on stages to explain each component
- Show raw data for technical committee members
- Monitor metrics dashboard for performance insights

### 3. Key Points to Highlight
- **Intent Recognition**: NLP pipeline with confidence scoring
- **Tool Discovery**: Graph-based exploration and capability matching
- **Q-Learning**: Adaptive tool selection improving over time
- **Parallel Execution**: Efficient multi-tool orchestration
- **Caching**: Performance optimization through intelligent caching

## API Endpoints

### Main Endpoints
- `GET /` - Main demo interface
- `POST /demo/process` - Submit query for processing
- `GET /demo/status/{session_id}` - Get processing status
- `GET /demo/results/{session_id}` - Get final results
- `GET /demo/metrics` - System metrics

### Utility Endpoints
- `DELETE /demo/sessions` - Clear all sessions
- `GET /static/*` - Static assets (CSS, JS)

## Troubleshooting

### Server Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use a different port
uvicorn demo_app:app --port 8001
```

### Dependencies Missing
```bash
# Install all requirements
pip install fastapi uvicorn[standard] jinja2
```

### No Tools Discovered
- Ensure mock servers are initialized
- Check orchestrator initialization in logs

## Architecture

```
src/web/
├── demo_app.py          # FastAPI application
├── static/
│   ├── demo.html        # Main UI
│   ├── demo.css         # Styling
│   └── demo.js          # Client-side logic
└── README.md            # This file
```

## Customization

### Adding New Sample Queries
Edit `demo.html` and add new buttons:
```html
<button class="sample-btn" onclick="setSampleQuery('Your query here')">
  Label
</button>
```

### Modifying Stage Display
Edit `demo.js` `formatStageContent()` functions to customize how each stage displays data.

### Changing Colors/Theme
Edit `demo.css` - primary colors are defined in the gradient backgrounds.

## Performance Considerations

- **Polling Interval**: Set to 500ms for smooth updates
- **Session Storage**: In-memory for demo (clears on restart)
- **Cache TTL**: 5 minutes for demo purposes
- **Max Tools**: Limited to 3 for clear visualization

## For Committee Presentation

### Estimated Demo Time: 10-15 minutes
1. **Introduction** (2 min): Explain the system architecture
2. **Basic Demo** (3 min): Show simple query processing
3. **Q-Learning Demo** (3 min): Demonstrate learning behavior
4. **Performance** (2 min): Show cache and metrics
5. **Complex Query** (3 min): Multi-tool orchestration
6. **Q&A** (2-5 min): Interactive exploration

### Key Metrics to Show
- Intent recognition < 100ms (performance target)
- Cache hit rate improvement over time
- Q-learning convergence (decreasing exploration)
- Tool selection accuracy

## Support

For issues or questions about the demo, refer to the main project documentation or contact the development team.