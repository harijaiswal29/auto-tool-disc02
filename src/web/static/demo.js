// Global variables
let currentSessionId = null;
let pollInterval = null;
let startTime = null;

// Stage mapping
const stages = ['intent_recognition', 'tool_discovery', 'tool_selection', 'execution', 'results'];
const stageNames = {
    'intent_recognition': 'intent',
    'tool_discovery': 'discovery',
    'tool_selection': 'selection',
    'execution': 'execution',
    'results': 'results'
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    refreshMetrics();
    // Set up enter key handler for query input
    document.getElementById('queryInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitQuery();
        }
    });
});

// Submit query for processing
async function submitQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a query');
        return;
    }
    
    // Disable submit button
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';
    
    // Reset stages
    resetStages();
    
    // Start timer
    startTime = Date.now();
    
    try {
        const response = await fetch('/demo/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        currentSessionId = data.session_id;
        
        // Start polling for status updates
        startPolling();
        
    } catch (error) {
        console.error('Error submitting query:', error);
        alert('Error submitting query. Please try again.');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Process Query';
    }
}

// Start polling for status updates
function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    pollInterval = setInterval(async () => {
        if (!currentSessionId) return;
        
        try {
            const response = await fetch(`/demo/status/${currentSessionId}`);
            const data = await response.json();
            
            updateWorkflowDisplay(data);
            
            // Stop polling if completed or error
            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(pollInterval);
                pollInterval = null;
                
                // Re-enable submit button
                const submitBtn = document.getElementById('submitBtn');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Process Query';
                
                // Update metrics
                refreshMetrics();
            }
            
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 500); // Poll every 500ms
}

// Update workflow display based on status
function updateWorkflowDisplay(data) {
    // Update each stage
    for (const [stageName, stageData] of Object.entries(data.stages)) {
        const displayName = stageNames[stageName];
        updateStage(displayName, stageData);
    }
    
    // Update total time if completed
    if (data.status === 'completed' && startTime) {
        const totalTime = Date.now() - startTime;
        document.getElementById('total-time').textContent = `${totalTime}ms`;
    }
}

// Update individual stage display
function updateStage(stageName, stageData) {
    const stageElement = document.getElementById(`stage-${stageName}`);
    const statusElement = document.getElementById(`status-${stageName}`);
    const contentElement = document.getElementById(`content-${stageName}`);
    
    // Update stage status class
    stageElement.className = `stage ${stageData.status}`;
    
    // Update status text
    statusElement.textContent = stageData.status.charAt(0).toUpperCase() + stageData.status.slice(1);
    statusElement.className = `stage-status ${stageData.status}`;
    
    // Update content based on stage and status
    if (stageData.status === 'pending') {
        contentElement.innerHTML = '<p class="waiting-text">Waiting to start...</p>';
    } else if (stageData.status === 'active') {
        contentElement.innerHTML = '<div class="loading"></div> Processing...';
    } else if (stageData.status === 'completed' && stageData.data) {
        contentElement.innerHTML = formatStageContent(stageName, stageData.data);
    } else if (stageData.status === 'error') {
        contentElement.innerHTML = '<p style="color: red;">Error occurred</p>';
    }
}

// Format stage content based on stage type
function formatStageContent(stageName, data) {
    let html = '';
    
    switch(stageName) {
        case 'intent':
            html = formatIntentContent(data);
            break;
        case 'discovery':
            html = formatDiscoveryContent(data);
            break;
        case 'selection':
            html = formatSelectionContent(data);
            break;
        case 'execution':
            html = formatExecutionContent(data);
            break;
        case 'results':
            html = formatResultsContent(data);
            break;
    }
    
    return html;
}

// Format intent recognition content
function formatIntentContent(data) {
    let html = '<div class="intent-info">';
    html += `<strong>Intent Type:</strong> ${data.type}`;
    html += `<span class="intent-confidence">${(data.confidence * 100).toFixed(1)}%</span><br>`;
    
    if (data.keywords && data.keywords.length > 0) {
        html += '<strong>Keywords:</strong> ';
        html += '<div class="keywords">';
        data.keywords.forEach(keyword => {
            html += `<span class="keyword">${keyword}</span>`;
        });
        html += '</div>';
    }
    
    if (data.secondary_intents && data.secondary_intents.length > 0) {
        html += '<br><strong>Secondary Intents:</strong><br>';
        data.secondary_intents.forEach(intent => {
            html += `<small>${intent.type} (${(intent.confidence * 100).toFixed(1)}%)</small><br>`;
        });
    }
    
    html += '</div>';
    return html;
}

// Format tool discovery content
function formatDiscoveryContent(data) {
    let html = `<strong>Discovered ${data.discovered_count} tools</strong><br><br>`;
    
    if (data.tools && data.tools.length > 0) {
        data.tools.forEach(tool => {
            html += '<div class="tool-card">';
            html += `<span class="tool-name">${tool.name}</span>`;
            html += `<span class="tool-score">${(tool.relevance_score * 100).toFixed(1)}%</span>`;
            html += `<br><small>Type: ${tool.type}</small>`;
            html += '</div>';
        });
    }
    
    return html;
}

// Format tool selection content
function formatSelectionContent(data) {
    let html = '<div class="q-learning-info">';
    html += `<strong>Selection Method:</strong> ${data.selection_method}<br>`;
    html += `<strong>Selected ${data.selected_count} tools</strong><br>`;
    
    if (data.exploration_rate !== null) {
        html += `<strong>Exploration Rate:</strong> ${(data.exploration_rate * 100).toFixed(1)}%<br>`;
    }
    
    if (data.selected_tools && data.selected_tools.length > 0) {
        html += '<br><strong>Selected Tools:</strong><br>';
        data.selected_tools.forEach(toolId => {
            html += `<div class="tool-card selected">`;
            html += `<span class="tool-name">${toolId}</span>`;
            if (data.q_values && data.q_values[toolId]) {
                const qValue = data.q_values[toolId];
                const barWidth = Math.max(10, qValue * 100);
                html += `<br>Q-value: ${qValue.toFixed(3)}`;
                html += `<span class="q-value-bar" style="width: ${barWidth}px"></span>`;
            }
            html += '</div>';
        });
    }
    
    html += '</div>';
    return html;
}

// Format execution content
function formatExecutionContent(data) {
    let html = '';
    
    if (data.parallel_execution) {
        html += '<strong>Parallel Execution Enabled</strong><br><br>';
    }
    
    if (data.executed_tools && data.executed_tools.length > 0) {
        data.executed_tools.forEach(tool => {
            html += '<div class="tool-card">';
            html += `<strong>${tool.tool_name}</strong><br>`;
            
            if (tool.success) {
                html += `<span style="color: green;">✓ Success</span>`;
            } else {
                html += `<span style="color: red;">✗ Failed</span>`;
                if (tool.error) {
                    html += `<br><small>Error: ${tool.error}</small>`;
                }
            }
            
            html += `<br><small>Time: ${tool.execution_time_ms.toFixed(2)}ms</small>`;
            html += '</div>';
        });
    }
    
    return html;
}

// Format results content
function formatResultsContent(data) {
    const className = data.success ? 'result-summary' : 'result-summary error';
    let html = `<div class="${className}">`;
    
    if (data.success) {
        html += '<strong style="color: green;">✓ Query Processed Successfully</strong><br><br>';
    } else {
        html += '<strong style="color: red;">✗ Query Processing Failed</strong><br><br>';
    }
    
    html += `<strong>Summary:</strong><br>${data.summary}<br>`;
    
    if (data.total_time_ms) {
        html += `<div class="execution-time">Total Time: ${data.total_time_ms.toFixed(2)}ms</div>`;
    }
    
    if (data.cache_hit) {
        html += '<br><span style="color: blue;">📦 Result served from cache</span>';
    }
    
    html += '</div>';
    
    // Add button to view raw data
    html += '<br><button onclick="showRawData()">View Raw Data</button>';
    
    return html;
}

// Reset all stages to pending
function resetStages() {
    stages.forEach(stage => {
        const displayName = stageNames[stage];
        const stageElement = document.getElementById(`stage-${displayName}`);
        const statusElement = document.getElementById(`status-${displayName}`);
        const contentElement = document.getElementById(`content-${displayName}`);
        
        stageElement.className = 'stage pending';
        statusElement.textContent = 'Pending';
        statusElement.className = 'stage-status';
        contentElement.innerHTML = '<p class="waiting-text">Waiting to start...</p>';
    });
}

// Set sample query
function setSampleQuery(query) {
    document.getElementById('queryInput').value = query;
}

// Refresh metrics
async function refreshMetrics() {
    try {
        const response = await fetch('/demo/metrics');
        const data = await response.json();
        
        // Update cache metrics
        if (data.cache) {
            document.getElementById('cache-hit-rate').textContent = 
                data.cache.hit_rate ? `${(data.cache.hit_rate * 100).toFixed(1)}%` : '0%';
            document.getElementById('cache-total').textContent = 
                data.cache.total_queries || '0';
        }
        
        // Update Q-learning metrics
        if (data.q_learning) {
            document.getElementById('exploration-rate').textContent = 
                data.q_learning.exploration_rate ? 
                `${(data.q_learning.exploration_rate * 100).toFixed(1)}%` : 'N/A';
            document.getElementById('episodes-count').textContent = 
                data.q_learning.episodes_completed || '0';
        }
        
        // Update active sessions
        document.getElementById('active-sessions').textContent = 
            data.active_sessions || '0';
        
    } catch (error) {
        console.error('Error refreshing metrics:', error);
    }
}

// Clear all sessions
async function clearSessions() {
    if (!confirm('Clear all session data?')) return;
    
    try {
        await fetch('/demo/sessions', { method: 'DELETE' });
        alert('All sessions cleared');
        resetStages();
        currentSessionId = null;
    } catch (error) {
        console.error('Error clearing sessions:', error);
    }
}

// Show raw data
async function showRawData() {
    if (!currentSessionId) return;
    
    try {
        const response = await fetch(`/demo/results/${currentSessionId}`);
        const data = await response.json();
        
        document.getElementById('rawDataContent').textContent = 
            JSON.stringify(data, null, 2);
        document.getElementById('rawDataSection').style.display = 'block';
        
    } catch (error) {
        console.error('Error fetching raw data:', error);
    }
}

// Hide raw data
function hideRawData() {
    document.getElementById('rawDataSection').style.display = 'none';
}

// Auto-refresh metrics every 5 seconds
setInterval(refreshMetrics, 5000);