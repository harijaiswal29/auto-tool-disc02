# Autonomous Tool Discovery and Integration Through Model Context Protocol (MCP)

## 🎓 M.Tech Dissertation Project

An AI system that enables autonomous agents to discover, learn, and optimize tool usage through reinforcement learning. This research demonstrates how Q-learning agents can achieve **50.33% task completion rate** with **7.5% improvement over baseline strategies**, validating the feasibility of autonomous tool discovery.

## 📊 Key Research Achievements

- Superior task completion rate achieved by Q-learning agents
- Improvement over baseline strategy average
- Better tool selection accuracy than random selection
- Statistical significance across all hypotheses
- **600** training episodes
- **476-dimensional** state vectors for comprehensive context representation

## 🏗️ System Architecture

The system implements a 5-layer architecture for autonomous tool discovery:

1. **Intent Recognition Layer** - Natural language understanding with sentence-transformers
2. **Tool Discovery Layer** - Graph-based exploration and capability matching
3. **Tool Selection & Learning Layer** - Q-learning with neural network approximation
4. **Execution & Monitoring Layer** - Asynchronous parallel execution with performance tracking
5. **Learning & Adaptation Layer** - Pattern mining and continuous improvement

## ⚙️ Prerequisites

### System Requirements
- **Python**: 3.8 or higher (tested with 3.12.3)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 16GB RAM for training with GPUs preferebbly 
- **Storage**: 2GB free space for models and data

### Required Software
- Git for version control
- SQLite (included with Python)
- Virtual environment support (venv or conda)

## 🚀 Quick Start Setup

### 1. Clone the Repository
```bash
git clone https://github.com/harijaiswal29/auto-tool-disc02.git
cd auto-tool-disc02
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Download sentence-transformer model (happens automatically on first run)
# Model: all-MiniLM-L6-v2 (~80MB)
```

### 4. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your API keys (optional for testing)
# The system works with mock servers if API keys are not provided
nano .env  # or use any text editor
```

### 5. Configure MCP Servers (Optional)
```bash
# Automatic setup of MCP servers based on available API keys
python scripts/setup_mcp_servers.py

# Test the configured servers
python scripts/setup_mcp_servers.py --test

# Force mock servers for testing
python scripts/setup_mcp_servers.py --use-mock
```

The system includes comprehensive configuration for tool-to-server mapping:
- **Automatic routing**: Tools are routed to appropriate MCP servers
- **Fallback support**: Gracefully falls back to mock servers when APIs unavailable
- **Configuration files**: `config/tool_server_mapping.json` and `config/mcp_servers_config.json`

### 6. Verify Installation
```bash
# Test the installation by importing key modules
python -c "import src.main; print('✓ Core modules installed')"
python -c "from sentence_transformers import SentenceTransformer; print('✓ Sentence transformers available')"
python -c "import asyncio, aiosqlite; print('✓ Async libraries installed')"
python -c "import networkx, sklearn; print('✓ ML libraries installed')"

# Check if configuration exists
python -c "import json; json.load(open('config/config.json')); print('✓ Configuration loaded')"
```

### Command Line Interface
```bash
# Run the main system (interactive mode)
python src/main.py
# Then type your query when prompted, e.g.: "Find weather information for New York"

## 📈 Training and Evaluation

### Training with State Vector Collection

#### Configuration Setup

1. **Update `config/config.json`** with the following evaluation settings:
```json
{
  "evaluation": {
    "save_state_vectors": true,
    "state_sampling_rate": 1,
    "max_states_per_checkpoint": 5000,
    "checkpoint_interval": 50,
    "state_collection_strategies": [
      "q_learning_tabular", "q_learning_dqn", "random",
      "popular", "fixed_policy", "greedy", "context_agnostic"
    ],
    "strategy_sampling_rates": {
      "q_learning_tabular": 1.0,    // Collect 100% for positive examples
      "q_learning_dqn": 1.0,         // Collect 100% for positive examples
      "random": 0.1,                 // Sample 10% for negative examples
      "popular": 0.1,                // Sample 10% for baseline
      "fixed_policy": 0.2,           // Sample 20% for comparison
      "greedy": 0.2,                 // Sample 20% for comparison
      "context_agnostic": 0.2,       // Sample 20% for comparison
      "others": 0.2                  // Default for unlisted strategies
    },
    "strategy_configs": {
      "random": {"requires_intent_processing": false, "embedding_mode": "mock"},
      "popular": {"requires_intent_processing": false, "embedding_mode": "mock"},
      "fixed_policy": {"requires_intent_processing": false, "embedding_mode": "mock"},
      "greedy": {"requires_intent_processing": false, "embedding_mode": "mock"},
      "context_agnostic": {"requires_intent_processing": false, "embedding_mode": "fast_real"},
      "q_learning_tabular": {"requires_intent_processing": true, "embedding_mode": "full_real"},
      "q_learning_dqn": {"requires_intent_processing": true, "embedding_mode": "full_real"}
    }
  }
}
```

2. **Update `tests/dissertation_test_suite/data/experiment_config.yaml`**:
```yaml
baseline_comparison:
  episodes: 600                 # Use 600 episodes full training
  checkpoint_interval: 50         # Save every 50 episodes
  runs_per_strategy: 1            # Single run for state collection
```

#### Running the Training

Execute the optimized baseline comparison with state vector collection:

```bash
# Full command with all parameters
python tests/dissertation_test_suite/scripts/run_baseline_comparison_optimized.py \
    --episodes 600 \
    --checkpoint-dir tests/state_vector_training_20ep \
    --checkpoint-interval 50 \
    --success-criteria strict \
    --use-graded-rewards \
    --use-real-servers \
    --query-set dissertation_core

# Parameters explained:
# --episodes 600: Number of training episodes (use 600+ for full training)
# --checkpoint-dir: Directory to save checkpoints with state vectors
# --checkpoint-interval 5: Save checkpoint every 50 episodes
# --success-criteria strict: All optimal tools must be selected for success
# --use-graded-rewards: Use sophisticated reward calculation
# --use-real-servers: Attempt to use real MCP servers (falls back to mock if unavailable)
# --query-set dissertation_core: Use balanced set of 25 core queries
```

#### Verifying State Vector Collection

After training, verify that state vectors are being collected:

```bash
# Check checkpoint files
ls -lh tests/state_vector_training_20ep/checkpoint_*.pkl

# Verify state vectors in a checkpoint
python -c "
import pickle
with open('tests/state_vector_training_20ep/checkpoint_q_learning_dqn_ep20_*.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
print(f'States collected: {len(checkpoint.get(\"episode_states\", []))}')
print(f'State dimensions: {checkpoint.get(\"state_dimensions\", 0)}')
"
```

#### Expected Output

- **Checkpoint Files**: One per strategy per checkpoint interval
  - Format: `checkpoint_{strategy}_ep{episode}_{timestamp}.pkl`
  - Size: ~350KB for Q-learning strategies (with 100 states)
  
- **State Vector Structure**: Each state contains:
  - `state_vector`: 476-dimensional numpy array
  - `intent_embedding`: 384-dimensional sentence transformer embedding
  - `tools_selected`: List of selected tools
  - `optimal_tools`: Ground truth tools
  - `success`: Boolean outcome
  - `reward`: Calculated reward value
  - `episode`, `query_idx`: Training position


## 🔧 Configuration

### Main Configuration File
Edit `config/config.json` to customize:

```json
{
  "learning": {
    "algorithm": "dqn",
    "parameters": {
      "learning_rate": 0.3,
      "discount_factor": 0.99,
      "epsilon": 0.5,
      "epsilon_decay": 0.995,
      "epsilon_min": 0.005
    }
  },
  "intent_recognition": {
    "model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "cache_size": 1000
  }
}
```

### MCP Server Configuration
The system includes advanced configuration for tool-to-server mapping:

**`config/tool_server_mapping.json`** - Maps tools to servers:
- Direct tool ID to server ID mappings
- Pattern-based matching for tool families
- Fallback strategies for handling failures

**`config/mcp_servers_config.json`** - Server specifications:
- Real server commands and requirements
- Mock server implementations
- Auto-initialization settings
- Priority and capability definitions

### Environment Variables
Required API keys (optional for testing with mock servers):
- `GITHUB_TOKEN` - GitHub API access
- `BRAVE_API_KEY` - Brave Search API
- `POSTGRES_CONNECTION_STRING` - PostgreSQL database
- `OPENWEATHER_API_KEY` - Weather data
- `FINANCIAL_DATASETS_API_KEY` - Financial data


## 📁 Project Structure

```
auto-tool-disc02/
├── src/                      # Source code
│   ├── agents/              # AI agent implementations
│   ├── core/                # Core MCP integration
│   ├── learning/            # Q-learning and DQN implementations
│   ├── pipeline/            # Processing pipeline stages
│   ├── tools/               # Tool implementations and mocks
│   └── web/                 # Web interface (FastAPI)
├── tests/                    # Test suites
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── dissertation_test_suite/  # Research validation
├── scripts/                  # Training and utility scripts
├── demos/                    # Demonstration scripts
├── config/                   # Configuration files
├── data/                     # Training data and models
├── dissertation_results/     # Research findings and analysis
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Ensure virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. Model Download Issues
```bash
# Manually download sentence-transformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### 3. Database Errors
```bash
# Reset database
rm -rf data/registry/tool_registry.db
# Database will be recreated automatically on next run
python src/main.py
```

#### 4. Memory Issues During Training
```bash
# Reduce batch size in config.json
# Or use lighter model:
python scripts/run_training_with_states.py --batch-size 16
```

#### 5. Port Already in Use (Web Interface)
```bash
# Kill the process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn src.web.demo_app:app --port 8001
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### Setup Guides
- [Zerodha MCP Setup](docs/setup/zerodha-mcp-setup.md) - Zerodha trading platform MCP setup
- [Notion Integration Setup](docs/setup/notion-mcp-setup.md) - Notion integration MCP setup
- [PostgreSQL Setup Guide](docs/setup/postgresql-setup-guide.md) - PostgreSQL database setup for MCP testing

### Architecture & Design
- [System Architecture](docs/architecture/system-architecture.md) - Component architecture and design principles
- [MCP Communication](docs/architecture/mcp-communication.md) - MCP protocol details and message formats
- [Workflows](docs/architecture/workflows.md) - Key system workflows and processes
- [Database Schema](docs/architecture/database-schema.md) - Complete database schema and tables
- [Retry Architecture](docs/architecture/retry-architecture.md) - Retry and resilience patterns
- [Result Caching](docs/architecture/result-caching.md) - Result caching architecture

### Implementation Details
- [Implementation Status](docs/implementation/implementation-status.md) - Detailed implementation tracking
- [Learning System](docs/implementation/learning-system.md) - Q-learning, rewards, pattern mining
- [Q-Learning Implementation](docs/implementation/q_learning_implementation.md) - Q-learning engine implementation
- [Deep Q-Learning](docs/implementation/deep-q-learning.md) - Deep Q-Learning with neural networks
- [Intent Recognition](docs/implementation/intent-recognition.md) - NLP pipeline and classification
- [Tool Discovery](docs/implementation/tool-discovery.md) - Discovery algorithms and caching
- [Execution Engine](docs/implementation/execution-engine.md) - Task management and monitoring
- [Learning System Updates](docs/implementation/learning-system-updates.md) - Summary of learning system enhancements
- [Advanced Reward Strategies](docs/implementation/advanced-reward-strategies.md) - Advanced reward calculation strategies

### API & Data Models
- [REST API](docs/api/rest-api.md) - RESTful endpoints and specifications
- [WebSocket API](docs/api/websocket-api.md) - Real-time communication protocols
- [Data Models](docs/api/data-models.md) - Core data models and schemas

### Project Management
- [Phase Completion](docs/project/phase-completion.md) - Development phases and accomplishments
- [Project Structure](docs/project/project-structure.md) - Complete directory structure

### Development
- [Commands Reference](docs/development/commands-reference.md) - Development commands and scripts

### Evaluation
- [Evaluation Targets](docs/evaluation/evaluation-targets.md) - Performance targets and baselines
- [Baseline Comparisons](docs/evaluation/baseline-comparisons.md) - Baseline strategy comparisons
- [A/B Testing Framework](docs/evaluation/ab-testing-framework.md) - A/B testing documentation
- [Strategy Details](docs/evaluation/strategy-details.md) - Detailed strategy implementation

### Testing & Deployment
- [Test Suite Documentation](tests/README.md) - Comprehensive test suite documentation and ALL test commands
- [Dissertation Test Suite](tests/dissertation_test_suite/) - Dissertation-focused test suite for validating research hypotheses
  - [Testing Strategy](tests/dissertation_test_suite/dissertation-testing-strategy.md) - Pragmatic testing strategy for dissertation goals
- [Test Summary](docs/testing/test-summary.md) - Test coverage summary and metrics
- [Coverage Summary](docs/testing/coverage_summary.md) - Detailed test coverage report
- [Deployment Requirements](docs/deployment/requirements.md) - Non-functional requirements and SLOs
- [Infrastructure](docs/deployment/infrastructure.md) - Container specs and CI/CD pipelines
- [Security](docs/deployment/security.md) - Security architecture and best practices
- [Deployment Configuration](docs/deployment/configuration.md) - Configuration guide for learning system

### Configuration
- [MCP Server Configuration Guide](docs/configuration/mcp-server-configuration-guide.md) - Tool-to-server mapping setup

### Monitoring
- [Cache Monitoring](docs/monitoring/cache-monitoring.md) - Cache monitoring and optimization

### Troubleshooting
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions

### Web Interface
- [Web Interface Guide](src/web/README.md) - FastAPI-based demonstration interface

### Dissertation Results
- [Dissertation Results](dissertation_results/DISSERTATION_RESULTS.md) - Comprehensive experimental results, analysis, and summary
- [Result Visualizations](dissertation_results/figures/) - Learning curves, performance metrics, and hypothesis validation
- [Visualization Generator](dissertation_results/generate_visualizations.py) - Script to regenerate result visualizations

### Demo Scripts
- [Demo Scripts README](demos/README.md) - Detailed documentation of all available demos



## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{autonomous_tool_discovery_2024,
  title={Autonomous Tool Discovery Through Model Context Protocol Using Reinforcement Learning},
  author={[Hari Jaiswal]},
  school={Birla Institute of Technology and Science},
  year={2025},
  type={M.Tech Dissertation}
}
```

## 📝 License

This project is part of an M.Tech dissertation at BITS Pilani. All rights reserved.

## 👤 Author

**[Your Name]**
- M.Tech Student, BITS Pilani
- Email: [2023aa05106@wilp.bits-pilani.ac.in]
- GitHub: [@harijaiswal29](https://github.com/harijaiswal29)
- LinkedIn: [Hari Jaiswal](https://www.linkedin.com/in/harijaiswal/)

## 🙏 Acknowledgments

- Dr. D Venkata Subramanian
- Mr. Mohan Singh Arora
- Anthropic for Model Context Protocol specification
- Open source community for tools and libraries

---

*This project demonstrates the feasibility of autonomous tool discovery through reinforcement learning, achieving statistically significant improvements over traditional approaches.*