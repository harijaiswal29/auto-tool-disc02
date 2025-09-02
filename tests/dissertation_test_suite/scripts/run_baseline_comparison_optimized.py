#!/usr/bin/env python3
"""
Optimized Baseline Comparison Script with Performance Improvements.

This version includes:
- Smart state vector collection (only for relevant strategies)
- Intent caching to avoid redundant processing
- Shared model initialization
- Batch embedding processing
"""

import os
# DISABLE CUDA to prevent errors in WSL2
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import json
import pickle
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import numpy as np
import yaml
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.evaluation.evaluation_engine import EvaluationEngine
from src.utils.logger import get_logger
from tests.dissertation_test_suite.data.test_queries import get_all_queries, get_evaluation_sets
from src.learning.tool_optimized_reward_calculator import (
    ToolOptimizedRewardCalculator, ExecutionMetrics
)

logger = get_logger(__name__)


@dataclass
class TestQuery:
    """Test query with optimal tool annotations."""
    query: str
    intent_type: str
    optimal_tools: List[str]
    category: str
    expected_slots: Dict[str, Any] = None


class IntentCacheManager:
    """Manages caching of intent processing results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def _get_key(self, query: str, context: Dict) -> str:
        """Generate cache key from query and context."""
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
        
        try:
            # Sanitize context to ensure it's serializable
            sanitized_context = {}
            for k, v in context.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    sanitized_context[k] = v
                else:
                    sanitized_context[k] = str(v)
            context_str = json.dumps(sanitized_context, sort_keys=True)
        except (TypeError, ValueError) as e:
            logger.debug(f"Context serialization failed: {e}")
            # If context can't be serialized, use a simple string
            context_str = str(context)
        combined = f"{query}:{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context: Dict) -> Optional[Tuple[Any, np.ndarray]]:
        """Get cached intent result and embedding."""
        key = self._get_key(query, context)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, query: str, context: Dict, intent_result: Any, embedding: np.ndarray):
        """Store intent result and embedding in cache."""
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._get_key(query, context)
        self.cache[key] = (intent_result, embedding)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


class CheckpointManager:
    """Manages experiment checkpoints for resumable runs."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, state: Dict[str, Any], strategy_name: str, episode: int):
        """Save checkpoint state."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{strategy_name}_ep{episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Backup previous checkpoint if exists
        existing_checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_{strategy_name}_*.pkl"))
        if existing_checkpoints:
            latest = max(existing_checkpoints, key=lambda p: p.stat().st_mtime)
            backup_file = latest.with_suffix('.pkl.bak')
            latest.rename(backup_file)
            logger.info(f"Backed up previous checkpoint to {backup_file.name}")
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Checkpoint saved to {checkpoint_file.name} (episode {episode})")
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load checkpoint state."""
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path.name}")
        logger.info(f"  Resume from episode: {state.get('episode', 0)}")
        logger.info(f"  Strategy: {state.get('strategy_name', 'unknown')}")
        
        return state


class OptimizedBaselineComparisonRunner:
    """Optimized orchestrator for baseline comparison experiments."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent.parent / "data" / "experiment_config.yaml"
        self.load_config()
        self.results_dir = Path(__file__).parent.parent / "results" / "raw_data"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluation_engine = None
        self.orchestrator = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_retries = False
        
        # Checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.resume_state: Optional[Dict[str, Any]] = None
        
        # Optimization: Shared models and caching
        self.intent_agent = None
        self._sentence_model = None
        self.intent_cache = IntentCacheManager()
        
        # Load project config for optimization settings
        project_config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.json"
        with open(project_config_path) as f:
            self.project_config = json.load(f)
        
        # Optimization settings from config
        eval_config = self.project_config.get('evaluation', {})
        self.save_state_vectors = eval_config.get('save_state_vectors', False)
        self.state_collection_strategies = eval_config.get('state_collection_strategies', 
                                                          ['q_learning_tabular', 'q_learning_dqn', 'context_agnostic'])
        self.use_intent_cache = eval_config.get('use_intent_cache', True)
        self.batch_embeddings = eval_config.get('batch_embeddings', True)
        
        # Initialize reward calculator
        self.reward_calculator = ToolOptimizedRewardCalculator(self.project_config)
        self.use_graded_rewards = True  # Will be configurable via command line
        self.success_criteria = 'strict'  # Will be configurable via command line
        self.use_real_servers = False  # Will be configurable via command line
        
        # Get sampling rates from config
        self.strategy_sampling_rates = eval_config.get('strategy_sampling_rates', {
            'q_learning_tabular': 1.0,
            'q_learning_dqn': 1.0,
            'random': 0.1,
            'popular': 0.1,
            'fixed_policy': 0.2,
            'greedy': 0.2,
            'context_agnostic': 0.2,
            'others': 0.2
        })
        
        # Load strategy-specific configurations
        self.strategy_configs = eval_config.get('strategy_configs', {})
        self.default_embedding_mode = eval_config.get('default_embedding_mode', 'mock')
        self.embedding_mode_override = eval_config.get('embedding_mode_override', None)
        
        # Validate strategy configurations
        self.validate_strategy_config()
        
    def load_config(self):
        """Load experiment configuration."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        self.exp_config = self.config['experiments']['baseline_comparison']
    
    def validate_strategy_config(self):
        """Validate strategy configurations and warn about potential conflicts."""
        logger.info("Validating strategy configurations...")
        
        warnings_found = []
        
        for strategy_name, config in self.strategy_configs.items():
            requires_intent = config.get('requires_intent_processing', True)
            embedding_mode = config.get('embedding_mode', self.default_embedding_mode)
            
            # Override check
            if self.embedding_mode_override:
                embedding_mode = self.embedding_mode_override
            
            # Conflict 1: full_real embedding without intent processing
            if embedding_mode == 'full_real' and not requires_intent:
                warning_msg = (f"Strategy '{strategy_name}': Using 'full_real' embeddings without intent processing "
                             "is inefficient. Consider using 'fast_real' for real embeddings without full intent pipeline.")
                warnings_found.append(warning_msg)
                logger.warning(warning_msg)
            
            # Conflict 2: mock embedding with intent processing
            elif embedding_mode == 'mock' and requires_intent:
                warning_msg = (f"Strategy '{strategy_name}': Using 'mock' embeddings with intent processing "
                             "wastes computation. Consider using 'full_real' for complete processing or "
                             "disable intent processing.")
                warnings_found.append(warning_msg)
                logger.warning(warning_msg)
            
            # Conflict 3: State collection with mock embeddings
            if strategy_name in self.state_collection_strategies and embedding_mode == 'mock':
                warning_msg = (f"Strategy '{strategy_name}': Collecting state vectors with 'mock' embeddings "
                             "may reduce encoder training quality. Consider 'fast_real' or 'full_real' for "
                             "meaningful state vectors.")
                warnings_found.append(warning_msg)
                logger.warning(warning_msg)
            
            # Info: Optimal configurations
            if not requires_intent and embedding_mode == 'mock':
                logger.debug(f"Strategy '{strategy_name}': Optimal for speed (no intent, mock embeddings)")
            elif requires_intent and embedding_mode == 'full_real':
                logger.debug(f"Strategy '{strategy_name}': Optimal for quality (full intent and embeddings)")
            elif not requires_intent and embedding_mode == 'fast_real':
                logger.debug(f"Strategy '{strategy_name}': Balanced approach (real embeddings, no intent)")
        
        # Summary
        if warnings_found:
            logger.info(f"Configuration validation completed with {len(warnings_found)} warning(s)")
        else:
            logger.info("Configuration validation completed successfully (no conflicts found)")
        
        return warnings_found
    
    def requires_intent_processing(self, strategy_name: str) -> bool:
        """Check if a strategy requires intent processing."""
        # First check strategy-specific config
        if strategy_name in self.strategy_configs:
            return self.strategy_configs[strategy_name].get('requires_intent_processing', True)
        # Fallback to checking if it's in state_collection_strategies
        return strategy_name in self.state_collection_strategies
    
    def get_embedding_mode(self, strategy_name: str) -> str:
        """Get embedding mode for a strategy.
        
        Returns one of: 'mock', 'fast_real', 'full_real'
        
        Priority order:
        1. Command line override (if set)
        2. Strategy-specific config
        3. Default embedding mode
        """
        # Check for global override first
        if self.embedding_mode_override:
            return self.embedding_mode_override
        
        # Check strategy-specific config
        if strategy_name in self.strategy_configs:
            return self.strategy_configs[strategy_name].get('embedding_mode', self.default_embedding_mode)
        
        # Use default
        return self.default_embedding_mode
    
    def _initialize_shared_models(self):
        """Initialize shared models for intent processing (called once)."""
        # Get ALL strategies that will be evaluated, not just state_collection_strategies
        all_strategy_names = [s['name'] for s in self.exp_config.get('strategies', [])]
        logger.info(f"Checking initialization needs for all {len(all_strategy_names)} strategies: {all_strategy_names}")
        
        # Check if any strategy actually needs intent processing
        strategies_needing_intent = [s for s in all_strategy_names 
                                     if self.requires_intent_processing(s)]
        
        # Check if any strategy needs sentence transformer (fast_real or full_real)
        strategies_needing_embeddings = [s for s in all_strategy_names 
                                         if self.get_embedding_mode(s) in ['fast_real', 'full_real']]
        
        # Log what we found
        if strategies_needing_embeddings:
            logger.info(f"Strategies requiring real embeddings: {strategies_needing_embeddings}")
            for strategy in strategies_needing_embeddings:
                mode = self.get_embedding_mode(strategy)
                logger.info(f"  - {strategy}: embedding_mode={mode}")
        else:
            logger.info("No strategies require real embeddings")
        
        # Initialize intent agent if needed (for any strategy, not just state collection)
        if strategies_needing_intent and self.intent_agent is None:
            from src.agents.intent_recognition_agent import IntentRecognitionAgent
            self.intent_agent = IntentRecognitionAgent()
            logger.info(f"Initialized shared IntentRecognitionAgent for {len(strategies_needing_intent)} strategies")
        
        # Initialize sentence transformer if ANY strategy needs real embeddings
        if strategies_needing_embeddings and self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
                self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info(f"✓ Successfully initialized SentenceTransformer for {len(strategies_needing_embeddings)} strategies")
                
                # Verify model works by encoding a test sentence
                test_embedding = self._sentence_model.encode("test", convert_to_numpy=True)
                logger.info(f"  Model verification: embedding shape={test_embedding.shape}, dtype={test_embedding.dtype}")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer: {e}")
                logger.warning("Strategies will fall back to mock embeddings!")
                self._sentence_model = None
    
    async def _process_intent_with_cache(self, query: str, context: Dict) -> Tuple[Any, np.ndarray]:
        """Process intent with caching to avoid redundant computation."""
        # Check cache first
        if self.use_intent_cache:
            cached = self.intent_cache.get(query, context)
            if cached is not None:
                return cached
        
        # Process intent
        intent_result = await self.intent_agent.process_query(query, context)
        
        # Get or generate embedding
        embedding = intent_result.metadata.get('embedding')
        if embedding is None:
            if self._sentence_model is not None:
                embedding = self._sentence_model.encode(query, convert_to_numpy=True)
            else:
                # Fallback to mock embedding if sentence model not available
                import hashlib
                query_hash = hashlib.md5(query.encode()).hexdigest()
                seed = int(query_hash[:8], 16)
                rng = np.random.RandomState(seed)
                embedding = rng.randn(384).astype(np.float32) * 0.1
            intent_result.metadata['embedding'] = embedding
        
        # Cache the result
        if self.use_intent_cache:
            self.intent_cache.put(query, context, intent_result, embedding)
        
        return intent_result, embedding
    
    async def _batch_process_embeddings(self, queries: List[str]) -> Dict[str, np.ndarray]:
        """Batch process embeddings for multiple queries."""
        if not self.batch_embeddings or not queries:
            return {}
        
        # Check if sentence model is available
        if self._sentence_model is None:
            logger.warning("Sentence model not initialized for batch processing, returning empty map")
            return {}
        
        # Ensure queries are strings, not objects
        if queries and not isinstance(queries[0], str):
            logger.warning(f"Expected strings in queries, got {type(queries[0])}")
            # Try to extract string from objects
            if hasattr(queries[0], 'query'):
                queries = [q.query if hasattr(q, 'query') else str(q) for q in queries]
            else:
                queries = [str(q) for q in queries]
        
        # Get unique queries (now guaranteed to be strings)
        unique_queries = list(set(queries))
        
        # Batch encode
        embeddings = self._sentence_model.encode(unique_queries, convert_to_numpy=True, 
                                                 show_progress_bar=False)
        
        # Create mapping
        embedding_map = {query: embedding for query, embedding in zip(unique_queries, embeddings)}
        return embedding_map
    
    async def initialize_components(self):
        """Initialize evaluation engine and orchestrator."""
        # Configure retries
        project_config = self._configure_retries(self.project_config)
        
        # Map strategies to evaluation baselines
        strategies = self.exp_config.get('strategies', [])
        strategy_names = [s['name'] for s in strategies]
        project_config['evaluation'] = project_config.get('evaluation', {})
        project_config['evaluation']['baselines'] = strategy_names
        
        # Initialize evaluation engine
        self.evaluation_engine = EvaluationEngine(project_config)
        
        # Register Q-learning strategies if needed
        from src.evaluation.dqn_strategy import QLearningTabularStrategy, QLearningDQNStrategy
        
        if 'q_learning_tabular' in strategy_names and 'q_learning_tabular' not in self.evaluation_engine.strategies:
            self.evaluation_engine.strategies['q_learning_tabular'] = QLearningTabularStrategy(project_config)
            
        if 'q_learning_dqn' in strategy_names and 'q_learning_dqn' not in self.evaluation_engine.strategies:
            self.evaluation_engine.strategies['q_learning_dqn'] = QLearningDQNStrategy(project_config)
        
        logger.info(f"Initialized strategies: {list(self.evaluation_engine.strategies.keys())}")
        
        # Get available tools
        self.available_tools = list(project_config.get('tools', {}).keys())
        
        # Create tool mapping
        self.tool_mapping = {
            'filesystem_mcp': 'filesystem',
            'search_mcp': 'search',
            'sqlite_mcp': 'database',
            'postgres_mcp': 'database',
            'github_mcp': 'github',
            'weather_mcp': 'search',
            'system_mcp': 'filesystem',
            'financial_mcp': 'financial',
            'notion_mcp': 'notion'
        }
        
        # Only initialize orchestrator if needed for Q-learning strategies
        strategies_needing_orchestrator = ['q_learning_tabular', 'q_learning_dqn']
        if any(s in strategy_names for s in strategies_needing_orchestrator):
            self.orchestrator = OrchestratorAgent(project_config)
            await self.orchestrator.initialize()
            
            # Initialize mock servers and tools
            await self._initialize_mock_servers()
            await self._initialize_mock_tools()
        else:
            logger.info("Skipping orchestrator initialization (not needed for selected strategies)")
            self.orchestrator = None
        
        # Initialize shared models if needed
        self._initialize_shared_models()
    
    def _configure_retries(self, config: Dict) -> Dict:
        """Configure retry settings based on experiment needs."""
        if not self.enable_retries:
            for tool_config in config.get('tools', {}).values():
                if 'retry_config' in tool_config:
                    tool_config['retry_config']['max_attempts'] = 1
        return config
    
    async def _initialize_mock_servers(self):
        """Initialize MCP servers (real or mock based on configuration)."""
        if not self.orchestrator:
            return
            
        mcp = self.orchestrator.mcp_integration
        
        if self.use_real_servers:
            logger.info("Initializing real MCP servers where available...")
            import os
            from dotenv import load_dotenv
            
            # Load environment variables from .env file
            load_dotenv()
            
            # Filesystem - use real server (safe with read-only operations)
            logger.info("Using real Filesystem server")
            await mcp.add_filesystem_server(use_mock=False, server_id="filesystem_real")
            
            # Search - use real if API key available
            if os.getenv('BRAVE_API_KEY'):
                logger.info("Using real Brave Search server")
                await mcp.add_search_server(use_mock=False, server_id="search_real")
            else:
                logger.info("Brave API key not found, using mock search server")
                await mcp.add_search_server(use_mock=True, server_id="search_mock")
            
            # PostgreSQL - use real if connection string available
            postgres_conn = os.getenv('POSTGRES_CONNECTION_STRING')
            if postgres_conn:
                logger.info("Using real PostgreSQL server")
                await mcp.add_postgres_server(
                    connection_string=postgres_conn,
                    use_mock=False, 
                    server_id="postgres_real"
                )
            else:
                logger.info("PostgreSQL connection not found, using mock database")
                await mcp.add_sqlite_server(db_path=":memory:", use_mock=True, server_id="database_mock")
            
            # GitHub - use real if token available
            if os.getenv('GITHUB_TOKEN'):
                logger.info("Using real GitHub server")
                await mcp.add_github_server(use_mock=False, server_id="github_real")
            else:
                logger.info("GitHub token not found, using mock GitHub server")
                await mcp.add_github_server(use_mock=True, server_id="github_mock")
            
            # Financial - use mock (no real API key in environment)
            await mcp.add_financial_datasets_server(use_mock=True, server_id="financial_mock")
            
            logger.info("MCP servers initialized (mix of real and mock)")
        else:
            logger.info("Initializing mock MCP servers...")
            await mcp.add_filesystem_server(use_mock=True, server_id="filesystem_mock")
            await mcp.add_search_server(use_mock=True, server_id="search_mock")
            await mcp.add_sqlite_server(db_path=":memory:", use_mock=True, server_id="database_mock")
            await mcp.add_github_server(use_mock=True, server_id="github_mock")
            await mcp.add_financial_datasets_server(use_mock=True, server_id="financial_mock")
            
            logger.info("Mock MCP servers initialized successfully")
    
    async def _initialize_mock_tools(self):
        """Initialize mock tools in registry."""
        if not self.orchestrator:
            return
        registry = self.orchestrator.mcp_integration.registry
        
        mock_tools = [
            {
                "id": "filesystem",
                "name": "Filesystem Tool",
                "server_type": "filesystem",
                "endpoint": "mock://filesystem",
                "description": "Mock filesystem operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["read", "write", "list"],
                    "semantic_tags": ["file", "directory", "filesystem"]
                }
            },
            {
                "id": "search",
                "name": "Search Tool",
                "server_type": "search",
                "endpoint": "mock://search",
                "description": "Mock search operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["search", "query"],
                    "semantic_tags": ["search", "web", "information"]
                }
            },
            {
                "id": "database",
                "name": "Database Tool",
                "server_type": "database",
                "endpoint": "mock://database",
                "description": "Mock database operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["query", "insert", "update", "delete"],
                    "semantic_tags": ["database", "sql", "data"]
                }
            },
            {
                "id": "github",
                "name": "GitHub Tool",
                "server_type": "github",
                "endpoint": "mock://github",
                "description": "Mock GitHub operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["repository", "issues", "pull_requests"],
                    "semantic_tags": ["git", "github", "repository", "version"]
                }
            },
            {
                "id": "financial",
                "name": "Financial Tool",
                "server_type": "financial",
                "endpoint": "mock://financial",
                "description": "Mock financial operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["data", "analysis", "market"],
                    "semantic_tags": ["finance", "market", "data", "analysis"]
                }
            },
            {
                "id": "notion",
                "name": "Notion Tool",
                "server_type": "notion",
                "endpoint": "mock://notion",
                "description": "Mock Notion operations",
                "input_schema": {},
                "capabilities": {
                    "operations": ["pages", "blocks", "databases"],
                    "semantic_tags": ["notion", "documentation", "pages"]
                }
            }
        ]
        
        for tool in mock_tools:
            registry.register_tool(tool)
        
        logger.info(f"Registered {len(mock_tools)} mock tools")
    
    async def run_strategy_evaluation(self, strategy_name: str, queries: List[TestQuery], 
                                     run_id: int, seed: int, checkpoint_enabled: bool = False) -> Dict[str, Any]:
        """Run optimized evaluation for a single strategy."""
        logger.info(f"Evaluating {strategy_name} (run {run_id}, seed {seed})")
        
        # Log processing configuration
        embedding_mode = self.get_embedding_mode(strategy_name)
        requires_intent = self.requires_intent_processing(strategy_name)
        
        if requires_intent:
            logger.info(f"  Intent processing: ENABLED for {strategy_name}")
        else:
            logger.info(f"  Intent processing: DISABLED for {strategy_name}")
        logger.info(f"  Embedding mode: {embedding_mode}")
        
        # Runtime validation check
        if embedding_mode == 'full_real' and not requires_intent:
            logger.warning(f"  ⚠️ Runtime validation: Strategy '{strategy_name}' uses 'full_real' embeddings "
                         "without intent processing. This is inefficient - consider 'fast_real' instead.")
        elif embedding_mode == 'mock' and requires_intent:
            logger.warning(f"  ⚠️ Runtime validation: Strategy '{strategy_name}' uses 'mock' embeddings "
                         "with intent processing. This wastes computation - consider 'full_real' or disable intent.")
        elif strategy_name in self.state_collection_strategies and embedding_mode == 'mock':
            logger.info(f"  ℹ️ Note: Collecting state vectors with mock embeddings for '{strategy_name}'. "
                       "Consider 'fast_real' or 'full_real' for better encoder training data.")
        
        # Set random seed
        np.random.seed(seed)
        
        # Check if this strategy needs state collection
        needs_state_collection = (self.save_state_vectors and 
                                 strategy_name in self.state_collection_strategies)
        
        # Pre-process embeddings if needed (batch processing)
        embedding_map = {}
        strategy_needs_intent = self.requires_intent_processing(strategy_name)
        
        if needs_state_collection and self.batch_embeddings and strategy_needs_intent:
            # Only batch process embeddings for strategies that need intent
            # Ensure we're extracting strings correctly
            query_texts = []
            for q in queries:
                if isinstance(q, str):
                    query_texts.append(q)
                elif hasattr(q, 'query'):
                    query_texts.append(q.query)
                else:
                    logger.warning(f"Unexpected query type: {type(q)}, using str()")
                    query_texts.append(str(q))
            
            embedding_map = await self._batch_process_embeddings(query_texts)
            logger.info(f"Pre-computed {len(embedding_map)} embeddings for {strategy_name} (intent-aware strategy)")
        
        # Initialize metrics
        completion_rates = []
        tool_accuracies = []
        episode_rewards = []
        execution_times = []
        episode_states = []  # For state vector storage
        tool_precisions = []  # NEW: Track precision
        tool_recalls = []     # NEW: Track recall
        tool_f1_scores = []   # NEW: Track F1 score
        
        # Get strategy instance
        strategy = self.evaluation_engine.strategies.get(strategy_name)
        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return {}
        
        # Run episodes
        episodes = self.exp_config.get('episodes', 10)
        for episode in range(episodes):
            episode_metrics = {
                'successes': 0,
                'correct_tools': 0,
                'execution_times': [],
                'tool_precisions': [],  # NEW
                'tool_recalls': [],      # NEW
                'tool_f1_scores': [],    # NEW
                'rewards': []            # NEW
            }
            
            # Clear intent cache between episodes for freshness
            if self.use_intent_cache:
                self.intent_cache.clear()
            
            # Shuffle queries for diversity (except first episode for baseline)
            shuffled_queries = queries.copy()
            if episode > 0:  # Keep first episode consistent for baseline
                np.random.shuffle(shuffled_queries)
            
            # Run each query
            for query_idx, query in enumerate(shuffled_queries):
                query_start = time.time()
                
                # Map optimal tools
                mapped_optimal = []
                for tool in query.optimal_tools:
                    if tool in self.tool_mapping:
                        mapped_optimal.append(self.tool_mapping[tool])
                    else:
                        mapped_optimal.append(tool)
                
                optimal_set = set(mapped_optimal)
                
                # Process query based on strategy type
                state_vector = None
                intent_embedding = None
                
                # Define context for all strategies
                # Extract domain name if it's a dict, otherwise use as-is
                domain_value = getattr(query, 'domain', getattr(query, 'category', 'general'))
                if isinstance(domain_value, dict) and 'name' in domain_value:
                    domain_value = domain_value['name']
                context = {'domain': domain_value}
                
                # Check if this strategy actually needs intent processing
                strategy_needs_intent = self.requires_intent_processing(strategy_name)
                
                if needs_state_collection:
                    # Only process intent for strategies that need it
                    
                    if strategy_needs_intent and strategy_name in ["q_learning_tabular", "q_learning_dqn"]:
                        # Q-learning strategies need full state processing
                        # Initialize state to avoid undefined variable error
                        state = None
                        try:
                            if hasattr(strategy, 'q_learning') and hasattr(strategy.q_learning, 'state_encoder'):
                                # Use cached intent processing with safe query extraction
                                query_text = query.query if hasattr(query, 'query') else str(query)
                                intent_result, intent_embedding = await self._process_intent_with_cache(
                                    query_text, context
                                )
                                # Don't modify the intent_result object to avoid circular references
                                # intent_embedding is already available as a separate variable
                                
                                # Build context for state encoding
                                context_data = {
                                    'domain': context.get('domain', 'general'),
                                    'query_count': episode + 1,
                                    'session_duration': (episode + 1) * 60,
                                    'total_queries': (episode + 1) * len(queries),
                                    'success_rate': completion_rates[-1] if completion_rates else 0.5,
                                    'metrics': {
                                        'avg_response_time': np.mean(episode_metrics['execution_times']) if episode_metrics['execution_times'] else 1000,
                                        'success_rate': completion_rates[-1] if completion_rates else 0.5,
                                        'error_rate': 1 - (completion_rates[-1] if completion_rates else 0.5),
                                        'tools_invoked': len(self.available_tools)
                                    }
                                }
                                
                                # Encode state using Q-learning's state encoder
                                history = []  # Simplified for now
                                # Debug context_data to ensure it's correct
                                if not isinstance(context_data.get('domain'), str):
                                    logger.warning(f"Domain is not a string: {context_data.get('domain')} (type: {type(context_data.get('domain'))})")
                                state = strategy.q_learning.state_encoder.encode_state(
                                    intent_result, context_data, history
                                )
                                state_vector = state.copy()
                            else:
                                # If no state encoder, create a basic state
                                query_text = query.query if hasattr(query, 'query') else str(query)
                                if query_text in embedding_map:
                                    intent_embedding = embedding_map[query_text]
                                else:
                                    # Generate embedding based on strategy's embedding mode
                                    embedding_mode = self.get_embedding_mode(strategy_name)
                                    if embedding_mode in ['fast_real', 'full_real'] and self._sentence_model is not None:
                                        # Use real sentence transformer embeddings
                                        intent_embedding = self._sentence_model.encode(query_text, convert_to_numpy=True)
                                    else:
                                        # Fallback to mock embedding
                                        if embedding_mode in ['fast_real', 'full_real']:
                                            logger.warning(f"Sentence model not available for {strategy_name}, using mock embedding")
                                        intent_embedding = np.random.randn(384).astype(np.float32) * 0.1
                                
                                state = np.zeros(476, dtype=np.float32)
                                state[:384] = intent_embedding[:384] if len(intent_embedding) >= 384 else np.pad(
                                    intent_embedding, (0, 384-len(intent_embedding))
                                )
                                state_vector = state.copy()
                        except (TypeError, AttributeError) as e:
                            logger.error(f"Error processing query for {strategy_name}: {e}")
                            # Fall back to basic processing with mock intent
                            
                            # Create a mock intent result for state encoding
                            from src.models.intent import Intent, IntentResult
                            query_text = query.query if hasattr(query, 'query') else str(query)
                            
                            # Get or create intent embedding
                            if query_text in embedding_map:
                                intent_embedding = embedding_map[query_text]
                            else:
                                # Create a basic embedding if not available
                                intent_embedding = np.random.randn(384).astype(np.float32) * 0.1
                            
                            # Create mock primary intent
                            primary_intent = Intent(
                                type=getattr(query, 'intent_type', 'query.search'),
                                confidence=0.8,
                                keywords=[],
                                entities=[]
                            )
                            
                            # Create mock intent result with correct parameters
                            mock_intent = IntentResult(
                                primary_intent=primary_intent,
                                raw_query=query_text,
                                processed_query=query_text,
                                metadata={'embedding': intent_embedding}
                            )
                            
                            # Build context
                            context_data = {
                                'domain': context.get('domain', 'general'),
                                'query_count': episode + 1,
                                'session_duration': (episode + 1) * 60,
                                'total_queries': (episode + 1) * len(queries),
                                'success_rate': completion_rates[-1] if completion_rates else 0.5,
                                'metrics': {
                                    'avg_response_time': np.mean(episode_metrics['execution_times']) if episode_metrics['execution_times'] else 1000,
                                    'success_rate': completion_rates[-1] if completion_rates else 0.5,
                                    'error_rate': 1 - (completion_rates[-1] if completion_rates else 0.5),
                                    'tools_invoked': len(self.available_tools)
                                }
                            }
                            
                            # Encode state with mock intent
                            history = []  # Simplified for now
                            if hasattr(strategy, 'q_learning') and hasattr(strategy.q_learning, 'state_encoder'):
                                state = strategy.q_learning.state_encoder.encode_state(
                                    mock_intent, context_data, history
                                )
                            else:
                                # Fallback if no state encoder
                                state = np.zeros(476, dtype=np.float32)
                                state[:384] = intent_embedding[:384] if len(intent_embedding) >= 384 else np.pad(
                                    intent_embedding, (0, 384-len(intent_embedding))
                                )
                            state_vector = state.copy()
                        
                        # Final safety check - ensure state is defined
                        if state is None:
                            state = np.zeros(476, dtype=np.float32)
                            state_vector = state.copy()
                    
                    elif strategy_needs_intent and strategy_name == "context_agnostic":
                        # Context-agnostic needs intent but simpler state
                        query_text = query.query if hasattr(query, 'query') else str(query)
                        if query_text in embedding_map:
                            # Use pre-computed embedding
                            intent_embedding = embedding_map[query_text]
                        else:
                            # Process intent with cache
                            intent_result, intent_embedding = await self._process_intent_with_cache(
                                query_text, context
                            )
                        
                        # Create basic state vector
                        state = np.zeros(476, dtype=np.float32)
                        state[:384] = intent_embedding[:384] if len(intent_embedding) >= 384 else np.pad(
                            intent_embedding, (0, 384-len(intent_embedding))
                        )
                        state_vector = state.copy()
                    elif not strategy_needs_intent:
                        # Strategy doesn't need full intent - check embedding mode
                        query_text = query.query if hasattr(query, 'query') else str(query)
                        embedding_mode = self.get_embedding_mode(strategy_name)
                        
                        # Use cached embedding if available
                        if query_text in embedding_map:
                            intent_embedding = embedding_map[query_text]
                        else:
                            # Generate embedding based on mode
                            if embedding_mode == 'mock':
                                # Create lightweight mock embedding (much faster than full intent processing)
                                # Use hash of query for deterministic but fast embedding
                                import hashlib
                                query_hash = hashlib.md5(query_text.encode()).hexdigest()
                                seed = int(query_hash[:8], 16)
                                rng = np.random.RandomState(seed)
                                intent_embedding = rng.randn(384).astype(np.float32) * 0.1
                            elif embedding_mode == 'fast_real':
                                # Use sentence transformer directly without full intent pipeline
                                if self._sentence_model is not None:
                                    intent_embedding = self._sentence_model.encode(query_text, convert_to_numpy=True)
                                else:
                                    # Fallback to mock if model not initialized
                                    logger.warning(f"Sentence model not initialized, falling back to mock for {strategy_name}")
                                    import hashlib
                                    query_hash = hashlib.md5(query_text.encode()).hexdigest()
                                    seed = int(query_hash[:8], 16)
                                    rng = np.random.RandomState(seed)
                                    intent_embedding = rng.randn(384).astype(np.float32) * 0.1
                            elif embedding_mode == 'full_real':
                                # Full intent processing (shouldn't happen if strategy_needs_intent is False)
                                logger.warning(f"Strategy {strategy_name} has full_real embedding but needs_intent=False")
                                # Process intent with cache
                                intent_result, intent_embedding = await self._process_intent_with_cache(
                                    query_text, context
                                )
                            else:
                                logger.warning(f"Unknown embedding mode '{embedding_mode}' for {strategy_name}, using mock")
                                # Default to mock
                                import hashlib
                                query_hash = hashlib.md5(query_text.encode()).hexdigest()
                                seed = int(query_hash[:8], 16)
                                rng = np.random.RandomState(seed)
                                intent_embedding = rng.randn(384).astype(np.float32) * 0.1
                        
                        # Create state vector
                        state = np.zeros(476, dtype=np.float32)
                        state[:384] = intent_embedding[:384] if len(intent_embedding) >= 384 else intent_embedding
                        state_vector = state.copy()
                else:
                    # For other strategies, create a basic state if state saving is enabled
                    if self.save_state_vectors:
                        # Process intent to get embedding for state vector
                        query_text = query.query if hasattr(query, 'query') else str(query)
                        if query_text in embedding_map:
                            intent_embedding = embedding_map[query_text]
                        else:
                            intent_result, intent_embedding = await self._process_intent_with_cache(
                                query_text, context
                            )
                        
                        # Create basic state vector with intent embedding
                        state = np.zeros(476, dtype=np.float32)
                        state[:384] = intent_embedding[:384] if len(intent_embedding) >= 384 else np.pad(
                            intent_embedding, (0, 384-len(intent_embedding))
                        )
                        state_vector = state.copy()
                    else:
                        # No state collection needed - use dummy state
                        state = np.zeros(476, dtype=np.float32)
                        state_vector = None
                
                # Execute strategy
                try:
                    if strategy_name in ["q_learning_tabular", "q_learning_dqn"]:
                        constraints = {}
                        tools_selected = await strategy.select_tools(
                            state,
                            self.available_tools,
                            constraints
                        )
                        tools_used = tools_selected
                        selected_set = set(tools_used)
                    else:
                        # Use direct strategy method call for other strategies
                        constraints = {}
                        tools_selected = await strategy.select_tools(
                            state,
                            self.available_tools,
                            constraints
                        )
                        tools_used = tools_selected
                        selected_set = set(tools_used)
                    
                    # Calculate reward and success based on configuration
                    if self.use_graded_rewards and tools_used:
                        # Create execution metrics for reward calculation
                        exec_metrics = []
                        for tool in tools_used:
                            exec_metrics.append(ExecutionMetrics(
                                tool_id=tool,
                                success=tool in optimal_set,  # Tool was needed
                                partial_success=False,
                                completion_percentage=1.0 if tool in optimal_set else 0.0,
                                execution_time_ms=100.0,  # Mock execution time
                                error_type=None if tool in optimal_set else 'wrong_tool',
                                result_quality=1.0 if tool in optimal_set else 0.0
                            ))
                        
                        # Calculate sophisticated reward
                        reward, reward_breakdown = self.reward_calculator.calculate_reward(
                            execution_results=exec_metrics,
                            context={'intent': getattr(query, 'intent_type', 'unknown'), 
                                    'domain': getattr(query, 'category', 'general')},
                            optimal_tools=list(mapped_optimal)
                        )
                    else:
                        # Simple binary reward
                        reward = 1.0 if len(optimal_set & selected_set) > 0 else -1.0
                    
                    # Determine success based on criteria
                    if self.success_criteria == 'strict':
                        # All optimal tools must be selected
                        success = optimal_set.issubset(selected_set)
                    elif self.success_criteria == 'reward-based':
                        # Positive reward indicates success
                        success = reward > 0
                    else:  # lenient
                        # Any overlap counts as success (current behavior)
                        success = len(optimal_set & selected_set) > 0
                    
                    # Update metrics
                    if success:
                        episode_metrics['successes'] += 1
                    
                    if selected_set == optimal_set:
                        episode_metrics['correct_tools'] += 1
                    
                    execution_time = time.time() - query_start
                    episode_metrics['execution_times'].append(execution_time)
                    episode_metrics['rewards'].append(reward)
                    
                    # Calculate precision, recall, F1
                    intersection = optimal_set & selected_set
                    tool_precision = len(intersection) / len(selected_set) if selected_set else 0
                    tool_recall = len(intersection) / len(optimal_set) if optimal_set else 0
                    tool_f1 = 2 * (tool_precision * tool_recall) / (tool_precision + tool_recall) if (tool_precision + tool_recall) > 0 else 0
                    
                    episode_metrics['tool_precisions'].append(tool_precision)
                    episode_metrics['tool_recalls'].append(tool_recall)
                    episode_metrics['tool_f1_scores'].append(tool_f1)
                    
                    # Store state if needed with sampling
                    if self.save_state_vectors and state_vector is not None:
                        # Get sampling rate for this strategy
                        sampling_rate = self.strategy_sampling_rates.get(
                            strategy_name, 
                            self.strategy_sampling_rates.get('others', 0.2)
                        )
                        
                        # Apply sampling
                        if np.random.random() <= sampling_rate:
                            # Ensure all data is serializable (no object references)
                            state_data = {
                                'query': str(query.query),  # Ensure string
                                'state_vector': state_vector.copy() if isinstance(state_vector, np.ndarray) else state_vector,
                                'intent_embedding': intent_embedding.copy() if isinstance(intent_embedding, np.ndarray) else intent_embedding,
                                'tools_selected': list(tools_used) if tools_used else [],  # Ensure list
                                'optimal_tools': list(mapped_optimal) if mapped_optimal else [],  # Ensure list
                                'success': bool(success),  # Ensure bool
                                'reward': float(reward),  # Use calculated reward
                                'execution_time': float(execution_time),  # Ensure float
                                'episode': int(episode),  # Ensure int
                                'query_idx': int(query_idx),  # Ensure int
                                'strategy': strategy_name  # Add strategy name for analysis
                            }
                            episode_states.append(state_data)
                    
                except Exception as e:
                    logger.error(f"{strategy_name} query failed: {e}")
                    episode_metrics['execution_times'].append(time.time() - query_start)
            
            # Calculate episode metrics
            completion_rate = episode_metrics['successes'] / len(queries)
            tool_accuracy = episode_metrics['correct_tools'] / len(queries)
            avg_execution_time = np.mean(episode_metrics['execution_times'])
            
            # Calculate average precision, recall, F1
            avg_precision = np.mean(episode_metrics['tool_precisions']) if episode_metrics['tool_precisions'] else 0
            avg_recall = np.mean(episode_metrics['tool_recalls']) if episode_metrics['tool_recalls'] else 0
            avg_f1 = np.mean(episode_metrics['tool_f1_scores']) if episode_metrics['tool_f1_scores'] else 0
            avg_reward = np.mean(episode_metrics['rewards']) if episode_metrics['rewards'] else 0
            
            completion_rates.append(completion_rate)
            tool_accuracies.append(tool_accuracy)
            episode_rewards.append(avg_reward)  # Use actual calculated rewards
            execution_times.append(avg_execution_time)
            tool_precisions.append(avg_precision)
            tool_recalls.append(avg_recall)
            tool_f1_scores.append(avg_f1)
            
            # Enhanced logging every 10 episodes
            if (episode + 1) % 10 == 0:
                # Calculate advanced metrics
                baseline_completion = completion_rates[0] if len(completion_rates) > 0 else 0
                
                # Calculate state diversity (entropy-like measure)
                state_diversity = 0.0
                if episode_states:
                    unique_states = len(set(tuple(s['state_vector'].flatten()) if isinstance(s['state_vector'], np.ndarray) else tuple(s['state_vector']) 
                                          for s in episode_states[-100:] if 'state_vector' in s))  # Last 100 states
                    state_diversity = unique_states / min(100, len(episode_states[-100:]))
                
                # Calculate embedding quality (average cosine similarity variance)
                embedding_quality = 0.0
                if episode_states and len(episode_states) > 1:
                    recent_embeddings = [s['intent_embedding'] for s in episode_states[-50:] 
                                       if 'intent_embedding' in s and s['intent_embedding'] is not None]
                    if len(recent_embeddings) >= 2:
                        from sklearn.metrics.pairwise import cosine_similarity
                        cos_sim = cosine_similarity(recent_embeddings)
                        embedding_quality = np.std(cos_sim[np.triu_indices_from(cos_sim, k=1)])
                
                # Calculate convergence indicator (moving average slope)
                convergence_indicator = 0.0
                if len(completion_rates) >= 5:
                    recent_rates = completion_rates[-5:]
                    x = np.arange(len(recent_rates))
                    convergence_indicator = np.polyfit(x, recent_rates, 1)[0]  # Slope of linear fit
                
                log_metrics = {
                    'strategy': strategy_name,
                    'episode': episode + 1,
                    'completion_rate': float(np.mean(completion_rates)),
                    'q_learning_improvement': float(np.mean(completion_rates) - baseline_completion),
                    'state_vector_diversity': float(state_diversity),
                    'embedding_quality': float(embedding_quality),
                    'convergence_indicator': float(convergence_indicator),
                    'states_collected': len(episode_states),
                    'unique_queries_seen': len(set(s.get('query', '') for s in episode_states)),
                    'avg_reward': float(np.mean(episode_rewards)),
                    'avg_f1_score': float(np.mean(tool_f1_scores))
                }
                
                logger.info(f"\n{strategy_name} - Episode {episode + 1}/{episodes}:")
                logger.info(f"  Completion Rate: {log_metrics['completion_rate']:.3f}")
                logger.info(f"  Improvement: {log_metrics['q_learning_improvement']:+.3f}")
                logger.info(f"  State Diversity: {log_metrics['state_vector_diversity']:.3f}")
                logger.info(f"  Embedding Quality: {log_metrics['embedding_quality']:.3f}")
                logger.info(f"  Convergence: {log_metrics['convergence_indicator']:+.4f}")
                logger.info(f"  States Collected: {log_metrics['states_collected']}")
                logger.info(f"  Average Reward: {log_metrics['avg_reward']:.2f}")
                logger.info(f"  Average F1: {log_metrics['avg_f1_score']:.3f}")
                
                # Log cache stats if using cache
                if self.use_intent_cache and needs_state_collection:
                    cache_stats = self.intent_cache.get_stats()
                    logger.info(f"  Cache: hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                              f"hit_rate={cache_stats['hit_rate']:.2%}")
            
            # Save checkpoint if enabled
            checkpoint_interval = self.exp_config.get('checkpoint_interval', 5)
            if checkpoint_enabled and self.checkpoint_manager and (episode + 1) % checkpoint_interval == 0:
                checkpoint_state = {
                    'strategy_name': strategy_name,
                    'episode': episode + 1,
                    'metrics': {
                        'completion_rates': completion_rates,
                        'tool_accuracies': tool_accuracies,
                        'episode_rewards': episode_rewards,
                        'execution_times': execution_times
                    },
                    'episode_states': episode_states[-100:] if self.save_state_vectors else [],  # Last 100 states
                    'save_state_vectors': self.save_state_vectors,
                    'state_dimensions': 476 if state_vector is not None else 0
                }
                self.checkpoint_manager.save_checkpoint(checkpoint_state, strategy_name, episode + 1)
        
        logger.info(f"Completed {strategy_name}: mean_completion={np.mean(completion_rates):.3f}")
        
        result = {
            'strategy': strategy_name,
            'run_id': run_id,
            'seed': seed,
            'episodes': episodes,
            'completion_rates': completion_rates,
            'tool_accuracies': tool_accuracies,
            'episode_rewards': episode_rewards,
            'execution_times': execution_times,
            'tool_precisions': tool_precisions,  # NEW
            'tool_recalls': tool_recalls,        # NEW
            'tool_f1_scores': tool_f1_scores,    # NEW
            'final_metrics': {
                'mean_completion_rate': float(np.mean(completion_rates)),
                'std_completion_rate': float(np.std(completion_rates)),
                'mean_tool_accuracy': float(np.mean(tool_accuracies)),
                'mean_execution_time': float(np.mean(execution_times)),
                'mean_precision': float(np.mean(tool_precisions)),  # NEW
                'mean_recall': float(np.mean(tool_recalls)),        # NEW
                'mean_f1_score': float(np.mean(tool_f1_scores)),    # NEW
                'mean_reward': float(np.mean(episode_rewards))      # NEW
            },
            'episode_states': episode_states if self.save_state_vectors else []
        }
        
        # Return the result directly - we'll clean it up when returning all_results
        return result
    
    async def run_comparison(self, checkpoint_dir: Optional[str] = None, 
                           checkpoint_file: Optional[str] = None) -> Dict[str, Any]:
        """Run optimized baseline comparison experiment."""
        logger.info("=" * 60)
        logger.info("OPTIMIZED BASELINE COMPARISON EXPERIMENT")
        logger.info("=" * 60)
        
        # Setup checkpointing if enabled
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(checkpoint_dir))
            checkpoint_interval = self.exp_config.get('checkpoint_interval', 5)
            logger.info(f"Checkpointing enabled: saving every {checkpoint_interval} episodes")
        
        # Load checkpoint if resuming
        if checkpoint_file:
            if not self.checkpoint_manager:
                # Create checkpoint manager if not already created
                checkpoint_dir = checkpoint_dir or str(Path(checkpoint_file).parent)
                self.checkpoint_manager = CheckpointManager(Path(checkpoint_dir))
            self.resume_state = self.checkpoint_manager.load_checkpoint(checkpoint_file)
        
        # Initialize components
        await self.initialize_components()
        
        # Get test queries based on query set
        query_set = getattr(self, 'query_set', 'dissertation_core')
        evaluation_sets = get_evaluation_sets()
        
        if query_set in evaluation_sets:
            queries = evaluation_sets[query_set]
            logger.info(f"Loaded {len(queries)} queries from '{query_set}' set")
        else:
            # Fallback to all queries
            queries = get_all_queries()
            logger.info(f"Loaded {len(queries)} test queries (all queries)")
        
        # Store original query order for baseline
        self.original_query_order = queries.copy()
        
        # Run evaluations for each strategy
        all_results = {}
        strategies = self.exp_config.get('strategies', [])
        num_runs = self.exp_config.get('runs_per_strategy', 5)
        
        for strategy_config in strategies:
            strategy_name = strategy_config['name']
            strategy_results = []
            
            for run_id in range(num_runs):
                seed = 42 + run_id  # Deterministic seeds
                
                result = await self.run_strategy_evaluation(
                    strategy_name, queries, run_id, seed,
                    checkpoint_enabled=(checkpoint_dir is not None)
                )
                
                strategy_results.append(result)
                
                # Save intermediate results with error handling
                try:
                    self.save_results({strategy_name: strategy_results})
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not save intermediate results for {strategy_name}: {e}")
            
            all_results[strategy_name] = strategy_results
        
        # Generate final report with error handling
        try:
            self.generate_report(all_results)
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not generate final report: {e}")
        
        # Log optimization statistics
        if self.use_intent_cache:
            cache_stats = self.intent_cache.get_stats()
            logger.info("=" * 60)
            logger.info("OPTIMIZATION STATISTICS")
            logger.info(f"Intent cache - Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
            logger.info(f"Intent cache - Hit rate: {cache_stats['hit_rate']:.2%}")
            logger.info(f"State collection strategies: {self.state_collection_strategies}")
            logger.info("=" * 60)
        
        # Create a clean copy without deep copy which might hit circular references
        clean_results = {}
        for strategy_name, strategy_results in all_results.items():
            clean_results[strategy_name] = []
            for result in strategy_results:
                # Create a new dict with only serializable data
                clean_result = {
                    'strategy': str(result.get('strategy', strategy_name)),
                    'run_id': result.get('run_id', 0),
                    'seed': result.get('seed', 0),
                    'episodes': result.get('episodes', 0),
                    'completion_rates': list(result.get('completion_rates', [])),
                    'tool_accuracies': list(result.get('tool_accuracies', [])),
                    'episode_rewards': list(result.get('episode_rewards', [])),
                    'execution_times': list(result.get('execution_times', [])),
                    'final_metrics': dict(result.get('final_metrics', {})),
                    # Don't include episode_states in the returned results
                }
                clean_results[strategy_name].append(clean_result)
        
        return clean_results
    
    def _sanitize_for_json(self, obj, seen=None):
        """Recursively sanitize objects for JSON serialization."""
        if seen is None:
            seen = set()
        
        # Handle circular references
        obj_id = id(obj)
        if obj_id in seen:
            return None  # Break circular reference
        
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Add to seen set for container types
        if isinstance(obj, (dict, list, tuple)):
            seen.add(obj_id)
        
        # Handle containers
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Skip if key is not JSON serializable
                if not isinstance(k, (str, int, float, bool, type(None))):
                    continue
                result[k] = self._sanitize_for_json(v, seen)
            return result
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item, seen) for item in obj]
        elif isinstance(obj, tuple):
            return [self._sanitize_for_json(item, seen) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Object with attributes - convert to string representation
            return str(obj)
        else:
            return obj
    
    def save_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """Save intermediate results to file."""
        for strategy_name, strategy_results in results.items():
            for result in strategy_results:
                result_file = self.results_dir / f"{strategy_name}_run{result['run_id']}_{self.timestamp}.json"
                try:
                    # Create a clean copy for JSON serialization
                    json_result = {}
                    for k, v in result.items():
                        if k == 'episode_states':
                            # Don't save full states in JSON to avoid size and serialization issues
                            continue
                        elif k == 'strategy':
                            # Ensure strategy is a string
                            json_result[k] = str(v) if not isinstance(v, str) else v
                        else:
                            # Sanitize all other fields
                            json_result[k] = self._sanitize_for_json(v)
                    
                    # Try to save with circular reference detection disabled temporarily
                    json_str = json.dumps(json_result, indent=2, default=str, check_circular=False)
                    with open(result_file, 'w') as f:
                        f.write(json_str)
                        
                except (TypeError, ValueError, RecursionError) as e:
                    if "Circular reference" in str(e):
                        logger.warning(f"Circular reference detected for {strategy_name}, using simplified save")
                        # Create minimal result without problematic fields
                        minimal_result = {
                            'strategy': strategy_name,
                            'run_id': result.get('run_id', 0),
                            'final_metrics': result.get('final_metrics', {})
                        }
                        try:
                            with open(result_file, 'w') as f:
                                json.dump(minimal_result, f, indent=2)
                        except:
                            pass
                    else:
                        logger.warning(f"Could not save JSON results for {strategy_name}: {e}")
                    
                    # Always save pickle as backup
                    pickle_file = self.results_dir / f"{strategy_name}_run{result['run_id']}_{self.timestamp}.pkl"
                    try:
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(result, f)
                    except Exception as pe:
                        logger.error(f"Could not save pickle results for {strategy_name}: {pe}")
    
    def generate_report(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """Generate comparison report."""
        report_file = self.results_dir.parent / f"baseline_comparison_final_{self.timestamp}.json"
        
        summary = {}
        for strategy_name, results in all_results.items():
            # Aggregate metrics across runs
            all_completion_rates = []
            all_tool_accuracies = []
            all_execution_times = []
            all_precisions = []
            all_recalls = []
            all_f1_scores = []
            all_rewards = []
            
            for result in results:
                all_completion_rates.extend(result['completion_rates'])
                all_tool_accuracies.extend(result['tool_accuracies'])
                all_execution_times.extend(result['execution_times'])
                if 'tool_precisions' in result:
                    all_precisions.extend(result['tool_precisions'])
                    all_recalls.extend(result['tool_recalls'])
                    all_f1_scores.extend(result['tool_f1_scores'])
                if 'episode_rewards' in result:
                    all_rewards.extend(result['episode_rewards'])
            
            summary[strategy_name] = {
                'runs': len(results),
                'episodes_per_run': results[0]['episodes'] if results else 0,
                'mean_completion_rate': float(np.mean(all_completion_rates)),
                'std_completion_rate': float(np.std(all_completion_rates)),
                'mean_tool_accuracy': float(np.mean(all_tool_accuracies)),
                'std_tool_accuracy': float(np.std(all_tool_accuracies)),
                'mean_execution_time': float(np.mean(all_execution_times)),
                'convergence_episode': self._find_convergence_episode(all_completion_rates)
            }
            
            # Add new metrics if available
            if all_precisions:
                summary[strategy_name]['mean_precision'] = float(np.mean(all_precisions))
                summary[strategy_name]['mean_recall'] = float(np.mean(all_recalls))
                summary[strategy_name]['mean_f1_score'] = float(np.mean(all_f1_scores))
            if all_rewards:
                summary[strategy_name]['mean_reward'] = float(np.mean(all_rewards))
        
        # Save summary with error handling
        try:
            with open(report_file, 'w') as f:
                # Sanitize summary to ensure JSON serialization works
                sanitized_summary = self._sanitize_for_json(summary)
                json.dump(sanitized_summary, f, indent=2, default=str)
            logger.info(f"Final report saved to {report_file}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not save JSON report: {e}")
            # Save as pickle as backup
            pickle_file = self.results_dir.parent / f"baseline_comparison_final_{self.timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(summary, f)
            logger.info(f"Report saved as pickle to {pickle_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        for strategy, metrics in summary.items():
            logger.info(f"\n{strategy}:")
            logger.info(f"  Completion Rate: {metrics['mean_completion_rate']:.3f} ± {metrics['std_completion_rate']:.3f}")
            logger.info(f"  Tool Accuracy: {metrics['mean_tool_accuracy']:.3f} ± {metrics['std_tool_accuracy']:.3f}")
            if 'mean_precision' in metrics:
                logger.info(f"  Tool Precision: {metrics.get('mean_precision', 0):.3f}")
                logger.info(f"  Tool Recall: {metrics.get('mean_recall', 0):.3f}")
                logger.info(f"  Tool F1 Score: {metrics.get('mean_f1_score', 0):.3f}")
                logger.info(f"  Mean Reward: {metrics.get('mean_reward', 0):.2f}")
            logger.info(f"  Execution Time: {metrics['mean_execution_time']:.3f}s")
            logger.info(f"  Convergence: Episode {metrics['convergence_episode']}")
    
    def _find_convergence_episode(self, rates: List[float], window: int = 10, threshold: float = 0.01) -> int:
        """Find episode where performance converges."""
        if len(rates) < window * 2:
            return len(rates)
        
        for i in range(window, len(rates) - window):
            window_before = rates[i-window:i]
            window_after = rates[i:i+window]
            
            if abs(np.mean(window_after) - np.mean(window_before)) < threshold:
                return i
        
        return len(rates)


async def main():
    """Main entry point for optimized baseline comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized baseline comparison experiment")
    parser.add_argument("--config", help="Path to experiment config file")
    parser.add_argument("--checkpoint-dir", help="Directory to save checkpoints")
    parser.add_argument("--resume", help="Path to checkpoint file to resume from")
    parser.add_argument("--resume-from", help="Alias for --resume for compatibility")
    parser.add_argument("--no-retries", action="store_true", help="Disable retries for faster testing")
    parser.add_argument("--query-set", default="dissertation_core",
                       choices=["quick_test", "simple_only", "complex_only", "ambiguous_only",
                               "full_evaluation", "dissertation_core", "hard_evaluation"],
                       help="Query set to use for evaluation")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Override number of episodes")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--runs", type=int, default=None,
                       help="Override runs per strategy")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                       help="Save checkpoint every N episodes (default: 5)")
    parser.add_argument("--embedding-mode", 
                       choices=["mock", "fast_real", "full_real"],
                       help="Override embedding mode for all strategies (mock=fast, fast_real=balanced, full_real=complete)")
    parser.add_argument("--success-criteria", 
                       choices=["lenient", "strict", "reward-based"],
                       default="strict",
                       help="Success criteria: lenient (any match), strict (all optimal), reward-based (positive reward)")
    parser.add_argument("--use-graded-rewards", 
                       action="store_true",
                       default=True,
                       help="Use sophisticated reward calculator instead of binary rewards")
    parser.add_argument("--use-real-servers",
                       action="store_true",
                       help="Use real MCP servers instead of mock servers when available")
    
    args = parser.parse_args()
    
    runner = OptimizedBaselineComparisonRunner(args.config)
    
    if args.no_retries:
        runner.enable_retries = False
    
    # Override settings from command line if provided
    if args.episodes is not None:
        runner.exp_config['episodes'] = args.episodes
    if args.runs is not None:
        runner.exp_config['runs_per_strategy'] = args.runs
    if args.checkpoint_interval is not None:
        runner.exp_config['checkpoint_interval'] = args.checkpoint_interval
    if args.output_dir:
        runner.results_dir = Path(args.output_dir)
        runner.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set embedding mode override if provided
    if args.embedding_mode:
        runner.embedding_mode_override = args.embedding_mode
        logger.info(f"Embedding mode override: {args.embedding_mode} (all strategies will use this mode)")
    
    # Set evaluation criteria
    runner.success_criteria = args.success_criteria
    runner.use_graded_rewards = args.use_graded_rewards
    runner.use_real_servers = args.use_real_servers
    logger.info(f"Success criteria: {args.success_criteria}")
    logger.info(f"Using graded rewards: {args.use_graded_rewards}")
    logger.info(f"Using real servers: {args.use_real_servers}")
    
    # Set query set
    runner.query_set = args.query_set
    
    # Handle resume-from alias
    resume_file = args.resume or args.resume_from
    
    try:
        results = await runner.run_comparison(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_file=resume_file
        )
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Don't re-raise to see if script completes
        import sys
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())