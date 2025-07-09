# Tool Discovery Architecture

## Overview

The Tool Discovery layer finds relevant tools through semantic search, graph exploration, and capability matching.

## Tool Registry & Discovery Architecture

### Tool Capability Schema

```json
{
  "tool_id": "filesystem_mcp_001",
  "capabilities": {
    "operations": [
      {
        "name": "read_file",
        "category": "file_io",
        "parameters": {
          "path": { "type": "string", "required": true },
          "encoding": { "type": "string", "default": "utf-8" }
        },
        "returns": { "type": "string" },
        "errors": ["FileNotFoundError", "PermissionError"]
      },
      {
        "name": "write_file",
        "category": "file_io",
        "parameters": {
          "path": { "type": "string", "required": true },
          "content": { "type": "string", "required": true },
          "mode": { "type": "string", "default": "w" }
        },
        "returns": { "type": "boolean" }
      }
    ],
    "constraints": {
      "max_file_size_mb": 100,
      "allowed_extensions": [".txt", ".json", ".xml", ".csv"],
      "restricted_paths": ["/etc", "/sys", "/proc"]
    },
    "semantic_tags": ["filesystem", "io", "storage", "data_access"]
  }
}
```

## Discovery Algorithms

### 1. Semantic Discovery
```python
class SemanticToolDiscovery:
    def __init__(self, model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model)
        self.capability_embeddings = {}
        self.similarity_threshold = 0.7
    
    async def discover_tools(self, intent_vector, context):
        # Encode intent if string
        if isinstance(intent_vector, str):
            intent_embedding = self.encoder.encode(intent_vector)
        else:
            intent_embedding = intent_vector
        
        # Calculate similarities
        candidates = []
        for tool_id, embeddings in self.capability_embeddings.items():
            max_similarity = max(
                cosine_similarity(intent_embedding, cap_emb)
                for cap_emb in embeddings
            )
            if max_similarity > self.similarity_threshold:
                candidates.append((tool_id, max_similarity))
        
        # Sort by similarity and return top-k
        return sorted(candidates, key=lambda x: x[1], reverse=True)
```

### 2. Graph-Based Discovery
```python
class GraphBasedDiscovery:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.build_tool_graph()
    
    def discover_complementary_tools(self, selected_tools, max_depth=2):
        complementary = set()
        
        for tool in selected_tools:
            # BFS to find related tools
            for neighbor in nx.bfs_tree(self.graph, tool, max_depth):
                edge_data = self.graph.get_edge_data(tool, neighbor)
                if edge_data and edge_data['type'] in ['complements', 'enhances']:
                    complementary.add(neighbor)
        
        return list(complementary - set(selected_tools))
```

## Search Strategies

1. **Exact Match**: Direct capability name matching
2. **Semantic Search**: Embedding-based similarity
3. **Category Browse**: Hierarchical category navigation
4. **Graph Traversal**: Relationship-based exploration
5. **Hybrid Search**: Combination of above strategies

## Caching Architecture

```python
class ToolRegistryCache:
    def __init__(self, ttl_seconds=300):
        self.cache = TTLCache(maxsize=1000, ttl=ttl_seconds)
        self.embedding_cache = {}
        self.graph_cache = None
        self.last_update = None
    
    async def get_tool(self, tool_id):
        if tool_id in self.cache:
            return self.cache[tool_id]
        
        # Fetch from database
        tool = await self.fetch_from_db(tool_id)
        self.cache[tool_id] = tool
        return tool
    
    def invalidate_tool(self, tool_id):
        self.cache.pop(tool_id, None)
        # Also invalidate related caches
        self.invalidate_relationships(tool_id)
```

## Discovery Pipeline

```
Query → Intent Analysis → Strategy Selection → 
Parallel Search (Semantic + Graph + Category) → 
Result Aggregation → Ranking → Filtering → 
Cache Update → Return Results
```

## Tool Relationships

### Relationship Types
- **complements**: Tools that work well together
- **requires**: Dependencies between tools
- **conflicts**: Tools that shouldn't be used together
- **enhances**: Tools that improve other tools' capabilities

### Relationship Strength
- Values between 0.0 and 1.0
- Based on historical usage patterns
- Updated through learning

## Capability Matching

### Matching Algorithm
1. Extract required capabilities from intent
2. Compare with tool capabilities
3. Score based on:
   - Exact matches
   - Partial matches
   - Semantic similarity
   - Constraint satisfaction

### Scoring Formula
```python
def score_tool(tool, intent_capabilities):
    exact_match_score = count_exact_matches(tool, intent_capabilities) * 1.0
    partial_match_score = count_partial_matches(tool, intent_capabilities) * 0.5
    semantic_score = calculate_semantic_similarity(tool, intent_capabilities) * 0.3
    constraint_score = check_constraints(tool, intent_capabilities) * 0.2
    
    return exact_match_score + partial_match_score + semantic_score + constraint_score
```

## Performance Optimization

### Indexing Strategy
- Capability name index for exact matching
- Semantic embedding index for similarity search
- Category hierarchy index for browsing
- Graph adjacency index for traversal

### Parallel Search
```python
async def parallel_discovery(intent, context):
    tasks = [
        semantic_search(intent),
        graph_search(context.selected_tools),
        category_search(intent.categories),
        exact_match_search(intent.capabilities)
    ]
    
    results = await asyncio.gather(*tasks)
    return aggregate_results(results)
```

## Configuration

- **Similarity Threshold**: 0.7
- **Max Search Results**: 10
- **Cache TTL**: 300 seconds
- **Max Graph Depth**: 2
- **Parallel Search Timeout**: 5 seconds