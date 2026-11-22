---
name: apex-memory
description: >
  Sovereign omniversal semantic memory, vector store, Hebbian synapse & knowledge graph mesh.
  Use when: agent needs persistent memory across sessions, semantic search over stored memories,
  knowledge graph relationships, Hebbian synaptic reinforcement, or cryptographic shard consolidation.
  Triggers: "remember this", "recall", "memory search", "knowledge graph", "associate memories",
  "consolidate shard", "semantic memory", "vector store", "apex memory".
---

# APEX Memory SDK — Agent Skill

## Overview

APEX Memory SDK provides a sovereign, omniversal semantic memory system with:
- **SQLite Vector Store** with BM25 hybrid ranking and RRF fusion
- **Hebbian Synapse Network** with biological learning (reinforcement + decay + pruning)
- **Directed Knowledge Graph** with transitive traversal (BFS)
- **ANE-Accelerated Embeddings** on Apple Neural Engine (CoreML)
- **Cryptographic Shard Consolidation** with SHA-256 digests
- **MCP Server** (JSON-RPC 2.0 stdio) for agent integration
- **CLI** with lifecycle management (serve/stop/status/query/seed)

## Quick Start

```bash
pip install apex-memory-sdk
apex-memory info
apex-memory remember my-id "Content to remember" --tags tag1 tag2
apex-memory recall "query text" -k 5
apex-memory associate node_a node_b
apex-memory consolidate shard-1 -o shard.json
apex-memory mcp  # Run MCP server
apex-memory serve -b  # Start HTTP server
apex-memory status  # Check health
apex-memory query "search text"  # Query via HTTP
apex-memory seed --source docs/  # Seed from markdown
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store memory with vector + BM25 + synaptic relations |
| `memory_recall` | Hybrid semantic + keyword + synaptic recall |
| `memory_associate` | Reinforce Hebbian synaptic connection |
| `memory_graph_add` | Add entity/relation to knowledge graph |
| `memory_graph_query` | Query knowledge graph relations |
| `memory_consolidate` | Export cryptographic episodic shard |
| `memory_health` | Health check: counts and status |

## Architecture

```
Text Input → ANE Embeddings (CoreML) → Vector Store (SQLite)
                                    → BM25 Index
                                    → Hebbian Synapse Network
                                    → Knowledge Graph
```

## Key Patterns

### Remember + Associate
```python
store.put("item-1", "Python is great", tags=["lang"])
hebbian.fire("item-1", "python", delta=0.3)
```

### Hybrid Recall
```python
results = engine.recall("programming language", top_k=5, expand_synaptic=True)
```

### Knowledge Graph
```python
graph.add_entity("Python", "Language")
graph.add_entity("Django", "Framework")
graph.add_relation("Django", "built_with", "Python")
```

### Shard Consolidation
```python
shard = consolidator.consolidate("shard-1", "Agent snapshot")
# shard["shard_hash"] = SHA-256 digest
```

## Integration

### With apex-agent-sdk
```python
from apex_memory import VectorStore, HebbianNetwork
from apex_agent import Agent

agent = Agent(memory_store=VectorStore(), hebbian=HebbianNetwork())
```

### MCP Configuration
```json
{
  "mcpServers": {
    "apex-memory": {
      "command": "apex-memory",
      "args": ["mcp"]
    }
  }
}
```

## Performance

- ANE embedding: ~0.71ms per embedding (Apple Neural Engine)
- Semantic similarity: 0.997 for related concepts
- Test suite: 20 tests, <5s execution
- Module-level model cache: 0ms init on cache hit

## License

Apache-2.0
