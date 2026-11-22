# APEX Memory SDK — API Reference

## VectorStore

### `VectorStore(db_path: str = None)`

SQLite-backed vector store with BM25 hybrid ranking.

| Method | Signature | Description |
|--------|-----------|-------------|
| `put` | `put(item_id, content, tags=None)` | Store memory item with vector + BM25 indexing |
| `get` | `get(item_id)` | Retrieve item by ID |
| `delete` | `delete(item_id)` | Remove item and update indexes |
| `search` | `search(query, top_k=5)` | Hybrid vector + BM25 search with RRF fusion |
| `list_items` | `list_items()` | List all stored items |
| `count` | `count()` | Total item count |

### `SubwordEmbeddingEngine`

Hash-based n-gram embedding engine (fallback when ANE unavailable).

### `ANEEmbeddingEngine`

Apple Neural Engine accelerated embedding engine (CoreML).

| Property | Type | Description |
|----------|------|-------------|
| `dimensions` | `int` | Embedding dimensionality (default 384) |
| `_ane_ready` | `bool` | Whether ANE model loaded successfully |
| `_model_cache` | `dict` | Module-level cache (keyed by dimensions) |

## HebbianNetwork

### `HebbianNetwork()`

Biological synaptic learning network.

| Method | Signature | Description |
|--------|-----------|-------------|
| `fire` | `fire(node_a, node_b, delta=0.2)` | Reinforce connection, returns edge |
| `get_weight` | `get_weight(node_a, node_b)` | Get connection weight |
| `decay` | `decay(factor=0.95)` | Apply decay to all weights |
| `prune` | `prune(threshold=0.01)` | Remove weak connections |
| `edges` | property | Dict of all edges |

### `AssociativeRecallEngine`

Hybrid recall combining vector search + Hebbian expansion.

| Method | Signature | Description |
|--------|-----------|-------------|
| `recall` | `recall(query, top_k=3, expand_synaptic=True)` | Hybrid recall with synaptic expansion |

## KnowledgeGraph

### `KnowledgeGraph()`

Directed knowledge graph with transitive traversal.

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_entity` | `add_entity(id, label)` | Add entity node |
| `get_entity` | `get_entity(id)` | Get entity by ID |
| `add_relation` | `add_relation(source, relation, target)` | Add directed edge |
| `query` | `query(subject=None, predicate=None, obj=None)` | Query relations |
| `traverse` | `traverse(start, max_depth=3)` | BFS transitive traversal |
| `get_all_entities` | `get_all_entities()` | List all entities |

## MemoryConsolidator

### `MemoryConsolidator(store, hebbian, graph=None)`

Cryptographic shard consolidation.

| Method | Signature | Description |
|--------|-----------|-------------|
| `consolidate` | `consolidate(shard_id, label="")` | Export shard with SHA-256 hash |

Shard structure:
```json
{
  "shard_id": "shard-1",
  "label": "Agent Consolidation",
  "shard_hash": "sha256:...",
  "memory_count": 42,
  "synapse_count": 15,
  "graph_entity_count": 8,
  "created_at": "2026-11-21T...",
  "memories": [...],
  "synapses": [...],
  "graph": {...}
}
```

## MemoryMCPServer

### `MemoryMCPServer(db_path=None)`

MCP JSON-RPC 2.0 stdio server.

| Method | Description |
|--------|-------------|
| `get_tools()` | List available MCP tools |
| `dispatch_tool(name, args)` | Execute tool by name |
| `handle_request(req)` | Process JSON-RPC request |
| `run_stdio()` | Run stdio server loop |

### MCP Tools

| Tool | Required Args | Description |
|------|---------------|-------------|
| `memory_remember` | `item_id`, `content` | Store memory |
| `memory_recall` | `query` | Recall memories |
| `memory_associate` | `node_a`, `node_b` | Reinforce synapse |
| `memory_graph_add` | `source`, `relation`, `target` | Add graph relation |
| `memory_graph_query` | (none) | Query graph |
| `memory_consolidate` | `shard_id` | Export shard |
| `memory_health` | (none) | Health check |

## CLI Commands

| Command | Args | Description |
|---------|------|-------------|
| `remember` | `item_id content [--tags ...]` | Store memory |
| `recall` | `query [-k N]` | Recall memories |
| `associate` | `node_a node_b [-d delta]` | Reinforce synapse |
| `consolidate` | `shard_id [-o file]` | Export shard |
| `mcp` | (none) | Run MCP stdio server |
| `serve` | `[--port N] [-b]` | Start HTTP server |
| `stop` | (none) | Stop background server |
| `status` | `[--port N]` | Check health |
| `query` | `text [--limit N] [--port N]` | Query via HTTP |
| `seed` | `--source path [--dry-run]` | Seed from markdown |
| `info` | (none) | Display status |
