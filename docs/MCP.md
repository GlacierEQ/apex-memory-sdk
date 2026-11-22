# MCP Server Documentation

## Overview

APEX Memory SDK exposes a JSON-RPC 2.0 MCP server for agent integration.

## Start the Server

### Stdio mode (MCP default)
```bash
apex-memory mcp
```

### HTTP mode
```bash
apex-memory serve --port 18421
# Background:
apex-memory serve --background --port 18421
apex-memory status --port 18421
apex-memory stop
```

## Endpoints (HTTP mode)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check: status + memory count |
| POST | `/query` | Query memories: `{"text": "...", "limit": 5}` |
| POST | `/add` | Add memory: `{"item_id": "...", "content": "...", "tags": [...]}` |

## MCP Tools

| Tool | Required Args | Description |
|------|---------------|-------------|
| `memory_remember` | `item_id`, `content` | Store memory item |
| `memory_recall` | `query` | Recall via hybrid search |
| `memory_associate` | `node_a`, `node_b` | Reinforce Hebbian synapse |
| `memory_graph_add` | `source`, `relation`, `target` | Add graph relation |
| `memory_graph_query` | — | Query knowledge graph |
| `memory_consolidate` | `shard_id` | Export cryptographic shard |
| `memory_health` | — | Health check |

## MCP Configuration

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

## Health Check

```bash
# CLI
apex-memory status --port 18421

# HTTP
curl http://localhost:18421/health
# → {"status": "ok", "memory_count": 42}

# MCP tool
{"method": "tools/call", "params": {"name": "memory_health", "arguments": {}}}
```
