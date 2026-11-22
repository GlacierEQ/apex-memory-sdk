from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from apex_memory.federation.consolidator import MemoryConsolidator
from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork
from apex_memory.synapse.network import AssociativeRecallEngine


class MemoryMCPServer:
    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or os.environ.get(
            "APEX_MEMORY_DB", str(Path.home() / ".apex" / "memory.sqlite3")
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.store = VectorStore(db_path=path)
        self.hebbian = HebbianNetwork()
        self.engine = AssociativeRecallEngine(self.store, self.hebbian)
        self.graph = KnowledgeGraph()
        self.consolidator = MemoryConsolidator(self.store, self.hebbian, self.graph)

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_remember",
                "description": "Store a memory item with vector embeddings, BM25 indexing, and optional synaptic relations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "content": {"type": "string"},
                        "related_ids": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item_id", "content"],
                },
            },
            {
                "name": "memory_recall",
                "description": "Recall memories using hybrid semantic vector search, BM25 keyword matching, and Hebbian synaptic expansion",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 3},
                        "expand_synaptic": {"type": "boolean", "default": True},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_associate",
                "description": "Reinforce Hebbian synaptic connection weight between two memory entities",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node_a": {"type": "string"},
                        "node_b": {"type": "string"},
                        "delta": {"type": "number", "default": 0.2},
                    },
                    "required": ["node_a", "node_b"],
                },
            },
            {
                "name": "memory_graph_add",
                "description": "Add an entity and directed relation to the knowledge graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "source_label": {"type": "string", "default": "Concept"},
                        "relation": {"type": "string"},
                        "target": {"type": "string"},
                        "target_label": {"type": "string", "default": "Concept"},
                    },
                    "required": ["source", "relation", "target"],
                },
            },
            {
                "name": "memory_graph_query",
                "description": "Query relations from the knowledge graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                    },
                },
            },
            {
                "name": "memory_consolidate",
                "description": "Export current memory, synapses, and graph into a cryptographic episodic shard",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "shard_id": {"type": "string"},
                        "label": {"type": "string", "default": "Agent Consolidation"},
                    },
                    "required": ["shard_id"],
                },
            },
            {
                "name": "memory_health",
                "description": "Health check: returns store status, memory count, synapse count, and graph entity count",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "memory_remember":
            item = self.store.put(
                item_id=args["item_id"],
                content=args["content"],
                tags=args.get("tags"),
            )
            rel_ids = args.get("related_ids")
            if rel_ids:
                for r in rel_ids:
                    self.hebbian.fire(args["item_id"], r, delta=0.2)
            return {"status": "STORED", "item_id": item.item_id}

        if name == "memory_recall":
            results = self.engine.recall(
                query=args["query"],
                top_k=args.get("top_k", 3),
                expand_synaptic=args.get("expand_synaptic", True),
            )
            return {"count": len(results), "matches": results}

        if name == "memory_associate":
            edge = self.hebbian.fire(args["node_a"], args["node_b"], delta=args.get("delta", 0.2))
            return {"status": "REINFORCED", "edge": edge.to_dict()}

        if name == "memory_graph_add":
            if not self.graph.get_entity(args["source"]):
                self.graph.add_entity(args["source"], args.get("source_label", "Concept"))
            if not self.graph.get_entity(args["target"]):
                self.graph.add_entity(args["target"], args.get("target_label", "Concept"))
            rel = self.graph.add_relation(args["source"], args["relation"], args["target"])
            return {"status": "RELATION_ADDED", "relation": rel.to_dict()}

        if name == "memory_graph_query":
            rels = self.graph.query(
                subject=args.get("subject"),
                predicate=args.get("predicate"),
                obj=args.get("object"),
            )
            return {"count": len(rels), "relations": [r.to_dict() for r in rels]}

        if name == "memory_consolidate":
            shard = self.consolidator.consolidate(
                args["shard_id"], args.get("label", "Agent Consolidation")
            )
            return {
                "status": "CONSOLIDATED",
                "shard_id": shard["shard_id"],
                "shard_hash": shard["shard_hash"],
                "memory_count": shard["memory_count"],
            }

        if name == "memory_health":
            items = self.store.list_items() if hasattr(self.store, "list_items") else []
            synapses = getattr(self.hebbian, "edges", {})
            entities = self.graph.get_all_entities() if hasattr(self.graph, "get_all_entities") else []
            return {
                "status": "ok",
                "memory_count": len(items),
                "synapse_count": len(synapses),
                "graph_entity_count": len(entities),
                "db_path": str(self.store.db_path) if hasattr(self.store, "db_path") else None,
            }

        raise ValueError(f"Unknown tool: {name}")

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "apex-memory", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.get_tools()},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                res = self.dispatch_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e!s}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def run_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e!s}"},
                }
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


# HTTP server mode (transcribed from zer0dex pattern)
def _run_http_server(port: int, db_path: str) -> None:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import secrets

    server_instance = MemoryMCPServer(db_path=db_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            cl = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(cl)
            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return None

        def do_GET(self):
            if self.path == "/health":
                items = server_instance.store.list_items() if hasattr(server_instance.store, "list_items") else []
                self._send_json({"status": "ok", "memory_count": len(items)})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            data = self._read_body()
            if not isinstance(data, dict):
                self._send_json({"error": "invalid json"}, 400)
                return
            if self.path == "/query":
                text = data.get("text", "")
                limit = data.get("limit", 5)
                try:
                    results = server_instance.engine.recall(text, top_k=limit)
                    self._send_json({"memories": results})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            elif self.path == "/add":
                item_id = data.get("item_id", f"item-{int(time.time())}")
                content = data.get("content", "")
                try:
                    item = server_instance.store.put(item_id, content, tags=data.get("tags"))
                    self._send_json({"status": "STORED", "item_id": item.item_id})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({"error": "not found"}, 404)

    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"apex-memory HTTP server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18421)
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path or os.environ.get("APEX_MEMORY_DB", str(Path.home() / ".apex" / "memory.sqlite3"))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Check if stdin is a tty (interactive) - if not, run HTTP server
    if not sys.stdin.isatty():
        _run_http_server(args.port, db_path)
    else:
        # Run MCP stdio server
        server = MemoryMCPServer(db_path=db_path)
        server.run_stdio()
