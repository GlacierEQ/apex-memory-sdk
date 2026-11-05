from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from apex_memory.federation.consolidator import MemoryConsolidator
from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.mcp.server import MemoryMCPServer
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork
from apex_memory.synapse.network import AssociativeRecallEngine

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apex-memory",
        description="APEX Sovereign Omniversal Semantic Memory CLI",
    )
    sub = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    rem_p = sub.add_parser("remember", help="Store memory item")
    rem_p.add_argument("item_id")
    rem_p.add_argument("content")
    rem_p.add_argument("--tags", nargs="*", default=[])

    rec_p = sub.add_parser("recall", help="Recall memories via hybrid search")
    rec_p.add_argument("query")
    rec_p.add_argument("-k", "--top-k", type=int, default=3)

    assoc_p = sub.add_parser("associate", help="Strengthen synaptic connection")
    assoc_p.add_argument("node_a")
    assoc_p.add_argument("node_b")
    assoc_p.add_argument("-d", "--delta", type=float, default=0.2)

    cons_p = sub.add_parser("consolidate", help="Consolidate memory shard")
    cons_p.add_argument("shard_id")
    cons_p.add_argument("-o", "--output", default="")

    sub.add_parser("mcp", help="Run MCP JSON-RPC 2.0 stdio server")
    sub.add_parser("info", help="Display memory status")

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand or args.subcommand == "info":
        print("APEX Omniversal Semantic Memory SDK v1.0.0")
        print("Substrates: SQLite Vector Store + BM25 Hybrid Ranker + Hebbian Synapse Network")
        print("Graph: Directed Knowledge Graph with Transitive Traversals")
        return

    if args.subcommand == "mcp":
        server = MemoryMCPServer()
        server.run_stdio()
        return

    db_path = os.environ.get("APEX_MEMORY_DB", str(Path.home() / ".apex" / "memory.sqlite3"))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = VectorStore(db_path=db_path)
    hebbian = HebbianNetwork()
    engine = AssociativeRecallEngine(store, hebbian)

    if args.subcommand == "remember":
        item = store.put(args.item_id, args.content, tags=args.tags)
        print(json.dumps({"status": "STORED", "item_id": item.item_id}, indent=2))

    elif args.subcommand == "recall":
        results = engine.recall(args.query, top_k=args.top_k)
        print(json.dumps(results, indent=2))

    elif args.subcommand == "associate":
        edge = hebbian.fire(args.node_a, args.node_b, delta=args.delta)
        print(json.dumps({"status": "REINFORCED", "edge": edge.to_dict()}, indent=2))

    elif args.subcommand == "consolidate":
        consolidator = MemoryConsolidator(store, hebbian)
        shard = consolidator.consolidate(args.shard_id)
        if args.output:
            Path(args.output).write_text(json.dumps(shard, indent=2), encoding="utf-8")
            print(f"Shard saved to {args.output}")
        else:
            print(json.dumps(shard, indent=2))

if __name__ == "__main__":
    main()
