from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from apex_memory.federation.consolidator import MemoryConsolidator
from apex_memory.mcp.server import MemoryMCPServer
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork
from apex_memory.synapse.network import AssociativeRecallEngine

DEFAULT_PORT = 18421
DEFAULT_DB = str(Path.home() / ".apex" / "memory.sqlite3")
CONFIG_FILE = ".apex-memory.json"


def load_config() -> dict:
    p = Path(CONFIG_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_config(config: dict) -> None:
    Path(CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")


def port_is_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def server_state_file(config: dict) -> Path:
    db_path = config.get("db_path", DEFAULT_DB)
    return Path(db_path).parent / "server.json"


def load_server_state(config: dict) -> dict | None:
    sf = server_state_file(config)
    try:
        state = json.loads(sf.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(state, dict):
        return None
    if not isinstance(state.get("pid"), int) or state["pid"] < 1:
        return None
    if not isinstance(state.get("port"), int) or not (1 <= state["port"] <= 65535):
        return None
    return state


def save_server_state(pid: int, port: int, config: dict) -> None:
    sf = server_state_file(config)
    sf.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=sf.parent,
            prefix=f".{sf.name}.", suffix=".tmp", delete=False,
        ) as h:
            h.write(json.dumps({"pid": pid, "port": port}) + "\n")
            tmp = Path(h.name)
        tmp.replace(sf)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def clear_server_state(config: dict) -> None:
    try:
        server_state_file(config).unlink()
    except FileNotFoundError:
        pass


def wait_for_server(port: int, process: subprocess.Popen, timeout: int = 30) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"Error: server exited before readiness (exit {process.returncode}).")
            return False
        try:
            resp = urllib.request.urlopen(url, timeout=1)
            payload = json.loads(resp.read())
            if payload.get("status") == "ok":
                return True
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    print(f"Error: server did not become ready within {timeout} seconds.")
    return False


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

    # Lifecycle commands transcribed from zer0dex
    serve_p = sub.add_parser("serve", help="Start the memory HTTP server")
    serve_p.add_argument("--port", type=int, help="Server port")
    serve_p.add_argument("--background", "-b", action="store_true", help="Run in background")

    sub.add_parser("stop", help="Stop the project-managed background server")

    status_p = sub.add_parser("status", help="Check server health")
    status_p.add_argument("--port", type=int)

    query_p = sub.add_parser("query", help="Query memories from CLI")
    query_p.add_argument("text", help="Query text")
    query_p.add_argument("--limit", type=int, default=5)
    query_p.add_argument("--port", type=int)

    # Markdown seeding transcribed from zer0dex
    seed_p = sub.add_parser("seed", help="Seed memory store from markdown files")
    seed_p.add_argument("--source", action="append", required=True, help="Source file or directory")
    seed_p.add_argument("--dry-run", action="store_true", help="Show what would be seeded")

    sub.add_parser("info", help="Display memory status")

    return parser


def cmd_serve(args: argparse.Namespace) -> None:
    config = load_config()
    db_path = config.get("db_path", DEFAULT_DB)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    port = args.port or config.get("port", DEFAULT_PORT)

    if args.background:
        state = load_server_state(config)
        if state is not None:
            if process_is_running(state["pid"]):
                print("Error: a managed background server is already running; use 'stop' first.")
                sys.exit(1)
            clear_server_state(config)
        if port_is_in_use(port):
            print(f"Error: port {port} is already in use.")
            sys.exit(1)

        server_args = [
            sys.executable, "-m", "apex_memory.mcp.server",
            "--port", str(port),
            "--db-path", db_path,
        ]
        proc = subprocess.Popen(
            server_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            save_server_state(proc.pid, port, config)
        except OSError as exc:
            proc.terminate()
            clear_server_state(config)
            print(f"Error: could not record server state: {exc}")
            sys.exit(1)
        try:
            ready = wait_for_server(port, proc)
        except BaseException:
            proc.terminate()
            clear_server_state(config)
            raise
        if not ready:
            proc.terminate()
            clear_server_state(config)
            sys.exit(1)
        print(f"✅ apex-memory server started (PID {proc.pid}, port {port})")
    else:
        # Foreground mode: run MCP server directly
        server = MemoryMCPServer(db_path=db_path)
        server.run_stdio()


def cmd_stop(args: argparse.Namespace) -> None:
    config = load_config()
    state = load_server_state(config)
    if state is None:
        print("No managed background server is running.")
        return

    pid = state["pid"]
    if not process_is_running(pid):
        clear_server_state(config)
        print("Removed stale server state.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        clear_server_state(config)
        print("Removed stale server state.")
        return
    except PermissionError:
        print(f"Error: permission denied stopping PID {pid}.")
        sys.exit(1)

    deadline = time.monotonic() + 5
    while process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_is_running(pid):
        print(f"Error: server PID {pid} did not stop.")
        sys.exit(1)

    clear_server_state(config)
    print(f"✅ Server stopped (PID {pid}).")


def cmd_status(args: argparse.Namespace) -> None:
    config = load_config()
    port = args.port or config.get("port", DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}/health"
    try:
        resp = urllib.request.urlopen(url, timeout=3)
        result = json.loads(resp.read())
        print(f"✅ apex-memory running on port {port}")
        print(f"   Status: {result.get('status', '?')}")
        print(f"   Memories: {result.get('memory_count', '?')}")
    except urllib.error.URLError:
        print(f"❌ apex-memory not running on port {port}")
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    config = load_config()
    port = args.port or config.get("port", DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}/query"
    data = json.dumps({"text": args.text, "limit": args.limit}).encode()
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=5)
        result = json.loads(resp.read())
        memories = result.get("memories", [])
        if not memories:
            print("No relevant memories found.")
            return
        for m in memories:
            score = m.get("score", 0)
            text = m.get("text", "")
            print(f"  [{score:.3f}] {text}")
    except urllib.error.URLError:
        print(f"Error: server not running on port {port}. Run: apex-memory serve -b")
        sys.exit(1)


def cmd_seed(args: argparse.Namespace) -> None:
    """Seed memory store from markdown files."""
    from pathlib import Path as P

    files = []
    for src in args.source:
        p = P(src)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
            files.extend(sorted(p.rglob("*.txt")))
    if not files:
        print("No files found. Use --source <path>")
        sys.exit(1)

    print(f"Found {len(files)} file(s)")
    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        # Simple markdown chunking: split by headers
        chunks = []
        current = []
        for line in text.splitlines():
            if line.startswith("#") and current:
                chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join(current))
        all_chunks.extend(c for c in chunks if c.strip())
        print(f"  {f.name}: {len(chunks)} chunks")

    if args.dry_run:
        print(f"\n[DRY RUN] Would seed {len(all_chunks)} chunks. Exiting.")
        return

    db_path = os.environ.get("APEX_MEMORY_DB", str(Path.home() / ".apex" / "memory.sqlite3"))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = VectorStore(db_path=db_path)
    hebbian = HebbianNetwork()
    engine = AssociativeRecallEngine(store, hebbian)

    total = 0
    for i, chunk in enumerate(all_chunks, 1):
        print(f"  Seeding {i}/{len(all_chunks)}...", end=" ", flush=True)
        try:
            item = store.put(f"seed-{i}", chunk.strip())
            total += 1
            print(f"(stored)")
        except Exception as e:
            print(f"(error: {e})")

    print(f"\n✅ Seeded {total} memories from {len(files)} files.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand or args.subcommand == "info":
        print("APEX Omniversal Semantic Memory SDK v1.0.0")
        print("Substrates: SQLite Vector Store + BM25 Hybrid Ranker + Hebbian Synapse Network")
        print("Graph: Directed Knowledge Graph with Transitive Traversals")
        print("Commands: remember, recall, associate, consolidate, mcp, serve, stop, status, query, seed, info")
        return

    if args.subcommand == "mcp":
        server = MemoryMCPServer()
        server.run_stdio()
        return

    if args.subcommand == "serve":
        cmd_serve(args)
        return

    if args.subcommand == "stop":
        cmd_stop(args)
        return

    if args.subcommand == "status":
        cmd_status(args)
        return

    if args.subcommand == "query":
        cmd_query(args)
        return

    if args.subcommand == "seed":
        cmd_seed(args)
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
