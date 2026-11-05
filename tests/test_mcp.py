from __future__ import annotations

from apex_memory.mcp.server import MemoryMCPServer

def test_memory_mcp_server():
    server = MemoryMCPServer(db_path=":memory:")

    init_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["id"] == 1
    assert init_resp["result"]["serverInfo"]["name"] == "apex-memory"

    tools_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = [t["name"] for t in tools_resp["result"]["tools"]]
    assert "memory_remember" in tools
    assert "memory_recall" in tools
    assert "memory_associate" in tools
    assert "memory_graph_add" in tools
    assert "memory_graph_query" in tools
    assert "memory_consolidate" in tools

    rem_call = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "memory_remember",
            "arguments": {"item_id": "FACT-1", "content": "Autonomous memory persistence"},
        },
    }
    rem_resp = server.handle_request(rem_call)
    assert rem_resp["result"]["isError"] is False

    recall_call = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "memory_recall",
            "arguments": {"query": "memory persistence"},
        },
    }
    recall_resp = server.handle_request(recall_call)
    assert recall_resp["result"]["isError"] is False
