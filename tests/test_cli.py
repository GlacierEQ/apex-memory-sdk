from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from apex_memory.cli.main import build_parser, main


def test_cli_parser():
    parser = build_parser()

    args = parser.parse_args(["remember", "FACT-01", "Memory content", "--tags", "tag1", "tag2"])
    assert args.subcommand == "remember"
    assert args.item_id == "FACT-01"
    assert args.tags == ["tag1", "tag2"]

    args = parser.parse_args(["recall", "test search", "-k", "5"])
    assert args.subcommand == "recall"
    assert args.query == "test search"
    assert args.top_k == 5

    args = parser.parse_args(["associate", "A", "B", "-d", "0.3"])
    assert args.subcommand == "associate"
    assert args.node_a == "A"
    assert args.node_b == "B"
    assert args.delta == 0.3

    args = parser.parse_args(["consolidate", "SHARD-01", "-o", "shard.json"])
    assert args.subcommand == "consolidate"
    assert args.shard_id == "SHARD-01"
    assert args.output == "shard.json"

def test_cli_execution_info(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apex-memory", "info"])
    main()
    out = capsys.readouterr().out
    assert "APEX Omniversal Semantic Memory SDK" in out
    assert "SQLite Vector Store" in out

def test_cli_execution_remember_and_associate(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = str(Path(tmpdir) / "test_mem.sqlite3")
        monkeypatch.setenv("APEX_MEMORY_DB", db_file)

        monkeypatch.setattr(sys, "argv", ["apex-memory", "remember", "ITEM-1", "Test content payload"])
        main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "STORED"
        assert data["item_id"] == "ITEM-1"

        monkeypatch.setattr(sys, "argv", ["apex-memory", "associate", "ConceptA", "ConceptB", "-d", "0.5"])
        main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "REINFORCED"
        assert data["edge"]["weight"] == 0.5
