from __future__ import annotations

from apex_memory.cli.main import build_parser

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
