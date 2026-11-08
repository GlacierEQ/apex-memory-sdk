from __future__ import annotations

import tempfile
from pathlib import Path

from apex_memory.federation.consolidator import MemoryConsolidator
from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.graph.traversal import GraphTraversal
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork


def test_vector_store_delete_and_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test.db"
        store = VectorStore(db_path=db_file, dimensions=32)

        store.put("V1", "alpha node content")
        store.put("V2", "beta node content")
        assert len(store.list_all()) == 2

        deleted = store.delete("V1")
        assert deleted is True
        assert store.get("V1") is None
        assert len(store.list_all()) == 1

        store.close()


def test_graph_traversal_disconnected():
    graph = KnowledgeGraph()
    graph.add_entity("IslandA", "Node")
    graph.add_entity("IslandB", "Node")

    path = GraphTraversal.shortest_path(graph, "IslandA", "IslandB")
    assert path is None

    empty_sub = GraphTraversal.k_hop_subgraph(graph, "UnknownNode")
    assert empty_sub["entities"] == []


def test_hebbian_cluster_and_pruning():
    net = HebbianNetwork()
    nodes = ["C1", "C2", "C3", "C4"]
    net.fire_cluster(nodes, delta=0.2)
    assert len(net.edges) == 6

    for _ in range(20):
        net.apply_decay(factor=0.7)

    assert len(net.edges) == 0


def test_save_and_load_shard_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(db_path=":memory:", dimensions=32)
        net = HebbianNetwork()
        kg = KnowledgeGraph()

        store.put("E1", "Evidence payload 1")
        consolidator = MemoryConsolidator(store, net, kg)

        out_file = Path(tmpdir) / "shard.json"
        saved = consolidator.save_shard(out_file, "S-100")
        assert saved.exists()
        assert "S-100" in saved.read_text()
