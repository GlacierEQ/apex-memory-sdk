from __future__ import annotations

from apex_memory.federation.consolidator import MemoryConsolidator
from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork


def test_memory_consolidation_and_shard_restore():
    store1 = VectorStore(db_path=":memory:", dimensions=64)
    net1 = HebbianNetwork()
    kg1 = KnowledgeGraph()

    store1.put("ITEM1", "First consolidated piece of evidence")
    store1.put("ITEM2", "Second linked piece of evidence")
    net1.fire("ITEM1", "ITEM2", delta=0.5)

    kg1.add_entity("ITEM1", "Evidence")
    kg1.add_entity("ITEM2", "Evidence")
    kg1.add_relation("ITEM1", "CORROBORATES", "ITEM2")

    consolidator = MemoryConsolidator(store1, net1, kg1)
    shard = consolidator.consolidate("SHARD-001")
    assert shard["shard_id"] == "SHARD-001"
    assert "shard_hash" in shard
    assert shard["memory_count"] == 2

    store2 = VectorStore(db_path=":memory:", dimensions=64)
    net2 = HebbianNetwork()
    kg2 = KnowledgeGraph()

    restore_consolidator = MemoryConsolidator(store2, net2, kg2)
    loaded_count = restore_consolidator.load_shard(shard)
    assert loaded_count == 2
    assert store2.get("ITEM1") is not None
    assert store2.get("ITEM2") is not None
    assert len(net2.edges) == 1
    assert len(kg2.entities) == 2
    assert len(kg2.relations) == 1
